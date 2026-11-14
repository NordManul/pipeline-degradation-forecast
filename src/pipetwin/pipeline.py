"""CLI for audited historical benchmarks, saved-model inference and template export."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .diagnostics import input_shift, prediction_diagnostics
from .experiments import cross_validate, fit_models, group_metrics, predict_models, prediction_table
from .features import attach_labels, build_snapshot
from .geometry import build_grid
from .sources import load_sources, read_config
from .submission import export_submission
from .targets import build_targets
from .uncertainty import calibrate_interval, interval_metrics, predict_interval
from .validation import score_predictions, select_champion


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(val) for val in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, (pd.Timestamp, datetime)):
            return item.isoformat()
        return item
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_data(config):
    contract = read_config(config["contract"])
    sources = load_sources(config["data_dir"], contract)
    grid = build_grid(sources.passport, contract["segment_length_m"])
    targets, quarantine = build_targets(sources, grid, contract)
    return contract, sources, grid, targets, quarantine


def _provenance(config, contract, sources):
    source_dir = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "created_utc": datetime.now(timezone.utc), "git_commit": commit, "git_dirty": dirty,
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "matplotlib", "joblib"]},
        "source_sha256": {str(path.relative_to(source_dir)): sha256(path.read_bytes()).hexdigest() for path in sorted(source_dir.rglob("*.py"))},
        "input_manifest": sources.manifest, "config": config, "contract": contract,
        "evaluation": "historical_reconstruction_on_reported_labels",
        "prediction_bounds": "0 <= prediction_mm <= audited passport nominal_mm",
        "selection": "positive Pearson r, maximum squared Pearson r, then minimum MAE; development OOF only",
        "limitations": [
            "Scoring IDs, dates, official grid and label policy are unconfirmed.",
            "Development is spatial transfer within one transition, not a chronological backtest.",
            "Late and cold targets were audited before implementation; they are not pristine blind tests.",
            "All regime rows must precede T; actual future regime scenarios are not inferred.",
            "Unreported target thickness remains unknown and is excluded from every model's score.",
            "Cold transfer has one training main and one test main; intervals have no coverage guarantee.",
        ],
    }


def _run_benchmark(config, output):
    contract, sources, grid, targets, quarantine = load_data(config)
    if config["spatial_buffer_m"] < config["history_radius_m"]:
        raise ValueError("Spatial buffer must cover feature radius")
    for folder in ["predictions", "bundles", "plots"]:
        (output / folder).mkdir()
    write_json(output / "provenance.json", _provenance(config, contract, sources))
    quarantine.to_csv(output / "quarantine.csv", index=False)
    coverage = targets.groupby(["main_id", "date"], as_index=False).agg(cells=("observed", "size"), observed=("observed", "sum"))
    coverage.to_csv(output / "coverage.csv", index=False)
    prepared, selection, scores, per_group, per_fold, intervals, split_audit = {}, {}, [], [], [], {}, {}
    prediction_tables, training_frames, shifts, saturations = {}, {}, [], []
    # Complete selection/calibration for both scenarios before accessing their test labels.
    for scenario, key in [("between", "development"), ("cold", "cold_train")]:
        snapshot = build_snapshot(sources, grid, targets, config[key], config)
        frame = attach_labels(snapshot, targets)
        print(f"{scenario}: {len(frame)} cells, {frame.observed.sum()} observed targets", flush=True)
        cv = cross_validate(frame, config["models"], scenario, config, announce=True)
        ranking = score_predictions(frame, cv.predictions, "development")
        champion = select_champion(ranking)
        observed = frame.observed.to_numpy(dtype=bool)
        residual_quantiles = calibrate_interval(frame.loc[observed, "y_mm"], cv.predictions[champion][observed])
        models = fit_models(frame, config["models"], scenario, config["seed"])
        bundle = {"scenario": scenario, "models": models, "candidates": config["ensemble_candidates"],
                  "weights": cv.weights, "champion": champion, "residual_quantiles": residual_quantiles,
                  "config": config, "contract": contract}
        joblib.dump(bundle, output / "bundles" / f"{scenario}.joblib", compress=3)
        split = "development" if scenario == "between" else "cold_development"
        scores.append(ranking.assign(split=split))
        per_group.append(group_metrics(frame, cv.predictions, split))
        per_fold.append(cv.fold_metrics.assign(split=split))
        split_audit[scenario] = cv.audit
        table = prediction_table(frame, cv.predictions, split, cv.folds)
        table.to_csv(output / "predictions" / f"{split}.csv", index=False)
        prediction_tables[split] = table
        selection[scenario] = {
            "champion": champion, "weights": dict(zip(config["ensemble_candidates"], cv.weights)),
            "calibration_residual_q10_q90_mm": residual_quantiles,
            "development_score": ranking.loc[ranking.model.eq(champion)].iloc[0].to_dict(),
        }
        prepared[scenario] = bundle
        training_frames[scenario] = frame
        residuals = pd.DataFrame({name: frame.loc[observed, "y_mm"].to_numpy() - values[observed] for name, values in cv.predictions.items()})
        residuals.corr().to_csv(output / f"{scenario}_residual_correlations.csv")
        print(f"{scenario}: frozen selection = {champion}", flush=True)
    selection["frozen_before_test_scoring_utc"] = datetime.now(timezone.utc)
    write_json(output / "selection.json", selection)
    write_json(output / "spatial_folds.json", split_audit)
    selected_tables = []
    for scenario, key, split in [("between", "late", "late"), ("cold", "cold_test", "cold_test")]:
        bundle = prepared[scenario]
        frame = attach_labels(build_snapshot(sources, grid, targets, config[key], config), targets)
        shifts.append(input_shift(training_frames[scenario], frame, scenario, split))
        saturations.append(prediction_diagnostics(bundle["models"], frame, split))
        predictions = predict_models(bundle["models"], frame, bundle["candidates"], bundle["weights"])
        scores.append(score_predictions(frame, predictions, split))
        per_group.append(group_metrics(frame, predictions, split))
        champion = bundle["champion"]
        lower, upper = predict_interval(predictions[champion], bundle["residual_quantiles"], frame.nominal_mm)
        mask = frame.observed.to_numpy(dtype=bool)
        intervals[split] = {"model": champion, **interval_metrics(frame.loc[mask, "y_mm"], lower[mask], upper[mask])}
        table = prediction_table(frame, predictions, split)
        table["lower_80_mm"], table["upper_80_mm"] = lower, upper
        table.to_csv(output / "predictions" / f"{split}.csv", index=False)
        prediction_tables[split] = table
        chosen = frame[["id", "segment_id", "target_date"]].copy()
        chosen["scenario"], chosen["model"] = scenario, champion
        chosen["thickness_pred"] = predictions[champion]
        chosen["lower_80_mm"], chosen["upper_80_mm"] = lower, upper
        selected_tables.append(chosen)
    metrics = pd.concat(scores, ignore_index=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.concat(per_group, ignore_index=True).to_csv(output / "metrics_by_section.csv", index=False)
    pd.concat(per_fold, ignore_index=True).to_csv(output / "metrics_by_fold.csv", index=False)
    pd.concat(selected_tables, ignore_index=True).to_csv(output / "selected_predictions.csv", index=False)
    write_json(output / "intervals.json", intervals)
    pd.concat(shifts, ignore_index=True).to_csv(output / "feature_shift.csv", index=False)
    pd.concat(saturations, ignore_index=True).to_csv(output / "prediction_diagnostics.csv", index=False)
    from .reporting import create_report
    create_report(output, metrics, prediction_tables, selection, coverage, intervals)
    return metrics, selection


def run_benchmark(config_path, output_dir):
    config, output = read_config(config_path), Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Run output already exists; select a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    # Adapted from the reviewed E02 publication workflow: publish only a complete run.
    with tempfile.TemporaryDirectory(prefix=".pipetwin-", dir=output.parent) as staging:
        staged = Path(staging) / "run"
        staged.mkdir()
        with threadpool_limits(limits=1):
            result = _run_benchmark(config, staged)
        staged.rename(output)
    return result


def run_prediction(bundle_path, spec_path, output_path, raw_dir=None):
    bundle = joblib.load(bundle_path)
    config = dict(bundle["config"])
    if raw_dir:
        config["data_dir"] = str(raw_dir)
    _, sources, grid, targets, _ = load_data(config)
    if read_config(config["contract"]) != bundle["contract"]:
        raise ValueError("Data contract changed since training; retrain or restore the original contract")
    spec = read_config(spec_path)
    expected = "between" if "anchor_date" in spec else "cold"
    if bundle["scenario"] != expected:
        raise ValueError("Bundle scenario does not match the prediction specification")
    frame = build_snapshot(sources, grid, targets, spec, config)
    predictions = predict_models(bundle["models"], frame, bundle["candidates"], bundle["weights"])
    chosen = predictions[bundle["champion"]]
    lower, upper = predict_interval(chosen, bundle["residual_quantiles"], frame.nominal_mm)
    output = Path(output_path)
    if output.exists():
        raise ValueError("Prediction output exists; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = frame[["id", "segment_id", "target_date"]].assign(thickness_pred=chosen, lower_80_mm=lower, upper_80_mm=upper)
    result.to_csv(output, index=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser("prepare-cases", help="Clean and prepare both official case inputs without future data")
    preparation.add_argument("--data-dir", default="data/raw")
    preparation.add_argument("--contracts-dir", default="contracts")
    preparation.add_argument("--config", default="configs/forecast.yaml")
    preparation.add_argument("--output", required=True)
    official = sub.add_parser("case-benchmark", help="Compare models on shared mentor-confirmed case inputs")
    official.add_argument("--prepared", required=True)
    official.add_argument("--output", required=True)
    official.add_argument("--config", default="configs/case_benchmark.yaml")
    official.add_argument("--contracts-dir", default="contracts")
    official.add_argument("--data-dir", default="data/raw")
    transfer = sub.add_parser("case2-transfer", help="Fit case 2 using other mains under the retrospective protocol")
    transfer.add_argument("--data-dir", default="data/raw")
    transfer.add_argument("--contracts-dir", default="contracts")
    transfer.add_argument("--config", default="configs/forecast.yaml")
    transfer.add_argument("--output", required=True)
    benchmark = sub.add_parser("benchmark", help="Run all historical comparisons and charts")
    benchmark.add_argument("--config", default="configs/forecast.yaml")
    benchmark.add_argument("--output", required=True)
    prediction = sub.add_parser("predict", help="Predict from a locally trained bundle")
    prediction.add_argument("--bundle", required=True)
    prediction.add_argument("--spec", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--raw-dir")
    export = sub.add_parser("export", help="Validate exact template keys and write result.csv")
    export.add_argument("--predictions", required=True)
    export.add_argument("--template", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-cases":
            from .case_data import prepare_cases
            summaries = prepare_cases(args.data_dir, args.output, args.contracts_dir, args.config)
            print(pd.DataFrame(summaries).to_string(index=False))
        elif args.command == "case-benchmark":
            from .case_experiments import run_case_benchmark
            metrics, selection = run_case_benchmark(args.prepared, args.output, args.config, args.contracts_dir, args.data_dir)
            print(metrics.loc[metrics.output.eq("bounded")].to_string(index=False))
            print(f"Report: {Path(args.output).resolve() / 'report.html'}")
        elif args.command == "case2-transfer":
            from .case2_transfer import run
            run(args.data_dir, args.output, args.contracts_dir, args.config)
        elif args.command == "benchmark":
            metrics, selection = run_benchmark(args.config, args.output)
            print(metrics.to_string(index=False))
            print(f"Report: {Path(args.output).resolve() / 'report.html'}")
        elif args.command == "predict":
            result = run_prediction(args.bundle, args.spec, args.output, args.raw_dir)
            print(f"Wrote {len(result)} predictions to {Path(args.output).resolve()}")
        else:
            predictions = pd.read_csv(args.predictions, dtype={"id": str})
            template = pd.read_csv(args.template, dtype={"id": str})
            result = export_submission(predictions, template, args.output)
            print(f"Exported {len(result)} validated rows to {Path(args.output).resolve()}")
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
