"""Common case-cutoff benchmark: primary models, E03, E04 and fixed blends."""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .agent2_physics import make_model as make_physics
from .case_data import model_views
from .diagnostics import input_shift
from .matching import MatchingConfig, match_campaigns
from .models import make_model
from .models.matched_rates import MatchedRateModel
from .quality import load_case_sources
from .sources import read_config
from .validation import regression_metrics, select_champion


LABEL_COLUMNS = ["y_mm", "observed", "coverage_status", "label_available_at"]
KEY_COLUMNS = ["case_id", "sample_id", "id", "segment_id", "target_date"]


def query_view(frame):
    return frame.drop(columns=LABEL_COLUMNS, errors="ignore").copy()


def validate_package(package):
    """Fail closed on a stale historical bundle or overlapping/partial folds."""
    train, query = package["train_frame"], package["predict_frame"]
    cutoff = pd.Timestamp(package["cutoff_exclusive"])
    for frame in [train, query]:
        if frame.sample_id.duplicated().any() or not frame.case_id.eq(package["case_id"]).all():
            raise ValueError("Invalid case or duplicate sample keys")
        if not frame.scenario.eq(package["scenario"]).all():
            raise ValueError("Mixed scenarios")
    if not train.target_date.lt(cutoff).all() or not query.target_date.eq(cutoff).all():
        raise ValueError("Case cutoff violated")
    observed = train.loc[train.observed]
    if observed.empty or not observed.label_available_at.lt(cutoff).all():
        raise ValueError("Unavailable or absent training labels")
    if not np.isfinite(observed.y_mm).all() or observed.y_mm.le(0).any():
        raise ValueError("Invalid observed training labels")
    if any(column in query for column in LABEL_COLUMNS):
        raise ValueError("Official prediction inputs must not contain target labels")
    if set(package["feature_columns"]) & set(LABEL_COLUMNS):
        raise ValueError("Target metadata cannot be a model feature")
    if package["config"]["spatial_buffer_m"] < package["config"]["history_radius_m"]:
        raise ValueError("Fold buffer is smaller than history feature support")
    seen, numbers = np.zeros(len(train), dtype=int), set()
    for fold in package["folds"]:
        a, b = np.asarray(fold["train"], dtype=int), np.asarray(fold["valid"], dtype=int)
        if fold["number"] in numbers or not len(a) or not len(b):
            raise ValueError("Duplicate or empty folds")
        numbers.add(fold["number"])
        if min(a.min(), b.min()) < 0 or max(a.max(), b.max()) >= len(train):
            raise ValueError("Fold index out of range")
        if len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) & set(b):
            raise ValueError("Fold indices overlap or repeat")
        distance = np.abs(train.iloc[a].chain_position_m.to_numpy()[:, None] - train.iloc[b].chain_position_m.to_numpy()[None, :])
        if distance.min() < package["config"]["spatial_buffer_m"]:
            raise ValueError("Fold violates the spatial buffer")
        seen[b] += 1
    if not np.all(seen == 1):
        raise ValueError("Every training cell must have exactly one held-out prediction")


@dataclass
class CaseModel:
    estimator: object
    name: str
    case_id: str
    scenario: str
    cutoff: pd.Timestamp
    fallback_fraction: float

    def predict(self, frame, *, development=False):
        query = query_view(frame)
        if not query.case_id.eq(self.case_id).all() or not query.scenario.eq(self.scenario).all():
            raise ValueError("Fitted model belongs to a different case/scenario")
        dates = pd.to_datetime(query.target_date)
        if dates.isna().any() or dates.gt(self.cutoff).any():
            raise ValueError("Prediction date exceeds the fitted case cutoff")
        query["fallback_mm"] = query.nominal_mm * self.fallback_fraction
        if isinstance(self.estimator, MatchedRateModel):
            raw = self.estimator.predict(query, spatial_development=development)
        else:
            raw = self.estimator.predict(query)
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (len(query),) or not np.isfinite(raw).all():
            raise ValueError(f"{self.name} did not return a finite prediction for every requested cell")
        if self.scenario == "between":
            raw = np.where(query.anchor_observed, raw, query.fallback_mm)
        return raw


def fit_case_model(name, package, train, fallback, config, *, matched=None):
    observed = train.loc[train.observed].copy()
    if observed.empty:
        raise ValueError("A fold has no observed training labels")
    cutoff, scenario = pd.Timestamp(package["cutoff_exclusive"]), package["scenario"]
    if not observed.target_date.lt(cutoff).all() or not observed.label_available_at.lt(cutoff).all():
        raise ValueError("Training labels violate the case cutoff")
    # Root rate learners must not invent rates from nominal anchor placeholders.
    # Agent2 performs this check internally and reports its omitted-anchor count.
    direct_target = name == "ridge_direct" or name.startswith("e02_")
    eligible = observed.loc[observed.anchor_observed] if scenario == "between" and not direct_target else observed
    started = perf_counter()
    if name.startswith("e04_"):
        if scenario != "between" or matched is None:
            raise ValueError("E04 requires eligible historical pairs in a between case")
        variant, kind = name.removeprefix("e04_").split("__", 1)
        options, pairs = matched["config"], matched[variant].pairs
        estimator = MatchedRateModel(kind, options["min_rate_segments"], options["rate_bandwidth_m"], options["rate_prior_weight"])
        estimator.fit_pairs(pairs, train, package["training_date"])
        diagnostics = {"calibration_pairs": estimator.calibration_pair_count_, "calibration_segments": estimator.calibration_segment_count_,
                       "sparse_pair_fallback": estimator.fallback_used_, "global_rate_mm_year": estimator.rate_}
    elif name.startswith("agent2_"):
        estimator = make_physics(name.removeprefix("agent2_"), scenario=scenario, exposure=package["exposure"],
                                 cutoff_exclusive=package["cutoff_exclusive"], feature_columns=package["feature_columns"],
                                 seed=config["seed"], options=config["agent2_options"])
        estimator.fit(observed, observed.y_mm.to_numpy())
        diagnostics = dict(estimator.diagnostics_)
    else:
        if eligible.empty:
            raise ValueError("No observed anchors for warm model calibration")
        estimator = make_model(name, scenario, config["seed"]).fit(eligible, eligible.y_mm.to_numpy())
        diagnostics = {"fit_rows": len(eligible)}
    info = {"model": name, "fit_seconds": perf_counter() - started, "observed_train_rows": len(observed),
            "observed_train_anchors": int(observed.anchor_observed.sum()), "fallback": fallback, "diagnostics": diagnostics}
    return CaseModel(estimator, name, package["case_id"], scenario, cutoff, fallback["fraction"]), info


def predict_models(models, frame, blends, *, development=False):
    nominal = frame.nominal_mm.to_numpy(dtype=float)
    raw = {name: model.predict(frame, development=development) for name, model in models.items()}
    bounded = {name: np.clip(values, 0, nominal) for name, values in raw.items()}
    for name, weights in blends.items():
        if name in models or not weights or not set(weights).issubset(models):
            raise ValueError("Blend names/constituents must refer to distinct fitted base models")
        if any(not np.isfinite(w) or w < 0 for w in weights.values()) or not np.isclose(sum(weights.values()), 1):
            raise ValueError("Fixed blend weights must be nonnegative and sum to one")
        # Declared blend operates on each constituent's common bounded output.
        # Save the raw-component blend separately for extrapolation diagnostics.
        raw[name] = sum(raw[key] * w for key, w in weights.items())
        bounded[name] = sum(bounded[key] * w for key, w in weights.items())
    return raw, bounded


def metric_table(frame, raw, bounded, *, split="development", fold_id=-1):
    mask = frame.observed.to_numpy(dtype=bool)
    records = []
    for output, predictions in [("raw", raw), ("bounded", bounded)]:
        for name, values in predictions.items():
            metrics = regression_metrics(frame.loc[mask, "y_mm"], values[mask])
            records.append({"case_id": frame.case_id.iloc[0], "split": split, "fold_id": fold_id, "output": output,
                            "model": name, **metrics, "correlation_status": "defined" if np.isfinite(metrics["pearson_r"]) else "undefined_constant_target_or_prediction"})
    return pd.DataFrame(records)


def prediction_table(frame, raw, bounded, fold_ids):
    table = frame[KEY_COLUMNS].copy()
    table["fold_id"] = fold_ids
    if "observed" in frame:
        table["observed"], table["y_mm"] = frame.observed, frame.y_mm
    return pd.concat([table.assign(model=name, raw_prediction_mm=raw[name], prediction_mm=values)
                      for name, values in bounded.items()], ignore_index=True)


def range_diagnostics(frame, raw, bounded):
    nominal = frame.nominal_mm.to_numpy()
    return pd.DataFrame([{"case_id": frame.case_id.iloc[0], "model": name, "cells": len(frame),
                          "raw_min_mm": float(raw[name].min()), "raw_max_mm": float(raw[name].max()),
                          "bounded_min_mm": float(values.min()), "bounded_max_mm": float(values.max()),
                          "fraction_above_nominal": float((raw[name] > nominal).mean()),
                          "fraction_below_zero": float((raw[name] < 0).mean()),
                          "constant_after_bounds": bool(np.std(values) <= 1e-12)} for name, values in bounded.items()])


def matching_for_fold(package, train, sources, options, directory):
    from .pipeline import write_json
    matched = {"config": options}
    directory.mkdir(parents=True)
    for variant in options["predictive_variants"]:
        result = match_campaigns(sources.inspection, sources.passport, train.anchor_date.iloc[0],
                                 package["training_date"], train.id.unique(), MatchingConfig(**options["matching"]),
                                 available_before=package["cutoff_exclusive"], variant=variant, allowed_cells=train)
        matched[variant] = result
        for kind in ["pairs", "anchors", "unmatched"]:
            getattr(result, kind).to_csv(directory / f"{variant}_{kind}.csv.gz", index=False)
        write_json(directory / f"{variant}_summary.json", result.summary)
    return matched


def run_case(package, config, output, matching_sources=None):
    from .pipeline import write_json
    validate_package(package)
    output = Path(output)
    output.mkdir()
    names = list(config["models"])
    matching_options = None
    if package["scenario"] == "between" and config["include_matching"]:
        if matching_sources is None or matching_sources.manifest != package["input_manifest"]:
            raise ValueError("E04 source hashes differ from the prepared inputs")
        matching_options = read_config(config["matching_config"])
        names.extend(f"e04_{v}__{k}" for v in matching_options["predictive_variants"] for k in matching_options["rate_models"])
    if len(names) != len(set(names)):
        raise ValueError("Duplicate benchmark model names")
    blends = config["fixed_blends"]
    all_names = names + list(blends)
    n = len(package["train_frame"])
    oof_raw = {name: np.full(n, np.nan) for name in all_names}
    oof = {name: np.full(n, np.nan) for name in all_names}
    assignments = np.full(n, -1)
    fold_metrics, audit = [], []
    for fold in package["folds"]:
        print(f"{package['case_id']}: common spatial fold {fold['number'] + 1}/{len(package['folds'])}", flush=True)
        train, valid, fallback = model_views(package, fold["number"])
        matched = matching_for_fold(package, train, matching_sources, matching_options, output / "matching" / f"fold_{fold['number']}") if matching_options else None
        models, fit_info = {}, []
        for name in names:
            models[name], info = fit_case_model(name, package, train, fallback, config, matched=matched)
            fit_info.append(info)
        raw, bounded = predict_models(models, valid, blends, development=True)
        for name in all_names:
            oof_raw[name][fold["valid"]], oof[name][fold["valid"]] = raw[name], bounded[name]
        assignments[fold["valid"]] = fold["number"]
        fold_metrics.append(metric_table(valid, raw, bounded, fold_id=fold["number"]))
        audit.append({"fold_id": fold["number"], "train_sample_ids": train.sample_id.tolist(),
                      "valid_sample_ids": valid.sample_id.tolist(), "fallback": fallback, "fits": fit_info})
    if (assignments < 0).any() or any(not np.isfinite(values).all() for values in [*oof.values(), *oof_raw.values()]):
        raise ValueError("Incomplete OOF predictions")
    metrics = metric_table(package["train_frame"], oof_raw, oof)
    champion = select_champion(metrics.loc[metrics.output.eq("bounded")])
    # Freeze OOF selection BEFORE final prediction and its range diagnostics.
    selection = {"case_id": package["case_id"], "model": champion, "selection": config["selection"],
                 "frozen_utc": datetime.now(timezone.utc), "evaluation": package["evaluation"],
                 "score": metrics.loc[metrics.output.eq("bounded") & metrics.model.eq(champion)].iloc[0].to_dict()}
    write_json(output / "selection.json", selection)
    train, query, fallback = model_views(package)
    matched = matching_for_fold(package, train, matching_sources, matching_options, output / "matching" / "final") if matching_options else None
    models, final_fit = {}, []
    for name in names:
        models[name], info = fit_case_model(name, package, train, fallback, config, matched=matched)
        final_fit.append(info)
    raw, bounded = predict_models(models, query, blends)
    diagnostics = range_diagnostics(query, raw, bounded)
    chosen_diagnostics = diagnostics.loc[diagnostics.model.eq(champion)].iloc[0].to_dict()
    selection["forecast_diagnostics"] = chosen_diagnostics
    selection["forecast_review_required"] = bool(chosen_diagnostics["constant_after_bounds"] or
                                                    chosen_diagnostics["fraction_above_nominal"] == 1 or chosen_diagnostics["fraction_below_zero"] == 1)
    selection["submission_status"] = "provisional_grid_and_unvalidated_target_transfer"
    bundle = {"models": models, "fixed_blends": blends, "selection": selection, "fallback": fallback,
              "case_id": package["case_id"], "scenario": package["scenario"], "cutoff_exclusive": package["cutoff_exclusive"],
              "contract_version": package["contract_version"], "contract_sha256": package["contract_sha256"],
              "input_manifest": package["input_manifest"], "benchmark_config": config}
    joblib.dump(bundle, output / "models.joblib", compress=3)
    replay = joblib.load(output / "models.joblib")
    replay_raw, replay_bounded = predict_models(replay["models"], query, replay["fixed_blends"])
    for name in all_names:
        np.testing.assert_allclose(replay_raw[name], raw[name], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(replay_bounded[name], bounded[name], rtol=1e-12, atol=1e-12)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.concat(fold_metrics, ignore_index=True).to_csv(output / "metrics_by_fold.csv", index=False)
    prediction_table(package["train_frame"], oof_raw, oof, assignments).assign(contract_version=package["contract_version"]).to_csv(output / "oof_predictions.csv.gz", index=False)
    prediction_table(query, raw, bounded, -1).assign(contract_version=package["contract_version"]).to_csv(output / "forecast_predictions.csv.gz", index=False)
    diagnostics.to_csv(output / "forecast_diagnostics.csv", index=False)
    input_shift(train, query, package["scenario"], "official_request").to_csv(output / "feature_shift.csv", index=False)
    selected = query[KEY_COLUMNS + ["center_m", "nominal_mm", "anchor_mm", "anchor_observed"]].assign(model=champion, thickness_pred=bounded[champion])
    selected.to_csv(output / "selected_provisional_predictions.csv", index=False)
    write_json(output / "selection.json", selection)
    write_json(output / "fit_audit.json", {"folds": audit, "final": final_fit, "fallback": fallback})
    print(f"{package['case_id']}: OOF selection = {champion}; forecast review required = {selection['forecast_review_required']}", flush=True)
    return metrics, selection, selected


def run_case_benchmark(prepared_dir, output_dir, config_path="configs/case_benchmark.yaml", contracts_dir="contracts", data_dir="data/raw"):
    from .pipeline import write_json
    prepared, output = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    config = read_config(config_path)
    if output.exists():
        raise ValueError("Benchmark output exists; use a new run directory")
    if config["selection"] != "positive_r_then_pearson_squared_then_mae" or config["between_missing_anchor_policy"] != "shared_train_median_fraction":
        raise ValueError("Unsupported selection or fallback policy")
    summary = json.loads((prepared / "summary.json").read_text(encoding="utf-8"))
    if not summary["completed"]:
        raise ValueError("Prepared inputs are incomplete")
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(Path(contracts_dir).glob("*.yaml"))}
    if hashes != summary["contract_sha256"]:
        raise ValueError("Contracts changed since preparation; run prepare-cases again")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    from .case_reporting import create_case_report
    source = Path(__file__).resolve().parent
    provenance = {"created_utc": datetime.now(timezone.utc), "python": platform.python_version(),
                  "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
                  "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()),
                  "source_sha256": {str(p.relative_to(source)): sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob("*.py"))},
                  "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib", "matplotlib"]},
                  "prepared_dir": str(prepared), "contract_sha256": hashes, "config": config,
                  "config_sha256": sha256(Path(config_path).read_bytes()).hexdigest()}
    if config["include_matching"]:
        provenance["matching_config"] = read_config(config["matching_config"])
    tables, selections, forecasts = [], {}, []
    with tempfile.TemporaryDirectory(prefix=".case-benchmark-", dir=output.parent) as temp:
        staging = Path(temp) / "run"
        staging.mkdir()
        with threadpool_limits(limits=1):
            for case in summary["cases"]:
                package_path = prepared / case["case_id"] / "prepared.joblib"
                package = joblib.load(package_path)
                if package["contract_sha256"] != hashes or package["case_id"] != case["case_id"]:
                    raise ValueError("Case package provenance mismatch")
                provenance.setdefault("package_sha256", {})[case["case_id"]] = sha256(package_path.read_bytes()).hexdigest()
                sources = load_case_sources(data_dir, package["case_id"], contracts_dir).sources if config["include_matching"] and package["scenario"] == "between" else None
                metrics, selection, forecast = run_case(package, config, staging / package["case_id"], sources)
                tables.append(metrics)
                selections[package["case_id"]] = selection
                forecasts.append(forecast)
        metrics, forecast = pd.concat(tables, ignore_index=True), pd.concat(forecasts, ignore_index=True)
        if forecast.duplicated(["case_id", "id", "segment_id"]).any():
            raise ValueError("Duplicate official forecast keys")
        metrics.to_csv(staging / "metrics.csv", index=False)
        forecast.to_csv(staging / "selected_provisional_predictions.csv", index=False)
        write_json(staging / "provenance.json", provenance)
        write_json(staging / "selection.json", selections)
        create_case_report(staging, metrics, forecast, selections)
        write_json(staging / "summary.json", {"completed": True, "forecast_cells": len(forecast),
                                               "selection": selections, "status": "provisional_not_official_submission"})
        staging.rename(output)
    return metrics, selections
