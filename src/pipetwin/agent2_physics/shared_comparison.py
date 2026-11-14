"""Agent 2 comparison using the owner's prepared data, folds and evaluator.

No loader, target policy, scoring mask, fold generator or metric is implemented
here. The shared run_case does the fitting, bounds, scoring and model replay.
"""

import argparse
from datetime import datetime, timezone
import gc
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import tempfile
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from pipetwin.case_data import model_views
from pipetwin.case_experiments import predict_models, run_case, validate_package
from pipetwin.pipeline import write_json


SHARED_BASE = "d443e4fd3f6b217897f1d1e1d8bb6ee6cf8a052e"
PHYSICS = ["agent2_constant_rate", "agent2_positive_rate", "agent2_hybrid_residual", "agent2_hybrid_scale"]
CONTROLS = ["persistence", "global_rate", "gam", "catboost"]
GUARD_OPTIONS = {"residual_guard": "train-bounded", "residual_z_limit": 3.0,
                 "residual_min_relative_scale": 1e-6}


def comparison_config(policy, variant):
    """Fixed before looking at held predictions; no tuning or automatic retries."""
    if policy not in {"observed-only", "train-median"} or variant not in {"original", "train-bounded"}:
        raise ValueError("Unknown comparison variant")
    return {
        "seed": 42, "models": CONTROLS + PHYSICS if variant == "original" else ["agent2_hybrid_residual"],
        "agent2_options": {"missing_policy": policy, **(GUARD_OPTIONS if variant == "train-bounded" else {})},
        "fixed_blends": {}, "include_matching": False,
        "selection": "positive_r_then_pearson_squared_then_mae",
        "between_missing_anchor_policy": "shared_train_median_fraction",
    }


def verify_shared_predictions(package, table, *, forecast=False):
    """Assert the saved evaluator output has exactly the supplied keys and mask."""
    reference = package["predict_frame" if forecast else "train_frame"].set_index("sample_id")
    assignments = pd.Series(-1, index=reference.index)
    if not forecast:
        for fold in package["folds"]:
            assignments.iloc[fold["valid"]] = fold["number"]
    if table.empty:
        raise ValueError("Missing saved predictions")
    for _, group in table.groupby("model", sort=False):
        if group.sample_id.duplicated().any() or set(group.sample_id) != set(reference.index):
            raise ValueError("Saved model predictions changed shared keys")
        group = group.set_index("sample_id").loc[reference.index]
        for column in ["id", "case_id", "segment_id"]:
            if not group[column].astype(str).equals(reference[column].astype(str)):
                raise ValueError(f"Saved predictions changed {column}")
        # CSV parsing may change datetime64[s] to datetime64[ns] without changing
        # a single timestamp. Compare values at a common resolution.
        saved_dates = pd.to_datetime(group.target_date).to_numpy(dtype="datetime64[ns]")
        expected_dates = pd.to_datetime(reference.target_date).to_numpy(dtype="datetime64[ns]")
        if not np.array_equal(saved_dates, expected_dates):
            raise ValueError("Saved predictions changed target dates")
        if not np.array_equal(group.fold_id.to_numpy(), assignments.to_numpy()):
            raise ValueError("Saved predictions changed the shared fold assignment")
        if not forecast:
            if not np.array_equal(group.observed.to_numpy(), reference.observed.to_numpy()):
                raise ValueError("Saved predictions changed the common observed mask")
            np.testing.assert_allclose(group.y_mm, reference.y_mm, equal_nan=True, atol=1e-12)
        if not np.isfinite(group[["raw_prediction_mm", "prediction_mm"]].to_numpy()).all():
            raise ValueError("Saved model predictions are not complete and finite")


def run(prepared_dir, output_dir, contracts_dir="contracts"):
    prepared, output = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Choose a new comparison output directory")
    summary = json.loads((prepared / "summary.json").read_text(encoding="utf-8"))
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(Path(contracts_dir).glob("*.yaml"))}
    if not summary["completed"] or summary["contract_sha256"] != hashes:
        raise ValueError("Prepared inputs are incomplete or have stale contracts")
    configs = {f"{policy}/{variant}": comparison_config(policy, variant)
               for policy in ["observed-only", "train-median"] for variant in ["original", "train-bounded"]}
    source = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    provenance = {
        "created_utc": datetime.now(timezone.utc), "git_commit": commit, "shared_base": SHARED_BASE,
        "contract_version": summary["contract_version"], "contract_sha256": hashes, "configs": configs,
        "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "catboost"]}},
        "source_sha256": {str(p.relative_to(source)): sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob("*.py"))},
        "prepared_packages": {}, "evaluation": "shared_auxiliary_spatial_not_independent_transfer",
        "real_target_labels_available": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tables = {name: [] for name in ["metrics", "metrics_by_fold", "forecast_diagnostics", "oof_predictions", "forecast_predictions"]}
    runs = []
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix=".physics-comparison-", dir=output.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        # The configuration is written before any call to the common evaluator.
        write_json(staging / "frozen_configs.json", configs)
        with threadpool_limits(limits=1):
            for spec in summary["cases"]:
                case_id = spec["case_id"]
                path = prepared / case_id / "prepared.joblib"
                package = joblib.load(path)
                if package["case_id"] != case_id or package["contract_sha256"] != hashes:
                    raise ValueError("Prepared package provenance mismatch")
                validate_package(package)
                provenance["prepared_packages"][case_id] = {"sha256": sha256(path.read_bytes()).hexdigest(),
                                                          "input_manifest": package["input_manifest"]}
                for run_name, config in configs.items():
                    policy, variant = run_name.split("/")
                    directory = staging / policy / variant / case_id
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Agent2: {case_id}, {policy}, {variant}", flush=True)
                    tick = perf_counter()
                    run_case(package, config, directory)
                    elapsed = perf_counter() - tick
                    for name in tables:
                        extension = ".csv.gz" if name.endswith("predictions") else ".csv"
                        table = pd.read_csv(directory / (name + extension), dtype={"id": str, "sample_id": str})
                        if name.endswith("predictions"):
                            verify_shared_predictions(package, table, forecast=name.startswith("forecast"))
                        tables[name].append(table.assign(missing_policy=policy, variant=variant))
                    # The shared evaluator already replayed the joblib bundle. Also
                    # check the on-disk CSV against that bundle and export guard details.
                    bundle = joblib.load(directory / "models.joblib")
                    _, query, _ = model_views(package)
                    raw, bounded = predict_models(bundle["models"], query, {})
                    saved = tables["forecast_predictions"][-1]
                    replay_errors = {}
                    guard_diagnostics = {}
                    for name, model in bundle["models"].items():
                        part = saved.loc[saved.model.eq(name)].set_index("sample_id").loc[query.sample_id]
                        np.testing.assert_allclose(part.raw_prediction_mm, raw[name], rtol=1e-12, atol=1e-12)
                        np.testing.assert_allclose(part.prediction_mm, bounded[name], rtol=1e-12, atol=1e-12)
                        replay_errors[name] = float(np.max(np.abs(part.raw_prediction_mm.to_numpy() - raw[name])))
                        if hasattr(model.estimator, "diagnostics_"):
                            guard_diagnostics[name] = model.estimator.diagnostics_.get("residual_guard")
                    write_json(directory / "residual_guard_diagnostics.json", guard_diagnostics)
                    runs.append({"case_id": case_id, "missing_policy": policy, "variant": variant,
                                 "elapsed_seconds": elapsed, "serialization_csv_errors_mm": replay_errors})
                    del bundle, raw, bounded
                    gc.collect()
        for name, parts in tables.items():
            extension = ".csv.gz" if name.endswith("predictions") else ".csv"
            pd.concat(parts, ignore_index=True).to_csv(staging / (name + extension), index=False)
        provenance["elapsed_seconds"] = perf_counter() - started
        provenance["runs"] = runs
        write_json(staging / "provenance.json", provenance)
        write_json(staging / "summary.json", {"completed": True, "runs": len(runs),
                   "real_target_labels_available": False, "official_submission_created": False,
                   "shared_keys_and_masks_verified": True, "contract_version": summary["contract_version"]})
        staging.rename(output)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.prepared, args.output)
    print(f"Completed {len(report['runs'])} shared-evaluator runs in {report['elapsed_seconds']:.2f}s")
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
