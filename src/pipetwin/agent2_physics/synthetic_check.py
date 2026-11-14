"""Reproducible API/serialization exercise; never a real-case benchmark.

All inputs below are invented canonical examples. No raw sources, shared
cleaner, official labels, scoring masks or fold generation are used.
"""

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from . import CONTRACT_VERSION
from .adapter import REQUIRED_MODEL_NAMES, OPTIONAL_MODEL_NAMES, make_model
from .validation import DAYS_PER_YEAR, RATE_COLUMNS


def synthetic_inputs(scenario):
    """Two training windows and a separate 2020 leap-year inference window."""
    frames, histories = [], []
    definitions = [("train_a", "2018-01-01", "2019-01-01", 0.1),
                   ("train_b", "2019-01-01", "2020-01-01", 0.7),
                   ("query", "2020-01-01", "2021-01-01", 0.4)]
    for key, start, end, water in definitions:
        days = pd.date_range(start, end, inclusive="left")
        history = pd.DataFrame({
            "exposure_id": key, "id": "synthetic_pipe", "day": days,
            "available_at": days + pd.Timedelta(hours=12), "observed": True,
            "dt_years": 1 / DAYS_PER_YEAR, "water_fraction": water,
            "temperature_c": 20 + np.sin(np.arange(len(days)) / 30),
            "velocity_m_s": 0.5, "fill_coefficient": 0.8,
        })
        if key == "query":
            missing = history.day.ge("2020-07-01")
            history.loc[missing, "observed"] = False
            history.loc[missing, "available_at"] = pd.NaT
            history.loc[missing, list(RATE_COLUMNS)] = np.nan
        histories.append(history)
        frame = pd.DataFrame({
            "sample_id": [f"{scenario}-{key}-{i}" for i in range(8)],
            "id": "synthetic_pipe", "main_id": "synthetic_main", "segment_id": range(8),
            "scenario": scenario, "target_date": pd.Timestamp(end),
            "label_available_at": pd.Timestamp(end), "window_start": pd.Timestamp(start),
            "window_end": pd.Timestamp(end), "horizon_years": len(days) / DAYS_PER_YEAR,
            "nominal_mm": 8.0, "exposure_id": key, "safe_feature": np.linspace(-1, 1, 8),
        })
        if scenario == "between":
            frame["anchor_mm"], frame["anchor_observed"] = 7.0, True
            frame["anchor_date"], frame["fallback_mm"] = pd.Timestamp(start), 5.7
            frame.loc[[1, 5], "anchor_observed"] = False
            frame.loc[[1, 5], "anchor_mm"] = np.nan
        frames.append(frame)
    train = pd.concat(frames[:2], ignore_index=True)
    y = (7.0 if scenario == "between" else 8.0) - train.horizon_years * (
        train.exposure_id.map({"train_a": 0.15, "train_b": 0.35}) + 0.05 * train.safe_feature)
    return train, y.to_numpy(), frames[2].drop(columns="label_available_at"), pd.concat(histories, ignore_index=True)


def run(output_dir):
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Choose a new synthetic-check output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    records, predictions = [], []
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix=".synthetic-", dir=output.parent) as temporary:
        staged = Path(temporary) / "run"
        staged.mkdir()
        for scenario, cutoff in [("between", "2023-12-01"), ("cold", "2022-07-01")]:
            train, y, query, exposure = synthetic_inputs(scenario)
            train.assign(y_mm=y).to_csv(staged / f"{scenario}_train.csv", index=False)
            query.to_csv(staged / f"{scenario}_query.csv", index=False)
            exposure.to_csv(staged / f"{scenario}_exposure.csv", index=False)
            for missing_policy in ["observed-only", "train-median"]:
                for name in REQUIRED_MODEL_NAMES + OPTIONAL_MODEL_NAMES:
                    model = make_model(name, scenario=scenario, exposure=exposure,
                                       cutoff_exclusive=cutoff, feature_columns=["safe_feature"],
                                       options={"missing_policy": missing_policy}, seed=42)
                    fit_start = perf_counter()
                    model.fit(train, y)
                    seconds = perf_counter() - fit_start
                    prediction = model.predict(query)
                    key = f"{scenario}-{missing_policy}-{name}"
                    joblib.dump(model, staged / f"{key}.joblib", compress=3)
                    restored = joblib.load(staged / f"{key}.joblib").predict(query)
                    error = float(np.max(np.abs(restored - prediction)))
                    if error > 1e-12:
                        raise RuntimeError("Saved model did not reproduce its synthetic forecast")
                    table = query[["sample_id", "id", "segment_id", "target_date"]].copy()
                    table["case_id"], table["fold_id"] = f"synthetic_{scenario}", "synthetic_probe"
                    table["model"], table["missing_policy"] = name, missing_policy
                    table["prediction_mm"], table["contract_version"] = prediction, CONTRACT_VERSION
                    predictions.append(table)
                    records.append({"scenario": scenario, "model": name, "missing_policy": missing_policy,
                                    "fit_seconds": seconds, "rows": len(query),
                                    "max_serialization_error_mm": error, "diagnostics": model.diagnostics_})
        pd.concat(predictions, ignore_index=True).to_csv(staged / "predictions.csv", index=False)
        source_dir = Path(__file__).resolve().parent
        report = {
            "synthetic_only": True, "real_case_metrics_available": False,
            "contract_version": CONTRACT_VERSION,
            "base_commit": "0115916a0f07dbb849d7aee8cb022700911df42b",
            "versions": {"python": platform.python_version(), **{
                package: importlib.metadata.version(package)
                for package in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]}},
            "source_sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(source_dir.glob("*.py"))},
            "elapsed_seconds": perf_counter() - started, "runs": records,
        }
        (staged / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        staged.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = run(arguments.output)
    print(f"Synthetic only: {len(report['runs'])} fits and serialized replays; {report['elapsed_seconds']:.2f} seconds")
    print(f"Artifacts: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
