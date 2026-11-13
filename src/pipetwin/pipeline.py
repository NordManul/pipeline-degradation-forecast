"""Run the first E02 between-inspection development experiment."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile

import joblib
import numpy as np
import pandas as pd

from pipetwin.features import build_features, feature_blocks
from pipetwin.models.tabular import ridge_model
from pipetwin.sources import read_sources
from pipetwin.targets import observed_targets
from pipetwin.validation import metric_table, spatial_folds


def evaluate_ablation(frame: pd.DataFrame, blocks: dict, folds: int, buffer_m: float, alpha: float):
    splits = list(spatial_folds(frame, folds, buffer_m))
    predictions = []
    for fold, train, test in splits:
        held = frame.iloc[test]
        base = held[["id", "main_id", "segment_id", "thickness_mm"]].copy()
        base["fold"] = fold
        base["model"] = "persistence"
        base["prediction_mm"] = held.previous_thickness_mm
        predictions.append(base)
        for name, columns in blocks.items():
            model = ridge_model(alpha)
            model.fit(frame.iloc[train][columns], frame.iloc[train].thickness_mm)
            result = base.copy()
            result["model"] = name
            result["prediction_mm"] = model.predict(held[columns])
            predictions.append(result)
    return pd.concat(predictions, ignore_index=True)


def predict_models(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict an already-built feature frame for segments with prior observations."""
    if frame.empty or not np.isfinite(frame.previous_thickness_mm).all():
        raise ValueError("E02 between-VTD inference requires observed prior thickness for every row")
    result = frame[["id", "segment_id"]].copy()
    if result.duplicated().any():
        raise ValueError("Duplicate inference keys")
    result["persistence"] = frame.previous_thickness_mm
    for name, model in bundle["models"].items():
        result[name] = model.predict(frame[bundle["features"][name]])
    if not np.isfinite(result.drop(columns=["id", "segment_id"]).to_numpy()).all():
        raise ValueError("Nonfinite model predictions")
    return result


def run_experiment(
    raw_dir: Path, output_dir: Path, start: str = "2018-09-23", end: str = "2022-07-06",
    main_id: str = "MAIN-2", target_policy: str = "internal-corrosion", folds: int = 5,
    buffer_m: float = 50, alpha: float = 10, terrain_direction: str | None = None,
    terrain_radius_m: float = 50,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("Output directory already exists; select a new run directory")
    if terrain_direction and buffer_m < terrain_radius_m:
        raise ValueError("Spatial buffer must cover the terrain feature radius")
    sources = read_sources(raw_dir, terrain=bool(terrain_direction))
    frame, prior_audit = build_features(
        sources, start, end, main_id, target_policy, terrain_direction, terrain_radius_m
    )
    sections = sources["sections"].loc[sources["sections"].main_id == main_id]
    targets, target_audit = observed_targets(sources["vtd"], sections, end, target_policy)
    scored = frame.merge(targets, on=["id", "segment_id"], validate="one_to_one")
    missing_prior = int(scored.previous_thickness_mm.isna().sum())
    scored = scored.loc[scored.previous_thickness_mm.notna()].sort_values(["id", "segment_id"])
    scored = scored.reset_index(drop=True)
    blocks = feature_blocks(scored)
    predictions = evaluate_ablation(scored, blocks, folds, buffer_m, alpha)
    metrics = metric_table(predictions)
    config = {
        "start": start, "end": end, "main_id": main_id, "target_policy": target_policy,
        "folds": folds, "buffer_m": buffer_m, "alpha": alpha,
        "terrain_direction": terrain_direction, "terrain_radius_m": terrain_radius_m,
        "evaluation": "spatial_development_same_transition",
        "tech_window": "[start, end), last snapshot per day, no forward fill",
        "temperature_scenario": "endpoint mean in assumed Celsius; <=0.001 treated as missing",
    }
    models = {}
    for name, columns in blocks.items():
        models[name] = ridge_model(alpha).fit(scored[columns], scored.thickness_mm)
    bundle = {"models": models, "features": blocks, "config": config}
    # Exercise the same inference entry point used by consumers before writing artifacts.
    predict_models(bundle, scored)
    used_files = ["pot_south.csv", "vtd_south_train.csv",
                  "tech_mode_for_embedings_with_length_with_temperatures_ids16.csv"]
    if terrain_direction:
        used_files.append("altitudes_south.csv")
    report = {
        "config": config, "prior_audit": prior_audit, "target_audit": target_audit,
        "grid_segments": len(frame), "scored_segments": len(scored),
        "target_segments_without_prior": missing_prior,
        "features": blocks,
        "versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "joblib"]
        }},
        "source_sha256": {name: hashlib.sha256((Path(raw_dir) / name).read_bytes()).hexdigest()
                          for name in used_files},
        "metrics": json.loads(metrics.to_json(orient="records")),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete run together. A failed write leaves no partial run directory.
    with tempfile.TemporaryDirectory(prefix=".e02-", dir=output_dir.parent) as staging:
        staged_run = Path(staging) / "run"
        staged_run.mkdir()
        predictions.to_csv(staged_run / "oof_predictions.csv", index=False)
        metrics.to_csv(staged_run / "metrics.csv", index=False)
        (staged_run / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        joblib.dump(bundle, staged_run / "models.joblib")
        staged_run.rename(output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/e02"))
    parser.add_argument("--start", default="2018-09-23")
    parser.add_argument("--end", default="2022-07-06")
    parser.add_argument("--main-id", default="MAIN-2")
    parser.add_argument("--target-policy", choices=["internal-corrosion", "all-residual"], default="internal-corrosion")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--buffer-m", type=float, default=50)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--terrain-direction", choices=["increasing", "decreasing"])
    parser.add_argument("--terrain-radius-m", type=float, default=50)
    args = parser.parse_args()
    try:
        report = run_experiment(**vars(args))
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))
    metrics = pd.DataFrame(report["metrics"])
    print(metrics.loc[(metrics.scope == "oof") & (metrics.group == "all")].to_string(index=False))
    print(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
