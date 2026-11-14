"""E03 positive rates, ML and hybrid thickness forecasts on shared validation keys."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd

from pipetwin.features import build_features, feature_blocks
from pipetwin.models.physics import (
    RATE_COLUMNS, HybridScaleModel, ObservedThicknessModel, PositiveRateModel,
    build_exposure, validate_inputs,
)
from pipetwin.reporting import write_experiment
from pipetwin.sources import read_sources
from pipetwin.targets import observed_targets
from pipetwin.validation import metric_table, spatial_folds


def prepare_data(sources, start, end, main_id, policy, terrain_direction, terrain_radius_m):
    frame, prior_audit = build_features(sources, start, end, main_id, policy,
                                        terrain_direction, terrain_radius_m)
    sections = sources["sections"].loc[sources["sections"].main_id == main_id]
    labels, label_audit = observed_targets(sources["vtd"], sections, end, policy)
    scored = frame.merge(labels, on=["id", "segment_id"], validate="one_to_one")
    missing = int(scored.previous_thickness_mm.isna().sum())
    scored = scored.loc[scored.previous_thickness_mm.notna()].sort_values(["id", "segment_id"])
    scored = scored.reset_index(drop=True)
    if len(scored) < 2:
        raise ValueError("E03 requires at least two target segments with previous observations")
    exposure = build_exposure(sources["tech"], sections.id, start, end)
    audit = {"prior": prior_audit, "target": label_audit, "grid_segments": len(frame),
             "scored_segments": len(scored), "target_segments_without_prior": missing,
             "observed_thickness_increases": int((scored.thickness_mm > scored.previous_thickness_mm).sum())}
    return scored, exposure, audit


def fit_candidates(frame, exposure, seed=42, missing_policy="train-median"):
    columns = list(feature_blocks(frame).values())[-1]
    target = frame.thickness_mm.to_numpy()
    constant = PositiveRateModel(constant=True, missing_policy=missing_policy).fit(frame, exposure, target)
    physics = PositiveRateModel(missing_policy=missing_policy).fit(frame, exposure, target)
    models = {
        "constant_rate": constant, "parametric_rate": physics,
        "ml": ObservedThicknessModel(columns, seed=seed).fit(frame, exposure, target),
        "hybrid_scale": HybridScaleModel(physics, columns).fit(frame, exposure, target),
        "hybrid_residual": ObservedThicknessModel(columns, physics, seed).fit(frame, exposure, target),
    }
    return models, columns


def predict_e03(bundle, frame, exposure):
    """Inference needs prior VTD and dated TECH; target labels are unnecessary."""
    validate_inputs(frame, exposure)
    result = frame[["id", "segment_id"]].copy()
    result["persistence"] = frame.previous_thickness_mm.to_numpy()
    for name, model in bundle["models"].items():
        prediction = model.predict(frame, exposure)
        if not (np.isfinite(prediction) & (prediction >= 0)).all():
            raise ValueError(f"Invalid thickness forecast from {name}")
        result[name] = prediction
    return result


def evaluation_rows(models, frame, exposure, evaluation, fold):
    wide = predict_e03({"models": models}, frame, exposure)
    rows = []
    for name in wide.columns[2:]:
        part = frame[["id", "main_id", "segment_id", "previous_thickness_mm", "thickness_mm"]].copy()
        part["model"], part["evaluation"], part["fold"] = name, evaluation, fold
        part["prediction_mm"] = wide[name].to_numpy()
        part["latent_thickness_mm"] = np.nan
        if name in {"constant_rate", "parametric_rate", "hybrid_scale"}:
            part["latent_thickness_mm"] = part.prediction_mm
        if name == "hybrid_residual":
            part["latent_thickness_mm"] = models[name].physics.predict(frame, exposure)
        rows.append(part)
    return rows


def rate_summary(models, frame, exposure, evaluation):
    rows = []
    unique_sections = frame.drop_duplicates("id")
    for name in ["constant_rate", "parametric_rate", "hybrid_scale"]:
        model = models[name]
        loss = model.integrated_loss(frame, exposure)
        per_segment = pd.DataFrame({"id": frame.id, "loss": loss})
        for section in unique_sections.itertuples(index=False):
            history = exposure.loc[exposure.id == section.id]
            values = per_segment.loc[per_segment.id == section.id, "loss"]
            rows.append({"evaluation": evaluation, "model": name, "id": section.id,
                         "calendar_days": len(history), "observed_days": int(history.observed.sum()),
                         "complete_physical_days": int(history[RATE_COLUMNS].notna().all(axis=1).sum()),
                         "mean_integrated_loss_mm": float(values.mean()),
                         "min_integrated_loss_mm": float(values.min()),
                         "max_integrated_loss_mm": float(values.max())})
    return rows


def run_e03(raw_dir: Path, output_dir: Path, start="2018-09-23", end="2022-07-06",
            main_id="MAIN-2", late_date=None, target_policy="internal-corrosion", folds=5,
            buffer_m=50, seed=42, missing_policy="train-median", terrain_direction=None,
            terrain_radius_m=50):
    if Path(output_dir).exists():
        raise ValueError("Output directory already exists; select a new run directory")
    if late_date and pd.Timestamp(late_date) <= pd.Timestamp(end):
        raise ValueError("Late holdout date must follow the training target date")
    if terrain_direction and buffer_m < terrain_radius_m:
        raise ValueError("Buffer must cover the terrain radius")
    sources = read_sources(raw_dir, terrain=bool(terrain_direction))
    train, history, train_audit = prepare_data(sources, start, end, main_id, target_policy,
                                              terrain_direction, terrain_radius_m)
    rows, fits = [], []
    for fold, train_idx, test_idx in spatial_folds(train, folds, buffer_m):
        started = perf_counter()
        models, columns = fit_candidates(train.iloc[train_idx], history, seed, missing_policy)
        fits.append({"fold": fold, "train_n": len(train_idx), "test_n": len(test_idx),
                     "seconds": perf_counter() - started})
        rows.extend(evaluation_rows(models, train.iloc[test_idx], history, "spatial_oof", fold))
    started = perf_counter()
    models, columns = fit_candidates(train, history, seed, missing_policy)
    fits.append({"fold": "final_fit", "train_n": len(train), "test_n": 0,
                 "seconds": perf_counter() - started})
    exposure_rows = rate_summary(models, train, history, "development")
    audits = {"development": train_audit}
    if late_date:
        held, held_history, audits["temporal_holdout"] = prepare_data(
            sources, end, late_date, main_id, target_policy, terrain_direction, terrain_radius_m)
        rows.extend(evaluation_rows(models, held, held_history, "temporal_holdout", 0))
        exposure_rows.extend(rate_summary(models, held, held_history, "temporal_holdout"))
    predictions = pd.concat(rows, ignore_index=True)
    summaries = []
    for evaluation, part in predictions.groupby("evaluation", sort=False):
        summary = metric_table(part)
        if evaluation == "temporal_holdout":
            summary = summary.loc[summary.scope == "oof"].copy()
            summary["scope"] = "holdout"
        summary["evaluation"] = evaluation
        summaries.append(summary)
    metrics = pd.concat(summaries, ignore_index=True)
    physics = models["parametric_rate"]
    config = {
        "start": start, "end": end, "main_id": main_id, "late_date": late_date,
        "target_policy": target_policy, "folds": folds, "buffer_m": buffer_m, "seed": seed,
        "missing_policy": missing_policy, "days_per_year": 365.25,
        "terrain_direction": terrain_direction, "terrain_radius_m": terrain_radius_m,
        "rate_regularization": 0.25, "scale_regularization": 1.0, "robust_delta_mm": 0.5,
        "rate_log_intercept_bounds": [-18, float(np.log(10))], "rate_coefficient_bounds": [-2, 2],
        "standardized_channel_bounds": [-5, 5], "hybrid_log_scale_bounds": [-5, 5],
        "ml": {"family": "RandomForestRegressor", "trees": 200, "max_depth": 4, "min_samples_leaf": 8},
        "lower_thickness_bound_mm": 0, "measurement_residual": "signed annualized correction",
        "tech_window": "[previous VTD, target), last daily snapshot",
        "temperature_scenario": "endpoint Celsius mean; <=0.001 treated as missing",
    }
    files = ["pot_south.csv", "vtd_south_train.csv",
             "tech_mode_for_embedings_with_length_with_temperatures_ids16.csv"]
    if terrain_direction:
        files.append("altitudes_south.csv")
    report = {
        "experiment": "E03", "config": config, "audits": audits, "features": columns,
        "physical_rate_columns": RATE_COLUMNS,
        "rate_parameters": {"log_intercept": float(physics.parameters_[0]),
                            "coefficients": physics.parameters_[1:].tolist(),
                            "optimizer": physics.optimization_},
        "hybrid_scale_coefficients": models["hybrid_scale"].coefficients_.tolist(),
        "constant_rate_mm_year": float(np.exp(models["constant_rate"].parameters_[0])),
        "versions": {"python": platform.python_version(), **{p: importlib.metadata.version(p)
                      for p in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]}},
        "source_sha256": {name: hashlib.sha256((Path(raw_dir) / name).read_bytes()).hexdigest() for name in files},
        "metrics": json.loads(metrics.to_json(orient="records")),
    }
    bundle = {"experiment": "E03", "models": models, "features": columns, "config": config}
    write_experiment(output_dir, report, bundle, {"predictions.csv": predictions, "metrics.csv": metrics,
                                                "fits.csv": pd.DataFrame(fits),
                                                "exposure_summary.csv": pd.DataFrame(exposure_rows)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/e03"))
    parser.add_argument("--start", default="2018-09-23")
    parser.add_argument("--end", default="2022-07-06")
    parser.add_argument("--main-id", default="MAIN-2")
    parser.add_argument("--late-date")
    parser.add_argument("--target-policy", choices=["internal-corrosion", "all-residual"], default="internal-corrosion")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--buffer-m", type=float, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-policy", choices=["train-median", "observed-only"], default="train-median")
    parser.add_argument("--terrain-direction", choices=["increasing", "decreasing"])
    parser.add_argument("--terrain-radius-m", type=float, default=50)
    args = parser.parse_args()
    try:
        report = run_e03(**vars(args))
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    metrics = pd.DataFrame(report["metrics"])
    print(metrics.loc[metrics.scope.isin(["oof", "holdout"]) & (metrics.group == "all")].to_string(index=False))
    print(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
