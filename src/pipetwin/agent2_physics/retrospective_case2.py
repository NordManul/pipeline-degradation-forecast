"""User-authorized retrospective case2 study, isolated from strict contract v1.

Shared cleaning, canonical features, estimators, metrics and clipping are reused.
See RETROSPECTIVE_PROTOCOL.md: late labels of OTHER mains are now permitted;
each feature window still ends at its own state date.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import gc
from hashlib import sha256
from pathlib import Path
import subprocess
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from threadpoolctl import threadpool_limits

from pipetwin.case_data import canonical_snapshot, build_exposure
from pipetwin.case_experiments import (fit_case_model, predict_models, metric_table,
                                     prediction_table, range_diagnostics)
from pipetwin.diagnostics import input_shift
from pipetwin.features import feature_columns
from pipetwin.geometry import build_grid
from pipetwin.physical_features import ablation_columns
from pipetwin.pipeline import write_json
from pipetwin.quality import read_raw_sources, clean_sources
from pipetwin.sources import read_config
from pipetwin.targets import build_targets

from .shared_comparison import comparison_config, verify_shared_predictions


STATE_DATE = pd.Timestamp("2022-07-01")
TRAINING_CUTOFF = pd.Timestamp("2023-12-02")
TARGET_ID = "1250009583"
CONTEXT = "case2_retrospective"
MODELS = ["persistence", "global_rate", "gam", "catboost", "e02_terrain",
          "agent2_constant_rate", "agent2_hybrid_residual"]


def validate_pool(train, query):
    """Do not confuse the training-availability bound with the query state date."""
    for frame in [train, query]:
        if frame.empty or frame.sample_id.duplicated().any():
            raise ValueError("Empty or duplicate samples")
        if not frame.scenario.eq("cold").all() or not frame.case_id.eq(CONTEXT).all():
            raise ValueError("Retrospective study requires its own cold context")
        if not frame.window_end.eq(frame.target_date).all() or frame.anchor_observed.any():
            raise ValueError("Cold feature windows or anchors are inconsistent")
    if not query.id.eq(TARGET_ID).all() or not query.target_date.eq(STATE_DATE).all():
        raise ValueError("The requested case2 state date/ID must not move")
    if set(train.main_id) & set(query.main_id):
        raise ValueError("Target main inspections cannot enter training")
    measured = train.loc[train.observed]
    if measured.empty or not train.target_date.lt(TRAINING_CUTOFF).all():
        raise ValueError("Training campaigns exceed the allowed pool")
    if not measured.label_available_at.lt(TRAINING_CUTOFF).all():
        raise ValueError("Training labels were unavailable at the pool cutoff")
    if any(c in query for c in ["y_mm", "observed", "coverage_status", "label_available_at"]):
        raise ValueError("Prediction requests must not contain target labels")


def validate_group_split(train, valid):
    if set(train.main_id) & set(valid.main_id) or set(train.id) & set(valid.id):
        raise ValueError("All campaigns of a held main must stay out of training")
    if not train.observed.any() or not valid.observed.any():
        raise ValueError("Both groups require observed labels")


def prepare(raw, manifest):
    contract = deepcopy(read_config("contracts/cases.yaml"))
    target_contract = read_config("configs/data_contract.yaml")
    config = {**read_config("configs/forecast.yaml"), "temperature_policy": "observed_endpoints"}
    # This context's date is a LOADING bound, not the target state date. The
    # existing cleaner requires its cutoff/date to agree; its rules are reused.
    contract["cases"][CONTEXT] = {"id": TARGET_ID, "main_id": contract["cases"]["case2"]["main_id"],
                                "target_date": str(TRAINING_CUTOFF.date()),
                                "cutoff_exclusive": str(TRAINING_CUTOFF.date())}
    cleaned = clean_sources(raw, CONTEXT, contract, target_contract, manifest)
    target_main = cleaned.sources.passport.set_index("id").loc[TARGET_ID, "main_id"]
    excluded_ids = set(cleaned.sources.passport.loc[lambda d: d.main_id.eq(target_main), "id"])
    inspection = cleaned.sources.inspection.loc[lambda d: ~d.id.isin(excluded_ids)].copy()
    cleaned = replace(cleaned, sources=replace(cleaned.sources, inspection=inspection))
    grid = build_grid(cleaned.sources.passport, target_contract["segment_length_m"])
    targets, _ = build_targets(cleaned.sources, grid, target_contract)
    frames = [canonical_snapshot(cleaned, grid, targets,
              {"ids": group.id.unique().tolist(), "target_date": date}, config, target_contract, labelled=True)
              for (_, date), group in targets.groupby(["main_id", "date"])]
    train = pd.concat(frames, ignore_index=True)
    query = canonical_snapshot(cleaned, grid, targets, {"ids": [TARGET_ID], "target_date": STATE_DATE},
                               config, target_contract, labelled=False)
    validate_pool(train, query)
    exposure = build_exposure(cleaned.sources.regime, [train, query], TRAINING_CUTOFF)
    folds = [{"number": n, "train": a, "valid": b}
             for n, (a, b) in enumerate(LeaveOneGroupOut().split(train, groups=train.main_id))]
    for fold in folds:
        validate_group_split(train.iloc[fold["train"]], train.iloc[fold["valid"]])
    return {"case_id": CONTEXT, "scenario": "cold", "cutoff_exclusive": TRAINING_CUTOFF,
            "target_state_date": STATE_DATE, "training_available_until": TRAINING_CUTOFF,
            "feature_columns": feature_columns("cold"), "input_manifest": manifest,
            "train_frame": train, "predict_frame": query, "exposure": exposure, "folds": folds}


def compare_query_inputs(package, strict_reference):
    """Expanded labels/sources must not introduce post-T query features."""
    if package["input_manifest"] != strict_reference["input_manifest"]:
        raise ValueError("Strict reference came from different raw files")
    columns = list(dict.fromkeys(feature_columns("cold") + ablation_columns("terrain", "cold")))
    query, reference = package["predict_frame"], strict_reference["predict_frame"]
    for column in ["id", "segment_id", "target_date", "window_start", "window_end", "exposure_id"]:
        pd.testing.assert_series_equal(query[column], reference[column], check_dtype=False)
    np.testing.assert_allclose(query[columns], reference[columns], rtol=0, atol=0, equal_nan=True)
    history = package["exposure"].loc[lambda d: d.exposure_id.isin(query.exposure_id)].reset_index(drop=True)
    old = strict_reference["exposure"].loc[lambda d: d.exposure_id.isin(reference.exposure_id)].reset_index(drop=True)
    pd.testing.assert_frame_equal(history, old, check_dtype=False)
    return {"features": len(columns), "exposure_days": len(history), "exactly_unchanged": True}


def fallback(train):
    observed = train.loc[train.observed]
    return {"fraction": float(np.clip(np.median(observed.y_mm / observed.nominal_mm), 0, 1))}


def campaign_metrics(frame, raw, bounded, fold_id):
    """Keep each state date separate; all actual scoring is delegated."""
    tables = []
    for (main, date), _ in frame.groupby(["main_id", "target_date"]):
        mask = (frame.main_id.eq(main) & frame.target_date.eq(date)).to_numpy()
        tables.append(metric_table(frame.loc[mask], {k: v[mask] for k, v in raw.items()},
                                   {k: v[mask] for k, v in bounded.items()},
                                   split="leave_main_out", fold_id=fold_id).assign(held_main=main, target_date=date))
    return pd.concat(tables, ignore_index=True)


def run(output, strict_reference):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new retrospective output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw, manifest = read_raw_sources("data/raw")
    package = prepare(raw, manifest)
    del raw
    query_check = compare_query_inputs(package, joblib.load(strict_reference))
    train, query = package["train_frame"], package["predict_frame"]
    config = comparison_config("train-median", "train-bounded")
    config["models"] = list(MODELS)
    source = Path(__file__).resolve().parents[1]
    provenance = {"created_utc": datetime.now(timezone.utc), "mode": "user_authorized_retrospective_transfer",
                  "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
                  "source_sha256": {str(p.relative_to(source)): sha256(p.read_bytes()).hexdigest() for p in source.rglob("*.py")},
                  "protocol_sha256": sha256(Path(__file__).with_name("RETROSPECTIVE_PROTOCOL.md").read_bytes()).hexdigest(),
                  "input_manifest": manifest, "query_state_date": STATE_DATE, "training_available_until": TRAINING_CUTOFF,
                  "query_vs_strict_reference": query_check, "config": config,
                  "independent_main_count": int(train.main_id.nunique()), "evaluation_is_exploratory": True,
                  "hidden_target_accuracy_measured": False}
    with tempfile.TemporaryDirectory(prefix=".retro-", dir=output.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        write_json(staging / "frozen_protocol.json", provenance)
        train.to_csv(staging / "train_frame.csv.gz", index=False)
        query.to_csv(staging / "predict_frame.csv.gz", index=False)
        tables, predictions, audit = [], [], []
        with threadpool_limits(limits=1):
            for fold in package["folds"]:
                fit, valid = train.iloc[fold["train"]], train.iloc[fold["valid"]]
                validate_group_split(fit, valid)
                raw_predictions, bounded, fits = {}, {}, []
                for name in MODELS:
                    print(f"Retrospective: hold {valid.main_id.iloc[0]}, {name}", flush=True)
                    model, info = fit_case_model(name, package, fit, fallback(fit), config)
                    r, b = predict_models({name: model}, valid, {}, development=True)
                    raw_predictions.update(r)
                    bounded.update(b)
                    fits.append(info)
                    del model
                    gc.collect()
                tables.append(campaign_metrics(valid, raw_predictions, bounded, fold["number"]))
                predictions.append(prediction_table(valid, raw_predictions, bounded, fold["number"]))
                audit.append({"fold": fold["number"], "train_mains": fit.main_id.unique().tolist(),
                              "held_mains": valid.main_id.unique().tolist(), "fits": fits,
                              "train_sample_ids": fit.sample_id.tolist(), "valid_sample_ids": valid.sample_id.tolist()})
            oof = pd.concat(predictions, ignore_index=True)
            verify_shared_predictions(package, oof)
            pd.concat(tables, ignore_index=True).to_csv(staging / "metrics_by_campaign.csv", index=False)
            oof.to_csv(staging / "oof_predictions.csv.gz", index=False)
            write_json(staging / "group_fit_audit.json", audit)
            # Candidate names/options are already frozen; query ranges never
            # choose a winner. Save each estimator separately to limit memory.
            (staging / "models").mkdir()
            final_raw, final_bounded, final_audit = {}, {}, []
            replay_error = 0.0
            for name in MODELS:
                print(f"Retrospective: full training, {name}", flush=True)
                model, info = fit_case_model(name, package, train, fallback(train), config)
                r, b = predict_models({name: model}, query, {})
                final_raw.update(r)
                final_bounded.update(b)
                path = staging / "models" / f"{name}.joblib"
                joblib.dump(model, path, compress=3)
                del model
                replay = joblib.load(path)
                rr, bb = predict_models({name: replay}, query, {})
                np.testing.assert_allclose(rr[name], r[name], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(bb[name], b[name], rtol=1e-12, atol=1e-12)
                replay_error = max(replay_error, float(np.max(np.abs(rr[name] - r[name]))))
                if hasattr(replay.estimator, "diagnostics_"):
                    info["final_prediction_diagnostics"] = replay.estimator.diagnostics_
                final_audit.append(info)
                del replay
                gc.collect()
            forecasts = prediction_table(query, final_raw, final_bounded, -1)
            verify_shared_predictions(package, forecasts, forecast=True)
            forecasts.to_csv(staging / "candidate_forecasts.csv.gz", index=False)
            range_diagnostics(query, final_raw, final_bounded).to_csv(staging / "forecast_diagnostics.csv", index=False)
            input_shift(train, query, "cold", "retrospective_case2").to_csv(staging / "feature_shift.csv", index=False)
            write_json(staging / "final_fit_audit.json", final_audit)
            write_json(staging / "summary.json", {"completed": True, "observed_train_labels": int(train.observed.sum()),
                       "training_grid_cells": len(train), "training_mains": train.main_id.unique().tolist(),
                       "query_cells": len(query), "models": MODELS, "replay_max_error_mm": replay_error,
                       "query_features_and_history_unchanged": True, "hidden_target_accuracy_measured": False,
                       "official_submission_created": False, "proven_universal_winner": None})
        staging.rename(output)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strict-reference", required=True, type=Path)
    args = parser.parse_args()
    run(args.output, args.strict_reference)
