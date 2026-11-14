"""Shared training/evaluation engine with nested, buffered OOF stacking."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import make_model
from .models.ensemble import fit_stack_weights, prediction_matrix
from .validation import blocked_folds, score_predictions


@dataclass
class CVResult:
    predictions: dict
    folds: np.ndarray
    fold_metrics: pd.DataFrame
    audit: list
    weights: np.ndarray | None


def checked_prediction(model, frame):
    prediction = np.asarray(model.predict(frame), dtype=float)
    if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
        raise ValueError("A model did not return finite predictions for all requested cells")
    # Shared, predeclared physical envelope for the audited passport/metal epoch.
    return np.clip(prediction, 0, frame["nominal_mm"].to_numpy(dtype=float))


def fit_models(frame, names, scenario, seed):
    train = frame.loc[frame.observed].copy()
    if len(train) < 20:
        raise ValueError("At least 20 observed training segments are required")
    if train.duplicated(["id", "segment_id"]).any():
        raise ValueError("Training keys must be unique within a snapshot")
    models = {}
    for name in names:
        models[name] = make_model(name, scenario, seed).fit(train, train.y_mm)
    return models


def predict_models(models, frame, candidates=None, weights=None):
    predictions = {name: checked_prediction(model, frame) for name, model in models.items()}
    if candidates is not None:
        matrix = prediction_matrix(predictions, candidates)
        predictions["mean_blend"] = matrix.mean(axis=1)
        if weights is not None:
            predictions["oof_stack"] = matrix @ weights
    return predictions


def cross_validate(frame, names, scenario, config, nested=True, n_splits=None, announce=False):
    frame = frame.reset_index(drop=True)
    if config["spatial_buffer_m"] < config["history_radius_m"]:
        raise ValueError("Spatial buffer must cover the feature radius")
    candidates = config["ensemble_candidates"]
    if nested and not set(candidates).issubset(names):
        raise ValueError("Stack candidates must be included among the base models")
    splits = blocked_folds(frame, n_splits or config["folds"], config["spatial_buffer_m"])
    output_names = list(names) + (["mean_blend", "oof_stack"] if nested else [])
    predictions = {name: np.full(len(frame), np.nan) for name in output_names}
    assignments = np.full(len(frame), -1)
    metric_tables, audit = [], []
    for fold in splits:
        if announce:
            print(f"{scenario}: outer fold {fold.number + 1}/{len(splits)}", flush=True)
        train, valid = frame.iloc[fold.train].copy(), frame.iloc[fold.valid].copy()
        models = fit_models(train, names, scenario, config["seed"])
        held = predict_models(models, valid)
        entry = {
            "fold": fold.number, "left_m": fold.left_m, "right_m": fold.right_m,
            "train_cells": len(train), "train_observed": int(train.observed.sum()),
            "valid_cells": len(valid), "valid_observed": int(valid.observed.sum()),
            "train_keys": list(zip(train.id, train.segment_id.astype(int))),
            "valid_keys": list(zip(valid.id, valid.segment_id.astype(int))),
        }
        if nested:
            # These inner splits can only see the outer training frame, including
            # its geometry. No outer-held labels can enter the weight optimizer.
            inner = cross_validate(train, candidates, scenario, config, nested=False, n_splits=3)
            mask = train.observed.to_numpy(dtype=bool)
            matrix = prediction_matrix(inner.predictions, candidates)
            weights = fit_stack_weights(matrix[mask], train.loc[mask, "y_mm"])
            held_matrix = prediction_matrix(held, candidates)
            held["mean_blend"] = held_matrix.mean(axis=1)
            held["oof_stack"] = held_matrix @ weights
            entry["stack_weights"] = dict(zip(candidates, weights.tolist()))
            entry["inner_folds"] = inner.audit
        for name, values in held.items():
            predictions[name][fold.valid] = values
        assignments[fold.valid] = fold.number
        metric_tables.append(score_predictions(valid, held, "development").assign(fold=fold.number))
        audit.append(entry)
    if (assignments < 0).any() or any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("OOF coverage is incomplete")
    weights = None
    if nested:
        mask = frame.observed.to_numpy(dtype=bool)
        weights = fit_stack_weights(prediction_matrix(predictions, candidates)[mask], frame.loc[mask, "y_mm"])
    return CVResult(predictions, assignments, pd.concat(metric_tables, ignore_index=True), audit, weights)


def prediction_table(frame, predictions, split, folds=None):
    columns = ["id", "main_id", "segment_id", "target_date", "center_m", "chain_position_m",
               "nominal_mm", "anchor_mm", "anchor_observed", "y_mm", "observed", "coverage_status"]
    result = frame[columns].copy()
    result["split"] = split
    if folds is not None:
        result["fold"] = folds
    for name, values in predictions.items():
        result[name] = values
    return result


def group_metrics(frame, predictions, split):
    tables = []
    for section_id in frame.id.unique():
        mask = frame.id.eq(section_id).to_numpy()
        if frame.loc[mask, "observed"].any():
            table = score_predictions(frame.loc[mask], {key: value[mask] for key, value in predictions.items()}, split)
            tables.append(table.assign(id=section_id))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
