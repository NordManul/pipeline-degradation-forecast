"""Buffered spatial folds and common-mask metrics, including signed correlation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpatialFold:
    number: int
    train: np.ndarray
    valid: np.ndarray
    left_m: float
    right_m: float


def blocked_folds(frame, n_splits=4, buffer_m=100.0):
    """Continuous physical intervals; buffers exclude target labels from fitting.

    Geometry, including unlabelled cells, defines the folds. Only a single main
    pipeline is supported here; transfer to another main is evaluated separately.
    """
    if frame["main_id"].nunique() != 1:
        raise ValueError("Spatial CV expects one main pipeline; use a grouped outer split otherwise")
    if n_splits < 2 or buffer_m < 0 or len(frame) < n_splits:
        raise ValueError("Invalid fold count or buffer")
    position = frame["chain_position_m"].to_numpy(dtype=float)
    if not np.isfinite(position).all() or np.ptp(position) <= 0:
        raise ValueError("Finite, varying physical positions are required")
    boundaries = np.linspace(position.min() - 1e-6, position.max() + 1e-6, n_splits + 1)
    folds = []
    for number, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        valid = np.flatnonzero((position >= left) & (position < right))
        train = np.flatnonzero((position < left - buffer_m) | (position >= right + buffer_m))
        if not len(valid) or not len(train):
            raise ValueError("An empty spatial fold requires a smaller fold count or buffer")
        folds.append(SpatialFold(number, train, valid, float(left), float(right)))
    return folds


def _thin_membership(values, k):
    threshold = np.partition(values, k - 1)[k - 1]
    below, tied = values < threshold, values == threshold
    return below.astype(float) + tied * ((k - below.sum()) / tied.sum())


def regression_metrics(y_true, y_pred):
    y, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y.ndim != 1 or y.shape != pred.shape or not len(y):
        raise ValueError("Metrics require non-empty, equally shaped 1-D arrays")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Metrics do not silently drop non-finite predictions or labels")
    varying = np.std(y) > 1e-12 and np.std(pred) > 1e-12
    correlation = float(np.corrcoef(y, pred)[0, 1]) if varying else np.nan
    k = max(1, int(np.ceil(len(y) * 0.1)))
    residual = pred - y
    return {
        "n": len(y), "pearson_r": correlation, "pearson_squared": correlation ** 2,
        "mae_mm": float(np.abs(residual).mean()),
        "rmse_mm": float(np.sqrt(np.mean(residual ** 2))), "bias_mm": float(residual.mean()),
        # Fractional ties give a constant predictor chance-level recall.
        "recall_thin_10pct": float(np.dot(_thin_membership(y, k), _thin_membership(pred, k)) / k),
    }


def score_predictions(frame, predictions, split):
    observed = frame["observed"].to_numpy(dtype=bool)
    records = []
    for name, values in predictions.items():
        values = np.asarray(values, dtype=float)
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ValueError(f"Model {name} must predict every requested cell with finite values")
        records.append({"split": split, "model": name, **regression_metrics(frame.loc[observed, "y_mm"], values[observed])})
    return pd.DataFrame(records)
