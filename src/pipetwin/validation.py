"""Buffered spatial development folds, evaluated on identical observed keys."""

import numpy as np
import pandas as pd


def spatial_folds(frame: pd.DataFrame, n_splits: int = 5, buffer_m: float = 50):
    """Hold out a contiguous fraction of every section; purge its full interval."""
    if n_splits < 2 or not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("Need at least two folds and a nonnegative buffer")
    location = frame.center_m.to_numpy()
    length = frame.length_m.to_numpy()
    labels = np.minimum((location / length * n_splits).astype(int), n_splits - 1)
    for fold in range(n_splits):
        lower = fold * length / n_splits
        upper = (fold + 1) * length / n_splits
        test = labels == fold
        train = (location < lower - buffer_m) | (location > upper + buffer_m)
        if test.sum() < 2 or train.sum() < 2:
            raise ValueError(f"Fold {fold} has insufficient train/test observations")
        yield fold, np.flatnonzero(train), np.flatnonzero(test)


def regression_metrics(y, prediction) -> dict:
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != prediction.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("Metrics require equally sized nonempty one-dimensional arrays")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite observations and predictions")
    residual = prediction - y
    r = None
    if len(y) > 1 and np.ptp(y) > 1e-12 and np.ptp(prediction) > 1e-12:
        r = float(np.corrcoef(y, prediction)[0, 1])
    count = max(1, int(np.ceil(len(y) * 0.1)))
    # Stable key order resolves ties reproducibly.
    actual = set(np.argsort(y, kind="stable")[:count])
    predicted = set(np.argsort(prediction, kind="stable")[:count])
    return {
        "n": len(y), "pearson_r": r, "pearson_squared": None if r is None else r * r,
        "mae_mm": float(np.abs(residual).mean()),
        "rmse_mm": float(np.sqrt(np.mean(residual ** 2))),
        "bias_mm": float(residual.mean()), "recall_at_10pct": len(actual & predicted) / count,
    }


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model, rows in predictions.groupby("model", sort=False):
        for group, part in [("all", rows), *list(rows.groupby("id", sort=True))]:
            records.append({"model": model, "scope": "oof", "group": group,
                            **regression_metrics(part.thickness_mm, part.prediction_mm)})
        for fold, part in rows.groupby("fold"):
            records.append({"model": model, "scope": f"fold_{fold}", "group": "all",
                            **regression_metrics(part.thickness_mm, part.prediction_mm)})
    return pd.DataFrame(records)
