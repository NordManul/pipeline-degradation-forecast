"""Empirical OOF residual intervals; no coverage guarantee under distribution shift."""

import numpy as np


def calibrate_interval(y, oof_prediction):
    residual = np.asarray(y, dtype=float) - np.asarray(oof_prediction, dtype=float)
    if residual.ndim != 1 or len(residual) < 20 or not np.isfinite(residual).all():
        raise ValueError("Interval calibration needs at least 20 finite OOF residuals")
    return np.quantile(residual, [0.1, 0.9])


def predict_interval(prediction, residual_quantiles, nominal):
    prediction, nominal = np.asarray(prediction), np.asarray(nominal)
    low, high = residual_quantiles
    if low > high:
        raise ValueError("Reversed residual quantiles")
    return np.clip(prediction + low, 0, nominal), np.clip(prediction + high, 0, nominal)


def interval_metrics(y, lower, upper):
    y, lower, upper = np.asarray(y), np.asarray(lower), np.asarray(upper)
    if not (y.shape == lower.shape == upper.shape) or not np.isfinite([y, lower, upper]).all():
        raise ValueError("Interval arrays must be finite and share one shape")
    if (lower > upper).any():
        raise ValueError("Reversed prediction intervals")
    return {"coverage_80": float(np.mean((y >= lower) & (y <= upper))),
            "mean_width_mm": float(np.mean(upper - lower)), "n": len(y)}
