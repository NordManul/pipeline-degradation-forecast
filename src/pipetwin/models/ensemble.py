"""Fixed blending and a nonnegative, sum-to-one OOF stack."""

import numpy as np
from scipy.optimize import minimize


def prediction_matrix(predictions, candidates):
    matrix = np.column_stack([predictions[name] for name in candidates])
    if not np.isfinite(matrix).all():
        raise ValueError("Ensembling requires complete finite predictions")
    return matrix


def fit_stack_weights(matrix, y, penalty=0.02):
    matrix, y = np.asarray(matrix, dtype=float), np.asarray(y, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(y) or len(y) < 10 or not matrix.shape[1]:
        raise ValueError("Stack calibration needs at least 10 held-out labelled rows")
    if not np.isfinite(matrix).all() or not np.isfinite(y).all():
        raise ValueError("OOF stack inputs must be finite")
    uniform = np.full(matrix.shape[1], 1 / matrix.shape[1])
    variance = max(float(np.var(y)), 1e-6)

    def objective(weights):
        return np.mean((matrix @ weights - y) ** 2) / variance + penalty * np.sum((weights - uniform) ** 2)

    result = minimize(objective, uniform, method="SLSQP", bounds=[(0, 1)] * len(uniform),
                      constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
                      options={"ftol": 1e-10, "maxiter": 500})
    if not result.success:
        raise RuntimeError(f"OOF stacking failed: {result.message}")
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()
