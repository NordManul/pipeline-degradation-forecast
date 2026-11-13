"""Small fixed Ridge control for E02 feature comparisons."""

import math

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ridge_model(alpha: float = 10.0) -> Pipeline:
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be positive")
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])
