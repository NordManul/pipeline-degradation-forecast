"""Train-only feature support for the opt-in bounded residual correction.

This controls extrapolation of inputs, not the final wall-thickness output.
It neither learns from query rows nor promises accurate out-of-domain forecasts.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted


class TrainBoundedFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, feature_names, z_limit=3.0, min_relative_scale=1e-6):
        self.feature_names = feature_names
        self.z_limit = z_limit
        self.min_relative_scale = min_relative_scale

    def fit(self, X, y=None):
        self.imputer_ = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        values = self.imputer_.fit_transform(X)
        self.names_ = self.imputer_.get_feature_names_out(self.feature_names)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.threshold_ = self.min_relative_scale * np.maximum(1.0, np.abs(self.mean_))
        self.keep_ = self.scale_ > self.threshold_
        return self

    def _values(self, X):
        check_is_fitted(self, "keep_")
        values = self.imputer_.transform(X)
        z = (values[:, self.keep_] - self.mean_[self.keep_]) / self.scale_[self.keep_]
        return values, z

    def transform(self, X):
        values, z = self._values(X)
        # Ridge still learns its intercept if every feature is unsupported.
        if not self.keep_.any():
            return np.zeros((len(values), 1))
        return np.clip(z, -self.z_limit, self.z_limit)

    def fit_diagnostics(self):
        check_is_fitted(self, "keep_")
        return {
            "policy": "train-bounded", "z_limit": self.z_limit,
            "min_relative_scale": self.min_relative_scale,
            "feature_names": self.names_.tolist(), "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(), "threshold": self.threshold_.tolist(),
            "imputer_medians": self.imputer_.statistics_.tolist(),
            "kept_features": self.names_[self.keep_].tolist(),
            "dropped_features": self.names_[~self.keep_].tolist(),
            "intercept_only": bool(not self.keep_.any()),
        }

    def diagnose(self, X):
        values, z = self._values(X)
        clipped = np.abs(z) > self.z_limit
        shifted = np.abs(values[:, ~self.keep_] - self.mean_[~self.keep_]) > self.threshold_[~self.keep_]
        return {
            "rows": len(values), "clipped_rows": int(clipped.any(axis=1).sum()),
            "clipped_feature_values": int(clipped.sum()),
            "maximum_abs_unclipped_z": float(np.max(np.abs(z))) if z.size else 0.0,
            "clipped_features": self.names_[self.keep_][clipped.any(axis=0)].tolist(),
            "shifted_dropped_rows": int(shifted.any(axis=1).sum()),
            "shifted_dropped_features": self.names_[~self.keep_][shifted.any(axis=0)].tolist(),
        }
