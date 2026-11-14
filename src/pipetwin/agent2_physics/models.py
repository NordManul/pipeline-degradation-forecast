"""Daily nonnegative loss and train-only residual correction, contract 1.0.0.

The positive rate and pseudo-Huber calibration are adapted from E03 commit
659c251cba1f8e3e73c232758487099793537571. Contexts, validation, cold forecasts,
fallbacks and the public interface follow the shared contract instead.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import CONTRACT_VERSION
from .validation import (RATE_COLUMNS, feature_matrix, integrate_loss, timestamp,
                         validate_features, validate_inputs)


DEFAULT_OPTIONS = {
    "missing_policy": "observed-only",
    "rate_regularization": 0.25,
    "robust_delta_mm": 0.5,
    "residual_alpha": 100.0,
    "scale_regularization": 1.0,
    "max_nfev": 2000,
}


def parse_options(options):
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("options must be a mapping")
    supplied = dict(options or {})
    if set(supplied) - set(DEFAULT_OPTIONS):
        raise ValueError(f"Unknown model options: {sorted(set(supplied) - set(DEFAULT_OPTIONS))}")
    result = {**DEFAULT_OPTIONS, **supplied}
    if result["missing_policy"] not in {"observed-only", "train-median"}:
        raise ValueError("missing_policy must be observed-only or train-median")
    for key in ["rate_regularization", "robust_delta_mm", "residual_alpha", "scale_regularization"]:
        value = result[key]
        if isinstance(value, (str, bool)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        result[key] = float(value)
    if isinstance(result["max_nfev"], bool) or not isinstance(result["max_nfev"], (int, np.integer)) or result["max_nfev"] < 1:
        raise ValueError("max_nfev must be a positive integer")
    result["max_nfev"] = int(result["max_nfev"])
    return result


def robust_residual(error, delta):
    """Squared residual equals twice the pseudo-Huber loss, preserving signs."""
    error = np.asarray(error, dtype=float)
    return error * np.sqrt(2 / (np.sqrt(1 + (error / delta) ** 2) + 1))


def optimize(residual, initial, bounds, max_nfev):
    result = least_squares(residual, initial, bounds=bounds, max_nfev=max_nfev,
                           ftol=1e-9, xtol=1e-9, gtol=1e-9)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Physical rate optimization failed: {result.message}")
    return result.x, {"success": bool(result.success), "status": int(result.status),
                      "nfev": int(result.nfev), "cost": float(result.cost),
                      "message": str(result.message)}


class PhysicsEstimator:
    """A separate fitted context for one scenario, cutoff and canonical exposure."""

    def __init__(self, name, *, scenario, exposure, cutoff_exclusive,
                 feature_columns, seed=42, options=None):
        if name not in {"constant_rate", "positive_rate", "hybrid_residual", "hybrid_scale"}:
            raise ValueError(f"Unknown or unimplemented physical model: {name}")
        if scenario not in {"between", "cold"}:
            raise ValueError("scenario must be between or cold")
        if not isinstance(exposure, pd.DataFrame):
            raise ValueError("exposure must be a canonical DataFrame")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self.name, self.scenario = name, scenario
        self.cutoff = timestamp(cutoff_exclusive)
        self.feature_columns = validate_features(feature_columns, scenario)
        if name in {"hybrid_residual", "hybrid_scale"} and not self.feature_columns:
            raise ValueError(f"{name} requires explicit correction features")
        self.seed, self.options = int(seed), parse_options(options)
        # A caller's later edits cannot mutate the model's case context.
        self.exposure = exposure.copy(deep=True)
        self._fitted = False

    @staticmethod
    def _raw_rate_matrix(history):
        values = history[list(RATE_COLUMNS)].to_numpy(dtype=float).copy()
        values[:, 2] = np.log1p(values[:, 2])
        return values

    def _rate_matrix(self, history):
        if self.name == "constant_rate":
            return np.empty((len(history), 0))
        return np.clip(self.rate_transformer_.transform(self._raw_rate_matrix(history)), -5, 5)

    def _loss(self, frame, history):
        rates = np.exp(self.rate_parameters_[0] + self._rate_matrix(history) @ self.rate_parameters_[1:])
        losses = integrate_loss(history, rates, self.options["missing_policy"])
        return frame.exposure_id.map(losses).to_numpy(dtype=float)

    def _scale_matrix(self, frame):
        matrix = self.scale_transformer_.transform(feature_matrix(frame, self.feature_columns))
        return np.column_stack([np.ones(len(frame)), np.clip(matrix, -5, 5)])

    def fit(self, frame, y_mm):
        self._fitted = False
        self.diagnostics_ = {}
        self.last_prediction_ = None
        data = validate_inputs(frame, self.exposure, self.scenario, self.cutoff, y=y_mm)
        train = data.frame.loc[data.eligible]
        history = data.exposure.loc[data.exposure.exposure_id.isin(train.exposure_id)].copy()
        observed = history.loc[history.observed]
        if self.name != "constant_rate":
            if observed.empty:
                raise ValueError("Positive-rate calibration needs observed training TECH days")
            self.rate_transformer_ = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
            self.rate_transformer_.fit(self._raw_rate_matrix(observed))
        matrix = self._rate_matrix(history)
        durations = history.dt_years.to_numpy(dtype=float)
        if self.options["missing_policy"] == "observed-only":
            durations = durations * history.observed.to_numpy()
        identifiers = pd.Index(history.exposure_id.unique())
        day_group = identifiers.get_indexer(history.exposure_id)
        sample_group = identifiers.get_indexer(train.exposure_id)
        total_years = np.bincount(day_group, weights=durations, minlength=len(identifiers))[sample_group]
        if not (total_years > 0).any():
            raise ValueError("No nonzero exposure available for rate calibration in this missing-data scenario")
        initial, y = data.initial[data.eligible], data.y[data.eligible]
        # Keep signed observed loss: increasing measured thickness is not zero loss.
        observed_loss = initial - y
        start_rate = np.clip(np.median(observed_loss[total_years > 0] / total_years[total_years > 0]), 1e-4, 1)

        def residual(parameters):
            rates = np.exp(parameters[0] + matrix @ parameters[1:])
            loss = np.bincount(day_group, weights=rates * durations, minlength=len(identifiers))[sample_group]
            prediction = np.maximum(0, initial - loss)
            errors = robust_residual(prediction - y, self.options["robust_delta_mm"]) / np.sqrt(len(train))
            return np.r_[errors, np.sqrt(self.options["rate_regularization"]) * parameters[1:]]

        parameters = np.r_[np.log(start_rate), np.zeros(matrix.shape[1])]
        bounds = (np.r_[-18.0, np.full(matrix.shape[1], -2.0)],
                  np.r_[np.log(10.0), np.full(matrix.shape[1], 2.0)])
        self.rate_parameters_, status = optimize(residual, parameters, bounds, self.options["max_nfev"])
        loss = self._loss(train, history)
        physics = np.maximum(0, initial - loss)
        correction_diagnostics = None
        scale_diagnostics = None
        if self.name == "hybrid_residual":
            self.correction_ = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                StandardScaler(), Ridge(alpha=self.options["residual_alpha"]))
            # Signed mm/year correction transfers explicitly across horizons.
            response = (y - physics) / data.years[data.eligible]
            self.correction_.fit(feature_matrix(train, self.feature_columns), response)
            estimator = self.correction_[-1]
            correction_diagnostics = {"unit": "mm/year", "intercept": float(estimator.intercept_),
                                      "coefficients": estimator.coef_.tolist()}
        if self.name == "hybrid_scale":
            self.scale_transformer_ = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                StandardScaler())
            self.scale_transformer_.fit(feature_matrix(train, self.feature_columns))
            scale_matrix = self._scale_matrix(train)

            def scale_residual(coefficients):
                multiplier = np.exp(np.clip(scale_matrix @ coefficients, -5, 5))
                prediction = np.maximum(0, initial - loss * multiplier)
                errors = robust_residual(prediction - y, self.options["robust_delta_mm"]) / np.sqrt(len(train))
                return np.r_[errors, np.sqrt(self.options["scale_regularization"]) * coefficients]

            self.scale_parameters_, scale_status = optimize(
                scale_residual, np.zeros(scale_matrix.shape[1]), (-2.0, 2.0), self.options["max_nfev"])
            scale_diagnostics = {"coefficients": self.scale_parameters_.tolist(),
                                 "optimizer": scale_status, "log_multiplier_bounds": [-5.0, 5.0]}
        self.diagnostics_ = {
            "model": self.name, "scenario": self.scenario, "contract_version": CONTRACT_VERSION,
            "cutoff_exclusive": self.cutoff.isoformat(), "timezone": "Europe/Moscow",
            "seed": self.seed, "options": dict(self.options),
            "training_rows": len(data.frame), "calibration_rows": len(train),
            "training_histories": int(data.frame.exposure_id.nunique()),
            "calibration_histories": len(identifiers), "observed_days": len(observed),
            "calendar_days": len(history), "excluded_unobserved_anchors": int((~data.eligible).sum()),
            "negative_observed_losses": int((observed_loss < 0).sum()),
            "rate_columns": [] if self.name == "constant_rate" else list(RATE_COLUMNS),
            "feature_columns": list(self.feature_columns),
            "rate_parameters": self.rate_parameters_.tolist(), "optimizer": status,
            "residual_correction": correction_diagnostics, "last_predict_fallback_rows": 0,
            "scale_correction": scale_diagnostics,
            "physical_lower_bound_mm": 0, "shared_final_clipping_applied": False,
            "rate_log_intercept_bounds": [-18.0, float(np.log(10.0))],
            "rate_coefficient_bounds": [-2.0, 2.0], "standardized_channel_bounds": [-5.0, 5.0],
        }
        if self.name != "constant_rate":
            self.diagnostics_["rate_preprocessing"] = {
                "imputer_medians": self.rate_transformer_[0].statistics_.tolist(),
                "mean": self.rate_transformer_[1].mean_.tolist(),
                "scale": self.rate_transformer_[1].scale_.tolist(),
                "empty_training_channels": [c for c in RATE_COLUMNS if observed[c].isna().all()],
                "empty_channel_fill": 0.0,
            }
        self._fitted = True
        return self

    def predict(self, frame):
        if not self._fitted:
            raise RuntimeError("Fit the model successfully before prediction")
        data = validate_inputs(frame, self.exposure, self.scenario, self.cutoff)
        prediction = data.fallback.copy()
        physical = np.full(len(data.frame), np.nan)
        loss = np.full(len(data.frame), np.nan)
        eligible = data.eligible
        if eligible.any():
            active = data.frame.loc[eligible]
            loss[eligible] = self._loss(active, data.exposure)
            if self.name == "hybrid_scale":
                loss[eligible] *= np.exp(np.clip(self._scale_matrix(active) @ self.scale_parameters_, -5, 5))
            physical[eligible] = np.maximum(0, data.initial[eligible] - loss[eligible])
            prediction[eligible] = physical[eligible]
            if self.name == "hybrid_residual":
                correction = self.correction_.predict(feature_matrix(active, self.feature_columns))
                prediction[eligible] += correction * data.years[eligible]
        if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
            raise RuntimeError("Model produced a non-finite or incomplete prediction")
        self.diagnostics_["last_predict_fallback_rows"] = int((~eligible).sum())
        # Unobserved anchors have no physical state estimate; their fallback is exact.
        self.last_prediction_ = pd.DataFrame({
            "sample_id": data.frame.sample_id, "physical_prediction_mm": physical,
            "integrated_loss_mm": loss, "prediction_mm": prediction,
            "used_fallback": ~eligible,
        })
        return prediction
