"""Calibrated nonnegative exposure loss; not a mechanistic CO2 corrosion law."""

import numpy as np
from scipy.optimize import nnls
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features import feature_columns
from .baselines import initial_level


class ExposureModel:
    """Loss in mm = nonnegative coefficients (mm/year) x exposures (years).

    No pressure, chemistry, coupon rate, or future corrosion proxy is inferred.
    Missing observed-day exposures add no exposure; elapsed time remains a
    separate term, and missing coverage is retained in the ML feature table.
    """

    def __init__(self, scenario, regularization=1.0):
        self.scenario, self.regularization = scenario, regularization

    def design(self, frame):
        horizon = frame.horizon_years.to_numpy(dtype=float)
        active = frame.active_years_observed.fillna(0).to_numpy(dtype=float)
        water = frame.water_exposure_years.fillna(0).to_numpy(dtype=float)
        # z / 1 m is a dimensionless, bounded local accumulation proxy.
        depression = frame.lowpoint_depth_m.fillna(0).to_numpy(dtype=float)
        accumulation = depression / (1.0 + depression)
        return np.column_stack([horizon, active, water, water * accumulation])

    def fit(self, frame, y):
        design = self.design(frame)
        loss = initial_level(frame, self.scenario) - np.asarray(y)
        if self.scenario == "between":
            # Nominal fallback is not a measured old thickness and cannot
            # calibrate physical growth over the between-inspection interval.
            eligible = frame.anchor_observed.eq(1).to_numpy()
            if not eligible.any():
                raise ValueError("No observed anchors for physical growth calibration")
            design, loss = design[eligible], loss[eligible]
        self.scale_ = np.maximum(np.sqrt(np.mean(design ** 2, axis=0)), 1e-9)
        standardized = design / self.scale_
        augmented = np.vstack([standardized, np.sqrt(self.regularization) * np.eye(design.shape[1])])
        response = np.concatenate([loss, np.zeros(design.shape[1])])
        self.coefficients_, _ = nnls(augmented, response)
        self.rates_mm_per_year_ = self.coefficients_ / self.scale_
        self.calibration_rows_ = len(loss)
        return self

    def predict(self, frame):
        loss = self.design(frame) / self.scale_ @ self.coefficients_
        return np.maximum(0, initial_level(frame, self.scenario) - loss)


class PhysicsResidual:
    """Exposure forecast with a regularized signed correction for observed thickness."""

    def __init__(self, scenario):
        self.physics = ExposureModel(scenario)
        self.columns = feature_columns(scenario)
        self.correction = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(), Ridge(alpha=100.0))

    def fit(self, frame, y):
        self.physics.fit(frame, y)
        self.correction.fit(frame[self.columns], np.asarray(y) - self.physics.predict(frame))
        return self

    def predict(self, frame):
        return np.maximum(0, self.physics.predict(frame) + self.correction.predict(frame[self.columns]))
