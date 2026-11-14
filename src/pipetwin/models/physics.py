"""E03 positive rate integration and regularized corrections to observed VTD."""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipetwin.features import daily_mode


DAYS_PER_YEAR = 365.25
RATE_COLUMNS = ["water_fraction", "temperature_c", "velocity_m_s", "fill_coefficient"]


def build_exposure(tech, ids, start, end):
    """One day per section, including unknown days, with no forward filling."""
    observed = daily_mode(tech, start, end)
    observed["observed"] = True
    days = pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), freq="D")
    grid = pd.MultiIndex.from_product([pd.Index(ids).unique(), days], names=["id", "day"]).to_frame(index=False)
    exposure = grid.merge(observed[["id", "day", "observed", *RATE_COLUMNS]],
                          on=["id", "day"], how="left", validate="one_to_one")
    exposure["observed"] = exposure.observed.eq(True)
    exposure["dt_years"] = 1 / DAYS_PER_YEAR
    return exposure


def validate_inputs(frame, exposure, y=None):
    if frame.empty or frame.duplicated(["id", "segment_id"]).any():
        raise ValueError("E03 requires nonempty unique segment keys")
    if not (np.isfinite(frame.previous_thickness_mm) & (frame.previous_thickness_mm > 0)).all():
        raise ValueError("E03 requires positive observed previous thickness")
    if not (np.isfinite(frame.mode_horizon_days) & (frame.mode_horizon_days > 0)).all():
        raise ValueError("E03 requires a positive calendar horizon")
    if frame.groupby("id").mode_horizon_days.nunique().gt(1).any():
        raise ValueError("Each section must have one horizon per prediction call")
    if exposure.duplicated(["id", "day"]).any():
        raise ValueError("Duplicate exposure days would double count corrosion")
    if exposure.day.isna().any() or exposure.id.isna().any():
        raise ValueError("Exposure needs dated section keys")
    if not (np.isfinite(exposure.dt_years) & (exposure.dt_years > 0)).all():
        raise ValueError("Exposure intervals must have finite positive duration")
    duration = exposure.groupby("id").dt_years.sum()
    expected = frame.mode_horizon_days.to_numpy() / DAYS_PER_YEAR
    actual = frame.id.map(duration).to_numpy()
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-10):
        raise ValueError("Exposure must cover exactly the requested calendar horizon")
    if np.isinf(exposure[RATE_COLUMNS].to_numpy(dtype=float)).any():
        raise ValueError("Exposure contains infinite channel values")
    for column in ["water_fraction", "fill_coefficient"]:
        if not exposure[column].dropna().between(0, 1).all():
            raise ValueError(f"Invalid physical fraction in {column}")
    if exposure.velocity_m_s.dropna().lt(0).any():
        raise ValueError("Exposure velocity must be nonnegative")
    if y is not None:
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not (np.isfinite(y) & (y > 0)).all():
            raise ValueError("Labels must be finite positive observed thicknesses")


def robust_residual(error, delta=0.5):
    """Pseudo-Huber data residual; separately appended penalties remain quadratic."""
    return error * np.sqrt(2 / (np.sqrt(1 + (error / delta) ** 2) + 1))


def fitted_optimizer(residual, initial, bounds):
    result = least_squares(residual, initial, bounds=bounds, max_nfev=2000,
                           ftol=1e-9, xtol=1e-9, gtol=1e-9)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Rate optimization failed: {result.message}")
    return result


class PositiveRateModel:
    """r(t)=exp(intercept + beta*z(t)) mm/year, integrated on daily intervals."""

    def __init__(self, constant=False, regularization=0.25, missing_policy="train-median"):
        if regularization <= 0 or not np.isfinite(regularization):
            raise ValueError("Rate regularization must be finite and positive")
        if missing_policy not in {"train-median", "observed-only"}:
            raise ValueError("Unknown exposure missing-data scenario")
        self.constant, self.regularization = constant, regularization
        self.missing_policy = missing_policy

    @staticmethod
    def _raw_matrix(exposure):
        matrix = exposure[RATE_COLUMNS].copy()
        matrix["velocity_m_s"] = np.log1p(matrix.velocity_m_s)
        return matrix

    def _matrix(self, exposure):
        if self.constant:
            return np.empty((len(exposure), 0))
        # Bound extrapolation of standardized channels, not observed target differences.
        return np.clip(self.transformer_.transform(self._raw_matrix(exposure)), -5, 5)

    def _duration(self, exposure):
        if self.missing_policy == "observed-only":
            return exposure.dt_years.to_numpy() * exposure.observed.to_numpy()
        return exposure.dt_years.to_numpy()

    def fit(self, frame, exposure, y):
        validate_inputs(frame, exposure, y)
        history = exposure.loc[exposure.id.isin(frame.id)].copy()
        if not self.constant:
            observed = history.loc[history.observed]
            if observed.empty:
                raise ValueError("Parametric rate fitting requires observed training TECH days")
            self.transformer_ = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                               StandardScaler())
            self.transformer_.fit(self._raw_matrix(observed))
        matrix, duration = self._matrix(history), self._duration(history)
        ids = pd.Index(pd.unique(history.id))
        daily_group = ids.get_indexer(history.id)
        row_group = ids.get_indexer(frame.id)
        prior = frame.previous_thickness_mm.to_numpy()
        # Negative measured differences remain in the fitting residual.
        observed_rate = (prior - np.asarray(y)) / (frame.mode_horizon_days.to_numpy() / DAYS_PER_YEAR)
        initial_rate = np.clip(np.median(observed_rate), 1e-4, 1.0)

        def residual(parameters):
            rate = np.exp(parameters[0] + matrix @ parameters[1:])
            integrated = np.bincount(daily_group, weights=rate * duration, minlength=len(ids))
            predicted = np.maximum(0, prior - integrated[row_group])
            data = robust_residual(predicted - np.asarray(y)) / np.sqrt(len(frame))
            penalty = np.sqrt(self.regularization) * parameters[1:]
            return np.concatenate([data, penalty])

        initial = np.r_[np.log(initial_rate), np.zeros(matrix.shape[1])]
        lower = np.r_[-18.0, np.full(matrix.shape[1], -2.0)]
        upper = np.r_[np.log(10), np.full(matrix.shape[1], 2.0)]
        solution = fitted_optimizer(residual, initial, (lower, upper))
        self.parameters_ = solution.x
        self.optimization_ = {"nfev": solution.nfev, "cost": float(solution.cost),
                              "success": bool(solution.success)}
        return self

    def daily_rates(self, exposure):
        matrix = self._matrix(exposure)
        return np.exp(self.parameters_[0] + matrix @ self.parameters_[1:])

    def integrated_loss(self, frame, exposure):
        validate_inputs(frame, exposure)
        daily_loss = self.daily_rates(exposure) * self._duration(exposure)
        grouped = pd.Series(daily_loss, index=exposure.index).groupby(exposure.id).sum()
        return frame.id.map(grouped).to_numpy()

    def predict(self, frame, exposure):
        return np.maximum(0, frame.previous_thickness_mm.to_numpy() - self.integrated_loss(frame, exposure))


class HybridScaleModel:
    """Regularized segment-specific positive multiplier on the physical integral."""

    def __init__(self, physics, columns, regularization=1.0):
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("Scale regularization must be positive")
        self.physics, self.columns, self.regularization = physics, list(columns), regularization

    def _matrix(self, frame):
        values = np.clip(self.transformer_.transform(frame[self.columns]), -5, 5)
        return np.column_stack([np.ones(len(frame)), values])

    def fit(self, frame, exposure, y):
        validate_inputs(frame, exposure, y)
        self.transformer_ = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                           StandardScaler())
        self.transformer_.fit(frame[self.columns])
        matrix = self._matrix(frame)
        base_loss = self.physics.integrated_loss(frame, exposure)
        prior = frame.previous_thickness_mm.to_numpy()

        def residual(coefficients):
            multiplier = np.exp(np.clip(matrix @ coefficients, -5, 5))
            predicted = np.maximum(0, prior - base_loss * multiplier)
            data = robust_residual(predicted - np.asarray(y)) / np.sqrt(len(frame))
            return np.r_[data, np.sqrt(self.regularization) * coefficients]

        solution = fitted_optimizer(residual, np.zeros(matrix.shape[1]), (-2.0, 2.0))
        self.coefficients_ = solution.x
        return self

    def integrated_loss(self, frame, exposure):
        multiplier = np.exp(np.clip(self._matrix(frame) @ self.coefficients_, -5, 5))
        return self.physics.integrated_loss(frame, exposure) * multiplier

    def predict(self, frame, exposure):
        return np.maximum(0, frame.previous_thickness_mm.to_numpy() - self.integrated_loss(frame, exposure))


class ObservedThicknessModel:
    """ML signed annualized change, optionally correcting the physical prediction."""

    def __init__(self, columns, physics=None, seed=42):
        self.columns, self.physics, self.seed = list(columns), physics, seed

    def _base(self, frame, exposure):
        if self.physics is None:
            return frame.previous_thickness_mm.to_numpy()
        return self.physics.predict(frame, exposure)

    def fit(self, frame, exposure, y):
        validate_inputs(frame, exposure, y)
        years = frame.mode_horizon_days.to_numpy() / DAYS_PER_YEAR
        target = (np.asarray(y) - self._base(frame, exposure)) / years
        self.estimator_ = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=8,
                                  random_state=self.seed, n_jobs=1),
        )
        self.estimator_.fit(frame[self.columns], target)
        return self

    def predict(self, frame, exposure):
        validate_inputs(frame, exposure)
        years = frame.mode_horizon_days.to_numpy() / DAYS_PER_YEAR
        correction = self.estimator_.predict(frame[self.columns]) * years
        return np.maximum(0, self._base(frame, exposure) + correction)
