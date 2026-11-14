"""Reference forecasts and robust signed rates of *observed* thickness change."""

import numpy as np


def initial_level(frame, scenario):
    return frame["anchor_mm" if scenario == "between" else "nominal_mm"].to_numpy(dtype=float)


def response_scale(frame, scenario):
    horizon = frame["horizon_years"].to_numpy(dtype=float)
    # Fraction of nominal wall per year transfers between different cold-start nominals.
    return horizon if scenario == "between" else horizon * frame["nominal_mm"].to_numpy(dtype=float)


def rate_target(frame, y, scenario):
    return (initial_level(frame, scenario) - np.asarray(y)) / response_scale(frame, scenario)


class Persistence:
    def __init__(self, scenario, smooth=False):
        self.scenario, self.smooth = scenario, smooth

    def fit(self, frame, y):
        return self

    def predict(self, frame):
        level = initial_level(frame, self.scenario)
        if self.smooth and self.scenario == "between":
            neighbor = frame["neighbor_mean_mm"].to_numpy(dtype=float)
            level = np.where(np.isfinite(neighbor), 0.5 * level + 0.5 * neighbor, level)
        return level


class GlobalRate:
    def __init__(self, scenario):
        self.scenario = scenario

    def fit(self, frame, y):
        rates = rate_target(frame, y, self.scenario)
        eligible = frame.anchor_observed.eq(1).to_numpy() if self.scenario == "between" else np.ones(len(frame), dtype=bool)
        if not eligible.any():
            raise ValueError("No observed anchors for rate calibration")
        self.rate_ = float(np.median(rates[eligible]))
        return self

    def predict(self, frame):
        return np.maximum(0, initial_level(frame, self.scenario) - self.rate_ * response_scale(frame, self.scenario))


class LocalRate(GlobalRate):
    """Shrink a spatial kernel estimate to the global rate; no new-main borrowing."""

    def __init__(self, scenario, bandwidth_m=300.0, prior_weight=20.0):
        super().__init__(scenario)
        self.bandwidth_m, self.prior_weight = bandwidth_m, prior_weight

    def fit(self, frame, y):
        super().fit(frame, y)
        eligible = frame.anchor_observed.eq(1).to_numpy() if self.scenario == "between" else np.ones(len(frame), dtype=bool)
        self.positions_ = frame.loc[eligible, "chain_position_m"].to_numpy()
        self.mains_ = frame.loc[eligible, "main_id"].to_numpy()
        self.rates_ = rate_target(frame, y, self.scenario)[eligible]
        return self

    def predict(self, frame):
        distance = frame.chain_position_m.to_numpy()[:, None] - self.positions_[None, :]
        weights = np.exp(-0.5 * (distance / self.bandwidth_m) ** 2)
        weights *= frame.main_id.to_numpy()[:, None] == self.mains_[None, :]
        rates = (weights @ self.rates_ + self.prior_weight * self.rate_) / (weights.sum(axis=1) + self.prior_weight)
        return np.maximum(0, initial_level(frame, self.scenario) - rates * response_scale(frame, self.scenario))
