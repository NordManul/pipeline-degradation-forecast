"""Segment-balanced rates from historical one-to-one correspondence hypotheses."""

import numpy as np
import pandas as pd

from .baselines import GlobalRate


class MatchedRateModel:
    def __init__(self, kind="global", min_segments=10, bandwidth_m=300.0, prior_weight=20.0, segment_length_m=10.0):
        if kind not in {"global", "local", "local_nonnegative"}:
            raise ValueError("Unknown matched-rate model")
        if min_segments < 1 or bandwidth_m <= 0 or prior_weight <= 0 or segment_length_m <= 0:
            raise ValueError("Invalid matched-rate parameters")
        self.kind, self.min_segments = kind, min_segments
        self.bandwidth_m, self.prior_weight, self.segment_length_m = bandwidth_m, prior_weight, segment_length_m

    def fit_pairs(self, pairs, training_frame, training_target_date):
        self.trained_through_ = pd.Timestamp(training_target_date)
        if pd.isna(self.trained_through_) or training_frame.duplicated(["id", "segment_id"]).any():
            raise ValueError("Finite training date and unique training cells are required")
        if not training_frame.target_date.eq(self.trained_through_).all():
            raise ValueError("Training snapshot date disagrees with the declared training target")
        if pairs[["earlier_date", "later_date"]].isna().any().any():
            raise ValueError("Pair dates must be finite")
        if len(pairs) and (pd.to_datetime(pairs.later_date) > self.trained_through_).any():
            raise ValueError("Future pairs cannot calibrate a historical model")
        if len(pairs) and (pd.to_datetime(pairs.earlier_date) >= pd.to_datetime(pairs.later_date)).any():
            raise ValueError("Pair dates must increase")
        train = training_frame.loc[training_frame.observed]
        fallback = GlobalRate("between").fit(train, train.y_mm).rate_
        known = pairs.loc[pairs.growth_eligible].copy()
        if len(known):
            if not np.isfinite(known[["earlier_distance_m", "later_distance_m", "observed_loss_rate_mm_year"]].to_numpy(dtype=float)).all():
                raise ValueError("Eligible growth rates and coordinates must be finite")
            allowed = pd.MultiIndex.from_frame(training_frame[["id", "segment_id"]])
            # Purge BOTH endpoints: an old defect in a held-out cell may migrate
            # into a train cell after alignment, which must not bypass the fold.
            for side in ["earlier", "later"]:
                cells = known[["id"]].assign(segment_id=np.floor(known[f"{side}_distance_m"] / self.segment_length_m).astype(int))
                if not pd.MultiIndex.from_frame(cells).isin(allowed).all():
                    raise ValueError("Pair endpoint outside the training fold")
            known["segment_id"] = np.floor(known.later_distance_m / self.segment_length_m).astype(int)
            grouped = known.groupby(["id", "segment_id"], as_index=False).agg(
                rate=("observed_loss_rate_mm_year", "median"), pairs=("earlier_row_id", "size"))
            # All candidate growth calibration must lie in the allowed training grid.
            rates = grouped.merge(training_frame[["id", "segment_id", "main_id", "chain_position_m"]],
                                  on=["id", "segment_id"], how="left", validate="one_to_one")
            if rates.chain_position_m.isna().any():
                raise ValueError("Pair rates outside the training fold")
        else:
            rates = pd.DataFrame(columns=["id", "segment_id", "rate", "pairs", "main_id", "chain_position_m"])
        self.calibration_pair_count_ = len(known)
        self.calibration_segment_count_ = len(rates)
        self.fallback_used_ = len(rates) < self.min_segments
        self.rate_ = fallback if self.fallback_used_ else float(rates.rate.median())
        self.rates_ = np.empty(0) if self.fallback_used_ else rates.rate.to_numpy(dtype=float)
        self.positions_ = np.empty(0) if self.fallback_used_ else rates.chain_position_m.to_numpy(dtype=float)
        self.mains_ = np.empty(0, dtype=object) if self.fallback_used_ else rates.main_id.to_numpy()
        self.segment_rates_ = rates
        if self.kind == "local_nonnegative":
            self.rate_ = max(0, self.rate_)
            self.rates_ = np.maximum(0, self.rates_)
        return self

    def predict(self, frame, *, spatial_development=False):
        target = pd.to_datetime(frame.target_date)
        if target.isna().any():
            raise ValueError("Prediction dates must be finite")
        if (target < self.trained_through_).any() or ((target == self.trained_through_).any() and not spatial_development):
            raise ValueError("Prediction date must follow the training target; same-date use is spatial development only")
        if not frame.scenario.eq("between").all():
            raise ValueError("Matching rates require a between-inspection scenario")
        rates = np.full(len(frame), self.rate_)
        if self.kind != "global" and len(self.rates_):
            distance = frame.chain_position_m.to_numpy()[:, None] - self.positions_[None, :]
            weights = np.exp(-0.5 * (distance / self.bandwidth_m) ** 2)
            weights *= frame.main_id.to_numpy()[:, None] == self.mains_[None, :]
            rates = (weights @ self.rates_ + self.prior_weight * self.rate_) / (weights.sum(axis=1) + self.prior_weight)
        prediction = frame.anchor_mm.to_numpy() - frame.horizon_years.to_numpy() * rates
        unknown = ~frame.anchor_observed.to_numpy(dtype=bool)
        if unknown.any():
            if "fallback_mm" not in frame or not np.isfinite(frame.loc[unknown, "fallback_mm"]).all():
                raise ValueError("Unknown anchors require a finite shared fallback_mm")
            prediction[unknown] = frame.loc[unknown, "fallback_mm"]
        return prediction
