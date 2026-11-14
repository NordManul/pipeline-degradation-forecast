"""Small controlled alternatives for warm profiles and cold physical loss."""

import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..agent2_physics import make_model as make_physics
from ..case_data import _fallback_fraction
from ..case_experiments import query_view
from ..validation import blocked_folds
from .matched_rates import MatchedRateModel
from .tabular import make_estimator


SEED = 42
ZONE_LENGTH_M = 1000.0
PRIOR_CELLS = 20
CORRECTION_ALPHA = 100.0
LOG_FACTOR_LIMIT = np.log(2.)
RATIO_FLOOR_MM = .1


def anchored(frame):
    return frame.loc[frame.observed & frame.anchor_observed]


def warm_bounds(query, values, fallback):
    nominal = query.nominal_mm.to_numpy()
    return np.clip(np.where(query.anchor_observed, values, nominal * fallback), 0, nominal)


def pooled_mean(values, n, prior, k):
    return (n * values + k * prior) / (n + k)


def direct_components(fit, query, columns):
    models = [make_estimator(kind, SEED).fit(fit[columns], fit.y_mm) for kind in ("gam", "catboost")]
    return [model.predict(query[columns]) for model in models]


def warm_regressors(train, query, columns, fallback):
    """Change one objective or the response centering at a time."""
    fit = anchored(train)
    x, x_query = fit[columns], query[columns]
    y = fit.y_mm.to_numpy()

    def predict(kind, response, **params):
        estimator = make_estimator(kind, SEED)
        if params:
            estimator[-1].set_params(**params)
        return estimator.fit(x, response).predict(x_query)

    gam, cat = [warm_bounds(query, values, fallback) for values in direct_components(fit, query, columns)]
    out = {"baseline": .5 * (gam + cat), "gam_direct": gam, "catboost_rmse": cat}
    for alpha, tag in [(.05, "q05"), (.10, "q10")]:
        pred = warm_bounds(query, predict("catboost", y, loss_function=f"Quantile:alpha={alpha:g}"), fallback)
        out[f"quantile_{tag}"] = pred
        out[f"quantile_{tag}_with_catboost"] = .5 * (pred + cat)

    by_campaign = fit.groupby(["id", "target_date"]).y_mm
    within = fit.y_mm - by_campaign.transform("mean")
    # The current warm case has one training campaign. If there are several,
    # the most recent available training level is applied per simple section.
    levels = by_campaign.agg(["mean", "size"]).reset_index().sort_values("target_date").drop_duplicates("id", keep="last").set_index("id")
    levels["shrunk"] = pooled_mean(levels["mean"], levels["size"], y.mean(), PRIOR_CELLS)
    section_level = query.id.map(levels.shrunk).fillna(y.mean()).to_numpy()
    spatial_gam, spatial_cat = predict("gam", within), predict("catboost", within)
    out["level_profile"] = warm_bounds(query, section_level + .5 * (spatial_gam + spatial_cat), fallback)
    return out


class PairwiseThickness:
    """YetiRank with nonnegative-slope millimetre calibration from inner OOF."""

    def __init__(self, columns, buffer_m):
        self.columns = columns
        self.buffer_m = buffer_m

    def _fit_ranker(self, frame):
        frame = anchored(frame).sort_values(["id", "target_date", "segment_id"])
        groups = frame.groupby(["id", "target_date"], sort=False).ngroup().to_numpy()
        relevance = (1 - frame.y_mm / frame.nominal_mm).clip(0, 1)
        ranker = CatBoostRanker(loss_function="YetiRank", iterations=350, depth=4, learning_rate=.04,
                                 l2_leaf_reg=10, random_seed=SEED, thread_count=1, verbose=False, allow_writing_files=False)
        x = frame[self.columns].to_numpy(dtype=float)
        ranker.fit(x, relevance, group_id=groups)
        raw = -ranker.predict(x)
        return ranker, raw.mean(), max(raw.std(), 1e-6)

    def _score(self, ranker, center, scale, frame):
        raw = -ranker.predict(frame[self.columns].to_numpy(dtype=float))
        return (raw - center) / scale

    def fit(self, train):
        scores = np.full(len(train), np.nan)
        for split in blocked_folds(train, n_splits=3, buffer_m=self.buffer_m):
            ranker = self._fit_ranker(train.iloc[split.train])
            scores[split.valid] = self._score(*ranker, query_view(train.iloc[split.valid]))
        mask = (train.observed & train.anchor_observed).to_numpy()
        score, y = scores[mask], train.loc[mask, "y_mm"].to_numpy()
        if not np.isfinite(score).all():
            raise ValueError("Pairwise calibration needs complete inner out-of-fold ranks")
        centered = score - score.mean()
        self.slope_ = max(0., np.dot(centered, y - y.mean()) / max(np.dot(centered, centered), 1e-12))
        self.intercept_ = y.mean() - self.slope_ * score.mean()
        self.ranker_ = self._fit_ranker(train)
        self.calibration_rows_ = len(y)
        return self

    def predict(self, frame):
        return self.intercept_ + self.slope_ * self._score(*self.ranker_, frame)


class PooledMatchedRates:
    """Use signs of observed changes; shrink zone estimates by distinct cells."""

    @staticmethod
    def _zones(frame):
        return (frame.chain_position_m // ZONE_LENGTH_M).astype(int)

    def fit(self, train, pairs, segment_length_m):
        observed = anchored(train)
        rates = (observed.anchor_mm - observed.y_mm) / observed.horizon_years
        self.prior_ = rates.median()
        zones = rates.groupby(self._zones(observed)).agg(["median", "size"])
        self.zone_rate_ = pooled_mean(zones["median"], zones["size"], self.prior_, PRIOR_CELLS)
        reliable = pairs.loc[pairs.alignment_supported]
        model = MatchedRateModel(segment_length_m=segment_length_m)
        matched = model.fit_pairs(reliable, train, train.target_date.iloc[0]).segment_rates_
        zones = matched.groupby(self._zones(matched)).rate.agg(["median", "size"])
        prior = self.zone_rate_.reindex(zones.index).fillna(self.prior_)
        self.matched_rate_ = pooled_mean(zones["median"], zones["size"], prior, PRIOR_CELLS)
        self.matched_cells_ = len(matched)
        self.fallback_ = _fallback_fraction(train)
        return self

    def predict(self, frame):
        zones = self._zones(frame)
        zone_rate = zones.map(self.zone_rate_).fillna(self.prior_).to_numpy()
        matched = zones.map(self.matched_rate_).fillna(pd.Series(zone_rate, index=frame.index)).to_numpy()
        values = [zone_rate, matched, np.maximum(0., matched)]
        names = ["zone_rate", "pooled_matched", "pooled_matched_positive"]
        return {name: warm_bounds(frame, frame.anchor_mm - frame.horizon_years * rate, self.fallback_)
                for name, rate in zip(names, values)}


class MultiplicativeColdPrior:
    """A five-parameter daily rate; terrain Ridge corrects its log multiplier."""

    def __init__(self, exposure, cutoff, feature_columns, missing_policy):
        self.feature_columns = feature_columns
        self.prior = make_physics("positive_rate", scenario="cold", exposure=exposure,
                                  cutoff_exclusive=cutoff, feature_columns=[], seed=SEED,
                                  options={"missing_policy": missing_policy})

    def _loss(self, frame):
        self.prior.predict(frame)
        return self.prior.last_prediction_.integrated_loss_mm.to_numpy()

    def _values(self, frame):
        return np.clip(self.scale_.transform(self.impute_.transform(frame[self.feature_columns])), -3., 3.)

    def fit(self, frame, weights):
        self.prior.fit(frame, frame.y_mm.to_numpy())
        loss = np.maximum(frame.nominal_mm - frame.y_mm, RATIO_FLOOR_MM)
        prior_loss = np.maximum(self._loss(query_view(frame)), RATIO_FLOOR_MM)
        response = np.clip(np.log(loss / prior_loss), -LOG_FACTOR_LIMIT, LOG_FACTOR_LIMIT)
        self.impute_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(frame[self.feature_columns])
        self.scale_ = StandardScaler().fit(self.impute_.transform(frame[self.feature_columns]))
        self.ridge_ = Ridge(alpha=CORRECTION_ALPHA).fit(self._values(frame), response, sample_weight=weights)
        return self

    def predict(self, frame):
        factor = np.exp(np.clip(self.ridge_.predict(self._values(frame)), -LOG_FACTOR_LIMIT, LOG_FACTOR_LIMIT))
        return frame.nominal_mm.to_numpy() - self._loss(frame) * factor
