"""Small tabular estimators; all preprocessing is fitted on the training fold."""

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from ..features import feature_columns
from .baselines import initial_level, rate_target, response_scale


def make_estimator(kind, seed):
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if kind in {"ridge", "ridge_direct"}:
        return make_pipeline(imputer, StandardScaler(), Ridge(alpha=30.0))
    if kind == "huber":
        return make_pipeline(imputer, StandardScaler(), HuberRegressor(alpha=1.0, max_iter=2000))
    if kind == "gam":
        return make_pipeline(imputer, VarianceThreshold(1e-12),
                             SplineTransformer(n_knots=4, degree=2, knots="uniform", extrapolation="constant", include_bias=False),
                             StandardScaler(), Ridge(alpha=100.0))
    if kind == "random_forest":
        return make_pipeline(imputer, RandomForestRegressor(
            n_estimators=200, max_depth=7, min_samples_leaf=8, max_features=0.8, random_state=seed, n_jobs=1))
    if kind == "catboost":
        return make_pipeline(imputer, CatBoostRegressor(
            iterations=350, depth=4, learning_rate=0.04, l2_leaf_reg=10,
            loss_function="RMSE", random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False))
    raise ValueError(f"Unknown tabular estimator: {kind}")


class TabularModel:
    def __init__(self, scenario, kind, seed=42, columns=None):
        self.scenario, self.kind = scenario, kind
        self.columns = feature_columns(scenario) if columns is None else columns
        self.estimator = make_estimator(kind, seed)

    def fit(self, frame, y):
        response = np.asarray(y) if self.kind == "ridge_direct" else rate_target(frame, y, self.scenario)
        self.estimator.fit(frame[self.columns], response)
        return self

    def predict(self, frame):
        response = self.estimator.predict(frame[self.columns])
        prediction = response if self.kind == "ridge_direct" else initial_level(frame, self.scenario) - response_scale(frame, self.scenario) * response
        return np.maximum(0, prediction)
