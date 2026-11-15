"""Cold-start thickness and thin-cell ranking from donor profiles."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


POSITION = ["position_fraction"]
PROFILE = POSITION + ["altitude_relative_m"]
FINAL_MODEL = "xgboost_knn"
BOOST_MODEL = "xgboost"
PRIORITY_MODEL = "random_forest"


def weights_by_campaign(train, *, balance_mains=True):
    """Equal campaign weights, optionally after assigning equal weight to each main."""
    count = train.groupby(["main_id", "target_date"]).sample_id.transform("size")
    if balance_mains:
        count *= train.groupby("main_id").target_date.transform("nunique")
    weight = 1.0 / count.to_numpy()
    return weight / weight.mean()


def blend_profiles(frame, base, local):
    """Align the KNN level and spread to XGBoost within each full section profile."""
    aligned = np.empty(len(frame))
    groups = frame.groupby(["id", "target_date"], dropna=False).indices
    for positions in groups.values():
        reference, shape = base[positions], local[positions]
        spread = shape.std()
        standardized = (shape - shape.mean()) / spread if spread > 1e-9 else np.zeros_like(shape)
        aligned[positions] = reference.mean() + reference.std() * standardized
    return .5 * base + .5 * aligned


class ColdProfile:
    """Fit XGBoost, KNN and a separate forest for the thin-cell list."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, train):
        # Histogram subsampling and KNN ties depend on row order.
        rows = train.sort_values(["main_id", "target_date", "id", "segment_id"])
        nominal = rows.nominal_mm.to_numpy()
        years = rows.horizon_years.to_numpy()
        fraction = rows.y_mm.to_numpy() / nominal
        if (not len(rows) or not np.isfinite(fraction).all() or (fraction <= 0).any()
                or not np.isfinite(years).all() or (years <= 0).any()):
            raise ValueError("Cold-profile training requires valid observed thickness and positive age")
        rate = -np.log(fraction) / years
        weight = weights_by_campaign(rows, balance_mains=False)
        self.boost_ = XGBRegressor(
            n_estimators=160, learning_rate=.04, grow_policy="lossguide", max_depth=0,
            max_leaves=15, min_child_weight=20, colsample_bytree=.7, subsample=.7,
            reg_lambda=10, tree_method="hist", n_jobs=1, random_state=self.seed,
        ).fit(rows[PROFILE], rate, sample_weight=weight)
        self.knn_ = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(), KNeighborsRegressor(n_neighbors=min(12, len(rows)), weights="distance"),
        ).fit(rows[POSITION], rows.y_mm.to_numpy())
        self.forest_ = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            RandomForestRegressor(n_estimators=64, max_depth=6, min_samples_leaf=10,
                                  max_features=.8, n_jobs=1, random_state=self.seed),
        ).fit(rows[PROFILE], rate, randomforestregressor__sample_weight=weight)
        return self

    def predict(self, frame):
        nominal = frame.nominal_mm.to_numpy()
        years = frame.horizon_years.to_numpy()
        boost = nominal * np.exp(np.clip(-years * self.boost_.predict(frame[PROFILE]), -100, 100))
        forest = nominal * np.exp(np.clip(-years * self.forest_.predict(frame[PROFILE]), -100, 100))
        local = self.knn_.predict(frame[POSITION])
        values = {BOOST_MODEL: boost, PRIORITY_MODEL: forest, FINAL_MODEL: blend_profiles(frame, boost, local)}
        if not all(np.isfinite(profile).all() for profile in values.values()):
            raise ValueError("Nonfinite cold-profile thickness")
        return {name: np.clip(profile, 0, nominal) for name, profile in values.items()}
