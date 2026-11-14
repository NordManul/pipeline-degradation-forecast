"""Small tabular estimators; all preprocessing is fitted on the training fold."""

from catboost import CatBoostRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


def make_estimator(kind, seed):
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if kind == "gam":
        return make_pipeline(imputer, VarianceThreshold(1e-12),
                             SplineTransformer(n_knots=4, degree=2, knots="uniform", extrapolation="constant", include_bias=False),
                             StandardScaler(), Ridge(alpha=100.0))
    if kind == "catboost":
        return make_pipeline(imputer, CatBoostRegressor(
            iterations=350, depth=4, learning_rate=0.04, l2_leaf_reg=10,
            loss_function="RMSE", random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False))
    raise ValueError(f"Unknown tabular estimator: {kind}")
