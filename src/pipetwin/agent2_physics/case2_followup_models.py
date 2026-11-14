"""Candidate estimators for the fixed case 2 follow-up comparison."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC

from .broad_case2 import decode, learner, TabularCandidate
from .residual_guard import TrainBoundedFeatures


class ProfileKernel(Kernel):
    """Smooth shared relative-position trend plus a linear height effect."""

    def __init__(self, length=0.2):
        self.length = length

    def __call__(self, X, Y=None, eval_gradient=False):
        left = np.asarray(X)
        right = left if Y is None else np.asarray(Y)
        distance = (left[:, None, 0] - right[None, :, 0]) / self.length
        covariance = np.exp(-0.5 * distance ** 2) + 0.25 * np.outer(left[:, 1], right[:, 1])
        if eval_gradient:
            if Y is not None:
                raise ValueError("Gradients require Y=None")
            return covariance, np.empty(covariance.shape + (0,))
        return covariance

    def diag(self, X):
        return 1.0 + 0.25 * np.asarray(X)[:, 1] ** 2

    def is_stationary(self):
        return False


class SmoothProfile:
    def __init__(self, spec, columns, length):
        self.spec, self.columns, self.length = spec, columns, length

    def matrix(self, frame):
        x = frame[self.columns].fillna(self.medians).to_numpy()
        x[:, 1] = np.clip((x[:, 1] - self.medians.iloc[1]) / self.height_scale, -3, 3)
        return x

    def fit(self, frame, y, weights):
        self.medians = frame[self.columns].median().fillna(0)
        self.height_scale = max(float(frame[self.columns[1]].std(ddof=0)), 0.01)
        self.gp = GaussianProcessRegressor(kernel=ProfileKernel(self.length), normalize_y=True,
                                           alpha=1.0 / weights, optimizer=None)
        self.gp.fit(self.matrix(frame), y)
        pred = self.gp.predict(self.matrix(frame))
        self.noise = float(np.sqrt(np.average((y - pred) ** 2, weights=weights)))
        return self

    def predict(self, frame, *, development=False):
        return decode(self.gp.predict(self.matrix(frame)), frame, self.spec.response)

    def interval(self, frame):
        mean, std = self.gp.predict(self.matrix(frame), return_std=True)
        spread = 1.2815515655 * np.hypot(std, self.noise)
        return decode(mean + spread, frame, self.spec.response), decode(mean - spread, frame, self.spec.response)


class PooledZones:
    """Regularized fixed effects and empirical shrinkage of shared zones.

    Source section effects absorb local offsets during estimation. They are
    set to zero on an unseen section. A zone is a fifth of relative length.
    """

    def __init__(self, spec, columns):
        self.spec, self.columns = spec, columns

    @staticmethod
    def zones(frame):
        return np.clip((frame.position_fraction.to_numpy() * 5).astype(int), 0, 4)

    @staticmethod
    def shrink(keys, values, weights, noise):
        table = pd.DataFrame({"key": keys, "wy": values * weights, "w": weights})
        sums = table.groupby("key")[["wy", "w"]].sum()
        means = sums.wy / sums.w
        variance = max(0.0, float(np.average(means ** 2 - noise / sums.w, weights=sums.w)))
        effect = means * np.divide(variance * sums.w, variance * sums.w + max(noise, 1e-16))
        return effect, variance

    def fit(self, frame, y, weights):
        self.transform = TrainBoundedFeatures(self.columns).fit(frame[self.columns].to_numpy())
        x = self.transform.transform(frame[self.columns].to_numpy())
        self.fixed = Ridge(alpha=100).fit(x, y, sample_weight=weights)
        residual = y - self.fixed.predict(x)
        noise = float(np.average(residual ** 2, weights=weights))
        section_effect, self.section_variance = self.shrink(frame.id.to_numpy(), residual, weights, noise)
        residual -= frame.id.map(section_effect).to_numpy()
        zone = self.zones(frame)
        self.zone_effect, self.zone_variance = self.shrink(zone, residual, weights, noise)
        residual -= pd.Series(zone).map(self.zone_effect).to_numpy()
        self.noise = float(np.average(residual ** 2, weights=weights))
        return self

    def rate(self, frame):
        x = self.transform.transform(frame[self.columns].to_numpy())
        return self.fixed.predict(x) + pd.Series(self.zones(frame)).map(self.zone_effect).fillna(0).to_numpy()

    def predict(self, frame, *, development=False):
        return decode(self.rate(frame), frame, self.spec.response)

    def interval(self, frame):
        mean = self.rate(frame)
        spread = 1.2815515655 * np.sqrt(self.noise + self.section_variance)
        return decode(mean + spread, frame, self.spec.response), decode(mean - spread, frame, self.spec.response)


class PairwiseRank:
    def __init__(self, spec, columns):
        self.spec, self.columns = spec, columns

    def fit(self, frame, y, weights):
        self.transform = TrainBoundedFeatures(self.columns).fit(frame[self.columns].to_numpy())
        x = self.transform.transform(frame[self.columns].to_numpy())
        rng = np.random.default_rng(self.spec.seed)
        differences, directions, pair_weights = [], [], []
        for indices in frame.groupby(["id", "target_date"]).indices.values():
            left, right = np.triu_indices(len(indices), k=1)
            if len(left) > 4000:
                picked = rng.choice(len(left), size=4000, replace=False)
                left, right = left[picked], right[picked]
            a, b = indices[left], indices[right]
            keep = y[a] != y[b]
            a, b = a[keep], b[keep]
            if not len(a):
                continue
            delta = x[a] - x[b]
            sign = np.sign(y[a] - y[b])
            differences.extend([delta, -delta])
            directions.extend([sign, -sign])
            pair_weights.extend([np.full(len(sign), 1.0 / len(sign))] * 2)
        w = np.concatenate(pair_weights)
        self.rank = LinearSVC(C=0.03, fit_intercept=False, dual=False).fit(
            np.vstack(differences), np.concatenate(directions), sample_weight=w / w.mean())
        self.calibration = IsotonicRegression(out_of_bounds="clip").fit(
            self.rank.decision_function(x), y, sample_weight=weights)
        return self

    def predict(self, frame, *, development=False):
        x = self.transform.transform(frame[self.columns].to_numpy())
        rate = self.calibration.predict(self.rank.decision_function(x))
        return decode(rate, frame, self.spec.response)


class AdditivePhysics:
    def __init__(self, physical, correction, columns):
        self.physical, self.correction, self.columns = physical, correction, columns

    def predict(self, frame, *, development=False):
        base = self.physical.predict(frame, development=development)
        years = frame.horizon_years.to_numpy()
        delta = np.clip(self.correction.predict(frame[self.columns]) * years, -0.5, 0.5)
        return base - delta


def boosted_estimator(name, seed):
    if name.startswith("xgb"):
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=160, learning_rate=0.04, grow_policy="lossguide",
                            max_depth=0, max_leaves=int(name[3:]), min_child_weight=20,
                            colsample_bytree=0.7, subsample=0.7, reg_lambda=10,
                            tree_method="hist", n_jobs=1, random_state=seed)
    return CatBoostRegressor(iterations=160, depth=3, learning_rate=0.04, loss_function="RMSE",
                             l2_leaf_reg=10, random_seed=seed, thread_count=1,
                             verbose=False, allow_writing_files=False)


def continued_boosting(spec, frame, columns, response, weights):
    first = frame.target_date.eq(frame.target_date.min()).to_numpy()
    model = boosted_estimator("catboost", spec.seed)
    model.fit(frame.loc[first, columns], response[first], sample_weight=weights[first])
    if (~first).any():
        model = CatBoostRegressor(iterations=80, depth=3, learning_rate=0.02, loss_function="RMSE",
                                  l2_leaf_reg=10, random_seed=spec.seed, thread_count=1,
                                  verbose=False, allow_writing_files=False).fit(
            frame.loc[~first, columns], response[~first], sample_weight=weights[~first], init_model=model)
    return TabularCandidate(spec, columns, model)


def fit_correction(physical, frame, columns, seed, weights):
    loss_residual = (physical.predict(frame) - frame.y_mm.to_numpy()) / frame.horizon_years.to_numpy()
    correction = learner("forest", columns, seed)
    correction.fit(frame[columns], loss_residual, randomforestregressor__sample_weight=weights)
    return AdditivePhysics(physical, correction, columns)
