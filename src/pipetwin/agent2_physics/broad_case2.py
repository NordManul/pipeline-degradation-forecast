"""Fixed factorial search for case2; scoring holds out whole mains."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
import tempfile
from time import perf_counter
import warnings

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from pipetwin.case_experiments import fit_case_model, predict_models, prediction_table, range_diagnostics
from pipetwin.models.tabular import make_estimator
from pipetwin.pipeline import write_json
from pipetwin.quality import read_raw_sources

from .broad_case2_features import add_features
from .residual_guard import TrainBoundedFeatures
from .retrospective_case2 import (
    campaign_metrics, compare_query_inputs, fallback, prepare, validate_group_split,
)
from .shared_comparison import comparison_config, verify_shared_predictions


MODES = ("direct", "loss_fraction", "mm_rate", "fraction_rate", "log_rate")
FAMILIES = ("ridge", "bounded_ridge", "huber", "gam", "kernel", "knn", "forest",
            "extra_trees", "hist", "catboost_small")
PHYSICS = ("constant_rate", "positive_rate", "hybrid_residual", "hybrid_scale")
BLENDS = {
    "blend_cat_gam": ("reference_catboost", "reference_gam"),
    "blend_cat_guard": ("reference_catboost", "reference_guard"),
}


@dataclass(frozen=True)
class Candidate:
    name: str
    learner: str
    features: str = "canonical"
    response: str = "fraction_rate"
    pool: str = "all"
    policy: str = "train-median"
    seed: int = 42


def candidates(groups):
    controls = [Candidate("reference_" + name, name) for name in
                ["persistence", "global_rate", "gam", "catboost", "e02_terrain"]]
    controls += [Candidate("reference_guard", "agent2_hybrid_residual")]
    trials = [Candidate(f"{family}__{block}__{response}", family, block, response)
              for family, block, response in product(FAMILIES, groups, MODES)]
    # Matched coverage of campaign ablations with scalable learners. All fixed
    # options and responses in this subgrid are shared across its three pools.
    trials += [Candidate(f"{family}__{block}__{response}__{pool}", family, block, response, pool)
               for family, block, response, pool in product(
                   ["bounded_ridge", "hist", "catboost_small"], ["canonical", "terrain", "all"],
                   ["direct", "fraction_rate"], ["first", "last", "balanced"])]
    trials += [Candidate(f"physics_{model}__{block}__{policy}__{basis}", "agent2_" + model,
                         block, basis, policy=policy)
               for model, policy, basis in product(PHYSICS, ["observed-only", "train-median"],
                                                  ["absolute", "relative"])
               for block in (["canonical", "terrain", "daily_terrain"] if model.startswith("hybrid")
                             else ["canonical"])]
    trials += [Candidate("heuristic_" + name, "heuristic", name) for name in
               ["basin_30", "basin_100", "basin_250", "relief_100", "slope_abs_100",
                "daily_wet_slow_mean", "basin_100_by_wet_slow", "profile_percentile_100"]]
    return controls + trials


def training_rows(frame, pool):
    measured = frame.loc[frame.observed]
    weights = None
    if pool in {"first", "last"}:
        grouped = measured.groupby("main_id").target_date
        date = grouped.transform("min" if pool == "first" else "max")
        measured = measured.loc[measured.target_date.eq(date)]
    if pool == "balanced":
        counts = measured.groupby(["main_id", "target_date"]).sample_id.transform("size")
        weights = 1.0 / counts.to_numpy()
        weights /= weights.mean()
    return measured, weights


def encode(frame, mode):
    y = frame.y_mm.to_numpy()
    nominal = frame.nominal_mm.to_numpy()
    years = frame.horizon_years.to_numpy()
    loss = nominal - y
    if mode == "direct":
        return y
    if mode == "loss_fraction":
        return loss / nominal
    if mode == "mm_rate":
        return loss / years
    if mode == "log_rate":
        return -np.log(y / nominal) / years
    return loss / (nominal * years)


def decode(response, frame, mode):
    nominal = frame.nominal_mm.to_numpy()
    years = frame.horizon_years.to_numpy()
    if mode == "direct":
        return response
    if mode == "loss_fraction":
        return nominal * (1 - response)
    if mode == "mm_rate":
        return nominal - years * response
    if mode == "log_rate":
        # Stable evaluation of the exponential, with shared output clipping
        # downstream. The exponent cap only prevents numerical overflow.
        return nominal * np.exp(np.clip(-years * response, -100, 100))
    return nominal * (1 - years * response)


def learner(kind, columns, seed=42):
    if kind == "bounded_ridge":
        return make_pipeline(TrainBoundedFeatures(columns), Ridge(alpha=30))
    if kind == "gam":
        return make_estimator("gam", seed)
    if kind == "catboost_small":
        regressor = CatBoostRegressor(iterations=120, depth=3, learning_rate=0.05, l2_leaf_reg=10,
                                      thread_count=1, random_seed=seed, verbose=False, allow_writing_files=False)
    else:
        regressors = {
            "ridge": Ridge(alpha=30),
            "huber": HuberRegressor(alpha=1.0, max_iter=1000),
            "kernel": KernelRidge(alpha=2.0, kernel="rbf", gamma=0.5 / len(columns)),
            "knn": KNeighborsRegressor(n_neighbors=12, weights="distance"),
            "forest": RandomForestRegressor(n_estimators=64, max_depth=6, min_samples_leaf=10,
                                            max_features=0.8, n_jobs=1, random_state=seed),
            "extra_trees": ExtraTreesRegressor(n_estimators=64, max_depth=6, min_samples_leaf=10,
                                              max_features=0.8, n_jobs=1, random_state=seed),
            "hist": HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=9, l2_regularization=2,
                                                 early_stopping=False, random_state=seed),
        }
        regressor = regressors[kind]
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), regressor)


class TabularCandidate:
    def __init__(self, spec, columns, estimator, constant=None):
        self.spec = spec
        self.columns = columns
        self.estimator = estimator
        self.constant = constant

    def predict(self, frame, *, development=False):
        response = (np.full(len(frame), self.constant) if self.estimator is None
                    else self.estimator.predict(frame[self.columns]))
        return decode(response, frame, self.spec.response)


class TerrainHeuristic:
    def __init__(self, column, scale, level):
        self.column, self.scale, self.level = column, scale, level

    def predict(self, frame, *, development=False):
        risk = frame[self.column].fillna(0).to_numpy() / self.scale
        return frame.nominal_mm.to_numpy() * (self.level - 0.05 * np.clip(risk, 0, 4))


class FixedAverage:
    def __init__(self, models):
        self.models = models

    def predict(self, frame, *, development=False):
        # Same policy as shared fixed blends: average already bounded models.
        _, bounded = predict_models(dict(enumerate(self.models)), frame, {}, development=development)
        return np.mean(list(bounded.values()), axis=0)


def fit(spec, package, frame, groups):
    measured, weights = training_rows(frame, spec.pool)
    if spec.name.startswith("reference_") or spec.learner.startswith("agent2_"):
        config = comparison_config(spec.policy, "train-bounded")
        config["seed"] = spec.seed
        if spec.response == "relative":
            config["agent2_options"]["cold_nominal_scaling"] = "train-relative"
        context = {**package, "feature_columns": groups[spec.features]}
        return fit_case_model(spec.learner, context, measured, fallback(measured), config)[0]
    if spec.learner == "heuristic":
        scale = max(float(measured[spec.features].fillna(0).abs().quantile(0.9)), 1e-6)
        return TerrainHeuristic(spec.features, scale, fallback(measured)["fraction"])
    columns = groups[spec.features]
    response = encode(measured, spec.response)
    # GAM's variance selector needs at least one varying feature. With no
    # information the fitted response is the training mean.
    if spec.learner == "gam" and not measured[columns].var(ddof=0).gt(1e-12).any():
        return TabularCandidate(spec, columns, None, float(response.mean()))
    model = learner(spec.learner, columns, spec.seed)
    kwargs = {} if weights is None else {model.steps[-1][0] + "__sample_weight": weights}
    model.fit(measured[columns], response, **kwargs)
    return TabularCandidate(spec, columns, model)


def ranking(metrics, failed_fits):
    bounded = metrics.loc[metrics.output.eq("bounded")].copy()
    by_main = bounded.assign(r=bounded.pearson_r.fillna(0)).groupby(
        ["model", "held_main"], sort=False)[["r", "mae_mm", "rmse_mm", "recall_thin_10pct"]].mean()
    result = by_main.groupby("model").mean().rename(columns={"r": "mean_main_r", "mae_mm": "mean_main_mae",
                                                              "rmse_mm": "mean_main_rmse",
                                                              "recall_thin_10pct": "mean_main_recall"})
    groups = bounded.groupby("model").pearson_r
    result["min_campaign_r"] = groups.agg(lambda r: r.fillna(0).min())
    result["positive_campaigns"] = groups.agg(lambda r: r.gt(0).sum())
    result["defined_campaigns"] = groups.count()
    result["campaigns"] = groups.size()
    result["fit_warning"] = result.index.isin(failed_fits)
    ref = result.loc["reference_catboost"]
    result["passes_gate"] = (
        ~result.fit_warning & result.positive_campaigns.eq(result.campaigns)
        & result.min_campaign_r.gt(ref.min_campaign_r)
        & result.mean_main_r.gt(ref.mean_main_r)
        & result.mean_main_mae.lt(ref.mean_main_mae)
        & result.mean_main_recall.ge(ref.mean_main_recall)
    )
    return result.reset_index().sort_values(
        ["min_campaign_r", "mean_main_r", "mean_main_mae", "model"], ascending=[False, False, True, True])


def evaluate(specs, package, groups, blends=BLENDS):
    train = package["train_frame"]
    tables, predictions = [], []
    timings = []
    for fold in package["folds"]:
        fit_frame, valid = train.iloc[fold["train"]], train.iloc[fold["valid"]]
        validate_group_split(fit_frame, valid)
        raw, bounded = {}, {}
        for index, spec in enumerate(specs, 1):
            started = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                fitted = fit(spec, package, fit_frame, groups)
            convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            for warning in caught:
                if not issubclass(warning.category, ConvergenceWarning):
                    warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
            r, b = predict_candidate(spec.name, fitted, valid, development=True)
            raw.update(r)
            bounded.update(b)
            timings.append({"fold": fold["number"], "model": spec.name,
                            "seconds": perf_counter() - started, "convergence_warning": "\n".join(convergence)})
            if index % 50 == 0:
                print(f"Hold {valid.main_id.iloc[0]}: {index}/{len(specs)}", flush=True)
        for name, (left, right) in blends.items():
            raw[name] = (raw[left] + raw[right]) / 2
            bounded[name] = (bounded[left] + bounded[right]) / 2
        tables.append(campaign_metrics(valid, raw, bounded, fold["number"]))
        predictions.append((valid, raw, bounded, fold["number"]))
    return pd.concat(tables, ignore_index=True), predictions, timings


def predict_candidate(name, model, frame, *, development=False):
    raw, bounded = predict_models({name: model}, frame, {}, development=development)
    if raw[name].shape != (len(frame),) or not np.isfinite(raw[name]).all():
        raise ValueError(f"Invalid forecast from {name}")
    return raw, bounded


def run(output, strict_reference):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw, manifest = read_raw_sources("data/raw")
    package = prepare(raw, manifest)
    del raw
    reference_check = compare_query_inputs(package, joblib.load(strict_reference))
    for key in ["train_frame", "predict_frame"]:
        package[key], groups = add_features(package[key], package["exposure"])
    specs = candidates(groups)
    names = [spec.name for spec in specs] + list(BLENDS)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate candidate names")
    train, query = package["train_frame"], package["predict_frame"]
    with tempfile.TemporaryDirectory(prefix=".case2-search-", dir=output.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        sources = [Path(__file__), Path(__file__).with_name("broad_case2_features.py"),
                   Path(__file__).with_name("broad_case2_protocol.md")]
        write_json(staging / "frozen_grid.json", {"configurations": [asdict(s) for s in specs],
                   "feature_groups": groups, "blends": BLENDS, "input_manifest": manifest,
                   "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in sources},
                   "target_state_date": package["target_state_date"],
                   "training_available_until": package["training_available_until"],
                   "reference_check": reference_check})
        with threadpool_limits(limits=1):
            metrics, predictions, timings = evaluate(specs, package, groups)
            board = ranking(metrics, {t["model"] for t in timings if t["convergence_warning"]})
            if len(board) != len(names) or not board.campaigns.eq(4).all():
                raise ValueError("Incomplete experiment matrix")
            metrics.to_csv(staging / "metrics.csv", index=False)
            board.to_csv(staging / "leaderboard.csv", index=False)
            pd.DataFrame(timings).to_csv(staging / "timings.csv", index=False)
            converged = board.loc[~board.fit_warning]
            corr = converged.loc[converged.defined_campaigns.eq(converged.campaigns)].iloc[0].model
            mae = converged.sort_values(["mean_main_mae", "model"]).iloc[0].model
            qualified = board.loc[board.passes_gate, "model"].tolist()
            recommendation = qualified[0] if qualified else None
            best_qualified_mae = converged.loc[converged.passes_gate].sort_values(
                ["mean_main_mae", "model"]).model.head(1).tolist()
            chosen = list(dict.fromkeys([corr, mae, *qualified[:1], *best_qualified_mae,
                                         "reference_catboost", "reference_global_rate"]))
            write_json(staging / "selection.json", {"correlation_leader": corr, "mae_leader": mae,
                       "gate_passes": qualified, "recommended_candidate": recommendation,
                       "target_vtd_available": False, "status": "exploratory_search"})
            parts = [prediction_table(valid, {n: raw[n] for n in chosen},
                                      {n: bounded[n] for n in chosen}, number)
                     for valid, raw, bounded, number in predictions]
            oof = pd.concat(parts, ignore_index=True)
            verify_shared_predictions(package, oof)
            oof.to_csv(staging / "leader_oof.csv.gz", index=False)
            indexed = {s.name: s for s in specs}
            (staging / "models").mkdir()
            final_raw, final_bounded = {}, {}
            for name in chosen:
                model = (FixedAverage([fit(indexed[n], package, train, groups) for n in BLENDS[name]])
                         if name in BLENDS else fit(indexed[name], package, train, groups))
                r, b = predict_candidate(name, model, query)
                path = staging / "models" / f"{name}.joblib"
                joblib.dump(model, path, compress=3)
                del model
                rr, bb = predict_candidate(name, joblib.load(path), query)
                np.testing.assert_allclose(rr[name], r[name], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(bb[name], b[name], rtol=1e-12, atol=1e-12)
                final_raw.update(r)
                final_bounded.update(b)
            forecast = prediction_table(query, final_raw, final_bounded, -1)
            verify_shared_predictions(package, forecast, forecast=True)
            forecast.to_csv(staging / "candidate_forecasts.csv.gz", index=False)
            range_diagnostics(query, final_raw, final_bounded).to_csv(staging / "forecast_ranges.csv", index=False)
            query.to_csv(staging / "query_features.csv.gz", index=False)
            write_json(staging / "summary.json", {"configurations": len(names), "fits": len(timings),
                       "train_labels": int(train.observed.sum()), "query_cells": len(query),
                       "held_mains": sorted(train.main_id.unique()), "forecast_models_replayed": chosen})
        staging.rename(output)
    return board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strict-reference", required=True, type=Path)
    args = parser.parse_args()
    board = run(args.output, args.strict_reference)
    print(board.head(8).to_string(index=False))


if __name__ == "__main__":
    # Ensure joblib stores importable class names even when launched with -m.
    from pipetwin.agent2_physics.broad_case2 import main as run_cli
    run_cli()
