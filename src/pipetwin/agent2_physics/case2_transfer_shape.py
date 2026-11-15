"""Audit transfer of normalized terrain features and spatial forecasts."""

import argparse
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import product
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

from pipetwin.case2_data import prepare_transfer
from pipetwin.case_experiments import prediction_table, range_diagnostics
from pipetwin.pipeline import write_json
from pipetwin.quality import clean_case2_donors, read_raw_sources
from pipetwin.sources import read_config

from .broad_case2 import TabularCandidate, encode, predict_candidate, training_rows
from .case2_followup import fit_trial
from .case2_followup_models import boosted_estimator
from .case2_release_checks import SEEDS, spec
from .data_quality_features import valid_minima
from .residual_guard import TrainBoundedFeatures
from .retrospective_case2 import campaign_metrics, compare_query_inputs, validate_group_split
from .shared_comparison import verify_shared_predictions


POSITION = ["position_fraction", "altitude_relative_m"]
SHAPE = ["height_percentile", "detrend_relative", "height_relative_range"]
LAND = [f"{feature}_{radius}" for radius in [100, 250] for feature in ["below_saddle", "valley_width", "relief_ratio"]]
FEATURES = {"reference": POSITION, "rank": ["position_fraction", "height_percentile"],
            "shape": ["position_fraction", *SHAPE], "land": [*SHAPE, *LAND],
            "combined": [*POSITION, *SHAPE, *LAND]}
ARCHITECTURES = ("ridge2", "xgb", "xgb_centered")
SMOOTH_RADII = (30, 100, 250)
REFERENCE = "xgb15"
POOL_RESPONSES = ("log_rate", "fraction_rate", "mm_rate")
POOL_WEIGHTS = ("campaign", "section", "half")
POOL_DATES = ("all", "first_two", "without_first", "without_middle")


def add_shape(frame):
    result = frame.copy()
    for column in SHAPE + LAND:
        result[column] = np.nan
    for _, section in frame.groupby("exposure_id", sort=False):
        section = section.sort_values("center_m")
        x = section.center_m.to_numpy()
        z = section.altitude_relative_m.to_numpy()
        finite = np.isfinite(z)
        if finite.sum() < 2:
            continue
        x, z, index = x[finite], z[finite], section.index[finite]
        if np.any(np.diff(x) <= 0):
            raise ValueError("Expected a strictly increasing profile")
        height_range = max(float(np.quantile(z, .95) - np.quantile(z, .05)), .2)
        chord = np.interp(x, x[[0, -1]], z[[0, -1]])
        rank = (pd.Series(z).rank(method="average").to_numpy() - .5) / len(z)
        result.loc[index, SHAPE] = np.column_stack([rank, (z - chord) / height_range, z / height_range])
        for radius in [100, 250]:
            offsets = np.linspace(-radius, radius, 21)
            sampled = np.interp(x[:, None] + offsets, x, z)
            saddle = np.minimum(sampled[:, :10].max(axis=1), sampled[:, 11:].max(axis=1))
            depth = np.maximum(0, saddle - z)
            below = sampled < saddle[:, None]
            # Fraction of the along-profile window in the contiguous basin
            # around its center, stopping at the first estimated saddle.
            left = np.cumprod(below[:, 10::-1], axis=1).sum(axis=1)
            right = np.cumprod(below[:, 10:], axis=1).sum(axis=1)
            width = np.maximum(0, left + right - 1) / len(offsets)
            relief = np.ptp(sampled, axis=1) / height_range
            cols = [f"{feature}_{radius}" for feature in ["below_saddle", "valley_width", "relief_ratio"]]
            result.loc[index, cols] = np.column_stack([depth / height_range, width, relief])
    return result


def smooth_profile(frame, values, radius_m):
    result = np.empty(len(frame))
    for indices in frame.groupby("exposure_id", sort=False).indices.values():
        x = frame.center_m.to_numpy()[indices]
        order = indices[np.argsort(x)]
        position = frame.segment_id.to_numpy()[order]
        if len(position) > 1 and not np.all(np.diff(position) == 1):
            raise ValueError("Spatial smoothing requires the complete canonical grid")
        result[order] = uniform_filter1d(np.asarray(values)[order], size=1 + 2 * (radius_m // 10), mode="nearest")
    return result


def attach_defect_types(frame, minima):
    extra = minima[["id", "segment_id", "date", "is_corrosion"]].rename(columns={"date": "target_date"})
    frame = frame.merge(extra, on=["id", "segment_id", "target_date"], how="left", validate="one_to_one")
    if frame.loc[frame.observed, "is_corrosion"].isna().any():
        raise ValueError("Every training minimum must resolve to an inspection record")
    return frame


@dataclass
class ShapeModel:
    columns: list
    model: object
    level: float

    def predict(self, frame, *, development=False):
        rate = self.level + self.model.predict(frame[self.columns])
        return frame.nominal_mm.to_numpy() * np.exp(np.clip(-frame.horizon_years.to_numpy() * rate, -100, 100))


def fit_variant(kind, columns, frame, seed, reweight):
    train, weights = training_rows(frame, "balanced")
    if reweight:
        relative = pd.Series(np.where(train.is_corrosion.eq(True), 1., .5), index=train.index)
        mean = relative.groupby([train.main_id, train.target_date]).transform("mean")
        weights *= (relative / mean).to_numpy()
    response = encode(train, "log_rate")
    level = 0.
    if kind == "xgb_centered":
        response = pd.Series(response, index=train.index)
        level = float(np.average(response, weights=weights))
        response -= response.groupby([train.id, train.target_date]).transform("mean")
    if kind == "ridge2":
        model = make_pipeline(TrainBoundedFeatures(columns), PolynomialFeatures(2, include_bias=False),
                              Ridge(alpha=100.))
        model.fit(train[columns], response, ridge__sample_weight=weights)
    else:
        model = boosted_estimator("xgb15", seed).fit(train[columns], response, sample_weight=weights)
    return ShapeModel(columns, model, level)


def fit_models(package, train, seed):
    yield REFERENCE, fit_trial(spec("xgb15", seed), package, train, {"position": POSITION})
    for kind, (block, columns), reweight in product(ARCHITECTURES, FEATURES.items(), [False, True]):
        if kind == "xgb" and block == "reference" and not reweight:
            continue
        name = f"{kind}__{block}__{'type_weight' if reweight else 'all'}"
        yield name, fit_variant(kind, columns, train, seed, reweight)


def donor_rows(frame, policy):
    if policy == "all":
        return frame
    retained = []
    for _, part in frame.groupby("main_id"):
        dates = pd.DatetimeIndex(part.loc[part.observed, "target_date"].unique()).sort_values()
        if len(dates) == 1:
            retained.append(part)
        elif len(dates) == 3:
            kept = {"first_two": dates[:2], "without_first": dates[1:],
                    "without_middle": dates[[0, 2]]}[policy]
            retained.append(part.loc[part.target_date.isin(kept)])
        else:
            raise ValueError("The frozen donor comparison expected three campaigns or one")
    return pd.concat(retained).sort_index()


def donor_weights(train, campaign_weights, policy):
    if policy == "campaign":
        return campaign_weights
    group_size = train.groupby(["main_id", "target_date", "id"]).sample_id.transform("size")
    section_count = train.groupby(["main_id", "target_date"]).id.transform("nunique")
    weights = 1.0 / (group_size * section_count).to_numpy()
    weights /= weights.mean()
    return weights if policy == "section" else (weights + campaign_weights) / 2


def pool_configurations():
    # first_two and without_last coincide in the supplied three-date pool.
    for response, policy in product(POOL_RESPONSES, POOL_WEIGHTS):
        yield response, policy, "all"
    for date in POOL_DATES[1:]:
        yield "log_rate", "campaign", date


def fit_pool_models(package, frame, seed):
    for response, weighting, split in pool_configurations():
        name = f"{response}__{weighting}__{split}"
        if (response, weighting, split) == ("log_rate", "campaign", "all"):
            name = REFERENCE
        train, weights = training_rows(donor_rows(frame, split), "balanced")
        y = encode(train, response)
        model = boosted_estimator("xgb15", seed).fit(
            train[POSITION], y, sample_weight=donor_weights(train, weights, weighting))
        candidate = replace(spec("xgb15", seed), name=name, response=response)
        yield name, TabularCandidate(candidate, POSITION, model)


def add_smoothed(query, raw, bounded):
    for radius in SMOOTH_RADII:
        smooth = smooth_profile(query, raw[REFERENCE], radius)
        raw[f"smooth_{radius}"] = smooth
        raw[f"mix_smooth_{radius}"] = 0.5 * (smooth + raw[REFERENCE])
        for name in [f"smooth_{radius}", f"mix_smooth_{radius}"]:
            bounded[name] = np.clip(raw[name], 0, query.nominal_mm.to_numpy())


def write_priority_comparison(query, prediction, output):
    """Save the exploratory list using deterministic ties on segment ID."""
    profile = query[["id", "segment_id"]].assign(thickness_pred=prediction)
    ordered = profile.sort_values(["thickness_pred", "segment_id"])
    ordered.head(int(np.ceil(0.10 * len(profile)))).to_csv(
        output / "exploratory_priority_smooth250.csv", index=False)


def score_summary(metrics):
    metrics = metrics.loc[metrics.output.eq("bounded")]
    groups = metrics.groupby(["seed", "model", "held_main"])
    scores = groups[["pearson_r", "pearson_squared", "mae_mm", "recall_thin_10pct"]].agg(
        lambda values: values.mean(skipna=False))
    summary = scores.groupby(["seed", "model"]).agg(lambda values: values.mean(skipna=False))
    compare = scores.pearson_squared.unstack()
    for main, name in [("MAIN-2", "main2_r2"), ("MAIN-3", "main3_r2")]:
        summary[name] = compare[main]
    subset = metrics.set_index(["seed", "held_main", "target_date", "model"]).pearson_r.unstack()
    summary["campaign_wins"] = subset.gt(subset[REFERENCE], axis=0).groupby("seed").sum().stack()
    summary["positive_campaigns"] = subset.gt(0).groupby("seed").sum().stack()
    ref = summary.xs(REFERENCE, level="model").pearson_squared
    gains = summary.pearson_squared.sub(ref, level="seed")
    summary["improved_seeds"] = (gains.gt(0).groupby("model").sum()).reindex(
        summary.index.get_level_values("model")).to_numpy()
    baseline = summary.loc[(42, REFERENCE)]
    fixed = summary.xs(42).copy()
    fixed["passes"] = (fixed.positive_campaigns.eq(4) & fixed.campaign_wins.ge(3)
                       & fixed.main2_r2.gt(baseline.main2_r2) & fixed.main3_r2.gt(baseline.main3_r2)
                       & fixed.mae_mm.le(baseline.mae_mm + .02) & fixed.improved_seeds.ge(2))
    fixed = fixed.sort_values("pearson_squared", ascending=False)
    return summary.reset_index(), fixed.reset_index()


def run(output, strict_reference, experiment):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw, manifest = read_raw_sources("data/raw")
    cases, target, protocol, config = (read_config(p) for p in [
        "contracts/cases.yaml", "configs/data_contract.yaml", "contracts/case2/retrospective.yaml", "configs/forecast.yaml"])
    train, query, exposure, _, _ = prepare_transfer(raw, manifest, cases, target, protocol, config)
    donors = clean_case2_donors(raw, cases, target, protocol["pool_cutoff_exclusive"], manifest)
    minima = valid_minima(donors, target["segment_length_m"])
    del raw, donors
    train = attach_defect_types(train, minima).sort_values(
        ["main_id", "target_date", "id", "segment_id"]).reset_index(drop=True)
    train, query = add_shape(train), add_shape(query)
    folds = [{"number": number, "valid": np.flatnonzero(train.main_id.eq(main))}
             for number, main in enumerate(sorted(train.main_id.unique()))]
    package = {"train_frame": train, "predict_frame": query, "folds": folds, "input_manifest": manifest,
               "exposure": exposure}
    query_check = compare_query_inputs(package, joblib.load(strict_reference))
    factory = fit_models if experiment == "shape" else fit_pool_models
    with tempfile.TemporaryDirectory(prefix=".case2-shape-", dir=output.parent) as directory:
        staging = Path(directory) / "run"
        staging.mkdir()
        write_json(staging / "protocol.json", {"experiment": experiment, "seeds": SEEDS, "features": FEATURES,
                   "smoothing_radii": SMOOTH_RADII, "pool_configurations": list(pool_configurations()),
                   "protocol_sha256": sha256(Path(__file__).with_name("CASE2_TRANSFER_SHAPE_PROTOCOL.md").read_bytes()).hexdigest(),
                   "manifest": manifest, "query_invariance": query_check,
                   "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest()})
        metrics, oof = [], []
        with threadpool_limits(limits=1):
            for fold, seed in product(folds, SEEDS):
                valid = train.iloc[fold["valid"]]
                fit = train.loc[~train.main_id.eq(valid.main_id.iloc[0])]
                validate_group_split(fit, valid)
                raw, bounded = from_models(dict(factory(package, fit, seed)), valid, smooth=experiment == "shape")
                metrics.append(campaign_metrics(valid, raw, bounded, fold["number"]).assign(seed=seed))
                oof.append(prediction_table(valid, raw, bounded, fold["number"]).assign(seed=seed))
                print(f"Held {valid.main_id.iloc[0]}, seed {seed}: {len(raw)} variants", flush=True)
            results = pd.concat(metrics, ignore_index=True)
            predictions = pd.concat(oof, ignore_index=True)
            for _, group in predictions.groupby("seed"):
                verify_shared_predictions(package, group)
            if experiment == "pool":
                seed_blends = []
                for fold in folds:
                    valid = train.iloc[fold["valid"]]
                    data = predictions.loc[predictions.fold_id.eq(fold["number"])]
                    data = data.groupby(["model", "sample_id"])[["raw_prediction_mm", "prediction_mm"]].mean()
                    raw = {name: part.droplevel("model").loc[valid.sample_id, "raw_prediction_mm"].to_numpy()
                           for name, part in data.groupby(level="model")}
                    bounded = {name: part.droplevel("model").loc[valid.sample_id, "prediction_mm"].to_numpy()
                               for name, part in data.groupby(level="model")}
                    seed_blends.append(campaign_metrics(valid, raw, bounded, fold["number"]))
                pd.concat(seed_blends, ignore_index=True).to_csv(staging / "seed_ensemble_metrics.csv", index=False)
            by_seed, board = score_summary(results)
            good = board.loc[board.passes]
            winner = good.iloc[0].model if not good.empty else REFERENCE
            experimental = board.loc[board.model.ne(REFERENCE)].iloc[0].model
            names = list(dict.fromkeys([REFERENCE, winner, experimental]))
            if experiment == "shape":
                names.append("mix_smooth_250")
                names = list(dict.fromkeys(names))
            needed = [n for n in names if "smooth_" not in n]
            models = {n: model for n, model in factory(package, train, 42) if n in needed}
            joblib.dump(models, staging / "models.joblib", compress=3)
            forecast_raw, forecast_bounded = from_models(models, query, names, smooth=experiment == "shape")
            replay_raw, replay_bounded = from_models(joblib.load(staging / "models.joblib"), query, names,
                                                     smooth=experiment == "shape")
            for n in names:
                np.testing.assert_allclose(forecast_raw[n], replay_raw[n], atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(forecast_bounded[n], replay_bounded[n], atol=1e-12, rtol=1e-12)
        forecast = prediction_table(query, forecast_raw, forecast_bounded, -1)
        verify_shared_predictions(package, forecast, forecast=True)
        forecast.to_csv(staging / "forecasts.csv.gz", index=False)
        for name in names:
            part = forecast.loc[forecast.model.eq(name), ["id", "segment_id", "prediction_mm"]]
            part.rename(columns={"prediction_mm": "thickness_pred"}).to_csv(staging / f"{name}.csv", index=False)
        if experiment == "shape":
            write_priority_comparison(query, forecast_bounded["mix_smooth_250"], staging)
        query.to_csv(staging / "query_features.csv.gz", index=False)
        range_diagnostics(query, forecast_raw, forecast_bounded).to_csv(staging / "forecast_ranges.csv", index=False)
        results.to_csv(staging / "metrics.csv", index=False)
        by_seed.to_csv(staging / "leaderboard_by_seed.csv", index=False)
        board.to_csv(staging / "leaderboard.csv", index=False)
        predictions.to_csv(staging / "oof.csv.gz", index=False)
        write_json(staging / "summary.json", {"recommended": winner, "highest_scoring_new": experimental,
                   "passing": good.model.tolist(), "configurations": len(board), "cells": len(query),
                   "held_campaigns": int(train.groupby(["main_id", "target_date"]).ngroups),
                   "hidden_target_accuracy_measured": False})
        staging.rename(output)
    print(board.head(12).round(6).to_string(index=False))


def from_models(models, query, subset=None, *, smooth=True):
    raw, bounded = {}, {}
    for name, model in models.items():
        r, b = predict_candidate(name, model, query)
        raw.update(r)
        bounded.update(b)
    if smooth:
        add_smoothed(query, raw, bounded)
    if subset is not None:
        return {n: raw[n] for n in subset}, {n: bounded[n] for n in subset}
    return raw, bounded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strict-reference", required=True, type=Path)
    parser.add_argument("--experiment", choices=["shape", "pool"], default="shape")
    args = parser.parse_args()
    run(args.output, args.strict_reference, args.experiment)


if __name__ == "__main__":
    # Preserve importable names in the persisted estimator.
    from pipetwin.agent2_physics.case2_transfer_shape import main as run_cli
    run_cli()
