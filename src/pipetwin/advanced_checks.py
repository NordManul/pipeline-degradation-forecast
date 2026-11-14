"""Compare five modeling approaches on the frozen case 1 and case 2 splits."""

import argparse
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from .agent2_physics import make_model as make_physics
from .case1_heuristics import FEATURE_SETS, history_features
from .case1_measurement import STABLE, comparable_history, individual_internal
from .case2_data import prepare_transfer
from .case2_transfer import ColdTabular, TABULAR, TERRAIN_FEATURES, ranking, weights_by_campaign
from .case_data import _fallback_fraction, canonical_snapshot, model_views
from .case_experiments import KEY_COLUMNS, query_view, validate_package
from .features import attach_labels
from .geometry import build_grid
from .matching import MatchingConfig, match_campaigns
from .models.advanced import (CORRECTION_ALPHA, LOG_FACTOR_LIMIT, PRIOR_CELLS, RATIO_FLOOR_MM, SEED, ZONE_LENGTH_M,
                              MultiplicativeColdPrior, PairwiseThickness, PooledMatchedRates, anchored,
                              direct_components, warm_bounds, warm_regressors)
from .pipeline import write_json
from .quality import clean_case1_peer_inspections, clean_sources, read_raw_sources
from .sources import read_config
from .targets import build_targets
from .uncertainty import calibrate_interval, interval_metrics, predict_interval
from .validation import blocked_folds, regression_metrics


def score(frame, predictions):
    """Spearman and Pearson on the same observed cells; center by section/date."""
    mask = frame.observed.to_numpy()
    actual = frame.loc[mask, ["id", "target_date", "y_mm"]].assign(pred=predictions[mask])
    metrics = regression_metrics(actual.y_mm, actual.pred)
    metrics["spearman_r"] = regression_metrics(rankdata(actual.y_mm), rankdata(actual.pred))["pearson_r"]
    centered = actual[["y_mm", "pred"]] - actual.groupby(["id", "target_date"])[["y_mm", "pred"]].transform("mean")
    centered_r = regression_metrics(centered.y_mm, centered.pred)["pearson_r"]
    return {**metrics, "centered_r": centered_r, "centered_r2": centered_r ** 2}


def warm_scores(frame, names, scope):
    return [{"scope": scope, "model": name, **score(frame, frame[name].to_numpy())} for name in names]


def warm_data(package, clean, target_contract):
    validate_package(package)
    if (package["case_id"] != "case1" or package["input_manifest"] != clean.sources.manifest
            or package["spec"] != clean.contract["cases"]["case1"]
            or package["label_policy"] != target_contract["target_policy"]):
        raise ValueError("Expected case 1 inputs prepared from the same raw files")
    history = clean.sources.inspection
    internal = individual_internal(history)
    peer_policy = read_config("contracts/case1/same-day-peers.yaml")
    peer_ids = peer_policy["peer_ids"]
    config = package["config"]
    grid = build_grid(clean.sources.passport, target_contract["segment_length_m"])
    spec = {"ids": peer_ids, "anchor_date": package["spec"]["anchor_date"], "target_date": package["spec"]["target_date"]}
    late = canonical_snapshot(clean, grid, None, spec, config, target_contract, labelled=False)

    def add_history(frame):
        preprocessed = history_features(frame, history, package["label_policy"])
        return comparable_history(preprocessed, internal)

    return {**package, "train_frame": add_history(package["train_frame"]), "predict_frame": add_history(package["predict_frame"])}, add_history(late)


def late_labels(late, raw, clean, target_contract):
    """Expose the same-day peers for final diagnostics only, after fitting."""
    ids = sorted(late.id.unique())
    policy = read_config("contracts/case1/same-day-peers.yaml")
    allowed, _ = clean_case1_peer_inspections(raw["inspection"], clean, ids, target_contract,
                                             pd.Timestamp(policy["label_pool_cutoff_exclusive"]))
    grid = build_grid(clean.sources.passport, target_contract["segment_length_m"])
    labels, _ = build_targets(replace(clean.sources, inspection=allowed), grid, target_contract)
    return attach_labels(late, labels)


def fit_pool(train, clean, matching, segment_length_m):
    date = train.target_date.iloc[0]
    matched = match_campaigns(clean.sources.inspection, clean.sources.passport, train.anchor_date.iloc[0], date,
                              train.id.unique(), matching, available_before=clean.cutoff, variant="anchored",
                              allowed_cells=train, segment_length_m=segment_length_m)
    return PooledMatchedRates().fit(train, matched.pairs, segment_length_m)


def warm_fold(train, query, pool, columns, buffer_m, fallback):
    outputs = warm_regressors(train, query, columns, fallback)
    ranker = PairwiseThickness(columns, buffer_m).fit(train)
    outputs["yetirank"] = warm_bounds(query, ranker.predict(query), fallback)
    outputs.update(pool.predict(query))
    outputs["blend_75ml_25pool"] = .75 * outputs["baseline"] + .25 * outputs["pooled_matched"]
    return outputs, ranker.calibration_rows_


def holdout_intervals(train, held, clean, columns, matching, segment_length_m, buffer_m, fold_id):
    """Calibrate on a block from outer train using a model that saw neither block."""
    split = blocked_folds(train, n_splits=3, buffer_m=buffer_m)
    # Pick by geometry only. This fixed tied-break does not inspect y or results.
    longest = max(split, key=lambda f: len(f.valid))
    fit, calib = train.iloc[longest.train], train.iloc[longest.valid]
    n = int(calib.observed.sum())
    if n < 20:
        raise ValueError("Insufficient observations in the interval calibration block")
    query = pd.concat([query_view(calib), query_view(held)], ignore_index=True)
    fallback = _fallback_fraction(fit)
    base = np.mean([warm_bounds(query, y, fallback) for y in direct_components(anchored(fit), query, columns)], axis=0)
    pool = fit_pool(fit, clean, matching, segment_length_m).predict(query)
    calibrated = {"baseline": base, "pooled_matched": pool["pooled_matched"],
                  "pooled_matched_positive": pool["pooled_matched_positive"],
                  "blend_75ml_25pool": .75 * base + .25 * pool["pooled_matched"]}
    observed = calib.observed.to_numpy()
    assessed = held.observed.to_numpy()
    rows = []
    for name, p in calibrated.items():
        q = calibrate_interval(calib.loc[observed, "y_mm"], p[:len(calib)][observed])
        low, high = predict_interval(p[len(calib):][assessed], q, held.loc[assessed, "nominal_mm"])
        rows.append({"fold": fold_id, "model": name, "calibration_labels": n,
                     **interval_metrics(held.loc[assessed, "y_mm"], low, high)})
    return rows


def warm_run(package, raw, clean, target_contract):
    package, late = warm_data(package, clean, target_contract)
    columns = [*package["feature_columns"], *FEATURE_SETS["combined_context"], *STABLE]
    buffer_m = package["config"]["spatial_buffer_m"]
    segment_length_m = target_contract["segment_length_m"]
    matching = MatchingConfig(**read_config("configs/matching.yaml")["matching"])
    target_id = package["spec"]["id"]
    pieces, metrics, audits, intervals = [], [], [], []
    for split in package["folds"]:
        fold = split["number"]
        train, held, fallback = model_views(package, fold)
        print(f"Case 1 | outer spatial block {fold + 1}/{len(package['folds'])}", flush=True)
        pool = fit_pool(train, clean, matching, segment_length_m)
        pred, n_rank = warm_fold(train, query_view(held), pool, columns, buffer_m, fallback["fraction"])
        frame = held.assign(fold=fold + 1, **pred)
        pieces.append(frame)
        metrics.extend(warm_scores(frame, list(pred), f"fold_{fold + 1}"))
        metrics.extend(warm_scores(frame.loc[frame.id.eq(target_id)], list(pred), f"target_fold_{fold + 1}"))
        intervals.extend(holdout_intervals(train, held, clean, columns, matching, segment_length_m, buffer_m, fold + 1))
        audits.append({"fold": fold + 1, "matched_training_cells": pool.matched_cells_, "inner_calibration_rank_labels": n_rank})
    oof = pd.concat(pieces, ignore_index=True).sort_values(["id", "segment_id"]).reset_index(drop=True)
    names = list(pred)
    target = oof.loc[oof.id.eq(target_id)]
    metrics.extend(warm_scores(oof, names, "all"))
    metrics.extend(warm_scores(target, names, "target_section"))
    for fold in target.fold.unique():
        metrics.extend(warm_scores(target.loc[target.fold.ne(fold)], names, f"target_omit_{fold}"))
    historical = pd.DataFrame(metrics)
    train, query, fallback = model_views(package)
    pool = fit_pool(train, clean, matching, segment_length_m)
    full = pd.concat([query, late], ignore_index=True)
    full_pred, _ = warm_fold(train, full, pool, columns, buffer_m, fallback["fraction"])
    forecast = query[KEY_COLUMNS + ["center_m", "nominal_mm"]].assign(**{k: v[:len(query)] for k, v in full_pred.items()})
    known = late_labels(late, raw, clean, target_contract).assign(**{k: v[len(query):] for k, v in full_pred.items()})
    metrics = warm_scores(known, names, "late_diagnostic")
    for section, part in known.groupby("id"):
        metrics.extend(warm_scores(part, names, f"late_{section}"))
    stages = ("pearson_r", "pearson_squared", "mae_mm", "spearman_r", "recall_thin_10pct")
    main_scores = historical.loc[historical.scope.eq("target_section")].set_index("model")
    base = main_scores.loc["baseline"]
    by_fold = historical.loc[historical.scope.str.startswith("target_fold_")].pivot(index="scope", columns="model", values="pearson_r")
    omit_fold = historical.loc[historical.scope.str.startswith("target_omit_")].pivot(index="scope", columns="model", values="pearson_squared")
    comparison = main_scores[list(stages)].assign(delta_r2=main_scores.pearson_squared - base.pearson_squared,
                                                  delta_mae_mm=main_scores.mae_mm - base.mae_mm,
                                                  fold_wins=by_fold.gt(by_fold.baseline, axis=0).sum(),
                                                  omit_fold_worst_gain=omit_fold.sub(omit_fold.baseline, axis=0).min())
    columns = KEY_COLUMNS + ["anchor_observed", "observed", "y_mm"]
    return {"case1_oof_metrics": historical, "case1_late_metrics": pd.DataFrame(metrics),
            "case1_comparison": comparison.reset_index(), "case1_intervals": pd.DataFrame(intervals),
            "case1_fold_audit": pd.DataFrame(audits), "case1_forecasts": forecast,
            "case1_oof_predictions": oof[columns + ["fold", *names]], "case1_late_predictions": known[columns + names]}


def cold_prediction(train, query, exposure, cutoff):
    features = TERRAIN_FEATURES
    weights = weights_by_campaign(train)
    nominal = query.nominal_mm.to_numpy()
    baseline = ColdTabular(*TABULAR["catboost_reference_rate"]).fit(train, weights)
    outputs = {"catboost_baseline": baseline.predict(query)}
    for policy, label in [("observed-only", "observed"), ("train-median", "imputed")]:
        prior = MultiplicativeColdPrior(exposure, cutoff, features, policy).fit(train, weights)
        outputs[f"physics_{label}"] = prior.prior.predict(query)
        outputs[f"physics_ridge_{label}"] = prior.predict(query)
    existing = make_physics("hybrid_scale", scenario="cold", exposure=exposure, cutoff_exclusive=cutoff,
                            feature_columns=features, seed=SEED, options={"missing_policy": "train-median"})
    outputs["agent2_scale_imputed"] = existing.fit(train, train.y_mm.to_numpy()).predict(query)
    bounded = {name: np.clip(y, 0, nominal) for name, y in outputs.items()}
    bounded["blend_75cat_25physics"] = .75 * bounded["catboost_baseline"] + .25 * bounded["physics_ridge_imputed"]
    outputs["blend_75cat_25physics"] = .75 * outputs["catboost_baseline"] + .25 * outputs["physics_ridge_imputed"]
    return bounded, outputs


def cold_scores(frame, pred, held_main):
    metrics = []
    for date, part in frame.groupby("target_date"):
        for name in pred:
            metrics.append({"held_main": held_main, "campaign": date, "model": name, **score(part, part[name].to_numpy())})
    return metrics


def cold_run(raw, manifest, cases, target_contract, config):
    policy = read_config("contracts/case2/retrospective.yaml")
    train, query, exposure, *_ = prepare_transfer(raw, manifest, cases, target_contract, policy, config)
    train = train.loc[train.observed].reset_index(drop=True)
    cutoff = policy["pool_cutoff_exclusive"]
    scores, expansion, held_predictions = [], [], []
    for main in sorted(train.main_id.unique()):
        print(f"Case 2 | holding out {main}", flush=True)
        rest = train.loc[train.main_id.ne(main)]
        held = train.loc[train.main_id.eq(main)].reset_index(drop=True)
        pred, _ = cold_prediction(rest, query_view(held), exposure, cutoff)
        frame = held.assign(**pred)
        scores.extend(cold_scores(frame, pred, main))
        held_predictions.append(frame[KEY_COLUMNS + ["main_id", "y_mm", *pred]])
        if main == "MAIN-3":
            early_pools = [("2018_only", rest.target_date.lt("2019-01-01")),
                           ("2018_plus_2022", rest.target_date.lt("2023-01-01"))]
            for label, mask in early_pools:
                sub = rest.loc[mask]
                values, _ = cold_prediction(sub, query_view(held), exposure, cutoff)
                expansion.extend({"training_pool": label, "training_labels": len(sub), **row}
                                 for row in cold_scores(held.assign(**values), values, main))
            expansion.extend({"training_pool": "all_main2", "training_labels": len(rest), **row}
                             for row in cold_scores(frame, pred, main))
    scores = pd.DataFrame(scores)
    pred, before_clamp = cold_prediction(train, query, exposure, cutoff)
    forecast = query[KEY_COLUMNS + ["center_m", "nominal_mm"]].assign(**pred)
    diag = pd.DataFrame([{"model": name, "min_mm": values.min(), "max_mm": values.max(), "std_mm": values.std(),
                          "outside_nominal_before_clamp": ((before_clamp[name] < 0) | (before_clamp[name] > query.nominal_mm)).mean()}
                         for name, values in pred.items()])
    return {"case2_transfer_metrics": scores, "case2_ranking": ranking(scores), "case2_expansion": pd.DataFrame(expansion),
            "case2_forecasts": forecast, "case2_forecast_ranges": diag,
            "case2_oof_predictions": pd.concat(held_predictions, ignore_index=True)}


def plot(output, warm, cold):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    case1 = warm["case1_comparison"].sort_values("pearson_squared")
    axes[0].barh(case1.model, case1.pearson_squared, color=["#203b59" if m == "baseline" else "#3c887e" for m in case1.model])
    axes[0].set(title="Case 1 | 508 historical target-section labels", xlabel="Pearson squared (spatial OOF)")
    case2 = cold["case2_transfer_metrics"].groupby(["model", "held_main"]).pearson_r.mean().unstack()
    case2.reindex(cold["case2_ranking"].model.iloc[::-1]).plot.barh(ax=axes[1], color=["#546c93", "#bd7544"])
    axes[1].set(title="Case 2 | entire pipelines held out", xlabel="Signed Pearson r (mean across campaigns)", ylabel="")
    axes[1].axvline(0, color="black", lw=.6)
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True, help="Frozen case 1 prepared.joblib")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    package = joblib.load(args.prepared)
    cases = read_config("contracts/cases.yaml")
    raw, manifest = read_raw_sources("data/raw")
    target = read_config(package["config"]["contract"])
    clean = clean_sources(raw, "case1", cases, target, manifest)
    with threadpool_limits(limits=1):
        warm = warm_run(package, raw, clean, target)
        cold = cold_run(raw, manifest, cases, target, package["config"])
    with tempfile.TemporaryDirectory(prefix=".advanced-checks-", dir=output.parent) as temp:
        staged = Path(temp) / "run"
        staged.mkdir()
        for name, table in {**warm, **cold}.items():
            if name.endswith("_forecasts") or name.endswith("_predictions"):
                keys = KEY_COLUMNS + (["fold"] if "fold" in table else [])
                if table.duplicated(keys).any():
                    raise ValueError(f"Duplicate samples in {name}")
            table.to_csv(staged / f"{name}.csv", index=False)
        configs = ["contracts/cases.yaml", "contracts/validation.yaml", "contracts/case1/same-day-peers.yaml",
                   "contracts/case2/retrospective.yaml", "configs/matching.yaml", package["config"]["contract"]]
        write_json(staged / "provenance.json", {"seed": SEED, "zone_length_m": ZONE_LENGTH_M, "pseudo_cells": PRIOR_CELLS,
                   "case2_ridge_alpha": CORRECTION_ALPHA, "case2_log_multiplier_limit": LOG_FACTOR_LIMIT,
                   "case2_loss_ratio_floor_mm": RATIO_FLOOR_MM,
                   "source_manifest": manifest, "prepared_sha256": sha256(args.prepared.read_bytes()).hexdigest(),
                   "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                                     for p in [Path(__file__), Path(__file__).with_name("models") / "advanced.py"]},
                   "config_sha256": {p: sha256(Path(p).read_bytes()).hexdigest() for p in configs},
                   "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "catboost", "matplotlib"]}})
        plot(staged, warm, cold)
        staged.rename(output)
    print(warm["case1_comparison"].round(5).to_string(index=False), flush=True)
    print(cold["case2_ranking"].round(5).to_string(index=False), flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
