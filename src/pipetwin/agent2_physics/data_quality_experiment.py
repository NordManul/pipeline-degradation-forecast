"""Compare data preparation hypotheses while keeping both verification targets fixed."""

import argparse
import gc
from hashlib import sha256
from itertools import product
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from pipetwin.case1_heuristics import FEATURE_SETS
from pipetwin.case1_measurement import STABLE, comparable_history, individual_internal
from pipetwin.case1_peers import prepare_inputs
from pipetwin.case2_data import prepare_transfer
from pipetwin.case2_transfer import ColdTabular, TABULAR, weights_by_campaign
from pipetwin.case_data import model_views, prepare_case
from pipetwin.case_experiments import prediction_table, query_view
from pipetwin.models.tabular import make_estimator
from pipetwin.pipeline import write_json
from pipetwin.quality import clean_case2_donors, clean_sources, read_raw_sources
from pipetwin.sources import read_config
from pipetwin.validation import regression_metrics

from .broad_case2 import TabularCandidate, decode, encode, predict_candidate, training_rows
from .case2_followup import fit_trial
from .case2_followup_models import boosted_estimator
from .case2_release_checks import aggregate, spec
from .data_quality_features import (alternate_cold_targets, anchor_features, cold_groups,
                                    exposure_statistics, inspect_campaigns, warm_groups)
from .data_quality_plots import write_plots
from .residual_guard import TrainBoundedFeatures
from .retrospective_case2 import campaign_metrics, validate_group_split
from .shared_comparison import verify_shared_predictions


WARM_KINDS = ["ols", "ridge", "bounded_ridge", "huber"]
COLD_KINDS = ["ridge", "bounded_ridge", "centered_ridge", "xgb"]
COLD_TARGETS = ["all", "corrosion", "individual", "local_nominal"]


def regression(kind, columns):
    if kind in {"bounded_ridge", "centered_ridge"}:
        return make_pipeline(TrainBoundedFeatures(columns), Ridge(alpha=30))
    learner = {"ols": LinearRegression(), "ridge": Ridge(alpha=30),
               "huber": HuberRegressor(alpha=1, max_iter=2000)}[kind]
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    return make_pipeline(imputer, StandardScaler(), learner)


def warm_predictions(package, train, query, fallback, groups):
    nominal, anchored = query.nominal_mm.to_numpy(), query.anchor_observed.to_numpy()
    base = nominal * fallback["fraction"]
    reference_columns = package["feature_columns"] + FEATURE_SETS["combined_context"] + STABLE
    reference_rows = train.loc[train.observed & train.anchor_observed]
    components = []
    for kind in ["gam", "catboost"]:
        fitted = make_estimator(kind, 42).fit(reference_rows[reference_columns], reference_rows.y_mm)
        components.append(np.clip(np.where(anchored, fitted.predict(query[reference_columns]), base), 0, nominal))
    reference = np.mean(components, axis=0)
    yield "gam_catboost_direct", reference
    yield "persistence", np.clip(np.where(anchored, query.anchor_mm, base), 0, nominal)
    for kind, (block, columns), sample in product(WARM_KINDS, groups.items(), ["anchored", "all"]):
        eligible = train.loc[train.observed]
        if sample == "anchored":
            eligible = eligible.loc[eligible.anchor_observed]
        x_fit, x_query = eligible[columns], query[columns]
        fitted = regression(kind, columns).fit(x_fit, eligible.y_mm.to_numpy())
        values = fitted.predict(x_query)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite warm prediction: {kind}, {block}, {sample}")
        if sample == "anchored":
            values = np.where(anchored, values, base)
        name = f"{kind}__{block}__{sample}"
        bounded = np.clip(values, 0, nominal)
        yield name, bounded
        if kind in {"huber", "ridge"} and block in {"all_reports", "quantiles_without_counts"} and sample == "all":
            yield "mix25__" + name, 0.75 * reference + 0.25 * bounded


def warm_scores(frame, values, scope):
    chosen = frame.observed.to_numpy()
    return [{"scope": scope, "model": name, **regression_metrics(frame.y_mm.to_numpy()[chosen], p[chosen])}
            for name, p in values.items()]


def warm_evaluation(prepared, late, groups):
    metrics, oof = [], []
    for fold in prepared["folds"]:
        number = fold["number"]
        train, query, fallback = model_views(prepared, number)
        values = dict(warm_predictions(prepared, train, query, fallback, groups))
        oof.append(prediction_table(query, values, values, number))
        metrics.extend(warm_scores(query, values, f"fold_{number + 1}"))
        target = query.id.eq(prepared["spec"]["id"]).to_numpy()
        metrics.extend(warm_scores(query.loc[target], {k: v[target] for k, v in values.items()},
                                   f"target_fold_{number + 1}"))
    oof = pd.concat(oof, ignore_index=True)
    verify_shared_predictions(prepared, oof)
    for scope, subset in [("all_oof", oof), ("target_oof", oof.loc[oof.id.eq(prepared["spec"]["id"])])]:
        for name, group in subset.groupby("model"):
            metrics.extend(warm_scores(group, {name: group.prediction_mm.to_numpy()}, scope))
    target_rows = oof.loc[oof.id.eq(prepared["spec"]["id"])]
    for number in sorted(target_rows.fold_id.unique()):
        for name, group in target_rows.loc[target_rows.fold_id.ne(number)].groupby("model"):
            metrics.extend(warm_scores(group, {name: group.prediction_mm.to_numpy()}, f"omit_target_fold_{number + 1}"))
    train, query, fallback = model_views(prepared)
    late_values = dict(warm_predictions(prepared, train, late, fallback, groups))
    metrics.extend(warm_scores(late, late_values, "late"))
    for section in late.id.unique():
        mask = late.id.eq(section).to_numpy()
        metrics.extend(warm_scores(late.loc[mask], {name: values[mask] for name, values in late_values.items()},
                                   f"late_{section}"))
    result = pd.DataFrame(metrics)
    target = result.loc[result.scope.eq("target_oof")].set_index("model")
    late_scores = result.loc[result.scope.eq("late")].set_index("model")
    reference = target.loc["gam_catboost_direct"]
    late_ref = late_scores.loc["gam_catboost_direct"]
    fold_r = result.loc[result.scope.str.startswith("target_fold_")].pivot(
        index="model", columns="scope", values="pearson_r")
    wins = fold_r.gt(fold_r.loc["gam_catboost_direct"]).sum(axis=1)
    board = target.drop(columns="scope").join(late_scores[["pearson_r", "pearson_squared", "mae_mm"]].add_prefix("late_"))
    board["target_fold_wins"] = wins
    board["passes"] = (board.pearson_r.gt(0) & board.pearson_squared.gt(reference.pearson_squared)
                       & board.mae_mm.le(reference.mae_mm + 0.01) & wins.ge(3)
                       & board.late_pearson_r.gt(0) & board.late_pearson_squared.ge(late_ref.pearson_squared)
                       & board.late_mae_mm.le(late_ref.mae_mm + 0.01))
    board = board.sort_values("pearson_squared", ascending=False)
    controlled = board.index[board.passes].tolist()
    best = controlled[0] if controlled else "gam_catboost_direct"
    forecasts = dict(warm_predictions(prepared, train, query, fallback, groups))
    selected = {k: forecasts[k] for k in dict.fromkeys(["gam_catboost_direct", best])}
    forecast_table = prediction_table(query, selected, selected, -1)
    verify_shared_predictions(prepared, forecast_table, forecast=True)
    return result, board.reset_index(), oof, forecast_table, best


def cold_training(frame, policy):
    if policy == "all":
        rows = frame.loc[frame.observed].copy()
    else:
        column = policy + "_y_mm"
        rows = frame.loc[frame.observed & frame[column].notna()].copy()
        rows["y_mm"] = rows[column]
    return rows


def cold_predictions(train, query, groups, policies):
    context = {"feature_columns": groups["position"]}
    reference = fit_trial(spec("xgb15", 42), context, train, groups)
    raw, _ = predict_candidate("xgb15", reference, query)
    yield "xgb15", raw["xgb15"]
    rows = cold_training(train, "all")
    model = ColdTabular(*TABULAR["catboost_reference_rate"]).fit(rows, weights_by_campaign(rows))
    yield "catboost_submission", model.predict(query)
    for policy in policies:
        rows = cold_training(train, policy)
        if rows.empty:
            raise ValueError(f"No eligible training observations for {policy}")
        weights = weights_by_campaign(rows)
        y = encode(rows, "log_rate")
        for kind, (block, columns) in product(COLD_KINDS, groups.items()):
            target = y
            level = 0.
            if kind == "centered_ridge":
                means = pd.Series(y, index=rows.index).groupby([rows.id, rows.target_date]).transform("mean")
                target = y - means.to_numpy()
                level = float(np.average(y, weights=weights))
            if kind == "xgb":
                learner = boosted_estimator("xgb15", 42).fit(rows[columns], y, sample_weight=weights)
                values = TabularCandidate(spec("xgb15", 42), columns, learner).predict(query)
            else:
                learner = regression(kind, columns)
                step = learner.steps[-1][0]
                learner.fit(rows[columns], target, **{f"{step}__sample_weight": weights})
                values = decode(learner.predict(query[columns]) + level, query, "log_rate")
            yield f"{kind}__{block}__{policy}", values


def cold_evaluation(train, query, minima):
    groups = cold_groups()
    # Match the documented agent baseline's order. With subsampling, a
    # permutation of the same rows can change the final booster.
    train = train.sort_values(["main_id", "target_date", "id", "segment_id"]).reset_index(drop=True)
    mains = sorted(train.main_id.unique())
    comparable = [policy for policy in COLD_TARGETS if all(
        not cold_training(train.loc[~train.main_id.eq(main)], policy).empty for main in mains)]
    folds, tables, predictions, audit = [], [], [], []
    for number, main in enumerate(mains):
        valid = train.loc[train.main_id.eq(main)]
        fit = train.loc[~train.main_id.eq(main)]
        validate_group_split(fit, valid)
        folds.append({"number": number, "valid": valid.index.to_numpy()})
        raw = dict(cold_predictions(fit, query_view(valid), groups, comparable))
        if any(not np.isfinite(values).all() for values in raw.values()):
            raise ValueError("Non-finite cold prediction")
        bounded = {k: np.clip(v, 0, valid.nominal_mm) for k, v in raw.items()}
        metrics = campaign_metrics(valid, raw, bounded, number)
        tables.append(metrics.assign(protocol="whole_main"))
        predictions.append(prediction_table(valid, raw, bounded, number))
        for policy in COLD_TARGETS:
            rows = cold_training(fit, policy)
            audit.append({"held_main": main, "policy": policy, "fit_labels": len(rows),
                          "fit_campaigns": int(rows.target_date.nunique()), "held_labels": int(valid.observed.sum()),
                          "evaluated_in_both_directions": policy in comparable})
    package = {"train_frame": train, "predict_frame": query, "folds": folds}
    results = pd.concat(tables, ignore_index=True)
    oof = pd.concat(predictions, ignore_index=True)
    verify_shared_predictions(package, oof)
    board = aggregate(results)
    metrics = results.loc[results.output.eq("bounded")]
    by_main = metrics.groupby(["model", "held_main"]).pearson_squared.agg(
        lambda x: x.mean(skipna=False)).unstack()
    board["main2_r2"], board["main3_r2"] = by_main["MAIN-2"], by_main["MAIN-3"]
    # A candidate that predicts a constant in a direction must not win via
    # skipna averaging that drops its weaker main or campaign.
    board["pearson_squared"] = by_main.mean(axis=1, skipna=False)
    board["pearson_r"] = metrics.groupby(["model", "held_main"]).pearson_r.agg(
        lambda x: x.mean(skipna=False)).unstack().mean(axis=1, skipna=False)
    board["defined_campaigns"] = metrics.groupby("model").pearson_r.count()
    board["positive_campaigns"] = metrics.groupby("model").pearson_r.agg(lambda x: x.gt(0).sum())
    ref = board.loc["xgb15"]
    board["passes"] = (board.positive_campaigns.eq(4) & board.main2_r2.gt(ref.main2_r2)
                       & board.main3_r2.gt(ref.main3_r2) & board.mae_mm.le(ref.mae_mm + 0.02))
    board = board.sort_values("pearson_squared", ascending=False)
    selected = board.index[board.passes].tolist()
    best = selected[0] if selected else "xgb15"
    forecasts = dict(cold_predictions(train, query, groups, comparable))
    raw = {name: forecasts[name] for name in dict.fromkeys(["xgb15", best])}
    bounded = {name: np.clip(p, 0, query.nominal_mm) for name, p in raw.items()}
    forecast = prediction_table(query, raw, bounded, -1)
    verify_shared_predictions(package, forecast, forecast=True)
    subtypes = oof.loc[oof.model.isin(["xgb15", "catboost_submission", best]) & oof.observed].merge(
        minima[["id", "segment_id", "date", "is_corrosion"]].rename(columns={"date": "target_date"}),
        on=["id", "segment_id", "target_date"], how="left", validate="many_to_one")
    if subtypes.is_corrosion.isna().any():
        raise ValueError("Observed minima lack the inspection type")
    subgroup = []
    for (date, name, corrosion), part in subtypes.groupby(["target_date", "model", "is_corrosion"]):
        subgroup.append({"target_date": date, "model": name, "minimum_is_corrosion": corrosion,
                         **regression_metrics(part.y_mm, part.prediction_mm)})
    return results, board.reset_index(), oof, forecast, pd.DataFrame(audit), pd.DataFrame(subgroup), best


def write_candidate(forecast, name, count, path):
    selected = forecast.loc[forecast.model.eq(name), ["id", "segment_id", "prediction_mm"]]
    if len(selected) != count or selected.id.nunique() != 1 or not selected.segment_id.is_unique:
        raise ValueError(f"Unexpected output keys for {name}")
    selected = selected.rename(columns={"prediction_mm": "thickness_pred"}).sort_values("segment_id")
    if not np.array_equal(selected.segment_id, np.arange(count)):
        raise ValueError(f"Incomplete segment sequence for {name}")
    selected.to_csv(path, index=False, float_format="%.9f")


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw, manifest = read_raw_sources("data/raw")
    cases = read_config("contracts/cases.yaml")
    config = read_config("configs/forecast.yaml")
    target = read_config(config["contract"])
    peers = read_config("contracts/case1/same-day-peers.yaml")
    transfer = read_config("contracts/case2/retrospective.yaml")
    strict = clean_sources(raw, "case1", cases, target, manifest)
    prepared, _ = prepare_case(strict, target, config)
    train1, query1, late1, _, _ = prepare_inputs(prepared, raw, manifest, cases, target, peers)
    known = strict.sources.inspection
    train1, query1, late1 = (anchor_features(comparable_history(frame, individual_internal(known)), known)
                             for frame in [train1, query1, late1])
    prepared = {**prepared, "train_frame": train1, "predict_frame": query1}
    train2, query2, exposure, query_audit, train_audit = prepare_transfer(raw, manifest, cases, target, transfer, config)
    donors = clean_case2_donors(raw, cases, target, transfer["pool_cutoff_exclusive"], manifest)
    counts, inspection, minima = inspect_campaigns(donors, target["segment_length_m"])
    train2 = alternate_cold_targets(train2, donors, target)
    availability = exposure_statistics(pd.concat([train2, query2], ignore_index=True), exposure)
    audit = pd.concat([strict.audit.assign(context="case1"), query_audit.assign(context="case2_query"),
                       train_audit.assign(context="case2_donors")])
    audit = audit.groupby(["context", "source_file", "rule_id", "action", "reason"]).size().rename("events").reset_index()
    wgroups = warm_groups(train1, prepared["feature_columns"])
    del raw, known, strict, donors, exposure, query_audit, train_audit
    gc.collect()
    with tempfile.TemporaryDirectory(prefix=".data-quality-", dir=output.parent) as directory:
        staging = Path(directory) / "run"
        staging.mkdir()
        write_json(staging / "protocol.json", {"source_manifest": manifest,
                   "protocol_sha256": sha256(Path(__file__).with_name("DATA_QUALITY_PROTOCOL.md").read_bytes()).hexdigest(),
                   "warm_groups": wgroups, "cold_groups": cold_groups(), "cold_targets": COLD_TARGETS, "seed": 42})
        with threadpool_limits(limits=1):
            case1, warm_board, warm_oof, warm_forecast, best1 = warm_evaluation(prepared, late1, wgroups)
            print("Case 1 complete", flush=True)
            case2, cold_board, cold_oof, cold_forecast, pool, subtypes, best2 = cold_evaluation(train2, query2, minima)
            print("Case 2 complete", flush=True)
        for table, name in [(audit, "cleaning_audit"), (counts, "inspection_campaigns"), (availability, "tech_coverage"),
                            (case1, "case1_metrics"), (case2, "case2_metrics"), (warm_board, "case1_ranking"),
                            (cold_board, "case2_ranking"), (warm_oof, "case1_oof"), (cold_oof, "case2_oof"),
                            (warm_forecast, "case1_forecast"), (cold_forecast, "case2_forecast"),
                            (pool, "case2_fit_pool"), (subtypes, "case2_subtype_diagnostics")]:
            suffix = ".csv.gz" if name.endswith("_oof") else ".csv"
            table.to_csv(staging / (name + suffix), index=False)
        write_plots(staging, counts, inspection, availability, train1, warm_board, cold_board)
        write_candidate(warm_forecast, best1, len(query1), staging / "case1_candidate.csv")
        write_candidate(cold_forecast, best2, len(query2), staging / "case2_candidate.csv")
        write_json(staging / "summary.json", {"case1_candidate": best1, "case2_candidate": best2,
                   "official_hidden_scores_available": False, "case1_models": len(warm_board),
                   "case2_models": len(cold_board), "case1_forecast_keys": len(query1),
                   "case2_forecast_keys": len(query2)})
        staging.rename(output)
    print("Candidates:", best1, best2, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
