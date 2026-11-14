"""Paired checks of six submission proposals, isolated from model selection."""

import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .case1_heuristics import FEATURE_SETS
from .case1_peers import prepare_inputs
from .case2_data import prepare_transfer
from .case2_transfer import ColdTabular, TABULAR, weights_by_campaign
from .case_data import model_views
from .case_experiments import query_view
from .models import make_model
from .models.tabular import make_estimator
from .physical_features import ablation_columns
from .quality import read_raw_sources
from .sources import read_config
from .uncertainty import calibrate_interval, interval_metrics, predict_interval
from .validation import regression_metrics


def warm_predictions(train, query, columns, fallback):
    nominal = query.nominal_mm.to_numpy()
    anchor = query.anchor_mm.to_numpy()
    measured = train.loc[train.observed & train.anchor_observed]
    loss = measured.anchor_mm - measured.y_mm
    years = query.horizon_years.to_numpy()
    anchored = query.anchor_observed.to_numpy()

    def bounded(pred):
        return np.clip(np.where(anchored, pred, nominal * fallback), 0, nominal)

    def regression(kind, names, response, scale):
        fit = make_estimator(kind, 42).fit(measured[names], response)
        return anchor - scale * fit.predict(query[names])

    combined = columns + FEATURE_SETS["combined_context"]
    rates = [regression(kind, combined, loss / measured.horizon_years, years) for kind in ("gam", "catboost")]
    result = {
        "context_baseline": np.mean([bounded(p) for p in rates], axis=0),
        "context_clip_growth": np.mean([bounded(np.minimum(p, anchor)) for p in rates], axis=0),
        "persistence": bounded(anchor),
    }
    for name, target in [("delta_signed", loss), ("delta_nonnegative_train", loss.clip(lower=0))]:
        components = [regression(kind, combined, target, 1.) for kind in ("gam", "catboost")]
        if name == "delta_signed":
            result[name] = np.mean([bounded(p) for p in components], axis=0)
        result[f"{name}_clip"] = np.mean([bounded(np.minimum(p, anchor)) for p in components], axis=0)
    # One historical training horizon. The raw delta is constant over this horizon;
    # the annual rate is multiplied by the actual query horizon.
    mode = [c for c in ablation_columns("mode", "between") if c.startswith("e02_")]
    for name, features in [("gam_full", columns), ("gam_mode", mode), ("gam_mode_anchor", mode + ["anchor_mm"])]:
        result[name] = bounded(regression("gam", features, loss / measured.horizon_years, years))
    result["e02_mode_direct"] = bounded(make_model("e02_mode", "between", 42).fit(
        train.loc[train.observed], train.loc[train.observed, "y_mm"]).predict(query))
    return result


class QuantileGate:
    """Fold-local ranges; values outside are replaced by train medians."""

    def fit(self, x):
        self.lower, self.upper = x.quantile(.01), x.quantile(.99)
        self.median = x.median()
        return self

    def transform(self, x):
        outside = x.notna() & (x.lt(self.lower) | x.gt(self.upper) | self.median.isna())
        values = x.mask(outside).fillna(self.median)
        return values.join(outside.astype(float).add_prefix("is_out_of_range__")), outside


def cold_checks(train, query):
    _, _, cols = TABULAR["catboost_reference_rate"]
    model_columns = train[cols]
    gate = QuantileGate().fit(model_columns)
    gate_train, _ = gate.transform(model_columns)
    gate_query, outside = gate.transform(query[cols])
    # The gate operates on predictors only. Keep actual nominal and horizon
    # for the response conversion back into millimetres.
    gate_train = gate_train.add_prefix("gated__")
    gate_query = gate_query.add_prefix("gated__")
    weights = weights_by_campaign(train)
    nominal = query.nominal_mm.to_numpy()
    raw = {}
    for kind in ("catboost", "huber"):
        raw[kind] = ColdTabular(kind, "rate", cols).fit(train, weights).predict(query)
        raw[f"{kind}_gated"] = ColdTabular(kind, "rate", list(gate_train)).fit(
            train.join(gate_train), weights).predict(query.join(gate_query))
    scores = {k: np.clip(y, 0, nominal) for k, y in raw.items()}
    for kind in ("catboost", "huber"):
        scores[f"{kind}_proposal_clamp"] = np.clip(raw[kind], .5 * nominal, 1.05 * nominal)
    return scores, raw, outside


def measure(frame, predictions, scope):
    mask = frame.observed.to_numpy() if "observed" in frame else np.ones(len(frame), dtype=bool)
    return [{"scope": scope, "model": name, **regression_metrics(frame.loc[mask, "y_mm"], value[mask])}
            for name, value in predictions.items()]


def interval_check(frame, baseline_q, groups, scenario, prediction="context_baseline", *, buffer_m=None):
    """Compare fixed historical intervals to residual cross-calibration on transfer data."""
    lower, upper = predict_interval(frame[prediction], baseline_q, frame.nominal_mm)
    residual = frame.y_mm - frame[prediction]
    # Always exclude the held calibration group. Tiny reference sets fall back
    # to the historical quantiles instead of fitting unstable tail estimates.
    cross_lo, cross_hi = np.empty(len(frame)), np.empty(len(frame))
    n_small = 0
    for group in groups.unique():
        held = groups.eq(group)
        eligible = ~held
        if buffer_m is not None:
            distance = np.abs(frame.chain_position_m.to_numpy()[:, None] - frame.loc[held, "chain_position_m"].to_numpy())
            eligible &= (distance >= buffer_m).all(axis=1)
        other = residual.loc[eligible]
        q = np.quantile(other, [.1, .9]) if len(other) >= 20 else baseline_q
        n_small += int(held.sum()) if len(other) < 20 else 0
        cross_lo[held], cross_hi[held] = predict_interval(frame.loc[held, prediction], q, frame.loc[held, "nominal_mm"])
    return [{"scenario": scenario, "calibration": "historical", **interval_metrics(frame.y_mm, lower, upper), "fallback_n": 0},
            {"scenario": scenario, "calibration": "cross_calibration_on_transfer", **interval_metrics(frame.y_mm, cross_lo, cross_hi), "fallback_n": n_small}]


def cold_diagnostics(query, outputs, raw, outside, scope):
    rows = []
    for name, values in {**outputs, **{f"{k}_raw": y for k, y in raw.items()}}.items():
        rows.append({"scope": scope, "model": name, "min_mm": values.min(), "max_mm": values.max(),
                     "unique_predictions": np.unique(values).size, "above_nominal": (values > query.nominal_mm).sum(),
                     "outside_feature_cells": outside.to_numpy().sum()})
    return rows


def run(prepared, output):
    from .pipeline import write_json

    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    prepared = joblib.load(prepared)
    raw, manifest = read_raw_sources("data/raw")
    config = prepared["config"]
    cases = read_config("contracts/cases.yaml")
    target = read_config(config["contract"])
    policy = read_config("contracts/case1/same-day-peers.yaml")
    train, query, late, *_ = prepare_inputs(prepared, raw, manifest, cases, target, policy)
    cold, cold_query, *_ = prepare_transfer(raw, manifest, cases, target, read_config("contracts/case2/retrospective.yaml"), config)
    cold = cold.loc[cold.observed].reset_index(drop=True)
    with threadpool_limits(limits=1):
        package = {**prepared, "train_frame": train, "predict_frame": query}
        pieces, warm_rows, cold_rows, diagnostics, gate_rows, interval_rows = [], [], [], [], [], []
        for fold in package["folds"]:
            fit, hold, fallback = model_views(package, fold["number"])
            results = warm_predictions(fit, query_view(hold), package["feature_columns"], fallback["fraction"])
            pieces.append(hold.assign(**results))
            warm_rows.extend(measure(hold, results, f"fold_{fold['number'] + 1}"))
        oof = pd.concat(pieces).sort_index()
        names = list(results)
        for scope, group in [("all", oof), ("target_section", oof.loc[oof.id.eq(package["spec"]["id"])])]:
            warm_rows.extend(measure(group, {k: group[k].to_numpy() for k in names}, scope))
        fit, official, fallback = model_views(package)
        final_warm = warm_predictions(fit, official, package["feature_columns"], fallback["fraction"])
        late_preds = warm_predictions(fit, query_view(late), package["feature_columns"], fallback["fraction"])
        warm_rows.extend(measure(late, late_preds, "late_diagnostic"))
        for section, group in late.assign(**late_preds).groupby("id"):
            warm_rows.extend(measure(group, {k: group[k].to_numpy() for k in names}, f"late_{section}"))
        warm_q = calibrate_interval(oof.loc[oof.observed, "y_mm"], oof.loc[oof.observed, "context_baseline"])
        late_interval = late.assign(**late_preds).loc[late.observed].reset_index(drop=True)
        interval_rows.extend(interval_check(late_interval, warm_q, late_interval.id, "case1_late_by_section"))
        block = np.floor(late_interval.chain_position_m / 100).astype(int)
        interval_rows.extend(interval_check(late_interval, warm_q, block, "case1_late_100m_buffered", buffer_m=100.))
        late_q = calibrate_interval(late_interval.y_mm, late_interval.context_baseline)

        held_cold = []
        for main in sorted(cold.main_id.unique()):
            fit = cold.loc[cold.main_id.ne(main)]
            hold = cold.loc[cold.main_id.eq(main)].reset_index(drop=True)
            pred, before, gate = cold_checks(fit, query_view(hold))
            for date, group in hold.groupby("target_date"):
                cold_rows.extend(measure(group, {name: values[group.index] for name, values in pred.items()}, f"{main}_{date.date()}"))
            diagnostics.extend(cold_diagnostics(hold, pred, before, gate, main))
            held_cold.append(hold.assign(catboost=pred["catboost"]))
        pred, before, gate = cold_checks(cold, cold_query)
        diagnostics.extend(cold_diagnostics(cold_query, pred, before, gate, "official_case2"))
        gate_rows.extend({"feature": k, "target_out_of_range_fraction": float(v)} for k, v in gate.mean().items())

        # Recheck the actual failure regime with the original 2018-only fit.
        early = cold.loc[cold.target_date.eq(pd.Timestamp("2018-09-23"))]
        old_pred, old_raw, old_gate = cold_checks(early, cold_query)
        diagnostics.extend(cold_diagnostics(cold_query, old_pred, old_raw, old_gate, "official_case2_2018_only"))

    warm_metrics = pd.DataFrame(warm_rows)
    cold_metrics = pd.DataFrame(cold_rows)
    cold_oof = pd.concat(held_cold, ignore_index=True)
    # Within each held main all prediction rows come from one frozen model that
    # excludes that main; calibration excludes the complete query section.
    for main, group in cold_oof.groupby("main_id", sort=True):
        local = group.reset_index(drop=True)
        other = cold_oof.loc[cold_oof.main_id.ne(main)]
        other_q = calibrate_interval(other.y_mm, other.catboost)
        rows = interval_check(local, other_q, local.id, f"case2_{main}", "catboost")
        # The opposite direction uses a different fit and is a descriptive
        # comparator, unlike the within-main cross-calibrated row.
        rows[0]["calibration"] = "other_transfer_direction_diagnostic"
        interval_rows.extend(rows)
    cold_q = calibrate_interval(cold_oof.y_mm, cold_oof.catboost)
    with tempfile.TemporaryDirectory(prefix=".submission-checks-", dir=output.parent) as temp:
        staged = Path(temp) / "run"
        staged.mkdir()
        tables = {"case1_ablations": warm_metrics, "case2_ablations": cold_metrics, "case2_ranges": pd.DataFrame(diagnostics),
                  "case2_gate_audit": pd.DataFrame(gate_rows), "interval_metrics": pd.DataFrame(interval_rows)}
        for name, table in tables.items():
            table.to_csv(staged / f"{name}.csv", index=False)
        frames = []
        for frame, point, quantiles in [(official, final_warm["context_baseline"], warm_q), (cold_query, pred["catboost"], cold_q)]:
            low, high = predict_interval(point, quantiles, frame.nominal_mm)
            frames.append(frame[["id", "segment_id"]].assign(thickness_pred=point, lower_p10_mm=low, upper_p90_mm=high))
        pd.concat(frames).to_csv(staged / "diagnostic_intervals.csv", index=False)
        late_low, late_high = predict_interval(final_warm["context_baseline"], late_q, official.nominal_mm)
        official[["id", "segment_id"]].assign(thickness_pred=final_warm["context_baseline"], lower_p10_mm=late_low,
                                              upper_p90_mm=late_high).to_csv(staged / "case1_late_calibrated_experimental.csv", index=False)
        write_json(staged / "summary.json", {"case1_interval_calibration": "strict_2022_spatial_oof",
                   "case2_interval_calibration": "retrospective_leave_main_out", "case1_residual_quantiles_mm": warm_q,
                   "case1_experimental_same_day_late_quantiles_mm": late_q,
                   "case2_residual_quantiles_mm": cold_q, "case1_anchored_labels": int((train.observed & train.anchor_observed).sum()),
                   "case1_negative_observed_loss": int((train.observed & train.anchor_observed & train.y_mm.gt(train.anchor_mm)).sum())})
        staged.rename(output)
    print(warm_metrics.loc[warm_metrics.scope.isin(["target_section", "late_diagnostic"]), ["scope", "model", "pearson_r", "pearson_squared", "mae_mm"]].to_string(index=False))
    print(pd.DataFrame(interval_rows).to_string(index=False))
