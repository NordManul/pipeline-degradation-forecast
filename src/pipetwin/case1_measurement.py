"""Audit campaign differences and test comparable historical inputs for case 1."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .case1_heuristics import FEATURE_SETS
from .case1_peers import prepare_inputs
from .case_data import model_views
from .case_experiments import query_view
from .models.tabular import make_estimator
from .quality import clean_sources, read_raw_sources
from .sources import read_config
from .submission_checks import measure


STABLE = ["individual_min_mm", "deep_individual_min_30m_mm", "deep_individual_distance_m"]


def individual_internal(inspection):
    surface = inspection["Defect Type"].str.strip().str.lower()
    kind = inspection["Feature Description"].str.strip().str.lower()
    return inspection.loc[surface.isin(["внутр.", "внутренний"]) & ~kind.str.contains("кластер", na=False)
                          & inspection.is_corrosion & inspection.thickness_mm.notna()]


def comparable_history(frame, known):
    """Prior internal individual metal loss, with one fixed 20% severity screen."""
    result = frame.copy()
    result[STABLE] = np.nan
    for (section, anchor, target), group in frame.groupby(["id", "anchor_date", "target_date"]):
        rows = known.loc[known.id.eq(section) & known.date.eq(anchor) & known.available_at.lt(target) & known.date.lt(target)]
        x, remaining = rows.distance_m.to_numpy(), rows.thickness_mm.to_numpy()
        deep = rows.depth_fraction.to_numpy() >= .20
        for row in group.itertuples():
            inside = (x >= row.start_m) & (x < row.end_m)
            if inside.any():
                result.at[row.Index, STABLE[0]] = remaining[inside].min()
            distance = abs(x - row.center_m)
            local = deep & (distance <= 30)
            if local.any():
                level = remaining[local].min()
                result.at[row.Index, STABLE[1]] = level
                result.at[row.Index, STABLE[2]] = distance[local & (remaining == level)].min()
    return result


def candidates(train, query, columns, fallback):
    nominal, anchor = query.nominal_mm.to_numpy(), query.anchor_mm.to_numpy()
    eligible = train.loc[train.observed & train.anchor_observed]
    loss = (eligible.anchor_mm - eligible.y_mm) / eligible.horizon_years
    context = columns + FEATURE_SETS["combined_context"]
    result = {}
    for name, features, direct in [("baseline", context, False), ("comparable_anchors", context + STABLE, False),
                                   ("direct_thickness", context, True), ("direct_comparable", context + STABLE, True)]:
        predictions = []
        for kind in ["gam", "catboost"]:
            response = eligible.y_mm if direct else loss
            fit = make_estimator(kind, 42).fit(eligible[features], response)
            p = fit.predict(query[features])
            if not direct:
                p = anchor - p * query.horizon_years
            predictions.append(np.clip(np.where(query.anchor_observed, p, nominal * fallback), 0, nominal))
        result[name] = np.mean(predictions, axis=0)
    return result


def loss_stats(values):
    return {"n": len(values), "negative": int((values < -1e-8).sum()),
            "below_minus_0_2_mm": int((values < -.2 - 1e-8).sum()), "median_loss_mm": values.median()}


def audit(inspection, pairs_dir, ids):
    rows = inspection.loc[inspection.id.isin(ids) & inspection.date.isin(pd.to_datetime(["2018-09-23", "2022-07-06"])) & inspection.thickness_mm.notna()].copy()
    rows["cell"] = (rows.distance_m // 10).astype(int)
    counts = []
    for date, group in rows.groupby("date"):
        internal = individual_internal(group)
        counts.append({"date": date, "all_valid_rows": len(group), "label_cells": group.groupby(["id", "cell"]).ngroups,
                       "individual_internal_rows": len(internal), "minimum_depth_pct": 100 * internal.depth_fraction.min(),
                       "median_depth_pct": 100 * internal.depth_fraction.median(),
                       "exactly_10pct": int(np.isclose(internal.depth_fraction, .10).sum()),
                       "integer_pct_share": np.isclose(internal.depth_fraction * 100, np.round(internal.depth_fraction * 100)).mean()})
    old, new = [g.loc[g.groupby(["id", "cell"]).thickness_mm.idxmin()].set_index(["id", "cell"])
                for _, g in rows.groupby("date")]
    joined = old.join(new, lsuffix="_old", rsuffix="_new", how="inner")
    match = joined.nominal_observed_mm_old.eq(joined.nominal_observed_mm_new)
    loss = joined.thickness_mm_old - joined.thickness_mm_new
    summary = [{"group": "cell_minima", **loss_stats(loss)}, {"group": "cell_minima_same_nominal", **loss_stats(loss[match])}]
    depth_rows = []
    for method in ["geometry", "anchored"]:
        pairs = pd.read_csv(pairs_dir / f"{method}_pairs.csv.gz")
        pairs = pairs.loc[pairs.growth_eligible]
        values = pairs.earlier_thickness_mm - pairs.later_thickness_mm
        summary.append({"group": method, **loss_stats(values)})
        if method == "anchored":
            same = (pairs.earlier_distance_m // 10).eq(pairs.later_distance_m // 10)
            support = pairs.alignment_supported
            tight = support & same & pairs.distance_error_m.le(.25) & pairs.angle_error_deg.le(10) & pairs.margin.ge(.15)
            for name, mask in [("weld_supported", support), ("unsupported", ~support), ("weld_tight_same_cell", tight)]:
                summary.append({"group": name, **loss_stats(values[mask])})
                if name != "unsupported":
                    depth = pairs.loc[mask, "earlier_row_id"].map(rows.set_index("row_id").depth_fraction)
                    bins = pd.cut(depth, [0, .15, .20, .30, 1], include_lowest=True)
                    for band, subset in values[mask].groupby(bins, observed=True):
                        depth_rows.append({"group": name, "depth_fraction_2018": str(band), **loss_stats(subset)})
    return pd.DataFrame(counts), pd.DataFrame(summary), pd.DataFrame(depth_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = joblib.load(args.prepared)
    raw, manifest = read_raw_sources("data/raw")
    cases = read_config("contracts/cases.yaml")
    contract = read_config(package["config"]["contract"])
    policy = read_config("contracts/case1/same-day-peers.yaml")
    train, query, late, *_ = prepare_inputs(package, raw, manifest, cases, contract, policy)
    strict = clean_sources(raw, "case1", cases, contract, manifest)
    known = individual_internal(strict.sources.inspection)
    train, query, late = (comparable_history(frame, known) for frame in (train, query, late))
    package = {**package, "train_frame": train, "predict_frame": query}
    cols = package["feature_columns"]
    rows, pieces = [], []
    with threadpool_limits(limits=1):
        for fold in package["folds"]:
            fit, held, fallback = model_views(package, fold["number"])
            preds = candidates(fit, query_view(held), cols, fallback["fraction"])
            pieces.append(held.assign(**preds))
            rows.extend(measure(held, preds, f"fold_{fold['number'] + 1}"))
        oof = pd.concat(pieces).sort_index()
        for scope, frame in [("all", oof), ("target_section", oof.loc[oof.id.eq(package["spec"]["id"])])]:
            rows.extend(measure(frame, {k: frame[k].to_numpy() for k in preds}, scope))
        fit, query, fallback = model_views(package)
        values = candidates(fit, query_view(late), cols, fallback["fraction"])
        rows.extend(measure(late, values, "late_diagnostic"))
        final = candidates(fit, query, cols, fallback["fraction"])
    counts, loss, by_depth = audit(strict.sources.inspection, args.pairs, train.id.unique())
    metrics = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    counts.to_csv(args.output / "campaigns.csv", index=False)
    loss.to_csv(args.output / "negative_loss.csv", index=False)
    by_depth.to_csv(args.output / "pairs_by_depth.csv", index=False)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    oof[["id", "segment_id", "observed", "y_mm", *preds]].to_csv(args.output / "oof.csv", index=False)
    sensitivity = []
    for fold in package["folds"]:
        retained = oof.loc[~oof.index.isin(train.iloc[fold["valid"]].index) & oof.id.eq(package["spec"]["id"])]
        sensitivity.extend(measure(retained, {k: retained[k].to_numpy() for k in preds}, f"omit_fold_{fold['number'] + 1}"))
    pd.DataFrame(sensitivity).to_csv(args.output / "sensitivity.csv", index=False)
    query[["id", "segment_id", "target_date"]].assign(**final).to_csv(args.output / "diagnostic_forecasts.csv", index=False)
    print(metrics.loc[metrics.scope.isin(["target_section", "late_diagnostic"])].to_string(index=False))


if __name__ == "__main__":
    main()
