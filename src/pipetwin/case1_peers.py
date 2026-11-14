"""Exploratory transfer from other sections inspected on the case 1 target date."""

from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .case1_heuristics import BASELINE, BLEND, FEATURE_SETS, history_features
from .case_data import _fallback_fraction, canonical_snapshot
from .case_experiments import CaseModel, predict_models, query_view, validate_package
from .features import attach_labels
from .geometry import build_grid
from .models.tabular import TabularModel
from .pipeline import write_json
from .quality import clean_case1_peer_inspections, clean_sources, read_raw_sources
from .sources import read_config
from .targets import build_targets
from .validation import regression_metrics


METHODS = ["baseline", "pooled", "blend_50", "donor_residual"]
RESIDUAL_COLUMNS = ["anchor_mm", "edge_min_mm", "local_contrast_mm"]


def prepare_inputs(prepared, raw, manifest, cases, target_contract, policy):
    """Open the late label pool only after establishing the strict feature context."""
    validate_package(prepared)
    case = cases["cases"]["case1"]
    cutoff = pd.Timestamp(case["cutoff_exclusive"])
    if (prepared["case_id"] != "case1" or prepared["spec"] != case or prepared["input_manifest"] != manifest
            or prepared["label_policy"] != target_contract["target_policy"]
            or policy["case_id"] != "case1" or policy["target_id"] != case["id"] or policy["target_main"] != case["main_id"]
            or pd.Timestamp(policy["allowed_peer_label_date"]) != cutoff or pd.Timestamp(policy["target_date"]) != cutoff):
        raise ValueError("Prepared case 1 inputs, raw sources and peer scope disagree")
    cleaned = clean_sources(raw, "case1", cases, target_contract, manifest)
    grid = build_grid(cleaned.sources.passport, target_contract["segment_length_m"])
    spec = {"ids": policy["peer_ids"], "anchor_date": case["anchor_date"], "target_date": case["target_date"]}
    peer_features = canonical_snapshot(cleaned, grid, None, spec, prepared["config"], target_contract, labelled=False)
    historic = cleaned.sources.inspection
    train, query, peers = (history_features(frame, historic, target_contract["target_policy"])
                           for frame in (prepared["train_frame"], prepared["predict_frame"], peer_features))
    inspection, audit = clean_case1_peer_inspections(raw["inspection"], cleaned, policy["peer_ids"], target_contract,
                                                   pd.Timestamp(policy["label_pool_cutoff_exclusive"]))
    targets, quarantine = build_targets(replace(cleaned.sources, inspection=inspection), grid, target_contract)
    labelled = attach_labels(peers, targets)
    if not labelled.observed.any():
        raise ValueError("No eligible peer inspection labels")
    return train, query, labelled, audit, quarantine


def fit_blend(train, columns, cutoff, seed):
    eligible = train.loc[train.observed & train.anchor_observed]
    if eligible.empty:
        raise ValueError("The between-inspection model needs eligible historical anchors")
    fraction = _fallback_fraction(train)
    models = {}
    for kind in BLEND[BASELINE]:
        estimator = TabularModel("between", kind, seed, columns=columns).fit(eligible, eligible.y_mm.to_numpy())
        models[kind] = CaseModel(estimator, kind, "case1", "between", cutoff, fraction)
    return models


def predict_blend(models, query):
    return predict_models(models, query, BLEND)[1][BASELINE]


def fit_comparison(base, peers, columns, as_of, policy, seed, *, historical_block=False):
    base_labels = base.loc[base.observed]
    if not historical_block and not (base_labels.target_date.lt(as_of) & base_labels.label_available_at.lt(as_of)).all():
        raise ValueError("Baseline labels must precede inference outside the historical block diagnostic")
    valid_peers = peers.loc[peers.observed]
    if (base.sample_id.isin(valid_peers.sample_id).any() or not valid_peers.target_date.eq(as_of).all()
            or valid_peers.id.eq(policy["target_id"]).any()
            or not valid_peers.label_available_at.lt(pd.Timestamp(policy["label_pool_cutoff_exclusive"])).all()):
        raise ValueError("Peer labels must be unique additions from the query campaign")
    reference = fit_blend(base, columns, as_of, seed)
    pooled = fit_blend(pd.concat([base, valid_peers], ignore_index=True), columns, as_of, seed)
    observed = valid_peers.loc[valid_peers.anchor_observed]
    correction, strength = None, 0.
    if len(observed):
        alpha = policy["residual_ridge_alpha"]
        correction = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=alpha))
        residual = observed.y_mm - predict_blend(reference, query_view(observed))
        correction.fit(observed[RESIDUAL_COLUMNS], residual)
        strength = len(observed) / (len(observed) + alpha)
    return {"reference": reference, "pooled": pooled, "correction": correction, "strength": strength}


def predict_comparison(fitted, query, policy):
    query = query_view(query)
    baseline = predict_blend(fitted["reference"], query)
    pooled = predict_blend(fitted["pooled"], query)
    blend = policy["fixed_blend_new_model_weight"] * (pooled - baseline) + baseline
    residual = baseline.copy()
    anchored = query.anchor_observed.to_numpy()
    if anchored.any() and fitted["correction"] is not None:
        limit = policy["residual_limit_mm"]
        adjust = fitted["correction"].predict(query.loc[anchored, RESIDUAL_COLUMNS]) * fitted["strength"]
        residual[anchored] += np.clip(adjust, -limit, limit)
    return dict(zip(METHODS, [baseline, pooled, blend, np.clip(residual, 0, query.nominal_mm.to_numpy())]))


def score_predictions(table, protocol, group_column):
    observed = table.loc[table.observed]
    groups = [("all", observed), *observed.groupby(group_column)]
    rows = [{"protocol": protocol, "split": split, "model": name, **regression_metrics(group.y_mm, group[name])}
            for split, group in groups for name in METHODS]
    if group_column == "id":
        cols = ["y_mm", *METHODS]
        centered = observed[cols] - observed.groupby("id")[cols].transform("mean")
        for name in METHODS:
            r = regression_metrics(centered.y_mm, centered[name])["pearson_r"]
            rows.append({"protocol": protocol, "split": "within_section_centered", "model": name, "n": len(observed),
                         "pearson_r": r, "pearson_squared": r ** 2})
    return pd.DataFrame(rows)


def evaluate(prepared, train, peers, columns, policy, seed):
    target_id = prepared["spec"]["id"]
    historical, late, provenance = [], [], []
    for fold in prepared["folds"]:
        fit_frame = train.iloc[fold["train"]]
        held = train.iloc[fold["valid"]].loc[lambda f: f.id.eq(target_id)]
        base, donors = fit_frame.loc[fit_frame.id.eq(target_id)], fit_frame.loc[fit_frame.id.ne(target_id)]
        fitted = fit_comparison(base, donors, columns, held.target_date.iloc[0], policy, seed, historical_block=True)
        historical.append(held.assign(fold=fold["number"] + 1, **predict_comparison(fitted, held, policy)))
        provenance.append({"experiment": "historical_2022", "held": fold["number"] + 1,
                           "donor_labels": int(donors.observed.sum()),
                           "anchored_donor_labels": int((donors.observed & donors.anchor_observed).sum())})
    for section_id, held in peers.groupby("id", sort=True):
        donors = peers.loc[peers.id.ne(section_id)]
        fitted = fit_comparison(train, donors, columns, held.target_date.iloc[0], policy, seed)
        late.append(held.assign(**predict_comparison(fitted, held, policy)))
        provenance.append({"experiment": "leave_one_peer_out_2023", "held": section_id,
                           "donor_labels": int(donors.observed.sum()),
                           "anchored_donor_labels": int((donors.observed & donors.anchor_observed).sum())})
    history, late = pd.concat(historical, ignore_index=True), pd.concat(late, ignore_index=True)
    scores = pd.concat([score_predictions(history, "historical_2022", "fold"),
                        score_predictions(late, "leave_one_peer_out_2023", "id")], ignore_index=True)
    # Omitting one scored block is a sensitivity check on the existing OOF, not a new fit.
    omitted = []
    for fold in sorted(history.fold.unique()):
        hold = history.loc[history.observed & history.fold.ne(fold)]
        for name in METHODS:
            omitted.append({"omitted_fold": fold, "model": name, **regression_metrics(hold.y_mm, hold[name])})
    return scores, history, late, pd.DataFrame(provenance), pd.DataFrame(omitted)


def create_plot(output, scores, forecast):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    labels = ["Baseline", "Add peer labels", "Blend 50/50", "Peer residual"]
    for ax, (protocol, title) in zip(axes, [("historical_2022", "2022 | 508 held target-section cells"),
                                           ("leave_one_peer_out_2023", "2023 | 53 held peer cells")]):
        data = scores.loc[scores.protocol.eq(protocol) & scores.split.eq("all")].set_index("model").loc[METHODS]
        ax.bar(labels, data.pearson_r, color=["#7b838d", "#2d7686", "#61a89c", "#be7647"])
        ax.axhline(0, color="#555555", lw=.8)
        ax.set_title(title)
        ax.set_ylabel("Signed Pearson r")
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("Same-campaign peer labels | exploratory comparison")
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    for name, label in zip(METHODS[1:], labels[1:]):
        ax.plot(forecast.center_m, forecast[name] - forecast.baseline, lw=1, label=label)
    ax.axhline(0, color="#777777", lw=.8)
    ax.set(title="Provisional 2023 profile | change relative to strict baseline", xlabel="Distance along section, m",
           ylabel="Prediction change, mm")
    ax.legend()
    fig.savefig(output / "forecast_changes.png", dpi=150)
    plt.close(fig)


def run(prepared_path, output, data_dir, contracts_dir):
    prepared_path, output, contracts_dir = map(Path, (prepared_path, output, contracts_dir))
    output = output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    policy_path = contracts_dir / "case1/same-day-peers.yaml"
    policy = read_config(policy_path)
    cases = read_config(contracts_dir / "cases.yaml")
    prepared = joblib.load(prepared_path)
    raw, manifest = read_raw_sources(data_dir)
    train, query, peers, audit, quarantine = prepare_inputs(prepared, raw, manifest, cases,
                                                           read_config(prepared["config"]["contract"]), policy)
    columns = prepared["feature_columns"] + FEATURE_SETS[policy["base_model"]]
    seed = read_config("configs/case_benchmark.yaml")["seed"]
    cutoff = pd.Timestamp(prepared["cutoff_exclusive"])
    with threadpool_limits(limits=1):
        scores, historic, late, folds, sensitivity = evaluate(prepared, train, peers, columns, policy, seed)
        fitted = fit_comparison(train, peers, columns, cutoff, policy, seed)
        values = predict_comparison(fitted, query, policy)
        forecast = query[["id", "segment_id", "center_m", "target_date", "nominal_mm"]].assign(**values)
        if len(forecast) != prepared["spec"]["expected_provisional_segments"] or forecast.duplicated(["id", "segment_id"]).any():
            raise ValueError("Unexpected case 1 experimental grid")
        if not np.isfinite(forecast[METHODS].to_numpy()).all():
            raise ValueError("Nonfinite final prediction")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".case1-peers-", dir=output.parent) as temp:
        staged = Path(temp) / "run"
        staged.mkdir()
        key_cols = ["sample_id", "id", "segment_id", "target_date", "anchor_observed", "observed", "y_mm"]
        tables = {"metrics": scores, "fold_audit": folds, "sensitivity": sensitivity, "forecast_experimental": forecast,
                  "historical_oof": historic[key_cols + ["fold", *METHODS]], "late_peer_oof": late[key_cols + METHODS],
                  "donor_counts": peers.loc[peers.observed].groupby("id").agg(labels=("observed", "size"), anchored_labels=("anchor_observed", "sum")).reset_index(),
                  "target_quarantine": quarantine}
        for name, table in tables.items():
            table.to_csv(staged / f"{name}.csv", index=False)
        audit.groupby(["rule_id", "action", "reason"]).size().rename("events").to_csv(staged / "audit.csv")
        create_plot(staged, scores, forecast)
        write_json(staged / "summary.json", {"policy": policy, "seed": seed, "source_manifest": manifest,
                   "prepared_sha256": sha256(prepared_path.read_bytes()).hexdigest(),
                   "policy_sha256": sha256(policy_path.read_bytes()).hexdigest(),
                   "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                   "observed_peer_labels": int(peers.observed.sum()),
                   "observed_anchored_peer_labels": int((peers.observed & peers.anchor_observed).sum()),
                   "forecast_cells": len(forecast)})
        staged.rename(output)
    main_scores = scores.loc[scores.split.eq("all"), ["protocol", "model", "n", "pearson_r", "pearson_squared", "mae_mm"]]
    print(main_scores.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Case 1 peer-label experiment: {output}")
