"""Case 1: predict the recorded minimum wall thickness from a previous ILI."""

from dataclasses import replace

import numpy as np
import pandas as pd

from .case_data import canonical_snapshot
from .features import attach_labels, feature_columns
from .geometry import build_grid
from .models.tabular import make_estimator
from .quality import clean_case1_peer_inspections, clean_sources
from .targets import build_targets
from .validation import blocked_folds, score_predictions


CONTEXT = ["edge_min_mm", "edge_min_distance_m", "local_contrast_mm"]
COMPARABLE = ["individual_min_mm", "deep_individual_min_30m_mm", "deep_individual_distance_m"]
FINAL_MODEL = "direct_blend"


def history_features(frame, inspection, target_policy):
    """Spatial context from the anchor campaign as known at each example date."""
    frame = frame.copy()
    frame[CONTEXT] = np.nan
    if not frame.index.is_unique or not frame.anchor_date.lt(frame.target_date).all():
        raise ValueError("Expected unique rows and an earlier anchor date")
    if target_policy not in {"reported_residual", "corrosion_only"}:
        raise ValueError("Unsupported target policy")
    for (section_id, anchor, target), section in frame.groupby(["id", "anchor_date", "target_date"]):
        points = inspection.loc[
            inspection.id.eq(section_id) & inspection.date.eq(anchor)
            & inspection.available_at.lt(target) & inspection.thickness_mm.notna()
        ]
        if target_policy == "corrosion_only":
            points = points.loc[points.is_corrosion]
        x, thickness = points.distance_m.to_numpy(), points.thickness_mm.to_numpy()
        centers = section.center_m.to_numpy()
        measured = section.anchor_mm.where(section.anchor_observed).to_numpy()
        for row in section.itertuples():
            inside = (x >= row.start_m) & (x < row.end_m)
            exact = thickness[inside].min() if inside.any() else np.nan
            if not np.isclose(exact, row.anchor_mm if row.anchor_observed else np.nan, equal_nan=True):
                raise ValueError(f"Historical points disagree with the anchor for {row.sample_id}")
            near = (x >= row.start_m - 2.0) & (x < row.end_m + 2.0)
            if near.any():
                level = thickness[near].min()
                distance = np.abs(x[near & (thickness == level)] - row.center_m).min()
                frame.loc[row.Index, CONTEXT[:2]] = [level, distance]
            distance = np.abs(centers - row.center_m)
            neighbors = measured[(distance > 0) & (distance <= 50.0) & np.isfinite(measured)]
            if row.anchor_observed and len(neighbors):
                frame.at[row.Index, "local_contrast_mm"] = row.anchor_mm - np.median(neighbors)
    return frame


def comparable_history(frame, inspection):
    """Individual internal corrosion, including ≥20% depth within 30 m."""
    surface = inspection["Defect Type"].str.strip().str.lower()
    description = inspection["Feature Description"].str.strip().str.lower()
    known = inspection.loc[
        surface.isin(["внутр.", "внутренний"]) & ~description.str.contains("кластер", na=False)
        & inspection.is_corrosion & inspection.thickness_mm.notna()
    ]
    result = frame.copy()
    result[COMPARABLE] = np.nan
    for (section, anchor, target), group in frame.groupby(["id", "anchor_date", "target_date"]):
        rows = known.loc[
            known.id.eq(section) & known.date.eq(anchor) & known.available_at.lt(target) & known.date.lt(target)
        ]
        x, remaining = rows.distance_m.to_numpy(), rows.thickness_mm.to_numpy()
        deep = rows.depth_fraction.to_numpy() >= .20
        for row in group.itertuples():
            inside = (x >= row.start_m) & (x < row.end_m)
            if inside.any():
                result.at[row.Index, COMPARABLE[0]] = remaining[inside].min()
            distance = abs(x - row.center_m)
            local = deep & (distance <= 30)
            if local.any():
                level = remaining[local].min()
                result.at[row.Index, COMPARABLE[1]] = level
                result.at[row.Index, COMPARABLE[2]] = distance[local & (remaining == level)].min()
    return result


def prepare_case1(raw, manifest, cases, target_contract, config):
    """Build training, submission and late feature frames with one strict clock."""
    cleaned = clean_sources(raw, "case1", cases, target_contract, manifest)
    case = cases["cases"]["case1"]
    grid = build_grid(cleaned.sources.passport, target_contract["segment_length_m"])
    targets, _ = build_targets(cleaned.sources, grid, target_contract)
    query_spec = {"ids": [case["id"]], "anchor_date": case["anchor_date"], "target_date": case["target_date"]}
    train = canonical_snapshot(cleaned, grid, targets, config["development"], config, target_contract, labelled=True)
    query = canonical_snapshot(cleaned, grid, None, query_spec, config, target_contract, labelled=False)
    late = canonical_snapshot(cleaned, grid, None, config["late"], config, target_contract, labelled=False)
    if len(query) != case["expected_provisional_segments"] or late.target_date.ne(cleaned.cutoff).any():
        raise ValueError("Case 1 grid or late diagnostic date disagrees with the contract")
    inspection = cleaned.sources.inspection
    train, query, late = (
        comparable_history(history_features(frame, inspection, target_contract["target_policy"]), inspection)
        for frame in (train, query, late)
    )
    return train, query, late, cleaned, grid


class BetweenInspections:
    """Fixed equal-weight GAM/CatBoost blend and a train-only fallback."""

    def __init__(self, seed, *, annual_loss=False):
        self.seed = seed
        self.annual_loss = annual_loss
        self.columns = feature_columns("between") + CONTEXT + ([] if annual_loss else COMPARABLE)

    def fit(self, train):
        observed = train.loc[train.observed]
        eligible = observed.loc[observed.anchor_observed]
        if eligible.empty:
            raise ValueError("The between-inspection model needs observed training anchors")
        self.fallback_fraction_ = float(np.clip(np.median(observed.y_mm / observed.nominal_mm), 0, 1))
        response = eligible.y_mm
        if self.annual_loss:
            response = (eligible.anchor_mm - response) / eligible.horizon_years
        self.estimators_ = [make_estimator(kind, self.seed).fit(eligible[self.columns], response)
                            for kind in ("gam", "catboost")]
        return self

    def predict(self, query):
        nominal = query.nominal_mm.to_numpy()
        predictions = []
        for estimator in self.estimators_:
            values = estimator.predict(query[self.columns])
            if self.annual_loss:
                values = query.anchor_mm.to_numpy() - values * query.horizon_years
            values = np.where(query.anchor_observed, values, nominal * self.fallback_fraction_)
            predictions.append(np.clip(values, 0, nominal))
        return np.mean(predictions, axis=0)

    def persistence(self, query):
        nominal = query.nominal_mm.to_numpy()
        return np.clip(np.where(query.anchor_observed, query.anchor_mm, nominal * self.fallback_fraction_), 0, nominal)


def fit_case1(train, query, late, config, target_id):
    """Score blocked 2022 holdouts and predict the two 2023 requests."""
    if config["spatial_buffer_m"] < max(config["history_radius_m"], 50.0):
        raise ValueError("The validation buffer must cover the history feature radius")
    folds = blocked_folds(train, config["folds"], config["spatial_buffer_m"])
    seed = config["seed"]
    pieces, scores = [], []
    for fold in folds:
        fit, held = train.iloc[fold.train], train.iloc[fold.valid]
        direct = BetweenInspections(seed).fit(fit)
        rate = BetweenInspections(seed, annual_loss=True).fit(fit)
        values = {"persistence": direct.persistence(held), "annual_loss_blend": rate.predict(held),
                  FINAL_MODEL: direct.predict(held)}
        pieces.append(held.assign(**values))
        target = held.id.eq(target_id)
        scores.append(score_predictions(held.loc[target], {name: pred[target] for name, pred in values.items()},
                                        f"spatial_fold_{fold.number + 1}"))
    oof = pd.concat(pieces).sort_index().loc[lambda frame: frame.id.eq(target_id)]
    scores.append(score_predictions(oof, {name: oof[name].to_numpy() for name in values}, "spatial_2022"))
    direct = BetweenInspections(seed).fit(train)
    rate = BetweenInspections(seed, annual_loss=True).fit(train)
    late_predictions = {"persistence": direct.persistence(late), "annual_loss_blend": rate.predict(late),
                        FINAL_MODEL: direct.predict(late)}
    observed = oof.loc[oof.observed, ["id", "segment_id", "y_mm", FINAL_MODEL]].rename(
        columns={FINAL_MODEL: "thickness_pred"}
    )
    return direct.predict(query), late_predictions, pd.concat(scores, ignore_index=True), observed


def score_late(late, predictions, raw, cleaned, grid, target_contract, config, freeze):
    """Join peer VTD on the target date only after fitting and prediction."""
    inspection, audit = clean_case1_peer_inspections(
        raw["inspection"], cleaned, config["late"]["ids"], target_contract, pd.Timestamp(freeze)
    )
    targets, _ = build_targets(replace(cleaned.sources, inspection=inspection), grid, target_contract)
    return score_predictions(attach_labels(late, targets), predictions, "late_2023"), audit
