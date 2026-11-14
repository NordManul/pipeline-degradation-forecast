"""Reported inspection minima and explicit unknown coverage."""

import numpy as np
import pandas as pd

from .geometry import assign_segments


def build_targets(sources, grid, contract):
    rows = assign_segments(sources.inspection, sources.passport, contract["segment_length_m"])
    rows["exclusion_reason"] = ""
    rows.loc[~rows["in_range"], "exclusion_reason"] = "outside_section"
    invalid = (
        ~np.isfinite(rows["thickness_mm"])
        | rows["thickness_mm"].le(0)
        | rows["nominal_observed_mm"].isna()
        | rows["thickness_mm"].gt(rows["nominal_observed_mm"] + contract["thickness_tolerance_mm"])
    )
    rows.loc[invalid & rows["exclusion_reason"].eq(""), "exclusion_reason"] = "invalid_or_missing_thickness"
    policy = contract["target_policy"]
    if policy not in {"reported_residual", "corrosion_only"}:
        raise ValueError(f"Unknown target policy: {policy}")
    if policy == "corrosion_only":
        rows.loc[~rows["is_corrosion"] & rows["exclusion_reason"].eq(""), "exclusion_reason"] = "other_feature_type"
    valid = rows.loc[rows["exclusion_reason"].eq("")].copy()
    valid["label_available_at"] = valid["available_at"] if "available_at" in valid else valid["date"]
    measured = valid.groupby(["id", "date", "segment_id"], as_index=False).agg(
        thickness_mm=("thickness_mm", "min"),
        nominal_observed_mm=("nominal_observed_mm", "median"),
        recorded_features=("row_id", "size"),
        label_available_at=("label_available_at", "max"),
    )
    campaigns = rows[["id", "date"]].drop_duplicates()
    targets = grid.merge(campaigns, on="id", validate="many_to_many").merge(
        measured, on=["id", "date", "segment_id"], how="left", validate="one_to_one"
    )
    targets["observed"] = targets["thickness_mm"].notna()
    targets["coverage_status"] = np.where(targets["observed"], "reported", "unknown")
    targets["pre_commission"] = targets["date"] < targets["commission_date"]
    quarantine = rows.loc[rows["exclusion_reason"].ne(""), [
        "row_id", "id", "date", "distance_m", "exclusion_reason"
    ]]
    return targets.sort_values(["date", "id", "segment_id"]).reset_index(drop=True), quarantine
