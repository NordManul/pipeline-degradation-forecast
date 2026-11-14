"""Canonical case frames with point-in-time inputs and observations."""

from dataclasses import replace

import pandas as pd

from .features import attach_labels, build_snapshot
from .targets import build_targets


def _prior_targets(cleaned, grid, target_contract, as_of):
    inspection = cleaned.sources.inspection
    known = inspection.loc[inspection.date.lt(as_of) & inspection.available_at.lt(as_of)].copy()
    return build_targets(replace(cleaned.sources, inspection=known), grid, target_contract)[0]


def canonical_snapshot(cleaned, grid, all_targets, spec, config, target_contract, *, labelled):
    target = pd.Timestamp(spec["target_date"])
    if target > cleaned.cutoff or (labelled and target >= cleaned.cutoff):
        raise ValueError("Requested snapshot violates the official case cutoff")
    # Reaggregate OLD labels using only records available before this sample's T.
    # Filtering a fully aggregated future minimum afterwards would erase known
    # past observations when one late-arriving record changes the minimum.
    history = _prior_targets(cleaned, grid, target_contract, target) if "anchor_date" in spec else None
    frame = build_snapshot(cleaned.sources, grid, history, spec, config)
    frame["anchor_observed"] = frame.anchor_observed.astype(bool)
    date_text = target.strftime("%Y-%m-%d")
    frame["sample_id"] = cleaned.case_id + "|" + frame.id + "|" + date_text + "|" + frame.segment_id.astype(str)
    if labelled:
        frame = attach_labels(frame, all_targets)
        if (frame.loc[frame.observed, "label_available_at"] >= cleaned.cutoff).any():
            raise ValueError("Training labels were not available at case cutoff")
    if frame.sample_id.duplicated().any():
        raise ValueError("Canonical sample IDs must be unique")
    return frame
