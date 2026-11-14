"""Case 2 transfer data with distinct donor and query clocks."""

import numpy as np
import pandas as pd

from .case_data import build_exposure, canonical_snapshot
from .geometry import build_grid
from .quality import clean_case2_donors, clean_sources
from .targets import build_targets


TERRAIN = ["slope_50m", "lowpoint_depth_m", "profile_available"] + [
    f"{kind}_{radius}m" for radius in (25, 100, 250) for kind in ("relief", "depression", "slope_magnitude")
]
WET_TERRAIN = ["wet_depression_100m", "wet_depression_250m"]
OPERATION = ["water_fraction_mean", "liquid_velocity_mean", "temperature_mean_c", "regime_coverage",
             "active_fraction", "e02_water_fill_mean", "e02_thermal_water_slow_flow_mean"]


def local_features(frame, profile):
    """Direction-free geometry and optional water × depression proxies."""
    frame = frame.copy()
    for section_id, part in frame.groupby("id", sort=False):
        points = profile.loc[profile.id.eq(section_id)].sort_values("distance_m")
        x, z = points.distance_m.to_numpy(), points.altitude_m.to_numpy()
        center = part.center_m.to_numpy()
        for radius in (25, 100, 250):
            relief = np.full(len(part), np.nan)
            depression, slope = relief.copy(), relief.copy()
            if len(x) >= 2:
                valid = (center - radius >= x[0]) & (center + radius <= x[-1])
                for i in np.flatnonzero(valid):
                    left, right = center[i] - radius, center[i] + radius
                    a, b = np.interp([left, right], x, z)
                    mid = np.interp(center[i], x, z)
                    relief[i] = np.ptp(np.r_[a, z[(x > left) & (x < right)], b])
                    depression[i] = max(0, min(a, b) - mid)
                    slope[i] = abs(b - a) / (2 * radius)
            frame.loc[part.index, [f"relief_{radius}m", f"depression_{radius}m", f"slope_magnitude_{radius}m"]] = np.column_stack([relief, depression, slope])
    for radius in (100, 250):
        frame[f"wet_depression_{radius}m"] = frame[f"depression_{radius}m"] * frame.e02_thermal_water_slow_flow_mean
    return frame


def prepare_transfer(raw, manifest, cases, target_contract, validation, config):
    case = cases["cases"]["case2"]
    if (case["main_id"] != validation["target_main"]
            or pd.Timestamp(case["target_date"]) != pd.Timestamp(validation["inference_cutoff_exclusive"])):
        raise ValueError("Case 2 transfer and inference contracts disagree")
    strict = clean_sources(raw, "case2", cases, target_contract, manifest)
    donors = clean_case2_donors(raw, cases, target_contract, validation["pool_cutoff_exclusive"], manifest)
    config = {**config, "temperature_policy": "observed_endpoints"}
    donor_grid = build_grid(donors.sources.passport, target_contract["segment_length_m"])
    targets, _ = build_targets(donors.sources, donor_grid, target_contract)
    usable = targets.observed & ~targets.main_id.isin(validation["unresolved_metal_epochs"])
    snapshots = []
    for date, campaign in targets.loc[usable].groupby("date", sort=True):
        spec = {"ids": sorted(campaign.id.unique()), "target_date": date}
        frame = canonical_snapshot(donors, donor_grid, targets, spec, config, target_contract, labelled=True)
        snapshots.append(local_features(frame, donors.sources.profile))
    train = pd.concat(snapshots, ignore_index=True)
    grid = build_grid(strict.sources.passport, target_contract["segment_length_m"])
    spec = {"ids": [case["id"]], "target_date": case["target_date"]}
    query = local_features(canonical_snapshot(strict, grid, None, spec, config, target_contract, labelled=False), strict.sources.profile)
    if (train.main_id.eq(case["main_id"]).any() or train.sample_id.duplicated().any()
            or len(query) != case["expected_provisional_segments"]):
        raise ValueError("Invalid target isolation or query grid")
    historical = build_exposure(donors.sources.regime, [train], donors.cutoff)
    forecast = build_exposure(strict.sources.regime, [query], strict.cutoff)
    return train, query, pd.concat([historical, forecast], ignore_index=True), strict.audit, donors.audit
