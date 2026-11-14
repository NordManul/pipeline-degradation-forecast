"""Explicit feature allowlists and historical snapshots with a strict date cutoff."""

import numpy as np
import pandas as pd

STATIC_FEATURES = [
    "nominal_mm", "diameter_mm", "age_years", "horizon_years", "position_fraction",
    "altitude_relative_m", "slope_50m", "lowpoint_depth_m", "profile_available",
]
REGIME_FEATURES = [
    "regime_coverage", "active_fraction", "water_fraction_mean", "water_fraction_std",
    "flow_mean_m3_day", "flow_p90_m3_day", "liquid_velocity_mean", "temperature_mean_c",
    "temperature_coverage", "active_years_observed", "water_exposure_years",
]
HISTORY_FEATURES = [
    "anchor_mm", "anchor_observed", "anchor_nominal_mm", "neighbor_mean_mm",
    "neighbor_min_mm", "neighbor_observed_fraction",
]


def feature_columns(scenario):
    if scenario not in {"between", "cold"}:
        raise ValueError(f"Unknown scenario: {scenario}")
    return STATIC_FEATURES + REGIME_FEATURES + (HISTORY_FEATURES if scenario == "between" else [])


def _profile_features(frame, profile, radius_m):
    profile = profile.sort_values("distance_m")
    centers = frame["center_m"].to_numpy()
    if len(profile) < 2:
        return np.full((len(frame), 3), np.nan), np.zeros(len(frame))
    x, z = profile["distance_m"].to_numpy(), profile["altitude_m"].to_numpy()
    altitude = np.interp(centers, x, z, left=np.nan, right=np.nan)
    left, right = np.maximum(centers - radius_m, x.min()), np.minimum(centers + radius_m, x.max())
    z_left, z_right = np.interp(left, x, z), np.interp(right, x, z)
    span = right - left
    slope = np.divide(z_right - z_left, span, out=np.full(len(frame), np.nan), where=span > 0)
    # A basin is below BOTH sides; a one-sided monotone slope is not a basin.
    lowpoint = np.maximum(0, np.minimum(z_left, z_right) - altitude)
    return np.column_stack([altitude - np.median(z), slope, lowpoint]), np.isfinite(altitude).astype(int)


def _regime_features(regime, start, end):
    eligible = regime.loc[regime.date.ge(start) & regime.date.lt(end)].copy()
    if "available_at" in eligible:
        eligible = eligible.loc[eligible.available_at.lt(end)].copy()
    # Repeated intraday records represent one day, with the last available record retained.
    eligible["day"] = eligible["date"].dt.normalize()
    daily = eligible.sort_values("date").drop_duplicates("day", keep="last")
    days = (end.normalize() - start.normalize()).days
    result = {key: np.nan for key in REGIME_FEATURES}
    result["regime_coverage"] = len(daily) / days if days > 0 else 0.0
    result["temperature_coverage"] = daily["t_mix"].notna().sum() / days if days > 0 else 0.0
    if daily.empty:
        return result
    q = daily["q"].where(daily["q"].ge(0))
    water = daily["water_fraction"].where(daily["water_fraction"].between(0, 1))
    active = q.gt(0).astype(float).where(q.notna())
    result.update({
        "active_fraction": active.mean(),
        "water_fraction_mean": water.mean(),
        "water_fraction_std": water.std(ddof=0),
        "flow_mean_m3_day": q.mean(),
        "flow_p90_m3_day": q.quantile(0.9),
        "liquid_velocity_mean": daily["v_liquid_true"].mean(),
        "temperature_mean_c": daily["t_mix"].mean(),
        "active_years_observed": active.sum(min_count=1) / 365.25,
        "water_exposure_years": (active * water).sum(min_count=1) / 365.25,
    })
    return result


def build_snapshot(sources, grid, targets, spec, config):
    """Produce prediction-time inputs; current target labels never enter this table.

    Regime observations before T are used for historical reconstruction. A true
    forecast beyond the available regime history requires an explicit future scenario.
    """
    ids = [str(value) for value in spec["ids"]]
    if len(ids) != len(set(ids)) or not set(ids).issubset(set(grid.id)):
        raise ValueError("Scenario IDs must be unique and present in the grid")
    end = pd.Timestamp(spec["target_date"]).normalize()
    anchor = pd.Timestamp(spec["anchor_date"]).normalize() if "anchor_date" in spec else None
    if anchor is not None and anchor >= end:
        raise ValueError("Anchor inspection must strictly precede the target")
    radius = float(config["history_radius_m"])
    if radius <= 0:
        raise ValueError("Feature radius must be positive")
    frames = []
    for section_id in ids:
        frame = grid.loc[grid.id.eq(section_id)].copy()
        commission = pd.Timestamp(frame["commission_date"].iloc[0])
        if end <= commission or (anchor is not None and anchor < commission):
            raise ValueError("Unresolved pre-commission inspection / metal epoch")
        start = anchor if anchor is not None else commission
        frame["target_date"] = end
        frame["scenario"] = "between" if anchor is not None else "cold"
        frame["age_years"] = (end - commission).days / 365.25
        frame["horizon_years"] = (end - start).days / 365.25
        frame["position_fraction"] = frame["center_m"] / frame["length_m"]
        # Main-chain coordinate is metadata for splits/plots, never a tabular predictor.
        frame["chain_position_m"] = frame["center_m"] + config.get("section_offsets_m", {}).get(section_id, 0.0)
        shape, available = _profile_features(frame, sources.profile.loc[sources.profile.id.eq(section_id)], radius)
        frame[["altitude_relative_m", "slope_50m", "lowpoint_depth_m"]] = shape
        frame["profile_available"] = available
        regime = sources.regime.loc[sources.regime.id.eq(section_id)]
        for key, value in _regime_features(regime, start, end).items():
            frame[key] = value
        frame["anchor_date"] = anchor
        frame["anchor_mm"] = frame["nominal_mm"]
        frame["anchor_nominal_mm"] = frame["nominal_mm"]
        frame["anchor_observed"] = 0
        frame["neighbor_mean_mm"] = np.nan
        frame["neighbor_min_mm"] = np.nan
        frame["neighbor_observed_fraction"] = 0.0
        if anchor is not None:
            previous = targets.loc[targets.id.eq(section_id) & targets.date.eq(anchor)]
            if previous.empty:
                raise ValueError(f"No anchor campaign for section {section_id} on {anchor.date()}")
            previous = previous.set_index("segment_id")
            raw_anchor = frame["segment_id"].map(previous["thickness_mm"])
            frame["anchor_observed"] = raw_anchor.notna().astype(int)
            frame["anchor_mm"] = raw_anchor.fillna(frame["nominal_mm"])
            frame["anchor_nominal_mm"] = frame["segment_id"].map(previous["nominal_observed_mm"]).fillna(frame["nominal_mm"])
            # All inputs here are from the past campaign, including in held-out locations.
            positions = previous["center_m"].to_numpy()
            thickness = previous["thickness_mm"].to_numpy()
            neighbor_values = []
            for center in frame["center_m"]:
                neighborhood = thickness[np.abs(positions - center) <= radius]
                observed = neighborhood[np.isfinite(neighborhood)]
                neighbor_values.append((
                    observed.mean() if len(observed) else np.nan,
                    observed.min() if len(observed) else np.nan,
                    len(observed) / len(neighborhood) if len(neighborhood) else 0.0,
                ))
            frame[["neighbor_mean_mm", "neighbor_min_mm", "neighbor_observed_fraction"]] = neighbor_values
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["id", "segment_id"]).reset_index(drop=True)


def attach_labels(snapshot, targets):
    columns = ["id", "segment_id", "date", "thickness_mm", "coverage_status"]
    if "label_available_at" in targets:
        columns.append("label_available_at")
    labels = targets[columns].rename(
        columns={"date": "target_date", "thickness_mm": "y_mm"}
    )
    result = snapshot.merge(labels, on=["id", "segment_id", "target_date"], how="left", validate="one_to_one")
    result["observed"] = result["y_mm"].notna()
    result["coverage_status"] = result["coverage_status"].fillna("unknown")
    return result
