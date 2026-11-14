"""Measurement diagnostics and isolated feature ablations for both cases."""

from dataclasses import replace

import numpy as np
import pandas as pd

from pipetwin.case1_measurement import individual_internal
from pipetwin.features import feature_columns
from pipetwin.geometry import assign_segments, build_grid
from pipetwin.targets import build_targets


WARM_BASE = ["anchor_mm", "anchor_observed", "neighbor_mean_mm", "neighbor_min_mm",
             "neighbor_observed_fraction", "edge_min_mm", "local_contrast_mm"]
POSITION = ["position_fraction", "altitude_relative_m"]
LOCAL = ["slope_50m", "lowpoint_depth_m", "e02_terrain_abs_slope",
         "e02_terrain_water_basin_m", "e02_terrain_slow_water_basin_m"]
MODE = ["water_fraction_mean", "liquid_velocity_mean", "temperature_mean_c",
        "regime_coverage", "temperature_coverage", "active_fraction"]


def valid_minima(cleaned, cell_size):
    rows = assign_segments(cleaned.sources.inspection, cleaned.sources.passport, cell_size)
    rows = rows.loc[rows.in_range & rows.thickness_mm.gt(0) & rows.nominal_observed_mm.notna()].copy()
    return rows.sort_values(["thickness_mm", "row_id"]).drop_duplicates(["id", "date", "segment_id"])


def inspect_campaigns(cleaned, cell_size):
    inspection = cleaned.sources.inspection.loc[lambda d: d.thickness_mm.notna()].copy()
    minima = valid_minima(cleaned, cell_size)
    main = cleaned.sources.passport.set_index("id").main_id
    nominal = cleaned.sources.passport.set_index("id").nominal_mm
    for rows in [inspection, minima]:
        rows["main_id"] = rows.id.map(main)
        rows["nominal_delta"] = rows.nominal_observed_mm - rows.id.map(nominal)
        rows["nominal_mismatch"] = ~np.isclose(rows.nominal_delta, 0, atol=0.05)
        rows["cluster"] = rows["Feature Description"].str.contains("кластер", case=False, na=False)
        rows["kind"] = np.select([rows.cluster, rows.is_corrosion], ["Cluster", "Metal loss"], default="Other")
    columns = ["main_id", "date"]
    record_counts = inspection.groupby(columns).agg(records=("id", "size"), corrosion_rows=("is_corrosion", "sum"),
                                                    median_depth=("depth_fraction", "median"))
    minima_counts = minima.groupby(columns).agg(cells=("id", "size"), corrosion_minima=("is_corrosion", "sum"),
                                               cluster_minima=("cluster", "sum"),
                                               nominal_mismatch_minima=("nominal_mismatch", "sum"))
    return record_counts.join(minima_counts).reset_index(), inspection, minima


def anchor_features(frame, known, radius=30.0):
    """Aggregate past reports only; record absence using counts and NaN."""
    output = frame.copy()
    comparable = individual_internal(known)
    comparable = comparable.loc[comparable.depth_fraction.ge(0.15)]
    for tag, source in [("all", known), ("comparable15", comparable)]:
        names = [f"{tag}_{s}" for s in ["count_cell", "count_local", "min_cell", "q25_local",
                                          "median_local", "min_local", "spread_local"]]
        output[names] = np.nan
        for (section, anchor, date), group in frame.groupby(["id", "anchor_date", "target_date"]):
            if not anchor < date:
                raise ValueError("A prior inspection must precede the target")
            rows = source.loc[source.id.eq(section) & source.date.eq(anchor)
                              & source.available_at.lt(date) & source.thickness_mm.notna()]
            x, values = rows.distance_m.to_numpy(), rows.thickness_mm.to_numpy()
            for cell in group.itertuples():
                inside = (x >= cell.start_m) & (x < cell.end_m)
                local = values[np.abs(x - cell.center_m) <= radius]
                q25, median, q75 = np.quantile(local, [0.25, 0.5, 0.75]) if len(local) else [np.nan] * 3
                output.loc[cell.Index, names] = [
                    np.log1p(inside.sum()), np.log1p(len(local)), values[inside].min() if inside.any() else np.nan,
                    q25, median, local.min() if len(local) else np.nan, q75 - q25,
                ]
    return output


def warm_groups(frame, canonical):
    all_reports = [c for c in frame if c.startswith("all_")]
    comparable = [c for c in frame if c.startswith("comparable15_")]
    return {"anchor": WARM_BASE, "canonical": canonical, "all_reports": WARM_BASE + all_reports,
            "quantiles_without_counts": WARM_BASE + [c for c in all_reports if "_count_" not in c],
            "comparable": WARM_BASE + comparable, "both_reports": WARM_BASE + all_reports + comparable,
            "both_with_mode": WARM_BASE + all_reports + comparable + MODE + POSITION + LOCAL}


def cold_groups():
    return {"position": POSITION, "terrain": POSITION + LOCAL, "terrain_mode": POSITION + LOCAL + MODE,
            "canonical": feature_columns("cold")}


def alternate_cold_targets(frame, cleaned, target_contract):
    """Return aligned optional training targets while retaining the scorer labels."""
    grid = build_grid(cleaned.sources.passport, target_contract["segment_length_m"])
    source = cleaned.sources
    keys = ["id", "segment_id", "target_date"]
    result = frame.copy()
    for name, selected in [("corrosion", source.inspection.loc[source.inspection.is_corrosion]),
                           ("individual", individual_internal(source.inspection))]:
        targets, _ = build_targets(replace(source, inspection=selected), grid, target_contract)
        aligned = frame[keys].merge(targets[["id", "segment_id", "date", "thickness_mm"]].rename(
            columns={"date": "target_date"}), on=keys, how="left", validate="one_to_one")
        result[name + "_y_mm"] = aligned.thickness_mm.to_numpy()
    minima = valid_minima(cleaned, target_contract["segment_length_m"])
    aligned = frame[keys].merge(minima[["id", "segment_id", "date", "nominal_observed_mm", "thickness_mm"]].rename(
        columns={"date": "target_date"}), on=keys, how="left", validate="one_to_one")
    np.testing.assert_allclose(aligned.thickness_mm, frame.y_mm, equal_nan=True, atol=1e-12)
    result["local_nominal_y_mm"] = (frame.nominal_mm.to_numpy() * aligned.thickness_mm
                                    / aligned.nominal_observed_mm)
    return result


def exposure_statistics(frame, exposure):
    static = frame.drop_duplicates("exposure_id").set_index("exposure_id")
    result = []
    for name, rows in exposure.groupby("exposure_id"):
        if name not in static.index:
            continue
        available = rows.loc[rows.observed].sort_values("day")
        gaps = available.day.diff().dt.days.dropna()
        info = static.loc[name]
        result.append({"exposure_id": name, "id": info.id, "main_id": info.main_id,
                       "target_date": info.target_date, "observed_days": len(available),
                       "calendar_days": len(rows), "coverage": len(available) / len(rows),
                       "median_gap_days": gaps.median(), "max_gap_days": gaps.max(),
                       "known_water_days": int(available.water_fraction.notna().sum()),
                       "first_known": available.day.min(), "last_known": available.day.max(),
                       "lead_gap_days": (available.day.min() - rows.day.min()).days,
                       "end_gap_days": (rows.day.max() - available.day.max()).days,
                       "known_water_fraction": available.water_fraction.mean()})
    return pd.DataFrame(result)
