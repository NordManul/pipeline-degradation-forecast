"""E02 feature ablations with calendar coverage and explicit unit conversions."""

import numpy as np
import pandas as pd

from pipetwin.geometry import segment_grid, terrain_features
from pipetwin.sources import TECH_COLUMNS
from pipetwin.targets import observed_targets


def daily_mode(tech: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Last snapshot per observed day in [start, end); no filling across gaps."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start != start.normalize() or end != end.normalize() or start >= end:
        raise ValueError("Window boundaries must be increasing midnight timestamps")
    columns = ["id" if c == "simple_section_id" else c for c in TECH_COLUMNS]
    rows = tech.loc[(tech.calc_date >= start) & (tech.calc_date < end), columns].copy()
    rows = rows.drop_duplicates()
    if rows.duplicated(["id", "calc_date"]).any():
        raise ValueError("Conflicting TECH snapshots at the same timestamp")
    rows["day"] = rows.calc_date.dt.normalize()
    rows = rows.sort_values("calc_date").drop_duplicates(["id", "day"], keep="last")
    for column in columns[2:]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    rows["q"] = rows.q.where(rows.q >= 0)
    rows["water_fraction"] = rows.watercut.where(rows.watercut.between(0, 100)) / 100
    rows["fill_coefficient"] = rows.fill_coefficient.where(rows.fill_coefficient.between(0, 1))
    rows["velocity_m_s"] = rows.v_liquid_true.where(rows.v_liquid_true >= 0)
    # A documented sensitivity scenario for tiny thermal-calculation placeholders.
    temperatures = rows[["t_start", "t_end"]].where(rows[["t_start", "t_end"]] > 0.001)
    rows["temperature_c"] = temperatures.mean(axis=1, skipna=False)
    rows["water_volume_m3"] = rows.q * rows.water_fraction
    rows["water_fill"] = rows.water_fraction * rows.fill_coefficient
    rows["water_slow_flow"] = rows.water_fraction / (1 + rows.velocity_m_s / 1.0)
    return rows


def mode_features(tech: pd.DataFrame, ids: pd.Series, start: str, end: str) -> pd.DataFrame:
    daily = daily_mode(tech, start, end)
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    records = []
    for section_id in ids:
        rows = daily.loc[daily.id == section_id]
        q = rows.q
        water = rows.water_fraction
        record = {
            "id": section_id,
            "mode_horizon_days": days,
            "mode_observed_fraction": len(rows) / days,
            "mode_flow_coverage": q.count() / days,
            "mode_active_fraction": (q > 0).sum() / days,
            "mode_stopped_fraction": (q == 0).sum() / days,
            "mode_flow_mean_m3_day": q.mean(),
            "mode_flow_std_m3_day": q.std(ddof=0),
            "water_coverage": water.count() / days,
            "water_fraction_mean": water.mean(),
            "water_fraction_days": water.sum(min_count=1),
            "water_volume_m3": rows.water_volume_m3.sum(min_count=1),
            "water_volume_coverage": rows.water_volume_m3.count() / days,
            "water_fill_mean": rows.water_fill.mean(),
            "water_fill_coverage": rows.water_fill.count() / days,
            "thermal_temperature_mean_c": rows.temperature_c.mean(),
            "thermal_temperature_coverage": rows.temperature_c.count() / days,
            "thermal_velocity_mean_m_s": rows.velocity_m_s.mean(),
            "thermal_velocity_p90_m_s": rows.velocity_m_s.quantile(0.9),
            "thermal_velocity_coverage": rows.velocity_m_s.count() / days,
            "thermal_water_slow_flow_mean": rows.water_slow_flow.mean(),
            "thermal_water_slow_flow_coverage": rows.water_slow_flow.count() / days,
        }
        records.append(record)
    return pd.DataFrame(records)


def build_features(
    sources: dict, start: str, end: str, main_id: str = "MAIN-2",
    policy: str = "internal-corrosion", terrain_direction: str | None = None,
    terrain_radius_m: float = 50,
) -> tuple[pd.DataFrame, dict]:
    """Construct inference rows using the prior VTD and TECH before the target date."""
    sections = sources["sections"].loc[sources["sections"].main_id == main_id]
    grid = segment_grid(sections)
    previous, audit = observed_targets(sources["vtd"], sections, start, policy)
    previous = previous.rename(columns={"thickness_mm": "previous_thickness_mm"})
    frame = grid.merge(previous, on=["id", "segment_id"], how="left", validate="one_to_one")
    frame["previous_available"] = frame.previous_thickness_mm.notna().astype(float)
    modes = mode_features(sources["tech"], sections.id, start, end)
    frame = frame.merge(modes, on="id", how="left", validate="many_to_one")
    if terrain_direction:
        terrain = terrain_features(frame, sources["profile"], terrain_direction, terrain_radius_m)
        frame["terrain_slope"] = terrain.slope
        frame["terrain_basin_depth_m"] = terrain.basin_depth_m
        frame["terrain_water_uphill"] = frame.water_fraction_mean * terrain.slope.clip(lower=0)
        frame["terrain_water_basin_m"] = frame.water_fill_mean * terrain.basin_depth_m
        frame["terrain_slow_water_basin_m"] = frame.thermal_water_slow_flow_mean * terrain.basin_depth_m
    return frame, audit


def feature_blocks(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Cumulative ablations; coordinates, IDs and target columns never enter X."""
    baseline = ["previous_thickness_mm", "previous_available"]
    baseline += [c for c in frame if c.startswith("mode_")]
    water = baseline + [c for c in frame if c.startswith("water_")]
    thermal = water + [c for c in frame if c.startswith("thermal_")]
    blocks = {"mode": baseline, "mode_water": water, "mode_water_thermal_flow": thermal}
    terrain = [c for c in frame if c.startswith("terrain_")]
    if terrain:
        blocks["mode_water_thermal_flow_terrain"] = thermal + terrain
    return blocks
