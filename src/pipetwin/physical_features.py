"""E02 features adapted from branch codex/e02-physical-features, commit f9e0ba6.

Endpoint temperature >0.001 C is an explicit sensitivity assumption.
No flow direction is inferred from the coordinate axis.
"""

import numpy as np
import pandas as pd

TECH_COLUMNS = ["simple_section_id", "calc_date", "q", "watercut", "fill_coefficient", "v_liquid_true", "t_start", "t_end"]


def ablation_columns(name, scenario):
    blocks = {
        "mode": ["mode_horizon_days", "mode_observed_fraction", "mode_flow_coverage", "mode_active_fraction",
                 "mode_stopped_fraction", "mode_flow_mean_m3_day", "mode_flow_std_m3_day"],
        "water": ["water_coverage", "water_fraction_mean", "water_fraction_days", "water_volume_m3",
                  "water_volume_coverage", "water_fill_mean", "water_fill_coverage"],
        "thermal": ["thermal_temperature_mean_c", "thermal_temperature_coverage", "thermal_velocity_mean_m_s",
                    "thermal_velocity_p90_m_s", "thermal_velocity_coverage", "thermal_water_slow_flow_mean",
                    "thermal_water_slow_flow_coverage"],
        "terrain": ["terrain_abs_slope", "terrain_basin_depth_m", "terrain_water_basin_m", "terrain_slow_water_basin_m"],
    }
    order = ["mode", "water", "thermal", "terrain"]
    if name not in order or scenario not in {"between", "cold"}:
        raise ValueError("Unknown E02 feature block / scenario")
    baseline = ["anchor_mm", "anchor_observed"] if scenario == "between" else ["nominal_mm", "age_years"]
    return baseline + ["e02_" + column for block in order[:order.index(name) + 1] for column in blocks[block]]


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
