"""Cold-case features derived from canonical altitude and daily histories."""

import numpy as np
import pandas as pd

from pipetwin.features import feature_columns
from pipetwin.physical_features import ablation_columns


def add_features(frame, exposure):
    result = frame.copy()
    terrain = []
    # Groups are all interpolated grid points of a section/state, including
    # unlabelled points. No inspection rows enter this calculation.
    for _, part in result.groupby("exposure_id", sort=False):
        part = part.sort_values("center_m")
        x = part.center_m.to_numpy()
        z = part.altitude_relative_m.to_numpy()
        features = {}
        for radius in [30, 100, 250]:
            # Equal spacing of sample offsets avoids using the survey density
            # as a feature. At boundaries sample the known part of the profile.
            offsets = np.linspace(-radius, radius, 11)
            sampled = np.interp(x[:, None] + offsets, x, z)
            left, right = sampled[:, 0], sampled[:, -1]
            features[f"basin_{radius}"] = np.maximum(0, np.minimum(left, right) - z)
            features[f"ridge_{radius}"] = np.maximum(0, z - np.maximum(left, right))
            features[f"relief_{radius}"] = np.ptp(sampled, axis=1)
            features[f"chord_depth_{radius}"] = (left + right) / 2 - z
            features[f"slope_abs_{radius}"] = np.abs(right - left) / (
                np.minimum(x + radius, x[-1]) - np.maximum(x - radius, x[0]) + 1e-12)
            features[f"profile_percentile_{radius}"] = (sampled <= z[:, None]).mean(axis=1)
        terrain.append(pd.DataFrame(features, index=part.index))
    terrain = pd.concat(terrain).reindex(result.index)
    result = result.join(terrain)
    observed = exposure.loc[exposure.observed].copy()
    daily = observed.assign(
        wet_fill=observed.water_fraction * observed.fill_coefficient,
        wet_slow=observed.water_fraction / (1 + observed.velocity_m_s),
        wet_warm=observed.water_fraction * np.exp(
            np.clip((observed.temperature_c - 20) / 20, -4, 4)),
        low_velocity=observed.velocity_m_s.le(0.15).astype(float).where(observed.velocity_m_s.notna()),
        wet_stationary=(observed.water_fraction * observed.velocity_m_s.le(0.15))
        .where(observed.velocity_m_s.notna()),
    )
    channels = ["wet_fill", "wet_slow", "wet_warm", "low_velocity", "wet_stationary",
                "water_fraction", "temperature_c", "velocity_m_s", "fill_coefficient"]
    groups = daily.groupby("exposure_id")[channels]
    summaries = {}
    for suffix, stats in [("mean", groups.mean()), ("p10", groups.quantile(0.1)),
                          ("p90", groups.quantile(0.9))]:
        for column in stats:
            summaries[f"daily_{column}_{suffix}"] = result.exposure_id.map(stats[column])
    result = result.join(pd.DataFrame(summaries, index=result.index))
    interactions = []
    for radius in [30, 100, 250]:
        for regime in ["wet_slow", "wet_fill", "wet_stationary"]:
            name = f"basin_{radius}_by_{regime}"
            result[name] = result[f"basin_{radius}"] * result[f"daily_{regime}_mean"]
            interactions.append(name)
    canonical = feature_columns("cold")
    geometry = ["altitude_relative_m", "slope_50m", "lowpoint_depth_m", *terrain.columns]
    e02 = ablation_columns("terrain", "cold")
    groups = {
        "canonical": canonical,
        "regime": [c for c in canonical if c not in {"position_fraction", *geometry}],
        "terrain": geometry,
        "position": ["position_fraction", "altitude_relative_m"],
        "altitude_only": ["altitude_relative_m"],
        "position_only": ["position_fraction"],
        "e02": e02,
        "terrain_regime": [c for c in canonical if c != "position_fraction"] + list(terrain),
        "daily_terrain": geometry + list(summaries) + interactions,
        "all": list(dict.fromkeys(canonical + e02 + list(terrain) + list(summaries) + interactions)),
    }
    return result, groups
