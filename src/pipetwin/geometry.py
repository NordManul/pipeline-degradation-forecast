"""Local ten-metre grids; no assumed ordering of sections inside MAIN."""

import numpy as np
import pandas as pd


def segment_grid(sections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for section in sections.itertuples(index=False):
        if not np.isfinite(section.length_m) or section.length_m <= 0:
            raise ValueError("Section length must be finite and positive")
        for segment in range(int(np.ceil(section.length_m / 10))):
            start = 10.0 * segment
            end = min(start + 10, section.length_m)
            rows.append({
                "id": section.id, "main_id": section.main_id,
                "segment_id": segment, "start_m": start, "end_m": end,
                "center_m": (start + end) / 2, "length_m": section.length_m,
            })
    if not rows:
        raise ValueError("No sections selected")
    return pd.DataFrame(rows)


def terrain_features(
    grid: pd.DataFrame, profile: pd.DataFrame, direction: str, radius_m: float = 50,
) -> pd.DataFrame:
    """Scenario slope along flow and basin depth within a fixed local radius."""
    if direction not in {"increasing", "decreasing"} or not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Supply a flow direction and positive terrain radius")
    sign = 1 if direction == "increasing" else -1
    result = pd.DataFrame(index=grid.index, columns=["slope", "basin_depth_m"], dtype=float)
    for section_id, segments in grid.groupby("id"):
        points = profile.loc[profile.id == section_id, ["total_distance", "z"]].copy()
        points = points.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        points = points.groupby("total_distance", as_index=False).z.mean()
        if len(points) < 2:
            continue
        x, z = points.total_distance.to_numpy(), points.z.to_numpy()
        # Audit found endpoint rounding up to 0.047 m. Permit at most 0.1 m.
        covered = (segments.start_m >= x[0] - 0.1) & (segments.end_m <= x[-1] + 0.1)
        start_z = np.interp(segments.start_m, x, z)
        end_z = np.interp(segments.end_m, x, z)
        slope = sign * (end_z - start_z) / (segments.end_m - segments.start_m)
        center_z = np.interp(segments.center_m, x, z)
        left_z = np.interp(np.maximum(segments.center_m - radius_m, x[0]), x, z)
        right_z = np.interp(np.minimum(segments.center_m + radius_m, x[-1]), x, z)
        basin = np.maximum(0, np.minimum(left_z, right_z) - center_z)
        result.loc[segments.index, "slope"] = np.where(covered, slope, np.nan)
        result.loc[segments.index, "basin_depth_m"] = np.where(covered, basin, np.nan)
    return result
