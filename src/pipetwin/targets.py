"""Observed minima only. Missing inspection coverage remains unknown."""

import numpy as np
import pandas as pd


def observed_targets(
    vtd: pd.DataFrame, sections: pd.DataFrame, date: str, policy: str = "internal-corrosion",
) -> tuple[pd.DataFrame, dict]:
    if policy not in {"internal-corrosion", "all-residual"}:
        raise ValueError("Unknown target policy")
    rows = vtd.loc[vtd.date == pd.Timestamp(date)].copy()
    rows = rows.merge(sections[["id", "length_m"]], on="id", validate="many_to_one")
    count = len(rows)
    distance = pd.to_numeric(rows.Distance, errors="coerce")
    thickness = pd.to_numeric(rows["Residual Wall Thickness"], errors="coerce")
    geometry_ok = np.isfinite(distance) & distance.between(0, rows.length_m)
    value_ok = np.isfinite(thickness) & (thickness > 0)
    description = rows["Feature Description"].fillna("").str.strip().str.lower()
    defect_type = rows["Defect Type"].fillna("").str.strip().str.lower()
    type_ok = pd.Series(True, index=rows.index)
    if policy == "internal-corrosion":
        type_ok = defect_type.str.startswith("внутр") & (
            description.str.startswith("потеря металла") | description.str.startswith("коррози")
        )
    valid = geometry_ok & value_ok & type_ok
    accepted = rows.loc[valid, ["id"]].copy()
    # Exact endpoint belongs to the last segment; out-of-range points are excluded.
    accepted["segment_id"] = np.minimum(
        np.floor(distance[valid] / 10), np.ceil(rows.loc[valid, "length_m"] / 10) - 1
    ).astype(int)
    accepted["thickness_mm"] = thickness[valid]
    targets = accepted.groupby(["id", "segment_id"], as_index=False).thickness_mm.min()
    audit = {
        "rows": count, "accepted_rows": int(valid.sum()),
        "invalid_geometry": int((~geometry_ok).sum()),
        "invalid_thickness": int((~value_ok).sum()),
        "excluded_type": int((~type_ok).sum()), "observed_segments": len(targets),
    }
    return targets, audit
