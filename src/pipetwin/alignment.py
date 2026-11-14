"""Conservative weld-geometry anchors, independent of defect depth and outcomes."""

import numpy as np
import pandas as pd


def infer_joints(rows, config):
    """Infer upstream joint positions from within-run relative distances.

    Section Number only groups records inside ONE campaign. Consecutive numbers
    are used to identify flanking lengths, never to identify a cross-run pair.
    """
    result = []
    for (section_id, number), group in rows.groupby(["id", "Section Number"], sort=True):
        relative = pd.to_numeric(group["Relative Distance"], errors="coerce")
        starts = (group.distance_m - relative).where(relative.ge(0)).dropna()
        trait = group["trait"].fillna("").str.strip().str.lower()
        explicit = group.loc[trait.eq("шов"), "distance_m"]
        starts = pd.concat([starts, explicit])
        starts = starts[np.isfinite(starts)]
        if not len(starts):
            continue
        spread = float(starts.max() - starts.min())
        result.append({"id": section_id, "joint_number": float(number),
                       "position_m": float(starts.median()), "support_rows": len(starts),
                       "spread_m": spread, "usable": spread <= config.joint_spread_tolerance_m})
    columns = ["id", "joint_number", "position_m", "support_rows", "spread_m", "usable"]
    joints = pd.DataFrame(result, columns=columns)
    joints["left_length_m"], joints["right_length_m"] = np.nan, np.nan
    for _, group in joints.groupby("id"):
        group = group.sort_values("position_m")
        number, position = group.joint_number.to_numpy(), group.position_m.to_numpy()
        step = np.isclose(np.diff(number), config.joint_number_step)
        usable = group.usable.to_numpy()
        lengths = np.diff(position)
        reliable = step & usable[:-1] & usable[1:] & (lengths > 0)
        joints.loc[group.index[1:], "left_length_m"] = np.where(reliable, lengths, np.nan)
        joints.loc[group.index[:-1], "right_length_m"] = np.where(reliable, lengths, np.nan)
    return joints


def find_anchors(earlier_rows, later_rows, config):
    """Mutual unique candidates with two consistent flanking joint lengths.

    A pair is only a geometry-supported hypothesis. No fuzzy reference-name or
    joint-number identity is asserted, and no measured thickness enters this step.
    """
    earlier, later = infer_joints(earlier_rows, config), infer_joints(later_rows, config)
    candidates = []
    for section_id, old in earlier.groupby("id"):
        new = later.loc[later.id.eq(section_id)]
        old = old.loc[old.usable].dropna(subset=["left_length_m", "right_length_m"])
        new = new.loc[new.usable].dropna(subset=["left_length_m", "right_length_m"])
        for old_row in old.itertuples():
            possible = new.loc[new.position_m.sub(old_row.position_m).abs().le(config.anchor_search_m)]
            for new_row in possible.itertuples():
                dl = abs(old_row.left_length_m - new_row.left_length_m)
                dr = abs(old_row.right_length_m - new_row.right_length_m)
                if max(dl, dr) <= config.anchor_length_tolerance_m:
                    cost = 0.45 * (dl + dr) / config.anchor_length_tolerance_m
                    cost += 0.1 * abs(old_row.position_m - new_row.position_m) / config.anchor_search_m
                    candidates.append({"id": section_id, "earlier_joint": old_row.joint_number,
                                       "later_joint": new_row.joint_number, "earlier_m": old_row.position_m,
                                       "later_m": new_row.position_m, "cost": cost})
    columns = ["id", "earlier_joint", "later_joint", "earlier_m", "later_m", "cost"]
    table = pd.DataFrame(candidates, columns=columns)
    table["accepted"] = False
    table["reason"] = "ambiguous_or_costly"
    for index, row in table.iterrows():
        same_id = table.id.eq(row.id)
        alternatives = table.loc[same_id & (table.earlier_joint.eq(row.earlier_joint) | table.later_joint.eq(row.later_joint)) & table.index.to_series().ne(index), "cost"]
        runner_up = min(0.8, float(alternatives.min())) if len(alternatives) else 0.8
        if row.cost < 0.6 and runner_up - row.cost >= 0.1:
            table.loc[index, ["accepted", "reason"]] = [True, "two_flank_geometry"]
    for _, group in table.loc[table.accepted].groupby("id"):
        group = group.sort_values("earlier_m")
        later_position = group.later_m.to_numpy()
        crossing = (later_position[:, None] >= later_position[None, :]) & (np.arange(len(group))[:, None] < np.arange(len(group))[None, :])
        bad = np.flatnonzero(crossing.any(axis=0) | crossing.any(axis=1))
        table.loc[group.index[bad], ["accepted", "reason"]] = [False, "nonmonotone_anchors"]
    return table, earlier, later


def align_positions(rows, anchors, config):
    aligned = rows.distance_m.to_numpy(dtype=float).copy()
    supported = np.zeros(len(rows), dtype=bool)
    for section_id in rows.id.unique():
        accepted = anchors.loc[anchors.id.eq(section_id) & anchors.accepted].sort_values("earlier_m")
        if len(accepted) < config.anchor_min_count:
            continue
        x, y = accepted.earlier_m.to_numpy(), accepted.later_m.to_numpy()
        positions = np.flatnonzero(rows.id.eq(section_id).to_numpy())
        original = rows.iloc[positions].distance_m.to_numpy()
        for left in range(len(x) - 1):
            gap = x[left + 1] - x[left]
            strain = (y[left + 1] - y[left]) / gap - 1 if gap > 0 else np.inf
            if gap <= 0 or gap > config.max_anchor_gap_m or abs(strain) > config.max_alignment_strain:
                continue
            inside = (original >= x[left]) & (original <= x[left + 1])
            aligned[positions[inside]] = y[left] + (original[inside] - x[left]) * (1 + strain)
            supported[positions[inside]] = True
    return aligned, supported
