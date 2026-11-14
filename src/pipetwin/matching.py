"""E04: abstaining one-to-one matching with explicit ambiguity and event hypotheses.

No depth/residual-thickness term enters candidate construction, cost, alignment
or assignment. Reported thickness change is computed only after pairs are fixed.
"""

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .alignment import align_positions, find_anchors


@dataclass(frozen=True)
class MatchingConfig:
    distance_tolerance_m: float = 0.25
    angle_tolerance_deg: float = 15.0
    max_distance_tolerance_m: float = 1.0
    max_angle_tolerance_deg: float = 45.0
    size_factor: float = 0.5
    max_size_ratio: float = 4.0
    max_pair_cost: float = 0.8
    min_margin: float = 0.08
    nominal_tolerance_mm: float = 0.1
    max_component_nodes: int = 500
    joint_spread_tolerance_m: float = 0.05
    joint_number_step: float = 10.0
    anchor_search_m: float = 5.0
    anchor_length_tolerance_m: float = 0.20
    anchor_min_count: int = 3
    max_anchor_gap_m: float = 100.0
    max_alignment_strain: float = 0.03

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Matching parameter {field.name} must be finite and positive")
        if self.max_distance_tolerance_m < self.distance_tolerance_m or self.max_angle_tolerance_deg < self.angle_tolerance_deg:
            raise ValueError("Maximum gates must cover their base gates")
        if self.max_angle_tolerance_deg > 180 or self.max_size_ratio <= 1 or self.min_margin >= self.max_pair_cost:
            raise ValueError("Invalid matching gate, size ratio or margin")


@dataclass
class MatchResult:
    pairs: pd.DataFrame
    candidates: pd.DataFrame
    unmatched: pd.DataFrame
    quarantine: pd.DataFrame
    events: pd.DataFrame
    anchors: pd.DataFrame
    joints: pd.DataFrame
    eligible: pd.DataFrame
    summary: dict


EDGE_COLUMNS = ["earlier_index", "later_index", "earlier_row_id", "later_row_id", "id",
                "distance_error_m", "angle_error_deg", "distance_gate_m", "angle_gate_deg",
                "cost", "alignment_supported", "earlier_kind", "later_kind"]
PAIR_COLUMNS = ["earlier_row_id", "later_row_id", "id", "earlier_date", "later_date",
                "earlier_distance_m", "later_distance_m", "cost", "margin", "distance_error_m",
                "angle_error_deg", "alignment_supported", "earlier_thickness_mm", "later_thickness_mm",
                "observed_loss_rate_mm_year", "growth_eligible"]


def circular_distance_deg(a, b):
    delta = np.abs(np.asarray(a) - np.asarray(b)) % 360
    return np.minimum(delta, 360 - delta)


def restrict_to_cells(rows, cells, segment_length_m=10.0):
    """Purge BOTH campaigns before alignment or candidate discovery in each fold."""
    if segment_length_m <= 0:
        raise ValueError("Invalid segment length")
    keys = set(zip(cells.id.astype(str), cells.segment_id.astype(int)))
    distance = rows.distance_m.to_numpy(dtype=float)
    mask = np.isfinite(distance)
    keep = np.zeros(len(rows), dtype=bool)
    indices = np.flatnonzero(mask)
    ids = rows.id.astype(str).to_numpy()
    segments = np.floor(distance[mask] / segment_length_m).astype(int)
    keep[indices] = [(section_id, segment) in keys for section_id, segment in zip(ids[mask], segments)]
    return rows.loc[keep].copy()


def prepare_geometry(rows, passport):
    if rows.row_id.duplicated().any():
        raise ValueError("Inspection row IDs must be unique")
    result = rows.copy()
    result["id"] = result.id.astype(str)
    result = result.merge(passport[["id", "length_m", "diameter_mm"]], on="id", how="left", validate="many_to_one")
    reason = pd.Series("", index=result.index)
    description = result["Feature Description"].fillna("").str.strip().str.lower()
    side = result["Defect Type"].fillna("").str.strip().str.lower()
    result["surface"] = np.select([side.str.startswith("внутр"), side.str.startswith("внеш")], ["internal", "external"], default="unknown")
    result["kind"] = np.where(description.str.contains("кластер"), "cluster", "individual")
    corrosion = description.str.contains("корроз|потеря металла", regex=True)
    reason.loc[~corrosion] = "not_corrosion"
    reason.loc[reason.eq("") & result.surface.eq("unknown")] = "unknown_surface"
    in_range = np.isfinite(result.distance_m) & result.distance_m.ge(0) & result.distance_m.lt(result.length_m)
    reason.loc[reason.eq("") & ~in_range] = "outside_section"
    angle = pd.to_numeric(result.Angle, errors="coerce")
    result["angle_deg"] = angle % 360
    valid_angle = np.isfinite(angle) & angle.between(0, 360)
    reason.loc[reason.eq("") & ~valid_angle] = "invalid_or_missing_angle"
    for raw, canonical in [("Feature Length", "length_mm"), ("Feature Width", "width_mm")]:
        result[canonical] = pd.to_numeric(result[raw], errors="coerce")
    valid_size = np.isfinite(result[["length_mm", "width_mm", "nominal_observed_mm", "diameter_mm"]]).all(axis=1)
    valid_size &= result[["length_mm", "width_mm", "nominal_observed_mm", "diameter_mm"]].gt(0).all(axis=1)
    reason.loc[reason.eq("") & ~valid_size] = "invalid_or_missing_size"
    result["exclusion_reason"] = reason
    eligible = result.loc[reason.eq("")].sort_values(["id", "row_id"]).reset_index(drop=True)
    quarantine = result.loc[reason.ne(""), ["row_id", "id", "date", "exclusion_reason"]]
    return eligible, quarantine


def candidate_edges(old, new, config, variant="geometry", include_clusters=False):
    if variant not in {"fixed", "geometry", "anchored"}:
        raise ValueError("Unknown matching variant")
    edges = []
    for section_id, old_group in old.groupby("id"):
        new_group = new.loc[new.id.eq(section_id)].sort_values("distance_m")
        distances = new_group.distance_m.to_numpy()
        for row in old_group.itertuples():
            if row.kind == "cluster" and not include_clusters:
                continue
            search = config.max_distance_tolerance_m if variant != "fixed" else config.distance_tolerance_m
            left, right = np.searchsorted(distances, [row.match_distance_m - search, row.match_distance_m + search], side="left")
            right = np.searchsorted(distances, row.match_distance_m + search, side="right")
            for other in new_group.iloc[left:right].itertuples():
                if other.kind == "cluster" and not include_clusters:
                    continue
                if row.surface != other.surface or abs(row.nominal_observed_mm - other.nominal_observed_mm) > config.nominal_tolerance_mm:
                    continue
                ratios = [max(row.length_mm, other.length_mm) / min(row.length_mm, other.length_mm),
                          max(row.width_mm, other.width_mm) / min(row.width_mm, other.width_mm)]
                if max(ratios) > config.max_size_ratio:
                    continue
                axial_gate, angular_gate = config.distance_tolerance_m, config.angle_tolerance_deg
                if variant != "fixed":
                    axial_gate = min(config.max_distance_tolerance_m, axial_gate + config.size_factor * (row.length_mm + other.length_mm) / 1000)
                    # Width/circumference is an angular tolerance proxy, not a certified footprint.
                    angular_gate = min(config.max_angle_tolerance_deg, angular_gate + config.size_factor * (row.width_mm + other.width_mm) * 360 / (np.pi * row.diameter_mm))
                dx = abs(row.match_distance_m - other.distance_m)
                da = float(circular_distance_deg(row.angle_deg, other.angle_deg))
                if dx > axial_gate or da > angular_gate:
                    continue
                size_cost = np.mean(np.log(ratios)) / np.log(config.max_size_ratio)
                cost = 0.5 * dx / axial_gate + 0.35 * da / angular_gate + 0.15 * size_cost
                edges.append((row.Index, other.Index, row.row_id, other.row_id, section_id,
                              dx, da, axial_gate, angular_gate, cost, row.alignment_supported, row.kind, other.kind))
    return pd.DataFrame(edges, columns=EDGE_COLUMNS)


def assign_edges(edges, config):
    """Optional assignment with a private no-match dummy for every earlier row.

    Geometrically ambiguous assignments are subsequently rejected, without
    forcing their freed endpoints into new low-quality pairs.
    """
    result = edges.copy()
    result["selected"], result["accepted"], result["margin"] = False, False, np.nan
    result["component"] = -1
    result["reason"] = "not_selected"
    if result.empty:
        return result, pd.DataFrame(columns=["component", "id", "earlier_count", "later_count", "hypothesis"])
    old_ids = sorted(result.earlier_index.unique())
    new_ids = sorted(result.later_index.unique())
    old_map = {value: i for i, value in enumerate(old_ids)}
    new_map = {value: i + len(old_ids) for i, value in enumerate(new_ids)}
    a = result.earlier_index.map(old_map).to_numpy()
    b = result.later_index.map(new_map).to_numpy()
    graph = coo_matrix((np.ones(2 * len(result)), (np.r_[a, b], np.r_[b, a])), shape=(len(old_ids) + len(new_ids),) * 2).tocsr()
    _, labels = connected_components(graph, directed=False)
    result["component"] = labels[a]
    events = []
    for component, group in result.groupby("component"):
        old = sorted(group.earlier_index.unique())
        new = sorted(group.later_index.unique())
        hypothesis = "one_to_one_candidate"
        if len(old) == 1 and len(new) > 1:
            hypothesis = "possible_split_or_dense_neighbors"
        elif len(old) > 1 and len(new) == 1:
            hypothesis = "possible_merge_or_dense_neighbors"
        elif len(old) > 1 and len(new) > 1:
            hypothesis = "complex_ambiguous_group"
        events.append({"component": int(component), "id": group.id.iloc[0],
                       "earlier_count": len(old), "later_count": len(new), "hypothesis": hypothesis})
        if len(old) + len(new) > config.max_component_nodes:
            result.loc[group.index, "reason"] = "component_too_large"
            continue
        oi, ni = {value: i for i, value in enumerate(old)}, {value: i for i, value in enumerate(new)}
        costs = np.full((len(old), len(new) + len(old)), 1e6)
        costs[np.arange(len(old)), len(new) + np.arange(len(old))] = config.max_pair_cost
        edge_lookup = {}
        for edge in group.itertuples():
            costs[oi[edge.earlier_index], ni[edge.later_index]] = edge.cost
            edge_lookup[(oi[edge.earlier_index], ni[edge.later_index])] = edge.Index
        row_indices, column_indices = linear_sum_assignment(costs)
        for i, j in zip(row_indices, column_indices):
            if j >= len(new) or costs[i, j] >= config.max_pair_cost:
                continue
            index = edge_lookup[(i, j)]
            edge = result.loc[index]
            alternatives = group.loc[(group.earlier_index.eq(edge.earlier_index) | group.later_index.eq(edge.later_index)) & group.index.to_series().ne(index), "cost"]
            runner_up = min(config.max_pair_cost, float(alternatives.min())) if len(alternatives) else config.max_pair_cost
            margin = runner_up - edge.cost
            accepted = margin >= config.min_margin
            result.loc[index, ["selected", "accepted", "margin", "reason"]] = [True, accepted, margin, "accepted" if accepted else "ambiguous"]
    return result, pd.DataFrame(events)


def match_campaigns(inspections, passport, earlier_date, later_date, ids, config, *, available_before, variant="geometry", allowed_cells=None, segment_length_m=10.0):
    earlier_date, later_date, cutoff = map(pd.Timestamp, [earlier_date, later_date, available_before])
    if not earlier_date < later_date < cutoff:
        raise ValueError("Matching requires earlier < later < available_before")
    ids = [str(value) for value in ids]
    if len(ids) != len(set(ids)) or not set(ids).issubset(set(passport.id)):
        raise ValueError("Matching IDs must be unique and present in the passport")
    rows = inspections.loc[inspections.id.astype(str).isin(ids) & inspections.date.isin([earlier_date, later_date])].copy()
    if allowed_cells is not None:
        rows = restrict_to_cells(rows, allowed_cells, segment_length_m)
    old_rows, new_rows = rows.loc[rows.date.eq(earlier_date)], rows.loc[rows.date.eq(later_date)]
    old, old_bad = prepare_geometry(old_rows, passport)
    new, new_bad = prepare_geometry(new_rows, passport)
    anchors = pd.DataFrame(columns=["id", "earlier_joint", "later_joint", "earlier_m", "later_m", "cost", "accepted", "reason"])
    joints = pd.DataFrame()
    old["match_distance_m"], old["alignment_supported"] = old.distance_m, False
    if variant == "anchored":
        lengths = passport.set_index("id").length_m
        anchor_old = old_rows.loc[old_rows.distance_m.ge(0) & old_rows.distance_m.lt(old_rows.id.map(lengths))]
        anchor_new = new_rows.loc[new_rows.distance_m.ge(0) & new_rows.distance_m.lt(new_rows.id.map(lengths))]
        anchors, old_joints, new_joints = find_anchors(anchor_old, anchor_new, config)
        joints = pd.concat([old_joints.assign(date=earlier_date), new_joints.assign(date=later_date)], ignore_index=True)
        old["match_distance_m"], old["alignment_supported"] = align_positions(old, anchors, config)
    assigned, events = assign_edges(candidate_edges(old, new, config, variant), config)
    accepted = assigned.loc[assigned.accepted]
    records = []
    years = (later_date - earlier_date).days / 365.25
    for edge in accepted.itertuples():
        a, b = old.loc[edge.earlier_index], new.loc[edge.later_index]
        values = np.array([a.thickness_mm, b.thickness_mm], dtype=float)
        usable = bool(np.isfinite(values).all() and (values > 0).all() and a.thickness_mm <= a.nominal_observed_mm + 0.05 and b.thickness_mm <= b.nominal_observed_mm + 0.05)
        records.append({"earlier_row_id": a.row_id, "later_row_id": b.row_id, "id": a.id,
                        "earlier_date": earlier_date, "later_date": later_date,
                        "earlier_distance_m": a.distance_m, "later_distance_m": b.distance_m,
                        "cost": edge.cost, "margin": edge.margin, "distance_error_m": edge.distance_error_m,
                        "angle_error_deg": edge.angle_error_deg, "alignment_supported": edge.alignment_supported,
                        "earlier_thickness_mm": a.thickness_mm, "later_thickness_mm": b.thickness_mm,
                        "observed_loss_rate_mm_year": (a.thickness_mm - b.thickness_mm) / years if usable else np.nan,
                        "growth_eligible": usable})
    pairs = pd.DataFrame(records, columns=PAIR_COLUMNS)
    unmatched = []
    for side, table, pair_column, edge_column in [("earlier", old, "earlier_row_id", "earlier_index"), ("later", new, "later_row_id", "later_index")]:
        paired = set(pairs[pair_column])
        counts = assigned.groupby(edge_column).size().to_dict()
        too_large = set(assigned.loc[assigned.reason.eq("component_too_large"), edge_column])
        for row in table.itertuples():
            if row.row_id in paired:
                continue
            count = counts.get(row.Index, 0)
            reason = "cluster_separate_hypothesis" if row.kind == "cluster" else ("no_candidate" if not count else "ambiguous_or_assignment_unmatched")
            if row.Index in too_large:
                reason = "component_too_large"
            unmatched.append({"side": side, "row_id": row.row_id, "id": row.id, "date": row.date, "reason": reason, "candidate_count": count})
    # Cluster links are diagnostic hypotheses only and never enter individual growth calibration.
    cluster_edges = candidate_edges(old, new, config, variant, include_clusters=True)
    cluster_edges = cluster_edges.loc[cluster_edges.earlier_kind.eq("cluster") | cluster_edges.later_kind.eq("cluster")].copy()
    cluster_edges["hypothesis"] = "cluster_related_candidate_unconfirmed"
    if len(cluster_edges):
        events = pd.concat([events, cluster_edges], ignore_index=True, sort=False)
    eligible = pd.concat([old.assign(campaign_role="earlier"), new.assign(campaign_role="later")], ignore_index=True)
    summary = {"variant": variant, "earlier_date": str(earlier_date.date()), "later_date": str(later_date.date()),
               "earlier_eligible_individuals": int(old.kind.eq("individual").sum()), "later_eligible_individuals": int(new.kind.eq("individual").sum()),
               "accepted_pairs": len(pairs), "growth_pairs": int(pairs.growth_eligible.sum()),
               "candidate_edges": len(assigned), "alignment_supported_old_rows": int(old.alignment_supported.sum()),
               "accepted_anchors": int(anchors.accepted.sum()), "unconfirmed_cluster_links": len(cluster_edges),
               "identity_accuracy": None}
    return MatchResult(pairs, assigned, pd.DataFrame(unmatched, columns=["side", "row_id", "id", "date", "reason", "candidate_count"]),
                       pd.concat([old_bad, new_bad], ignore_index=True), events, anchors, joints, eligible, summary)
