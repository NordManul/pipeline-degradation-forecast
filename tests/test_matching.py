from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pipetwin.alignment import align_positions, find_anchors
from pipetwin.matching import (EDGE_COLUMNS, MatchingConfig, assign_edges, circular_distance_deg,
                               match_campaigns, restrict_to_cells)


def records(positions, *, date="2020-01-01", row_start=0, angles=None, sizes=None, surface="Внутренний", kind="Коррозия", section_id="1"):
    n = len(positions)
    return pd.DataFrame({"row_id": np.arange(n) + row_start, "id": section_id, "date": pd.Timestamp(date),
                         "distance_m": positions, "Angle": np.full(n, 180.0) if angles is None else angles,
                         "Feature Length": np.full(n, 20.0) if sizes is None else sizes,
                         "Feature Width": 20.0, "Feature Description": kind, "Defect Type": surface,
                         "nominal_observed_mm": 8.0, "thickness_mm": 7.0, "Section Number": np.arange(n) * 10,
                         "Relative Distance": 0.5, "trait": "Потеря металла"})


def passport():
    return pd.DataFrame({"id": ["1", "2"], "length_m": [1000.0, 1000.0], "diameter_mm": [219.0, 219.0]})


def run(old, new, **kwargs):
    return match_campaigns(pd.concat([old, new], ignore_index=True), passport(), "2020-01-01", "2021-01-01",
                           ["1", "2"], kwargs.pop("config", MatchingConfig()), available_before="2022-01-01", **kwargs)


def test_wraparound_and_degree_boundaries():
    np.testing.assert_allclose(circular_distance_deg([359, 360, 90], [1, 0, 270]), [2, 0, 180])
    result = run(records([10], angles=[359]), records([10.01], date="2021-01-01", row_start=10, angles=[1]))
    assert len(result.pairs) == 1
    assert result.pairs.angle_error_deg.iloc[0] == 2


def test_units_and_dynamic_gates_are_bounded():
    old = records([10], sizes=[100])
    new = records([10.3], date="2021-01-01", row_start=10, sizes=[100])
    assert run(old, new, variant="fixed").candidates.empty
    result = run(old, new, variant="geometry")
    assert result.candidates.distance_gate_m.iloc[0] == pytest.approx(0.35)
    assert result.candidates.angle_gate_deg.max() <= 45


@pytest.mark.parametrize("change", ["distance", "angle", "surface", "section", "nominal"])
def test_impossible_correspondences_are_not_forced(change):
    old, new = records([10]), records([10], date="2021-01-01", row_start=10)
    if change == "distance":
        new["distance_m"] = 100
    elif change == "angle":
        new["Angle"] = 0
    elif change == "surface":
        new["Defect Type"] = "Внешний"
    elif change == "section":
        new["id"] = "2"
    else:
        new["nominal_observed_mm"] = 6
    result = run(old, new)
    assert result.pairs.empty and len(result.unmatched) == 2


def test_ambiguous_duplicates_stay_unmatched_and_event_is_only_a_hypothesis():
    result = run(records([10]), records([10, 10], date="2021-01-01", row_start=10))
    assert result.pairs.empty
    assert result.candidates.selected.sum() == 1
    assert result.events.hypothesis.iloc[0] == "possible_split_or_dense_neighbors"
    assert result.summary["identity_accuracy"] is None


def test_global_assignment_can_beat_greedy_without_claiming_confidence():
    edges = pd.DataFrame([(i, j, i, j + 10, "1", 0, 0, 1, 1, cost, False, "individual", "individual")
                          for i, j, cost in [(0, 0, .10), (0, 1, .20), (1, 0, .11), (1, 1, .79)]], columns=EDGE_COLUMNS)
    assigned, _ = assign_edges(edges, MatchingConfig())
    assert set(zip(assigned.loc[assigned.selected].earlier_index, assigned.loc[assigned.selected].later_index)) == {(0, 1), (1, 0)}
    assert assigned.loc[assigned.selected, "cost"].sum() == pytest.approx(.31)


def test_depth_outcomes_and_input_order_do_not_select_pairs():
    old = records([10, 20, 30])
    new = records([10.01, 20.01, 30.01], date="2021-01-01", row_start=10)
    first = run(old, new)
    new["thickness_mm"] = [1, 5, 7.9]
    new["Defect Depth in %"] = [99, 2, 4]
    new["Repair Method"] = "arbitrary future field"
    second = run(old.sample(frac=1, random_state=1), new.sample(frac=1, random_state=2))
    compare = ["earlier_row_id", "later_row_id", "cost", "margin"]
    pd.testing.assert_frame_equal(first.pairs[compare], second.pairs[compare])
    assert not np.allclose(first.pairs.observed_loss_rate_mm_year, second.pairs.observed_loss_rate_mm_year)
    assert second.pairs.observed_loss_rate_mm_year.min() < 0


def test_cluster_pairs_do_not_enter_individual_growth():
    result = run(records([10]), records([10], date="2021-01-01", row_start=10, kind="Коррозионный кластер"))
    assert result.pairs.empty
    assert result.summary["unconfirmed_cluster_links"] == 1
    assert "cluster_related_candidate_unconfirmed" in set(result.events.hypothesis)


def test_bad_geometry_quarantine_and_empty_campaign():
    old = records([-1, 10, 20], angles=[180, np.nan, 400])
    result = run(old, records([], date="2021-01-01", row_start=10))
    assert result.pairs.empty
    assert set(result.quarantine.exclusion_reason) == {"outside_section", "invalid_or_missing_angle"}
    with pytest.raises(ValueError, match="unique"):
        run(pd.concat([old, old]), records([], date="2021-01-01"))


def test_temporal_cutoff_is_explicit_and_future_rows_have_no_effect():
    old, new = records([10]), records([10], date="2021-01-01", row_start=10)
    data = pd.concat([old, new], ignore_index=True)
    with pytest.raises(ValueError, match="available_before"):
        match_campaigns(data, passport(), "2020-01-01", "2021-01-01", ["1"], MatchingConfig(), available_before="2021-01-01")
    future = records([999], date="2030-01-01", row_start=100)
    result = match_campaigns(pd.concat([data, future]), passport(), "2020-01-01", "2021-01-01", ["1"], MatchingConfig(), available_before="2022-01-01")
    pd.testing.assert_frame_equal(result.pairs, run(old, new).pairs)


def test_fold_filter_applies_before_matching_and_alignment():
    old = records([5, 15, 25, 35])
    new = records([5.01, 15.01, 25.01, 35.01], date="2021-01-01", row_start=10)
    cells = pd.DataFrame({"id": ["1", "1"], "segment_id": [0, 3]})
    result = run(old, new, allowed_cells=cells, variant="anchored")
    new.loc[new.distance_m.between(10, 30), ["distance_m", "Angle", "Relative Distance"]] = [999, 30, 100]
    changed = run(old, new, allowed_cells=cells, variant="anchored")
    pd.testing.assert_frame_equal(result.pairs, changed.pairs)
    assert set(result.pairs.earlier_row_id) <= {0, 3}
    assert set(result.pairs.later_row_id) <= {10, 13}


def test_joint_numbers_need_not_match_and_alignment_never_extrapolates():
    starts = np.array([0, 7, 15, 26, 40], dtype=float)
    old = records(starts + .5)
    new = records(starts + 1.5, date="2021-01-01", row_start=10)
    new["Section Number"] += 1000
    anchors, _, _ = find_anchors(old, new, MatchingConfig())
    assert anchors.accepted.sum() == 3
    query = pd.DataFrame({"id": "1", "distance_m": [2.0, 20.0, 50.0]})
    mapped, supported = align_positions(query, anchors, MatchingConfig())
    np.testing.assert_allclose(mapped, [2, 21, 50])
    np.testing.assert_array_equal(supported, [False, True, False])
    strained = anchors.copy()
    strained["later_m"] = strained.earlier_m * 2
    _, supported = align_positions(query, strained, MatchingConfig())
    assert not supported.any()


def test_large_component_abstains_and_parameter_validation():
    result = run(records([10, 10]), records([10, 10], date="2021-01-01", row_start=10), config=replace(MatchingConfig(), max_component_nodes=3))
    assert result.pairs.empty
    assert result.candidates.reason.eq("component_too_large").all()
    with pytest.raises(ValueError):
        MatchingConfig(distance_tolerance_m=0)
