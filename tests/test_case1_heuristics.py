import numpy as np
import pandas as pd
import pytest

from pipetwin.case1_heuristics import heuristic_predictions, history_features


def test_history_features_use_only_available_points_of_same_section_and_measured_neighbors():
    anchor, target = pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")
    frame = pd.DataFrame({"id": "a", "anchor_date": anchor, "target_date": target,
                          "sample_id": ["a0", "a1", "a2"], "start_m": [0., 10., 20.],
                          "end_m": [10., 20., 30.], "center_m": [5., 15., 25.],
                          "anchor_observed": [True, True, False], "anchor_mm": [6., 4., 8.]})
    inspection = pd.DataFrame([
        ("a", anchor, anchor, 6., 6.),
        ("a", anchor, anchor, 10.2, 4.),
        ("a", anchor, target, 9.9, 1.),  # Released at cutoff; too late for this example.
        ("a", target, target, 9.9, .5),
        ("b", anchor, anchor, 9.9, .1),
    ], columns=["id", "date", "available_at", "distance_m", "thickness_mm"])
    prior = history_features(frame, inspection, "reported_residual")
    assert prior.edge_min_mm.iloc[0] == 4.
    assert prior.edge_min_distance_m.iloc[0] == pytest.approx(5.2)
    assert prior.local_contrast_mm.iloc[0] == 2.
    assert prior.neighbor_median_50m_mm.iloc[2] == 5.
    assert np.isnan(prior.local_contrast_mm.iloc[2])
    assert np.isnan(prior.edge_min_mm.iloc[2])


def test_fixed_rule_preserves_fallback_at_missing_anchor_and_ignores_query_labels():
    train = pd.DataFrame({"observed": [True], "anchor_observed": [False],
                          "neighbor_median_50m_mm": [6.], "nominal_mm": [8.]})
    query = pd.DataFrame({"anchor_mm": [6., 8.], "anchor_observed": [True, False],
                          "nominal_mm": [8., 8.], "edge_min_mm": [4., 1.],
                          "neighbor_median_50m_mm": [5., 5.]})
    baseline = np.array([5.5, 6.0])
    first = heuristic_predictions(train, query, baseline)
    second = heuristic_predictions(train, query.assign(y_mm=[.01, 9.]), baseline)
    np.testing.assert_array_equal(first["edge_rule"], [5., 6.])
    np.testing.assert_array_equal(first["missing_anchor_rule"], [5.5, 5.5])
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
