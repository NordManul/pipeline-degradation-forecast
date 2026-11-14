"""Regression tests for chronology and training-only alternative labels."""

import numpy as np
import pandas as pd

from pipetwin.agent2_physics.data_quality_features import anchor_features


def test_anchor_statistics_only_use_reports_available_before_state_date():
    frame = pd.DataFrame({"id": ["a", "a"], "start_m": [0, 10], "end_m": [10, 20],
                          "center_m": [5, 15], "anchor_date": pd.Timestamp("2020-01-01"),
                          "target_date": pd.Timestamp("2021-01-01")})
    known = pd.DataFrame({"id": ["a"] * 4, "date": pd.to_datetime(
        ["2020-01-01", "2020-01-01", "2021-01-01", "2020-01-01"]),
        "available_at": pd.to_datetime(["2020-01-01", "2021-01-01", "2021-01-01", "2020-01-01"]),
        "distance_m": [2, 3, 4, 18], "thickness_mm": [6., 1., 1., 7.],
        "depth_fraction": [.25, .8, .8, .125], "is_corrosion": True,
        "Defect Type": "внутр.", "Feature Description": "Коррозия"})
    result = anchor_features(frame, known)
    np.testing.assert_array_equal(result.all_min_cell, [6., 7.])
    np.testing.assert_allclose(result.all_q25_local, [6.25, 6.25])
    np.testing.assert_allclose(result.comparable15_min_cell, [6., np.nan], equal_nan=True)
    assert not any(column.startswith("all_") for column in frame)
