"""Cases that could overstate campaign and seed robustness."""

import pandas as pd
import pytest

from pipetwin.agent2_physics.case2_release_checks import held_campaigns, seed_summary


def test_campaign_purge_removes_repeated_observed_cells_and_keeps_held_mask():
    frame = pd.DataFrame({
        "main_id": ["A", "A", "A", "A", "A", "A", "B"],
        "id": ["a", "a", "a", "a", "a", "a", "b"],
        "segment_id": [0, 1, 2, 0, 1, 2, 0],
        "target_date": pd.to_datetime(["2020-01-01"] * 3 + ["2021-01-01"] * 3 + ["2022-01-01"]),
        "observed": [True, False, True, True, True, False, True],
    })
    _, train, valid, overlap = next(held_campaigns(frame, purge=True))
    assert overlap == 1
    assert train.index.tolist() == [4, 6]
    assert valid.index.tolist() == [0, 1, 2]
    assert valid.observed.tolist() == [True, False, True]


def test_seed_gate_rejects_a_positive_mean_with_large_spread():
    averages = pd.DataFrame({"pearson_squared": [0.1, 0.1, 0.1, 0.11, 0.149, 0.101]},
                            index=["rf_42", "rf_123", "rf_2024", "xgb_42", "xgb_123", "xgb_2024"])
    _, gate = seed_summary(averages)
    assert gate["mean"]["delta_r2"] == pytest.approx(0.02)
    assert not gate["passes"]
