"""Spatial feature and smoothing checks."""

import numpy as np
import pandas as pd
import pytest

from pipetwin.agent2_physics.broad_case2 import training_rows
from pipetwin.agent2_physics.case2_transfer_shape import SHAPE, LAND, add_shape, donor_rows, donor_weights, smooth_profile


def profiles():
    return pd.DataFrame({"exposure_id": ["a"] * 3 + ["b"] * 3, "center_m": [5, 15, 25] * 2,
                         "segment_id": [0, 1, 2] * 2, "altitude_relative_m": [2, -1, 1, 3, 4, 5]})


def test_shape_uses_own_profile_independently_of_labels_and_global_height():
    frame = profiles()
    first = add_shape(frame)
    changed = frame.assign(altitude_relative_m=[102, 99, 101, 999, 1000, -100], y_mm=-999)
    second = add_shape(changed.iloc[::-1]).sort_index()
    np.testing.assert_allclose(first.loc[:2, SHAPE[:2] + LAND], second.loc[:2, SHAPE[:2] + LAND])
    assert first.loc[1, "below_saddle_100"] > 0


def test_smoothing_keeps_section_and_state_boundaries():
    frame = profiles().iloc[[5, 1, 3, 0, 2, 4]]
    values = np.where(frame.exposure_id.eq("a"), 3., 100.)
    np.testing.assert_array_equal(smooth_profile(frame, values, 250), values)
    with pytest.raises(ValueError, match="complete canonical grid"):
        smooth_profile(frame.loc[frame.segment_id.ne(1)], np.array([0., 0., 1., 1.]), 30)


def test_pool_ablation_preserves_other_main_and_balances_sections_inside_campaign():
    frame = pd.DataFrame({
        "main_id": ["A"] * 5 + ["B"], "id": ["x", "x", "y", "x", "y", "z"],
        "target_date": pd.to_datetime(["2018-01-01"] * 3 + ["2022-01-01", "2023-01-01", "2023-01-02"]),
        "sample_id": [str(i) for i in range(6)], "observed": True,
    })
    assert donor_rows(frame, "first_two").sample_id.tolist() == ["0", "1", "2", "3", "5"]
    rows, weights = training_rows(frame, "balanced")
    weighted = donor_weights(rows, weights, "section")
    np.testing.assert_allclose(weighted[:2].sum(), weighted[2])
    np.testing.assert_allclose(weighted[:3].sum(), weighted[3:])
