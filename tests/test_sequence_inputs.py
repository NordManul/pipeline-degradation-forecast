import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from pipetwin.models.sequence import SPATIAL_RADIUS_CELLS, spatial_windows, tech_windows


def test_profile_keeps_missing_cells_and_does_not_cross_sections():
    frame = pd.DataFrame({"id": ["a", "a", "b", "a"], "exposure_id": ["a", "a", "b", "a"],
                          "target_date": pd.Timestamp("2022-07-01"), "segment_id": [1, 0, 0, 3],
                          "altitude_relative_m": [20., 10., 999., 40.], "anchor_mm": [8., 5., 2., 4.],
                          "anchor_observed": [False, True, True, True]})
    windows = spatial_windows(frame, ["altitude_relative_m", "anchor_mm"])
    center = SPATIAL_RADIUS_CELLS
    np.testing.assert_allclose(windows[0, center - 1:center + 3, 0], [10., 20., np.nan, 40.], equal_nan=True)
    assert np.isnan(windows[0, center, 1])
    assert np.isnan(windows[2, center + 1, 0])


def test_tech_tokens_exclude_unknown_days_and_reject_late_availability():
    history = pd.DataFrame({"exposure_id": "a", "day": pd.to_datetime(["2022-06-29", "2022-06-30"]),
                            "available_at": pd.to_datetime(["2022-06-29 10:00", "2022-06-30 10:00"]),
                            "observed": [True, False], "water_fraction": [.2, .9], "temperature_c": [10., 99.],
                            "velocity_m_s": [.5, 99.], "fill_coefficient": [1., 1.]})
    frame = pd.DataFrame({"exposure_id": ["a"], "window_start": pd.Timestamp("2022-06-29"),
                          "window_end": pd.Timestamp("2022-07-01")})
    values = tech_windows(history, frame)
    assert np.isnan(values[0, 0, 0])
    assert values[0, -1, 0] == .2
    assert values[0, -1, -2] == 1 / 30
    with pytest.raises(ValueError, match="causal window"):
        tech_windows(history.assign(available_at=pd.Timestamp("2022-07-01")), frame)
