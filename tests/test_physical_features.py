import numpy as np
import pandas as pd
import pytest

from pipetwin.features import _profile_features
from pipetwin.physical_features import ablation_columns, daily_mode, mode_features


def tech_rows(dates, **overrides):
    values = dict(id="1", q=100.0, watercut=0.3, fill_coefficient=0.8,
                  v_liquid_true=1.0, t_start=20.0, t_end=10.0)
    values.update(overrides)
    return pd.DataFrame({"calc_date": pd.to_datetime(dates), **values})


def test_ported_e02_calendar_units_and_missing_days():
    tech = tech_rows(["2020-01-01 08:00", "2020-01-01 15:00", "2020-01-03 08:00"], q=[200, 100, 0])
    result = mode_features(tech, pd.Series(["1"]), "2020-01-01", "2020-01-05").iloc[0]
    assert result.water_fraction_mean == pytest.approx(0.003)
    assert result.water_volume_m3 == pytest.approx(0.3)
    assert result.mode_observed_fraction == 0.5
    assert result.mode_active_fraction == 0.25
    unknown = mode_features(tech, pd.Series(["1"]), "2019-01-01", "2019-01-02").iloc[0]
    assert np.isnan(unknown.water_volume_m3)
    assert unknown.mode_observed_fraction == 0


def test_ported_e02_invalid_values_and_duplicate_conflicts():
    invalid = tech_rows(["2020-01-01"], q=-1, watercut=101, fill_coefficient=2, v_liquid_true=-1, t_start=0.0002)
    rows = daily_mode(invalid, "2020-01-01", "2020-01-02")
    assert rows[["q", "water_fraction", "fill_coefficient", "velocity_m_s", "temperature_c"]].isna().all().all()
    with pytest.raises(ValueError, match="Conflicting"):
        daily_mode(tech_rows(["2020-01-01"] * 2, q=[1, 2]), "2020-01-01", "2020-01-02")


def test_no_false_basin_at_monotonic_profile_endpoints():
    frame = pd.DataFrame({"center_m": [5, 15, 24]})
    profile = pd.DataFrame({"distance_m": [0, 10, 25], "altitude_m": [0, 1, 2.5]})
    result, _ = _profile_features(frame, profile, 50)
    np.testing.assert_allclose(result[:, 1], 0.1)
    np.testing.assert_allclose(result[:, 2], 0)


def test_e02_cold_allowlist_has_no_inspection_features():
    for name in ["mode", "water", "thermal", "terrain"]:
        assert not any("anchor" in column or "neighbor" in column for column in ablation_columns(name, "cold"))
