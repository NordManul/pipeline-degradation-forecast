from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipetwin.quality import clean_sources, local_times, read_raw_sources
from pipetwin.sources import read_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def small_raw():
    raw, _ = read_raw_sources(ROOT / "data/raw")
    raw["profile"] = raw["profile"].groupby("id").head(3).copy()
    raw["inspection"] = raw["inspection"].loc[raw["inspection"].row_id.isin([32084, 3419])].copy()
    tech = raw["regime"]
    row = tech.loc[tech.simple_section_id.eq(1250009490) & tech.q.gt(0)].iloc[[0]].copy()
    row["calc_date"] = "2018-09-01 07:00:00"
    row["q_gas"] = 2.5
    row["watercut"] = 0.3
    row["q_oil"] = row.q * row.density_oil / 1000 * 0.997
    raw["regime"] = row
    raw["ultrasound"] = raw["ultrasound"].iloc[:0].copy()
    raw["coupons"] = raw["coupons"].iloc[:0].copy()
    return raw


def cleaned(raw, case="case2"):
    return clean_sources(raw, case, read_config(ROOT / "contracts/cases.yaml"),
                         read_config(ROOT / "configs/data_contract.yaml"))


def test_confirmed_units_and_input_immutability(small_raw):
    original = deepcopy(small_raw)
    result = cleaned(small_raw)
    row = result.sources.regime.iloc[0]
    assert row.gas_flow_m3_day_source_conditions == 2500
    assert row.water_fraction == pytest.approx(0.003)
    assert row.oil_volume_m3_day == pytest.approx(row.q * 0.997)
    assert "pit_corrosion_intensity" not in result.sources.regime
    for key in original:
        pd.testing.assert_frame_equal(small_raw[key], original[key])


@pytest.mark.parametrize("date", ["2022-07-01", "2022-07-06", "2023-12-01", "2099-01-01"])
def test_future_payload_cannot_change_case2_sources(small_raw, date):
    raw = deepcopy(small_raw)
    for source, date_column, id_column in [("inspection", "date", "id"), ("regime", "calc_date", "simple_section_id")]:
        future = raw[source].iloc[[0]].copy()
        future[date_column] = date
        future[id_column] = "invalid-future-id"
        future["Defect Depth in %" if source == "inspection" else "q"] = "unparseable-future-value"
        raw[source] = pd.concat([raw[source], future], ignore_index=True)
    first, second = cleaned(small_raw), cleaned(raw)
    pd.testing.assert_frame_equal(first.sources.inspection, second.sources.inspection)
    # A future-only extra column on TECH is not a predictor; compare canonical data.
    pd.testing.assert_frame_equal(first.sources.regime, second.sources.regime)
    assert second.audit.rule_id.eq("TIME").sum() > first.audit.rule_id.eq("TIME").sum()


def test_late_release_and_timezone_boundary_are_excluded(small_raw):
    raw = deepcopy(small_raw)
    raw["regime"]["available_at"] = "2022-06-30T21:00:00Z"  # July 1 midnight Moscow
    result = cleaned(raw)
    assert result.sources.regime.empty
    assert result.audit.reason.eq("available_on_or_after_case_cutoff").any()
    parsed = local_times(pd.Series(["2022-06-30 23:59:00", "2022-06-30T21:00:00Z", "invalid"]))
    assert parsed.iloc[0] == pd.Timestamp("2022-06-30 23:59:00")
    assert parsed.iloc[1] == pd.Timestamp("2022-07-01")
    assert pd.isna(parsed.iloc[2])


def test_optional_bad_channels_do_not_delete_valid_temperature_or_day(small_raw):
    raw = deepcopy(small_raw)
    raw["regime"]["watercut"] = 150.0
    raw["regime"]["q"] = -3.0
    raw["regime"]["t_start"] = -5.0
    raw["regime"]["t_end"] = -4.0
    result = cleaned(raw)
    assert len(result.sources.regime) == 1
    row = result.sources.regime.iloc[0]
    assert np.isnan(row.q) and np.isnan(row.water_fraction)
    assert row.t_start == -5.0 and row.t_end == -4.0
    assert set(result.audit.loc[result.audit.action.eq("mask_channel"), "rule_id"]) >= {"Q_FRACTION", "Q_FLOW"}


def test_invalid_depth_cannot_be_used_as_label_but_geometry_is_preserved(small_raw):
    raw = deepcopy(small_raw)
    mask = raw["inspection"].row_id.eq(32084)
    raw["inspection"].loc[mask, "Defect Depth in %"] = 4.0
    result = cleaned(raw)
    row = result.sources.inspection.set_index("row_id").loc[32084]
    assert np.isnan(row.thickness_mm)
    assert np.isfinite(row.distance_m)
    assert result.audit.rule_id.eq("Q_THICKNESS").any()


def test_conflicting_snapshot_is_quarantined_without_averaging(small_raw):
    raw = deepcopy(small_raw)
    second = raw["regime"].copy()
    second["q"] += 1
    raw["regime"] = pd.concat([raw["regime"], second], ignore_index=True)
    result = cleaned(raw)
    assert result.sources.regime.empty
    assert result.audit.reason.eq("conflicting_duplicate_key").sum() == 2


def test_exact_duplicates_are_logged_and_do_not_double_exposure(small_raw):
    raw = deepcopy(small_raw)
    raw["regime"] = pd.concat([raw["regime"], raw["regime"]], ignore_index=True)
    result = cleaned(raw)
    assert len(result.sources.regime) == 1
    assert result.audit.action.eq("deduplicate").sum() == 1


def test_zero_flow_is_an_observation_and_only_pressure_sentinel_is_missing(small_raw):
    raw = deepcopy(small_raw)
    raw["regime"][["q", "q_oil", "q_gas"]] = 0.0
    raw["regime"][["p_start_fact", "p_end_fact"]] = 0.0
    result = cleaned(raw)
    row = result.sources.regime.iloc[0]
    assert row.q == 0 and row.q_gas == 0
    assert pd.notna(row.date) and pd.isna(row.p_start_fact)


def test_known_bad_day_is_removed_and_other_days_remain(small_raw):
    raw = deepcopy(small_raw)
    bad = raw["regime"].copy()
    bad["simple_section_id"] = 1250009583
    bad["calc_date"] = "2018-10-07 08:00:00"
    raw["regime"] = pd.concat([raw["regime"], bad], ignore_index=True)
    result = cleaned(raw)
    assert len(result.sources.regime) == 1
    assert result.audit.rule_id.eq("Q_KNOWN_EVENT").sum() == 1
