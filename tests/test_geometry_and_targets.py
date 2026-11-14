import numpy as np
import pandas as pd
import pytest

from pipetwin.geometry import assign_segments, build_grid
from pipetwin.sources import canonical_ids
from pipetwin.targets import build_targets


def test_grid_boundaries_and_partial_end():
    passport = pd.DataFrame({"id": ["1"], "length_m": [25.0]})
    grid = build_grid(passport)
    assert grid["segment_length_m"].tolist() == [10, 10, 5]
    points = pd.DataFrame({"id": ["1"] * 6, "distance_m": [-0.1, 0, 9.99, 10, 24.99, 25]})
    assigned = assign_segments(points, passport, 10)
    assert assigned["in_range"].tolist() == [False, True, True, True, True, False]
    assert assigned.loc[assigned.in_range, "segment_id"].tolist() == [0, 0, 1, 2]
    with pytest.raises(ValueError):
        build_grid(passport, 0)


def test_ids_are_exact():
    assert canonical_ids(pd.Series([1250011604.0])).iloc[0] == "1250011604"
    with pytest.raises(ValueError):
        canonical_ids(pd.Series([1.5]))


def test_campaign_units(sources):
    rows = sources.inspection.set_index("row_id")
    assert rows.loc[32084, "depth_fraction"] == pytest.approx(0.285)
    assert rows.loc[3419, "depth_fraction"] == pytest.approx(0.14)


def test_real_data_audit_and_unknown_coverage(sources, contract):
    grid = build_grid(sources.passport)
    targets, _ = build_targets(sources, grid, contract)
    assert len(grid) == 4668
    august = targets[targets.date.eq(pd.Timestamp("2023-08-12"))]
    assert august["observed"].sum() == 94
    assert august.loc[~august.observed, "thickness_mm"].isna().all()
    assert august.loc[~august.observed, "coverage_status"].eq("unknown").all()
    regime = sources.regime
    good = regime.q.gt(0) & regime.density_oil.gt(0) & regime.q_oil.notna() & regime.watercut.notna()
    reconstructed = 100 * (1 - regime.oil_volume_m3_day / regime.q)
    assert np.max(np.abs(reconstructed[good] - regime.watercut[good])) < 0.02
    assert "Параметры перекачки_Рнач., атм" not in sources.passport.columns
