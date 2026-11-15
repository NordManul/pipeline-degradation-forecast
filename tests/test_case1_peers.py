import numpy as np
import pandas as pd
import pytest

from pipetwin.case1 import prepare_case1, score_late


@pytest.fixture(scope="module")
def warm(data_inputs):
    return prepare_case1(*data_inputs)


def test_peer_labels_join_after_building_strict_features(warm, data_inputs):
    train, query, late, cleaned, grid = warm
    raw, _, _, contract, config = data_inputs
    assert train.observed.sum() == 561 and train.loc[train.observed, "label_available_at"].lt(cleaned.cutoff).all()
    assert len(query) == 525 and query.anchor_observed.sum() == 508
    assert "y_mm" not in query and "y_mm" not in late
    assert set(late.id) == {"1250009490", "1250011604"}
    assert late.anchor_date.eq(pd.Timestamp("2022-07-06")).all()
    metrics, _, _ = score_late(late, {"example": np.linspace(7, 6, len(late))}, raw, cleaned, grid, contract, config,
                            config["late"]["label_pool_cutoff_exclusive"])
    assert metrics.n.tolist() == [53]


def test_target_and_peer_inspections_after_cutoff_do_not_change_warm_features(warm, data_inputs):
    raw, *configs = data_inputs
    changed = dict(raw)
    vtd = raw["inspection"]
    donor = vtd.loc[vtd.date.eq("2023-12-01")].copy()
    donor["Residual Wall Thickness"] += 1.0
    forged = donor.iloc[[0]].assign(id=1250022871, Distance_adjusted=1.)
    changed["inspection"] = pd.concat([vtd.loc[vtd.date.ne("2023-12-01")], donor, forged], ignore_index=True)
    tech = raw["regime"].copy()
    future = tech.simple_section_id.eq(1250022871) & pd.to_datetime(tech.calc_date).ge("2023-12-01")
    assert future.any()
    tech.loc[future, ["q", "t_mix"]] = 9999.
    changed["regime"] = tech
    replay = prepare_case1(changed, *configs)
    for expected, actual in zip(warm[:3], replay[:3]):
        pd.testing.assert_frame_equal(expected, actual)


def test_late_scope_rejects_the_target(warm, data_inputs):
    _, _, late, cleaned, grid = warm
    raw, _, cases, contract, config = data_inputs
    bad = {**config, "late": {**config["late"], "ids": [cases["cases"]["case1"]["id"]]}}
    with pytest.raises(ValueError, match="explicit non-target sections"):
        score_late(late, {}, raw, cleaned, grid, contract, bad, config["late"]["label_pool_cutoff_exclusive"])
