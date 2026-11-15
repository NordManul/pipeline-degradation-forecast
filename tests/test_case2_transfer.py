import numpy as np
import pandas as pd
import pytest

from pipetwin.case2 import ColdStart, prepare_transfer
from pipetwin.features import REGIME_FEATURES
from pipetwin.models.cold_profile import ColdProfile, blend_profiles, weights_by_campaign
from pipetwin.sources import read_config
from conftest import ROOT


@pytest.fixture(scope="module")
def prepared(data_inputs):
    raw, manifest, cases, target, config = data_inputs
    policy = read_config(ROOT / "contracts/case2/retrospective.yaml")
    inputs = manifest, cases, target, policy, config
    return raw, inputs, prepare_transfer(raw, *inputs)


def test_donor_clock_and_cold_features(prepared):
    train, query, *_ = prepared[2]
    assert train.observed.sum() == 1080 and (~train.observed).any()
    assert set(train.main_id) == {"MAIN-2", "MAIN-3"}
    assert (train.target_date.gt("2022-07-01") & train.observed).sum() == 708
    assert len(query) == 1475 and query.target_date.eq(pd.Timestamp("2022-07-01")).all()
    assert "y_mm" not in query and "label_available_at" not in query
    assert query[REGIME_FEATURES].drop_duplicates().shape[0] == 1
    assert query.slope_50m.nunique() > 10
    assert not any(c.startswith(("anchor", "neighbor", "target_", "label_")) for c in ColdStart(42).columns)


def test_balancing_is_equal_per_main_then_per_inspection(prepared):
    train = prepared[2][0].loc[lambda frame: frame.observed]
    train = train.assign(weight=weights_by_campaign(train))
    np.testing.assert_allclose(train.groupby("main_id").weight.sum(), [540., 540.])
    np.testing.assert_allclose(train.loc[train.main_id.eq("MAIN-2")].groupby("target_date").weight.sum(), [180.] * 3)
    train["weight"] = weights_by_campaign(train, balance_mains=False)
    np.testing.assert_allclose(train.groupby(["main_id", "target_date"]).weight.sum(), [270.] * 4)


def test_knn_alignment_is_local_and_includes_the_unlabelled_grid():
    frame = pd.DataFrame({
        "id": ["a", "a", "a", "b"], "target_date": ["2022", "2023", "2022", "2022"],
        "observed": [True, True, False, True],
    })
    boost = np.array([2., 5., 4., 9.])
    knn = np.array([200., 90., 100., 40.])
    np.testing.assert_allclose(blend_profiles(frame, boost, knn), [3., 5., 3., 9.])
    np.testing.assert_allclose(blend_profiles(frame, boost, np.full(4, 6.)), [2.5, 5., 3.5, 9.])


def test_target_inspections_and_future_tech_leave_predictions_invariant(prepared):
    raw, inputs, first = prepared
    altered = dict(raw)
    target_history = raw["inspection"].loc[raw["inspection"].id.eq(1250022871)].iloc[[0]].copy()
    target_history["row_id"], target_history["id"], target_history["date"] = 999999, 1250009583, "2021-01-01"
    altered["inspection"] = pd.concat([raw["inspection"], target_history], ignore_index=True)
    regime = raw["regime"].copy()
    future_target = regime.simple_section_id.eq(1250009583) & pd.to_datetime(regime.calc_date).ge("2022-07-01")
    assert future_target.any()
    regime.loc[future_target, "t_mix"] = 9999.
    altered["regime"] = regime
    second = prepare_transfer(altered, *inputs)
    for table_a, table_b in zip(first[:2], second[:2]):
        pd.testing.assert_frame_equal(table_a, table_b)
    trained = [ColdProfile(42).fit(train.loc[train.observed]).predict(query) for train, query, *_ in (first, second)]
    for name in trained[0]:
        np.testing.assert_array_equal(trained[0][name], trained[1][name])
