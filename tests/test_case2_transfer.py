from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipetwin.case2_data import OPERATION, prepare_transfer
from pipetwin.case2_transfer import ColdTabular, TABULAR, weights_by_campaign
from pipetwin.quality import read_raw_sources
from pipetwin.sources import read_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared():
    raw, manifest = read_raw_sources(ROOT / "data/raw")
    inputs = (manifest, read_config(ROOT / "contracts/cases.yaml"), read_config(ROOT / "configs/data_contract.yaml"),
              read_config(ROOT / "contracts/case2/retrospective.yaml"), read_config(ROOT / "configs/forecast.yaml"))
    return raw, inputs, prepare_transfer(raw, *inputs)


def test_real_pool_accepts_late_donors_and_keeps_the_official_query_clock(prepared):
    train, query, exposure, *_ = prepared[2]
    observed = train.loc[train.observed]
    assert len(observed) == 1080
    assert set(observed.main_id) == {"MAIN-2", "MAIN-3"}
    assert observed.target_date.gt("2022-07-01").sum() == 708
    assert len(query) == 1475 and query.target_date.eq(pd.Timestamp("2022-07-01")).all()
    assert "y_mm" not in query and "label_available_at" not in query
    assert query[OPERATION].drop_duplicates().shape[0] == 1  # TECH still at section scale.
    assert query.depression_250m.nunique() > 10
    windows = pd.concat([train, query]).drop_duplicates("exposure_id").set_index("exposure_id").window_end
    known = exposure.loc[exposure.observed]
    assert known.available_at.lt(known.exposure_id.map(windows)).all()
    for _, _, columns in TABULAR.values():
        assert not any(c.startswith(("anchor", "neighbor", "target_", "label_")) for c in columns)


def test_target_measurements_and_late_tech_cannot_change_the_transfer_prediction(prepared):
    raw, inputs, first = prepared
    altered = dict(raw)
    target_history = raw["inspection"].loc[raw["inspection"].id.eq(1250022871)].iloc[[0]].copy()
    target_history["row_id"], target_history["id"], target_history["date"] = 999999, 1250009583, "2022-07-06"
    altered["inspection"] = pd.concat([raw["inspection"], target_history], ignore_index=True)
    regime = raw["regime"].copy()
    future_target = regime.simple_section_id.eq(1250009583) & pd.to_datetime(regime.calc_date).ge("2022-07-01")
    assert future_target.any()
    regime.loc[future_target, "t_start"] = 9999.
    altered["regime"] = regime
    second = prepare_transfer(altered, *inputs)
    for table_a, table_b in zip(first[:3], second[:3]):
        pd.testing.assert_frame_equal(table_a, table_b)
    trained = []
    for train, query, *_ in [first, second]:
        observed = train.loc[train.observed]
        model = ColdTabular("bounded_ridge", "relative", ["relief_100m", "depression_250m"])
        trained.append(model.fit(observed, weights_by_campaign(observed)).predict(query))
    np.testing.assert_array_equal(*trained)
