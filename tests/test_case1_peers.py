from pathlib import Path

import pandas as pd
import pytest

from pipetwin.case1_peers import prepare_inputs
from pipetwin.case_data import prepare_case
from pipetwin.quality import clean_sources, read_raw_sources
from pipetwin.sources import read_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs():
    raw, manifest = read_raw_sources(ROOT / "data/raw")
    cases = read_config(ROOT / "contracts/cases.yaml")
    target = read_config(ROOT / "configs/data_contract.yaml")
    policy = read_config(ROOT / "contracts/case1/same-day-peers.yaml")
    clean = clean_sources(raw, "case1", cases, target, manifest)
    case, _ = prepare_case(clean, target, read_config(ROOT / "configs/forecast.yaml"))
    return case, raw, manifest, cases, target, policy


def test_peer_labels_join_after_feature_construction_and_target_campaign_isolated(inputs):
    strict_train, query, peers, *_ = prepare_inputs(*inputs)
    assert sum(strict_train.observed) == 561
    assert "y_mm" not in query and len(query) == 525
    assert sum(peers.observed) == 53
    assert sum(peers.observed & peers.anchor_observed) == 52
    assert set(peers.id) == {"1250009490", "1250011604"}
    assert peers.anchor_date.eq(pd.Timestamp("2022-07-06")).all()

    case, raw, *configs = inputs
    changed = dict(raw)
    vtd = raw["inspection"]
    donor = vtd.loc[vtd.date.eq("2023-12-01")]
    forged = donor.iloc[[0]].assign(id=1250022871, Distance_adjusted=1.)
    changed["inspection"] = pd.concat([vtd, forged], ignore_index=True)
    tech = raw["regime"].copy()
    future = tech.simple_section_id.eq(1250022871) & pd.to_datetime(tech.calc_date).ge("2023-12-01")
    assert future.any()
    tech.loc[future, "t_start"] = 9999.
    changed["regime"] = tech
    replay = prepare_inputs(case, changed, *configs)
    for expected, actual in zip([strict_train, query, peers], replay[:3]):
        pd.testing.assert_frame_equal(expected, actual)


def test_peer_scope_rejects_target_id(inputs):
    *args, policy = inputs
    with pytest.raises(ValueError, match="explicit non-target sections"):
        prepare_inputs(*args, {**policy, "peer_ids": [policy["target_id"]]})
