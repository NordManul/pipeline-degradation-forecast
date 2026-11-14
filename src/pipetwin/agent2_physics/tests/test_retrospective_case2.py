"""Retrospective authorization does not authorize future query features."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneGroupOut

from pipetwin.agent2_physics.retrospective_case2 import (
    CONTEXT, STATE_DATE, TARGET_ID, campaign_metrics, validate_pool, validate_group_split,
)
from pipetwin.agent2_physics.tests.test_contract_models import canonical, estimator, labels


@pytest.fixture
def pool():
    dates = pd.to_datetime(["2018-09-23", "2022-07-06", "2023-08-12", "2023-08-12"])
    train = pd.DataFrame({"sample_id": ["a18", "a22", "b23", "c23"], "id": ["a", "a", "b", "c"],
                          "main_id": ["MAIN-2", "MAIN-2", "MAIN-3", "MAIN-3"], "case_id": CONTEXT,
                          "scenario": "cold", "target_date": dates, "window_end": dates,
                          "label_available_at": dates, "anchor_observed": False,
                          "observed": True, "y_mm": [6.0, 6.1, 5.0, 5.2]})
    query = pd.DataFrame({"sample_id": ["target"], "id": TARGET_ID, "main_id": "MAIN-4",
                          "case_id": CONTEXT, "scenario": "cold", "target_date": STATE_DATE,
                          "window_end": STATE_DATE, "anchor_observed": False})
    return train, query


def test_late_other_main_labels_are_allowed_without_moving_query_date(pool):
    train, query = pool
    validate_pool(train, query)
    assert train.target_date.max() > query.target_date.max()
    with pytest.raises(ValueError, match="state date"):
        validate_pool(train, query.assign(target_date=pd.Timestamp("2023-12-02"), window_end=pd.Timestamp("2023-12-02")))
    with pytest.raises(ValueError, match="unavailable"):
        validate_pool(train.assign(label_available_at=pd.Timestamp("2024-01-01")), query)


@pytest.mark.parametrize("violation", ["target_main", "anchor", "query_labels", "duplicate"])
def test_retrospective_scope_does_not_relax_other_safeguards(pool, violation):
    train, query = pool
    if violation == "target_main":
        train.loc[0, "main_id"] = "MAIN-4"
    elif violation == "anchor":
        train.loc[0, "anchor_observed"] = True
    elif violation == "query_labels":
        query["y_mm"] = 4.0
    else:
        train.loc[0, "sample_id"] = train.loc[1, "sample_id"]
    with pytest.raises(ValueError):
        validate_pool(train, query)


def test_all_dates_of_the_held_main_leave_training(pool):
    train, _ = pool
    for a, b in LeaveOneGroupOut().split(train, groups=train.main_id):
        fit, valid = train.iloc[a], train.iloc[b]
        validate_group_split(fit, valid)
        assert set(fit.id).isdisjoint(valid.id)
    with pytest.raises(ValueError, match="All campaigns"):
        validate_group_split(train.iloc[[0, 2]], train.iloc[[1, 3]])


@pytest.mark.parametrize("name", ["constant_rate", "hybrid_residual"])
def test_later_fit_is_legal_but_each_prediction_history_still_ends_at_its_date(canonical, name):
    earlier, later, exposure = canonical
    earlier, later = earlier.assign(scenario="cold"), later.assign(scenario="cold")
    model = estimator(name, exposure, "cold", residual_guard="train-bounded").fit(later, labels(later, "cold"))
    np.testing.assert_allclose(model.predict(earlier), labels(earlier, "cold"), atol=1e-5)
    contaminated = exposure.copy()
    contaminated.loc[contaminated.exposure_id.eq("a"), "available_at"] = pd.Timestamp("2021-01-01")
    fitted = estimator(name, contaminated, "cold", residual_guard="train-bounded").fit(later, labels(later, "cold"))
    with pytest.raises(ValueError, match="availability"):
        fitted.predict(earlier)


def test_campaign_scores_do_not_reward_only_between_campaign_level_shift():
    frame = pd.DataFrame({"case_id": CONTEXT, "main_id": "MAIN-2", "observed": True,
                          "target_date": pd.to_datetime(["2018-09-23"] * 2 + ["2022-07-06"] * 2),
                          "y_mm": [1.0, 2.0, 9.0, 10.0]})
    prediction = np.array([1.2, 1.1, 9.2, 9.1])
    assert np.corrcoef(frame.y_mm, prediction)[0, 1] > 0.9
    scores = campaign_metrics(frame, {"candidate": prediction}, {"candidate": prediction}, 0)
    assert len(scores) == 4 and scores.n.eq(2).all()
    np.testing.assert_allclose(scores.pearson_r, -1.0)
