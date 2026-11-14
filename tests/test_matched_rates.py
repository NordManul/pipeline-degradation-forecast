import numpy as np
import pandas as pd
import pytest

from pipetwin.models.matched_rates import MatchedRateModel


def frame():
    return pd.DataFrame({"id": "1", "segment_id": np.arange(20), "main_id": "m",
                         "chain_position_m": 5 + np.arange(20) * 10, "observed": True,
                         "target_date": pd.Timestamp("2021-01-01"), "scenario": "between",
                         "anchor_mm": 7.0, "anchor_observed": 1, "horizon_years": 1.0, "y_mm": 6.8})


def pairs():
    return pd.DataFrame({"id": "1", "earlier_row_id": np.arange(20), "later_row_id": np.arange(20) + 100,
                         "earlier_distance_m": 5 + np.arange(20) * 10, "later_distance_m": 5 + np.arange(20) * 10,
                         "earlier_date": pd.Timestamp("2020-01-01"), "later_date": pd.Timestamp("2021-01-01"),
                         "growth_eligible": True, "observed_loss_rate_mm_year": np.linspace(-.1, .3, 20)})


def test_segment_balance_prevents_dense_defects_dominating_global_rate():
    data = pairs()
    duplicated = pd.concat([data, pd.concat([data.iloc[:1]] * 100)], ignore_index=True)
    first = MatchedRateModel().fit_pairs(data, frame(), "2021-01-01")
    second = MatchedRateModel().fit_pairs(duplicated, frame(), "2021-01-01")
    assert first.rate_ == pytest.approx(second.rate_)
    assert second.calibration_segment_count_ == 20


def test_matching_model_cannot_forecast_its_training_campaign_without_explicit_development_mode():
    model = MatchedRateModel().fit_pairs(pairs(), frame(), "2021-01-01")
    with pytest.raises(ValueError, match="spatial development"):
        model.predict(frame())
    assert np.isfinite(model.predict(frame(), spatial_development=True)).all()
    assert np.isfinite(model.predict(frame().assign(target_date=pd.Timestamp("2022-01-01")))).all()
    with pytest.raises(ValueError, match="Future pairs"):
        model.fit_pairs(pairs().assign(later_date=pd.Timestamp("2023-01-01")), frame(), "2021-01-01")


def test_signed_and_nonnegative_models_are_separate():
    data = pairs().assign(observed_loss_rate_mm_year=-.2)
    signed = MatchedRateModel("local").fit_pairs(data, frame(), "2021-01-01")
    latent = MatchedRateModel("local_nonnegative").fit_pairs(data, frame(), "2021-01-01")
    future = frame().assign(target_date=pd.Timestamp("2022-01-01"))
    np.testing.assert_allclose(signed.predict(future), 7.2)
    np.testing.assert_allclose(latent.predict(future), 7.0)


def test_sparse_pairs_fallback_is_explicit_and_future_labels_do_not_enter_prediction():
    model = MatchedRateModel("local").fit_pairs(pairs().iloc[:2], frame(), "2021-01-01")
    assert model.fallback_used_ and model.calibration_segment_count_ == 2
    future = frame().assign(target_date=pd.Timestamp("2022-01-01"))
    np.testing.assert_allclose(model.predict(future), model.predict(future.assign(y_mm=999)))
    np.testing.assert_allclose(model.predict(future), 6.8)


def test_growth_outside_training_fold_is_rejected():
    data = pairs()
    data.loc[0, "later_distance_m"] = 995
    with pytest.raises(ValueError, match="outside the training fold"):
        MatchedRateModel().fit_pairs(data, frame(), "2021-01-01")


def test_old_endpoint_alone_cannot_cross_into_training_fold():
    data = pairs()
    data.loc[0, "earlier_distance_m"] = 995
    with pytest.raises(ValueError, match="endpoint outside"):
        MatchedRateModel().fit_pairs(data, frame(), "2021-01-01")


def test_missing_anchor_uses_shared_fallback_and_rejects_missing_fallback():
    model = MatchedRateModel().fit_pairs(pairs(), frame(), "2021-01-01")
    query = frame().assign(target_date=pd.Timestamp("2022-01-01"), fallback_mm=5.5)
    query.loc[0, ["anchor_mm", "anchor_observed"]] = [999, 0]
    assert model.predict(query)[0] == 5.5
    with pytest.raises(ValueError, match="fallback_mm"):
        model.predict(query.drop(columns="fallback_mm"))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_eligible_rate_is_rejected(value):
    data = pairs()
    data.loc[0, "observed_loss_rate_mm_year"] = value
    with pytest.raises(ValueError, match="must be finite"):
        MatchedRateModel().fit_pairs(data, frame(), "2021-01-01")
