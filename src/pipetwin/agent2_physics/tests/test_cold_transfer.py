"""Physical units, nominal-transfer identifiability and terrain invariance."""

import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from pipetwin.agent2_physics.tests.test_contract_models import canonical, estimator, MODEL_NAMES


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("policy", ["observed-only", "train-median"])
def test_nominal_hypotheses_are_indistinguishable_at_train_nominal(canonical, name, policy):
    train, query, exposure = canonical
    train, query = train.assign(scenario="cold"), query.assign(scenario="cold")
    missing = exposure.exposure_id.eq("query") & exposure.day.ge("2020-07-01")
    exposure.loc[missing, "observed"] = False
    exposure.loc[missing, "available_at"] = pd.NaT
    exposure.loc[missing, ["water_fraction", "temperature_c", "velocity_m_s", "fill_coefficient"]] = np.nan
    y = train.nominal_mm - (0.2 + 0.1 * train.safe_feature) * train.horizon_years
    options = {"missing_policy": policy, "residual_guard": "train-bounded"}
    absolute = estimator(name, exposure, "cold", **options).fit(train, y)
    relative = estimator(name, exposure, "cold", cold_nominal_scaling="train-relative", **options).fit(train, y)
    np.testing.assert_array_equal(relative.predict(train), absolute.predict(train))
    prediction_at_eight = relative.predict(query)
    np.testing.assert_array_equal(prediction_at_eight, absolute.predict(query))
    shifted = query.assign(nominal_mm=4.5)
    expected = prediction_at_eight * 4.5 / 8
    np.testing.assert_allclose(relative.predict(shifted), expected, atol=1e-12)
    assert not np.allclose(relative.predict(shifted), absolute.predict(shifted))
    assert relative.reference_nominal_mm_ == 8.0
    assert relative.diagnostics_["rate_unit"] == "mm/year"
    np.testing.assert_allclose(relative.predict(shifted.iloc[::-1]), expected[::-1], atol=1e-12)
    stream = io.BytesIO()
    joblib.dump(relative, stream)
    stream.seek(0)
    np.testing.assert_allclose(joblib.load(stream).predict(shifted), expected, atol=1e-12)
    json.dumps(relative.diagnostics_, allow_nan=False)


def test_mixed_nominals_use_train_reference_and_correct_mm_units(canonical):
    train, query, exposure = canonical
    train = train.assign(scenario="cold")
    train.loc[train.exposure_id.eq("a"), "nominal_mm"] = 4.0
    query = query.assign(scenario="cold", nominal_mm=3.0)
    y = train.nominal_mm * (1 - 0.025 * train.horizon_years)
    model = estimator("constant_rate", exposure, "cold", cold_nominal_scaling="train-relative").fit(train, y)
    assert model.reference_nominal_mm_ == 6.0
    assert np.exp(model.rate_parameters_[0]) == pytest.approx(0.15, abs=1e-7)
    expected_loss = query.nominal_mm * 0.025 * query.horizon_years
    np.testing.assert_allclose(model.predict(query), query.nominal_mm - expected_loss, atol=1e-7)
    np.testing.assert_allclose(model.last_prediction_.integrated_loss_mm, expected_loss, atol=1e-7)
    model.predict(query.assign(nominal_mm=1e6))
    assert model.reference_nominal_mm_ == 6.0


def test_symmetric_terrain_ignores_direction_unlisted_columns_and_query_statistics(canonical):
    train, query, exposure = canonical
    train = train.assign(scenario="cold", slope_50m=lambda d: 0.05 * d.safe_feature,
                         lowpoint_depth_m=lambda d: np.maximum(0, d.safe_feature))
    query = query.assign(scenario="cold", slope_50m=lambda d: 0.1 * d.safe_feature,
                         lowpoint_depth_m=lambda d: np.maximum(0, d.safe_feature), nominal_mm=4.5)
    before_train, before_query = train.copy(deep=True), query.copy(deep=True)
    y = train.nominal_mm - (0.2 + train.slope_50m.abs()) * train.horizon_years
    options = {"residual_guard": "train-bounded", "residual_feature_map": "terrain-symmetric",
               "cold_nominal_scaling": "train-relative"}
    model = estimator("hybrid_residual", exposure, "cold", ["lowpoint_depth_m", "slope_50m"], **options).fit(train, y)
    prediction = model.predict(query)
    stats = model.correction_[0].fit_diagnostics()
    poisoned = query.assign(slope_50m=-query.slope_50m, safe_feature=1e20, position_fraction=-999,
                            y_mm=-100, label_available_at="2023-01-01", anchor_mm=100)
    np.testing.assert_array_equal(model.predict(poisoned), prediction)
    flipped = estimator("hybrid_residual", exposure, "cold", ["slope_50m", "lowpoint_depth_m"], **options).fit(
        train.assign(slope_50m=-train.slope_50m), y)
    np.testing.assert_array_equal(flipped.predict(query), prediction)
    model.predict(query.assign(slope_50m=-100, lowpoint_depth_m=100))
    assert model.correction_[0].fit_diagnostics() == stats
    assert stats["feature_names"] == ["abs_slope_50m", "lowpoint_depth_m"]
    pd.testing.assert_frame_equal(train, before_train)
    pd.testing.assert_frame_equal(query, before_query)
    with pytest.raises(ValueError, match="cutoff"):
        model.fit(train.assign(label_available_at="2022-07-06"), y)


def test_pre_search_serialized_models_keep_predictions(canonical):
    train, query, exposure = canonical
    train, query = train.assign(scenario="cold"), query.assign(scenario="cold", nominal_mm=4.5)
    model = estimator("hybrid_residual", exposure, "cold", residual_guard="train-bounded").fit(train, train.nominal_mm - 0.3)
    expected = model.predict(query)
    for option in ["cold_nominal_scaling", "residual_feature_map"]:
        model.options.pop(option)
    del model.reference_nominal_mm_
    stream = io.BytesIO()
    joblib.dump(model, stream)
    stream.seek(0)
    restored = joblib.load(stream)
    np.testing.assert_array_equal(restored.predict(query), expected)
    np.testing.assert_array_equal(restored.fit(train, train.nominal_mm - 0.3).predict(query), expected)


@pytest.mark.parametrize("name,scenario,columns,options", [
    ("constant_rate", "between", ["safe_feature"], {"cold_nominal_scaling": "train-relative"}),
    ("constant_rate", "cold", ["safe_feature"], {"cold_nominal_scaling": "query-relative"}),
    ("hybrid_residual", "cold", ["safe_feature"], {"residual_feature_map": "terrain-symmetric"}),
    ("hybrid_scale", "cold", ["slope_50m", "lowpoint_depth_m"], {"residual_feature_map": "terrain-symmetric"}),
    ("hybrid_residual", "cold", ["slope_50m", "lowpoint_depth_m", "safe_feature"], {"residual_feature_map": "terrain-symmetric"}),
])
def test_unsupported_transfer_combinations_fail_explicitly(canonical, name, scenario, columns, options):
    with pytest.raises(ValueError):
        estimator(name, canonical[2], scenario, columns, **options)
