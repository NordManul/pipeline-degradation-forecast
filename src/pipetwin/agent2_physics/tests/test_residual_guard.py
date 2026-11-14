"""Shift checks for the opt-in residual guard, without hidden-case labels."""

import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from pipetwin.agent2_physics.residual_guard import TrainBoundedFeatures
from pipetwin.agent2_physics.tests.test_contract_models import canonical, estimator, labels


def shifted_frames(canonical, scenario):
    train, query, exposure = canonical
    train, query = train.assign(scenario=scenario), query.assign(scenario=scenario)
    train["small_spread"] = 1.0 + 1e-4 * train.safe_feature
    train["almost_constant"] = 20.0 + 1e-9 * train.safe_feature
    query["small_spread"], query["almost_constant"] = 100.0, 50.0
    query["nominal_mm"] = 4.5
    query["anchor_mm"] = 4.0
    columns = ["safe_feature", "small_spread", "almost_constant", "nominal_mm"]
    return train, query, exposure, columns


@pytest.mark.parametrize("scenario", ["cold", "between"])
@pytest.mark.parametrize("missing_policy", ["observed-only", "train-median"])
def test_large_regime_shift_and_smaller_nominal_are_guarded(canonical, scenario, missing_policy):
    train, query, exposure, columns = shifted_frames(canonical, scenario)
    y = labels(train, scenario) - 0.4 * train.safe_feature.to_numpy()
    original = estimator("hybrid_residual", exposure, scenario, columns, missing_policy=missing_policy).fit(train, y)
    guarded = estimator("hybrid_residual", exposure, scenario, columns, missing_policy=missing_policy,
                        residual_guard="train-bounded").fit(train, y)
    raw = original.predict(query)
    stable = guarded.predict(query)
    assert np.max(np.abs(raw)) > 100
    assert np.max(np.abs(stable)) < 3 * query.nominal_mm.iloc[0]
    np.testing.assert_allclose(original.rate_parameters_, guarded.rate_parameters_, atol=1e-12)
    info = guarded.diagnostics_["residual_guard"]
    assert "almost_constant" in info["dropped_features"]
    assert "nominal_mm" in info["dropped_features"]
    assert info["last_predict"]["clipped_rows"] == len(query)
    assert "small_spread" in info["last_predict"]["clipped_features"]
    assert "nominal_mm" in info["last_predict"]["shifted_dropped_features"]
    assert guarded.diagnostics_["shared_final_clipping_applied"] is False
    stream = io.BytesIO()
    joblib.dump(guarded, stream)
    stream.seek(0)
    np.testing.assert_allclose(joblib.load(stream).predict(query), stable, atol=1e-12)
    np.testing.assert_allclose(guarded.predict(query.iloc[::-1]), stable[::-1], atol=1e-12)
    json.dumps(guarded.diagnostics_, allow_nan=False)


def test_guard_has_no_query_fitted_statistics_and_ignores_labels(canonical):
    train, query, exposure, columns = shifted_frames(canonical, "cold")
    y = labels(train, "cold") - 0.4 * train.safe_feature.to_numpy()
    guarded = estimator("hybrid_residual", exposure, "cold", columns,
                        residual_guard="train-bounded").fit(train, y)
    parameters = guarded.correction_[0].fit_diagnostics()
    expected = guarded.predict(query)
    contaminated = query.assign(y_mm=-1000, label_available_at="2030-01-01", target_count=9999,
                                anchor_mm=-1000, anchor_observed="invalid", neighbor_mean_mm=1e10)
    np.testing.assert_array_equal(guarded.predict(contaminated), expected)
    guarded.predict(query.assign(small_spread=-1e8, nominal_mm=1000))
    assert guarded.correction_[0].fit_diagnostics() == parameters
    future = exposure.copy()
    future.loc[future.exposure_id.eq("query"), "available_at"] = pd.Timestamp("2030-01-01")
    reference = estimator("hybrid_residual", future, "cold", columns,
                          residual_guard="train-bounded").fit(train, y)
    assert reference.correction_[0].fit_diagnostics() == parameters
    with pytest.raises(ValueError, match="availability"):
        reference.predict(query)
    with pytest.raises(ValueError, match="cutoff"):
        guarded.fit(train.assign(label_available_at="2022-07-06"), y)


def test_guard_uses_train_medians_and_support_not_query_ranges():
    X = np.array([[1, 10, np.nan], [2, 10, 5], [3, 10, 7]], dtype=float)
    guard = TrainBoundedFeatures(["varying", "constant", "missing"], z_limit=3.0).fit(X)
    before = guard.fit_diagnostics()
    values = guard.transform(np.array([[1e8, -1e8, np.nan]]))
    assert np.max(np.abs(values)) <= 3.0
    assert guard.imputer_.statistics_[2] == 6
    assert guard.fit_diagnostics() == before
    assert "constant" in before["dropped_features"]
    with_query = TrainBoundedFeatures(["varying", "constant", "missing"]).fit(
        np.vstack([X, [1e8, -1e8, 1000]]))
    assert with_query.fit_diagnostics() != before  # Sensitivity check of this isolation test.


def test_all_constant_or_missing_features_use_intercept_only(canonical):
    train, query, exposure = canonical
    train = train.assign(flat=1.0, absent=np.nan)
    query = query.assign(flat=1e10, absent=1e10)
    model = estimator("hybrid_residual", exposure, columns=["flat", "absent"],
                      residual_guard="train-bounded").fit(train, labels(train) + 0.1)
    assert np.isfinite(model.predict(query)).all()
    assert model.diagnostics_["residual_guard"]["intercept_only"]
    assert model.correction_[0].transform(query[["flat", "absent"]].to_numpy()).shape == (len(query), 1)


def test_guard_preserves_unanchored_fallback_and_signed_raw_output(canonical):
    train, query, exposure = canonical
    model = estimator("hybrid_residual", exposure, residual_guard="train-bounded").fit(
        train, train.anchor_mm + 2 * train.horizon_years)
    mixed = query.copy()
    mixed.loc[[1, 3], "anchor_observed"] = False
    mixed.loc[[1, 3], "fallback_mm"] = [-0.2, 9.0]
    prediction = model.predict(mixed)
    np.testing.assert_array_equal(prediction[[1, 3]], [-0.2, 9.0])
    assert (prediction[[0, 2, 4, 5]] > query.nominal_mm.iloc[0]).all()
    assert model.diagnostics_["residual_guard"]["last_predict"]["rows"] == 4
    all_fallback = mixed.assign(anchor_observed=False).drop(columns="safe_feature")
    np.testing.assert_array_equal(model.predict(all_fallback), all_fallback.fallback_mm)
    assert model.diagnostics_["residual_guard"]["last_predict"]["rows"] == 0


def test_guard_is_opt_in_and_options_are_explicit(canonical):
    train, query, exposure = canonical
    default = estimator("hybrid_residual", exposure).fit(train, labels(train))
    disabled = estimator("hybrid_residual", exposure, residual_guard="none").fit(train, labels(train))
    np.testing.assert_array_equal(default.predict(query), disabled.predict(query))
    assert default.diagnostics_["residual_guard"] is None
    for options in [{"residual_guard": "query-fit"}, {"residual_z_limit": 0},
                    {"residual_min_relative_scale": -1}]:
        with pytest.raises(ValueError):
            estimator("hybrid_residual", exposure, **options)


def test_pre_guard_serialized_instances_keep_original_predictions(canonical):
    train, query, exposure = canonical
    original = estimator("hybrid_residual", exposure).fit(train, labels(train))
    expected = original.predict(query)
    original.diagnostics_.pop("residual_guard")
    for key in ["residual_guard", "residual_z_limit", "residual_min_relative_scale"]:
        original.options.pop(key)
    stream = io.BytesIO()
    joblib.dump(original, stream)
    stream.seek(0)
    restored = joblib.load(stream)
    np.testing.assert_array_equal(restored.predict(query), expected)
    np.testing.assert_array_equal(restored.fit(train, labels(train)).predict(query), expected)
