"""Synthetic acceptance tests for the isolated contract-1.0.0 component."""

import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from pipetwin.agent2_physics.adapter import REQUIRED_MODEL_NAMES, OPTIONAL_MODEL_NAMES, make_model
from pipetwin.agent2_physics.validation import DAYS_PER_YEAR, RATE_COLUMNS, integrate_loss


CUTOFF = "2022-07-01"
MODEL_NAMES = REQUIRED_MODEL_NAMES + OPTIONAL_MODEL_NAMES


def history(key, start, end, water=0.3):
    days = pd.date_range(start, end, inclusive="left")
    return pd.DataFrame({
        "exposure_id": key, "id": "pipe", "day": days,
        "available_at": days + pd.Timedelta(hours=12), "observed": True,
        "dt_years": 1 / DAYS_PER_YEAR, "water_fraction": water,
        "temperature_c": 20 + np.sin(np.arange(len(days)) / 30),
        "velocity_m_s": 0.5, "fill_coefficient": 0.8,
    })


def sample(key, start, end, scenario="between", n=6):
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / DAYS_PER_YEAR
    return pd.DataFrame({
        "sample_id": [f"{key}-{i}" for i in range(n)], "id": "pipe", "main_id": "main",
        "segment_id": np.arange(n), "scenario": scenario, "exposure_id": key,
        "window_start": pd.Timestamp(start), "window_end": pd.Timestamp(end),
        "target_date": pd.Timestamp(end), "label_available_at": pd.Timestamp(end),
        "horizon_years": years, "nominal_mm": 8.0, "anchor_mm": 7.0,
        "anchor_observed": True, "anchor_date": pd.Timestamp(start), "fallback_mm": 5.7,
        "safe_feature": np.linspace(-1, 1, n),
    })


@pytest.fixture
def canonical():
    definitions = [("a", "2018-01-01", "2019-01-01", 0.1),
                   ("b", "2019-01-01", "2020-01-01", 0.7),
                   ("query", "2020-01-01", "2021-01-01", 0.4)]
    exposure = pd.concat([history(*definition) for definition in definitions], ignore_index=True)
    frames = [sample(*definition[:3]) for definition in definitions]
    return pd.concat(frames[:2], ignore_index=True), frames[2], exposure


def estimator(name, exposure, scenario="between", columns=("safe_feature",), **options):
    return make_model(name, scenario=scenario, exposure=exposure, cutoff_exclusive=CUTOFF,
                      feature_columns=columns, options=options)


def labels(frame, scenario="between"):
    initial = frame["anchor_mm" if scenario == "between" else "nominal_mm"].to_numpy()
    return initial - 0.2 * frame.horizon_years.to_numpy()


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("scenario", ["between", "cold"])
def test_public_api_permutation_diagnostics_and_serialization(canonical, name, scenario):
    train, query, exposure = canonical
    train["scenario"], query["scenario"] = scenario, scenario
    if scenario == "cold":
        train = train.drop(columns=[c for c in train if c.startswith("anchor")])
        query = query.drop(columns=[c for c in query if c.startswith("anchor")])
    model = estimator(name, exposure, scenario).fit(train, labels(train, scenario))
    prediction = model.predict(query)
    assert prediction.shape == (len(query),) and np.isfinite(prediction).all()
    np.testing.assert_allclose(model.predict(query.iloc[::-1]), prediction[::-1], atol=1e-12)
    np.testing.assert_allclose(prediction, labels(query, scenario), atol=1e-5)
    stream = io.BytesIO()
    joblib.dump(model, stream)
    stream.seek(0)
    np.testing.assert_allclose(joblib.load(stream).predict(query), prediction, atol=1e-12)
    json.dumps(model.diagnostics_, allow_nan=False)
    assert model.diagnostics_["contract_version"] == "1.0.0"
    assert model.diagnostics_["calibration_histories"] == 2


def test_daily_units_leap_year_and_separate_histories_of_same_id(canonical):
    train, query, exposure = canonical
    rates = exposure.exposure_id.map({"a": 0.2, "b": 0.4, "query": 0.6}).to_numpy()
    loss = integrate_loss(exposure, rates, "train-median")
    assert loss["a"] == pytest.approx(0.2 * 365 / 365.25)
    assert loss["b"] == pytest.approx(0.4 * 365 / 365.25)
    assert loss["query"] == pytest.approx(0.6 * 366 / 365.25)
    model = estimator("constant_rate", exposure).fit(train, labels(train))
    combined = pd.concat([query, train], ignore_index=True)
    np.testing.assert_allclose(model.predict(combined), labels(combined), atol=1e-5)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_preprocessing_uses_only_unique_train_histories(canonical, name):
    train, query, exposure = canonical
    reference = estimator(name, exposure).fit(train, labels(train))
    changed = exposure.copy()
    held = changed.exposure_id.eq("query")
    changed.loc[held, "temperature_c"] = 1e6
    changed.loc[held, "water_fraction"] = 0.99
    changed.loc[held, "available_at"] = pd.Timestamp("2024-01-01")
    model = estimator(name, changed).fit(train, labels(train))
    np.testing.assert_allclose(model.predict(train), reference.predict(train), atol=1e-12)
    np.testing.assert_allclose(model.rate_parameters_, reference.rate_parameters_, atol=1e-12)
    if name != "constant_rate":
        assert model.diagnostics_["rate_preprocessing"] == reference.diagnostics_["rate_preprocessing"]
    duplicated = pd.concat([train, train.loc[train.exposure_id.eq("a")].assign(sample_id=lambda d: "extra-" + d.sample_id)])
    repeated = estimator(name, exposure).fit(duplicated, labels(duplicated))
    assert repeated.diagnostics_["observed_days"] == reference.diagnostics_["observed_days"]
    if name != "constant_rate":
        assert repeated.diagnostics_["rate_preprocessing"] == reference.diagnostics_["rate_preprocessing"]
    with pytest.raises(ValueError, match="availability"):
        model.predict(query)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_unobserved_anchors_never_calibrate_and_fallback_is_exact(canonical, name):
    train, query, exposure = canonical
    reference = estimator(name, exposure).fit(train, labels(train))
    unobserved = query.assign(anchor_observed=False, anchor_mm=9999.0)
    augmented = pd.concat([train, unobserved], ignore_index=True)
    augmented_y = np.r_[labels(train), np.full(len(query), 0.01)]
    model = estimator(name, exposure).fit(augmented, augmented_y)
    np.testing.assert_allclose(model.predict(train), reference.predict(train), atol=1e-12)
    mixed = query.copy()
    mixed.loc[[1, 3], "anchor_observed"] = False
    mixed.loc[[1, 3], "anchor_mm"] = np.nan
    mixed.loc[[1, 3], "fallback_mm"] = [-0.2, 9.0]
    prediction = model.predict(mixed)
    np.testing.assert_array_equal(prediction[[1, 3]], [-0.2, 9.0])
    assert model.diagnostics_["last_predict_fallback_rows"] == 2
    assert model.diagnostics_["excluded_unobserved_anchors"] == len(query)
    assert model.last_prediction_.loc[[1, 3], "physical_prediction_mm"].isna().all()
    all_missing = mixed.assign(anchor_observed=False).drop(columns="safe_feature")
    np.testing.assert_array_equal(model.predict(all_missing), all_missing.fallback_mm)


def test_no_observed_training_anchor_is_an_explicit_error(canonical):
    train, _, exposure = canonical
    with pytest.raises(ValueError, match="No observed anchors"):
        estimator("positive_rate", exposure).fit(train.assign(anchor_observed=False), labels(train))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_cold_ignores_all_previous_inspection_columns_and_other_case_models(canonical, name):
    train, query, exposure = canonical
    cold_train, cold_query = train.assign(scenario="cold"), query.assign(scenario="cold")
    model = estimator(name, exposure, "cold").fit(cold_train, labels(cold_train, "cold"))
    prediction = model.predict(cold_query)
    poisoned = cold_query.assign(anchor_mm=-1e12, anchor_observed="invalid", anchor_date="not a date",
                                 neighbor_mean_mm=1e12, previous_thickness_mm=np.nan)
    np.testing.assert_array_equal(model.predict(poisoned), prediction)
    separate = estimator(name, exposure).fit(train, labels(train) - 1.0)
    separate.rate_parameters_[:] = 0.0
    np.testing.assert_array_equal(model.predict(cold_query), prediction)


@pytest.mark.parametrize("column", ["anchor_mm", "anchor_observed", "neighbor_mean_mm", "previous_thickness_mm"])
def test_cold_history_allowlist_is_rejected(canonical, column):
    with pytest.raises(ValueError, match="inspection-history"):
        estimator("hybrid_residual", canonical[2], "cold", columns=[column])


@pytest.mark.parametrize("column", ["y_mm", "target_depth_mm", "label_available_at", "id", "recorded_features"])
def test_target_and_metadata_allowlist_is_rejected(canonical, column):
    with pytest.raises(ValueError, match="Forbidden"):
        estimator("hybrid_residual", canonical[2], columns=[column])


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_prediction_never_reads_diagnostic_labels(canonical, name):
    train, query, exposure = canonical
    model = estimator(name, exposure).fit(train, labels(train))
    expected = model.predict(query.drop(columns="label_available_at"))
    poison = query.assign(y_mm=-1e9, thickness_mm=1e9, target_depth_mm=1e6,
                          label_available_at="2040-01-01", recorded_features=999)
    np.testing.assert_array_equal(model.predict(poison), expected)


def test_signed_increases_remain_in_robust_physical_objective(canonical):
    train, _, exposure = canonical
    signed_loss = np.tile([-0.5, 0.5], len(train) // 2)
    observed_y = train.anchor_mm.to_numpy() - signed_loss
    actual = estimator("constant_rate", exposure).fit(train, observed_y)
    clipped = estimator("constant_rate", exposure).fit(train, train.anchor_mm.to_numpy() - np.maximum(signed_loss, 0))
    assert actual.diagnostics_["negative_observed_losses"] == len(train) // 2
    assert np.exp(actual.rate_parameters_[0]) < 0.001
    assert np.exp(clipped.rate_parameters_[0]) > 0.1
    assert (actual.predict(train) <= train.anchor_mm).all()


def test_constant_rate_is_robust_to_a_large_measurement_residual(canonical):
    train, _, exposure = canonical
    frame = pd.concat([train.assign(sample_id=lambda x: str(i) + x.sample_id) for i in range(10)], ignore_index=True)
    y = labels(frame)
    y[0] -= 5
    model = estimator("constant_rate", exposure).fit(frame, y)
    assert np.exp(model.rate_parameters_[0]) == pytest.approx(0.2, abs=0.01)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_missing_scenarios_differ_and_zero_velocity_is_still_observed(canonical, name):
    train, query, exposure = canonical
    held = exposure.exposure_id.eq("query")
    missing = held & exposure.day.ge("2020-07-01")
    exposure.loc[held, "velocity_m_s"] = 0.0
    exposure.loc[missing, "observed"] = False
    exposure.loc[missing, "available_at"] = pd.NaT
    exposure.loc[missing, list(RATE_COLUMNS)] = np.nan
    lower = estimator(name, exposure, missing_policy="observed-only").fit(train, labels(train))
    filled = estimator(name, exposure, missing_policy="train-median").fit(train, labels(train))
    lower_pred, filled_pred = lower.predict(query), filled.predict(query)
    assert (lower_pred > filled_pred + 0.05).all()
    assert (lower_pred < query.anchor_mm - 0.05).all()
    assert lower.diagnostics_["options"]["missing_policy"] == "observed-only"


@pytest.mark.parametrize("scenario", ["between", "cold"])
def test_hybrid_returns_raw_signed_correction_without_hidden_clipping(canonical, scenario):
    train, query, exposure = canonical
    train, query = train.assign(scenario=scenario), query.assign(scenario=scenario)
    initial = train.anchor_mm if scenario == "between" else train.nominal_mm
    model = estimator("hybrid_residual", exposure, scenario).fit(train, initial + 2 * train.horizon_years)
    prediction = model.predict(query)
    assert (prediction > query.nominal_mm).all()
    assert (model.last_prediction_.physical_prediction_mm <= query.nominal_mm).all()
    assert model.diagnostics_["shared_final_clipping_applied"] is False


@pytest.mark.parametrize("change", ["same_day_label", "future_label", "case2_2022_07_06", "availability_before_target"])
def test_training_labels_respect_case_cutoff(canonical, change):
    train, _, exposure = canonical
    if change == "same_day_label":
        train["label_available_at"] = pd.Timestamp(CUTOFF)
    elif change == "future_label":
        train["label_available_at"] = pd.Timestamp("2023-08-12")
    elif change == "availability_before_target":
        train["label_available_at"] = train.target_date - pd.Timedelta(days=1)
    else:
        train["target_date"] = pd.Timestamp("2022-07-06")
        train["window_end"] = train.target_date
        train["horizon_years"] = (train.window_end - train.window_start).dt.days / DAYS_PER_YEAR
    with pytest.raises(ValueError, match="cutoff"):
        estimator("constant_rate", exposure).fit(train, labels(train))


@pytest.mark.parametrize("change", ["shift_days", "duplicate_day", "missing_day", "wrong_section", "bad_duration",
                                   "same_day_available", "future_available", "missing_available", "unknown_with_values"])
def test_invalid_canonical_histories_are_rejected(canonical, change):
    train, _, exposure = canonical
    selected = exposure.exposure_id.eq("a")
    first = exposure.index[selected][0]
    if change == "shift_days":
        exposure.loc[selected, "day"] += pd.Timedelta(days=1)
    elif change == "duplicate_day":
        exposure = pd.concat([exposure, exposure.loc[[first]]], ignore_index=True)
    elif change == "missing_day":
        exposure = exposure.drop(index=first)
    elif change == "wrong_section":
        exposure.loc[first, "id"] = "other-pipe"
    elif change == "bad_duration":
        exposure.loc[first, "dt_years"] = 1 / 365
    elif change == "same_day_available":
        exposure.loc[first, "available_at"] = pd.Timestamp("2019-01-01")
    elif change == "future_available":
        exposure.loc[first, "available_at"] = pd.Timestamp(CUTOFF)
    elif change == "missing_available":
        exposure.loc[first, "available_at"] = pd.NaT
    else:
        exposure.loc[first, "observed"] = False
    with pytest.raises(ValueError):
        estimator("constant_rate", exposure).fit(train, labels(train))


@pytest.mark.parametrize("column,value", [("water_fraction", 1.1), ("fill_coefficient", -0.1),
                                          ("velocity_m_s", -1), ("temperature_c", np.inf)])
def test_physical_ranges_are_checked(canonical, column, value):
    train, _, exposure = canonical
    exposure.loc[0, column] = value
    with pytest.raises(ValueError):
        estimator("positive_rate", exposure).fit(train, labels(train))


def test_negative_celsius_and_empty_training_channel_are_allowed(canonical):
    train, query, exposure = canonical
    exposure["temperature_c"] = -10.0
    exposure["fill_coefficient"] = np.nan
    model = estimator("positive_rate", exposure).fit(train, labels(train))
    assert np.isfinite(model.predict(query)).all()
    assert model.diagnostics_["rate_preprocessing"]["empty_training_channels"] == ["fill_coefficient"]
    json.dumps(model.diagnostics_, allow_nan=False)


def test_target_at_cutoff_is_valid_for_predict_but_not_fit(canonical):
    train, _, exposure = canonical
    query = sample("at-cutoff", "2022-06-01", CUTOFF)
    exposure = pd.concat([exposure, history("at-cutoff", "2022-06-01", CUTOFF)], ignore_index=True)
    model = estimator("positive_rate", exposure).fit(train, labels(train))
    assert np.isfinite(model.predict(query)).all()
    with pytest.raises(ValueError, match="cutoff"):
        model.fit(query, labels(query))
    with pytest.raises(RuntimeError, match="Fit"):
        model.predict(query)


def test_timezone_normalization_obeys_moscow_cutoff(canonical):
    train, query, exposure = canonical
    model = make_model("constant_rate", scenario="between", exposure=exposure,
                       cutoff_exclusive="2022-06-30T21:00:00Z", feature_columns=[]).fit(train, labels(train))
    assert model.diagnostics_["cutoff_exclusive"] == "2022-07-01T00:00:00"
    assert np.isfinite(model.predict(query)).all()


def test_factory_copies_context_and_validates_options(canonical):
    train, query, exposure = canonical
    model = estimator("positive_rate", exposure)
    exposure.loc[:, "water_fraction"] = 1000
    model.fit(train, labels(train))
    assert np.isfinite(model.predict(query)).all()
    for options in [{"missing_policy": "ffill"}, {"residual_alpha": 0}, {"max_nfev": 0}, {"typo": 1}]:
        with pytest.raises(ValueError):
            estimator("positive_rate", canonical[2], **options)


def test_rate_and_residual_learn_nonconstant_signals(canonical):
    train, query, exposure = canonical
    rates = train.exposure_id.map({"a": 0.1, "b": 0.9}).to_numpy()
    y = train.anchor_mm.to_numpy() - rates * train.horizon_years.to_numpy()
    physics = estimator("positive_rate", exposure).fit(train, y)
    prediction = physics.predict(train)
    assert prediction[:6].mean() > prediction[6:].mean() + 0.1
    assert np.linalg.norm(physics.rate_parameters_[1:]) > 0.01
    y = labels(train) - 0.5 * train.safe_feature.to_numpy()
    hybrid = estimator("hybrid_residual", exposure).fit(train, y)
    curve = hybrid.predict(query)
    assert curve[0] > curve[-1] + 0.01
    raw = hybrid.predict(query.assign(safe_feature=1000.0))
    assert (raw < 0).all()  # Final clipping belongs to the shared evaluator.


@pytest.mark.parametrize("change", ["sample_id", "scenario", "segment_id", "horizon_years", "anchor_date",
                                   "window_start", "nominal_mm", "conflicting_exposure_id"])
def test_invalid_frame_contract_is_rejected(canonical, change):
    train, _, exposure = canonical
    if change == "sample_id":
        train.loc[1, "sample_id"] = train.loc[0, "sample_id"]
    elif change == "scenario":
        train.loc[0, "scenario"] = "cold"
    elif change == "segment_id":
        train["segment_id"] = train.segment_id.astype(float)
        train.loc[0, "segment_id"] = 0.5
    elif change == "horizon_years":
        train.loc[0, "horizon_years"] = 100
    elif change == "anchor_date":
        train.loc[0, "anchor_date"] = train.loc[0, "target_date"]
    elif change == "window_start":
        train.loc[0, "window_start"] += pd.Timedelta(hours=1)
    elif change == "nominal_mm":
        train.loc[0, "nominal_mm"] = 0
    else:
        train["exposure_id"] = "a"
    with pytest.raises(ValueError):
        estimator("constant_rate", exposure).fit(train, labels(train))


def test_unknown_history_has_explicit_identifiability_limits(canonical):
    train, query, exposure = canonical
    exposure["observed"] = False
    exposure["available_at"] = pd.NaT
    exposure[list(RATE_COLUMNS)] = np.nan
    with pytest.raises(ValueError, match="nonzero exposure"):
        estimator("constant_rate", exposure).fit(train, labels(train))
    with pytest.raises(ValueError, match="observed training TECH"):
        estimator("positive_rate", exposure, missing_policy="train-median").fit(train, labels(train))
    constant = estimator("constant_rate", exposure, missing_policy="train-median").fit(train, labels(train))
    np.testing.assert_allclose(constant.predict(query), labels(query), atol=1e-5)


@pytest.mark.parametrize("scenario", ["between", "cold"])
def test_scale_is_positive_train_fitted_and_keeps_physical_monotonicity(canonical, scenario):
    train, query, exposure = canonical
    train, query = train.assign(scenario=scenario), query.assign(scenario=scenario)
    y = labels(train, scenario) - 0.15 * train.safe_feature.to_numpy()
    model = estimator("hybrid_scale", exposure, scenario).fit(train, y)
    prediction = model.predict(query)
    initial = query["anchor_mm" if scenario == "between" else "nominal_mm"].to_numpy()
    assert (prediction <= initial).all() and (prediction >= 0).all()
    assert prediction[0] > prediction[-1] + 0.001
    assert (model.last_prediction_.integrated_loss_mm >= 0).all()
    assert model.diagnostics_["scale_correction"]["optimizer"]["success"]
    assert np.linalg.norm(model.scale_parameters_) > 0
