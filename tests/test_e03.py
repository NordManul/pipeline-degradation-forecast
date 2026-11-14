import json

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pipetwin import e03
from pipetwin.features import build_features
from pipetwin.models.physics import (
    DAYS_PER_YEAR, RATE_COLUMNS, HybridScaleModel, ObservedThicknessModel,
    PositiveRateModel, build_exposure, robust_residual,
)
from test_e02 import inspections, sections, tech_rows


def rate_frame(count=20, days=365):
    return pd.DataFrame({
        "id": "1", "segment_id": np.arange(count), "main_id": "MAIN-2",
        "previous_thickness_mm": np.linspace(5, 8, count), "previous_available": 1.0,
        "mode_horizon_days": days,
    })


def history(days=365):
    dates = pd.date_range("2020-01-01", periods=days)
    return build_exposure(tech_rows(dates), pd.Series(["1"]), "2020-01-01",
                          dates[-1] + pd.Timedelta(days=1))


@pytest.mark.parametrize("days", [1, 365, 366])
def test_constant_rate_integrates_calendar_days_in_years(days):
    frame, exposure = rate_frame(days=days), history(days)
    model = PositiveRateModel(constant=True)
    model.parameters_ = np.array([np.log(0.2)])
    np.testing.assert_allclose(model.integrated_loss(frame, exposure), 0.2 * days / 365.25)
    np.testing.assert_allclose(model.predict(frame, exposure), frame.previous_thickness_mm - 0.2 * days / 365.25)


def test_missing_day_scenarios_and_daily_deduplication():
    tech = tech_rows(["2020-01-01 08:00", "2020-01-01 16:00", "2020-01-03 08:00"], q=[200, 100, 0])
    exposure = build_exposure(tech, ["1"], "2020-01-01", "2020-01-05")
    assert len(exposure) == 4
    assert exposure.observed.sum() == 2
    frame = rate_frame(days=4)
    complete = PositiveRateModel(constant=True)
    observed = PositiveRateModel(constant=True, missing_policy="observed-only")
    complete.parameters_ = observed.parameters_ = np.array([np.log(0.2)])
    np.testing.assert_allclose(complete.integrated_loss(frame, exposure), 0.2 * 4 / DAYS_PER_YEAR)
    np.testing.assert_allclose(observed.integrated_loss(frame, exposure), 0.2 * 2 / DAYS_PER_YEAR)


def test_future_tech_does_not_change_exposure():
    tech = tech_rows(["2020-01-01", "2020-01-03", "2021-01-01"])
    expected = build_exposure(tech, ["1"], "2020-01-01", "2020-01-03")
    tech.loc[1:, RATE_COLUMNS[0]] = 999
    tech.loc[1:, "watercut"] = 99
    assert_frame_equal(expected, build_exposure(tech, ["1"], "2020-01-01", "2020-01-03"))


def test_fit_recovers_known_constant_rate():
    frame, exposure = rate_frame(), history()
    y = frame.previous_thickness_mm - 0.15 * 365 / DAYS_PER_YEAR
    model = PositiveRateModel(constant=True).fit(frame, exposure, y)
    assert np.exp(model.parameters_[0]) == pytest.approx(0.15, abs=1e-6)
    np.testing.assert_allclose(model.predict(frame, exposure), y, atol=1e-6)


def test_variable_rate_is_integrated_daily_before_aggregation():
    frame, exposure = rate_frame(days=4), history(4)
    water = np.array([0.1, 0.2, 0.4, 0.9])
    exposure["water_fraction"] = water
    model = PositiveRateModel().fit(frame, exposure, frame.previous_thickness_mm - 0.01)
    model.parameters_ = np.array([np.log(0.1), 0.5, 0, 0, 0])
    standardized = (water - water.mean()) / water.std()
    expected = np.exp(np.log(0.1) + 0.5 * standardized).sum() / DAYS_PER_YEAR
    np.testing.assert_allclose(model.integrated_loss(frame, exposure), expected)
    np.testing.assert_allclose(model.integrated_loss(frame, exposure.sample(frac=1, random_state=5)), expected)
    assert expected > 0.1 * len(exposure) / DAYS_PER_YEAR


def test_measured_thickness_increase_is_kept_and_rate_stays_positive():
    frame, exposure = rate_frame(), history()
    model = PositiveRateModel().fit(frame, exposure, frame.previous_thickness_mm + 0.3)
    assert (model.daily_rates(exposure) > 0).all()
    assert (model.predict(frame, exposure) <= frame.previous_thickness_mm).all()
    assert model.integrated_loss(frame, exposure).max() < 1e-4


def test_physical_forecast_saturates_at_zero():
    frame, exposure = rate_frame(), history()
    model = PositiveRateModel(constant=True)
    model.parameters_ = np.array([np.log(100)])
    np.testing.assert_array_equal(model.predict(frame, exposure), 0)


def test_daily_integration_is_invariant_to_row_order():
    frame, exposure = rate_frame(), history()
    model = PositiveRateModel().fit(frame, exposure, frame.previous_thickness_mm - 0.2)
    shuffled = exposure.sample(frac=1, random_state=42)
    np.testing.assert_allclose(model.integrated_loss(frame, exposure), model.integrated_loss(frame, shuffled))


def test_negative_exposure_velocity_fails_before_log_transform():
    frame, exposure = rate_frame(), history()
    exposure.loc[0, "velocity_m_s"] = -2
    with pytest.raises(ValueError, match="velocity"):
        PositiveRateModel().fit(frame, exposure, frame.previous_thickness_mm - 0.1)


def test_training_medians_ignore_held_out_sections():
    frame, exposure = rate_frame(), history()
    held = exposure.copy()
    held["id"] = "2"
    held["water_fraction"] = 0.99
    all_history = pd.concat([exposure, held], ignore_index=True)
    model = PositiveRateModel().fit(frame, all_history, frame.previous_thickness_mm - 0.1)
    assert model.transformer_.named_steps["simpleimputer"].statistics_[0] == pytest.approx(0.003)
    # All-missing inference is defined by the fitted training-day medians.
    unknown = exposure.copy()
    unknown[RATE_COLUMNS] = np.nan
    unknown["observed"] = False
    assert np.isfinite(model.predict(frame, unknown)).all()


@pytest.mark.parametrize("damage", ["duplicate", "missing", "duration", "prior"])
def test_invalid_exposure_or_baseline_is_rejected(damage):
    frame, exposure = rate_frame(), history()
    model = PositiveRateModel(constant=True)
    model.parameters_ = np.array([np.log(0.1)])
    if damage == "duplicate":
        exposure = pd.concat([exposure, exposure.iloc[:1]], ignore_index=True)
    elif damage == "missing":
        exposure = exposure.iloc[:-1]
    elif damage == "duration":
        exposure.loc[0, "dt_years"] = -1
    else:
        frame.loc[0, "previous_thickness_mm"] = np.nan
    with pytest.raises(ValueError):
        model.predict(frame, exposure)


def test_robust_loss_preserves_sign_and_limits_outlier_influence():
    residual = robust_residual(np.array([-100, -1, 0, 1, 100.0]))
    assert (np.sign(residual) == [-1, -1, 0, 1, 1]).all()
    assert residual[-1] ** 2 < 100


def test_hybrid_outputs_and_serialized_inference(tmp_path):
    frame, exposure = rate_frame(), history()
    y = frame.previous_thickness_mm - np.linspace(-0.1, 0.3, len(frame))
    physics = PositiveRateModel().fit(frame, exposure, y)
    columns = ["previous_thickness_mm", "mode_horizon_days"]
    models = {
        "parametric_rate": physics,
        "hybrid_scale": HybridScaleModel(physics, columns).fit(frame, exposure, y),
        "hybrid_residual": ObservedThicknessModel(columns, physics).fit(frame, exposure, y),
        "ml": ObservedThicknessModel(columns).fit(frame, exposure, y),
    }
    expected = e03.predict_e03({"models": models}, frame, exposure)
    assert expected.hybrid_scale.le(frame.previous_thickness_mm).all()
    path = tmp_path / "bundle.joblib"
    joblib.dump({"models": models}, path)
    assert_frame_equal(expected, e03.predict_e03(joblib.load(path), frame, exposure))


@pytest.fixture
def sources():
    distance = np.arange(0, 1000, 10)
    old = inspections(distance, 6 + np.sin(distance / 100))
    current = inspections(distance, 5.8 + np.sin(distance / 100), "2022-07-06")
    late = inspections(distance, 5.7 + np.sin(distance / 100), "2023-12-01")
    return {"sections": sections(1000), "vtd": pd.concat([old, current, late], ignore_index=True),
            "tech": tech_rows(["2019-01-01", "2023-01-01"])}


def test_holdout_labels_cannot_change_features_or_rate_history(sources):
    frame, _ = build_features(sources, "2018-09-23", "2022-07-06")
    mask = sources["vtd"].date >= pd.Timestamp("2022-07-06")
    sources["vtd"].loc[mask, "Residual Wall Thickness"] = 999
    actual, _ = build_features(sources, "2018-09-23", "2022-07-06")
    assert_frame_equal(frame, actual)


def test_full_experiment_and_interrupted_publication(sources, tmp_path, monkeypatch):
    monkeypatch.setattr(e03, "read_sources", lambda *args, **kwargs: sources)
    for name in ["pot_south.csv", "vtd_south_train.csv",
                 "tech_mode_for_embedings_with_length_with_temperatures_ids16.csv"]:
        (tmp_path / name).write_text("fixture", encoding="utf-8")
    report = e03.run_e03(tmp_path, tmp_path / "run", late_date="2023-12-01")
    predictions = pd.read_csv(tmp_path / "run" / "predictions.csv")
    assert not predictions.duplicated(["evaluation", "model", "id", "segment_id"]).any()
    assert predictions.groupby(["evaluation", "model"]).size().eq(100).all()
    physical = predictions.loc[predictions.model.isin(["constant_rate", "parametric_rate", "hybrid_scale"])]
    assert physical.prediction_mm.ge(0).all()
    assert physical.prediction_mm.le(physical.previous_thickness_mm).all()
    assert json.loads((tmp_path / "run" / "report.json").read_text())["experiment"] == "E03"
    assert report["audits"]["temporal_holdout"]["scored_segments"] == 100
    with pytest.raises(ValueError, match="already exists"):
        e03.run_e03(tmp_path, tmp_path / "run")
    from pipetwin.reporting import write_experiment
    def fail_dump(*args):
        raise OSError("Interrupted model write")
    monkeypatch.setattr(joblib, "dump", fail_dump)
    with pytest.raises(OSError, match="Interrupted"):
        write_experiment(tmp_path / "broken", report, {}, {"metrics.csv": pd.DataFrame()})
    assert not (tmp_path / "broken").exists()
    assert not list(tmp_path.glob(".experiment-*"))
