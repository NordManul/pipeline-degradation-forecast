import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pipetwin.features import build_features, daily_mode, feature_blocks, mode_features
from pipetwin.geometry import segment_grid, terrain_features
from pipetwin.models.tabular import ridge_model
from pipetwin.pipeline import evaluate_ablation, predict_models
from pipetwin.sources import canonical_ids
from pipetwin.targets import observed_targets
from pipetwin.validation import regression_metrics, spatial_folds


def tech_rows(dates, **overrides):
    values = dict(id="1", q=100.0, watercut=0.3, fill_coefficient=0.8,
                  v_liquid_true=1.0, t_start=20.0, t_end=10.0)
    values.update(overrides)
    return pd.DataFrame({"calc_date": pd.to_datetime(dates), **values})


def sections(length=25.0):
    return pd.DataFrame({"id": ["1"], "main_id": ["MAIN-2"], "length_m": [length]})


def inspections(distances, thickness, date="2018-09-23"):
    return pd.DataFrame({
        "id": "1", "date": pd.Timestamp(date), "Distance": distances,
        "Residual Wall Thickness": thickness, "Feature Description": "потеря металла",
        "Defect Type": "внутр.", "trait": np.nan,
    })


def test_water_percent_and_daily_exposure():
    tech = tech_rows(["2020-01-01 08:00", "2020-01-01 15:00", "2020-01-03 08:00"], q=[200, 100, 0])
    result = mode_features(tech, pd.Series(["1"]), "2020-01-01", "2020-01-05").iloc[0]
    assert result.water_fraction_mean == pytest.approx(0.003)
    assert result.water_fraction_days == pytest.approx(0.006)
    assert result.water_volume_m3 == pytest.approx(0.3)
    assert result.mode_observed_fraction == 0.5
    assert result.mode_active_fraction == 0.25
    assert result.mode_stopped_fraction == 0.25
    assert result.thermal_temperature_mean_c == 15


def test_future_rows_cannot_change_daily_features():
    tech = tech_rows(["2020-01-01", "2020-01-03", "2021-01-01"], q=[100.0, 500.0, 999.0])
    before = daily_mode(tech, "2020-01-01", "2020-01-03")
    tech.loc[1:, "q"] = -1e30
    after = daily_mode(tech, "2020-01-01", "2020-01-03")
    assert_frame_equal(before, after)
    assert len(before) == 1


def test_unknown_days_are_distinct_from_zero_flow():
    unknown = mode_features(tech_rows(["2020-01-05"]), pd.Series(["1"]), "2020-01-01", "2020-01-03").iloc[0]
    stopped = mode_features(tech_rows(["2020-01-01"], q=0), pd.Series(["1"]), "2020-01-01", "2020-01-03").iloc[0]
    assert np.isnan(unknown.mode_flow_mean_m3_day)
    assert np.isnan(unknown.water_fraction_days)
    assert stopped.mode_flow_mean_m3_day == 0
    assert stopped.mode_stopped_fraction == 0.5


def test_invalid_physical_inputs_become_missing():
    rows = daily_mode(tech_rows(["2020-01-01"], q=-5, watercut=101, fill_coefficient=2,
                               v_liquid_true=-1, t_start=0.0002), "2020-01-01", "2020-01-02")
    assert rows[["q", "water_fraction", "fill_coefficient", "velocity_m_s", "temperature_c"]].isna().all().all()


def test_last_snapshot_keeps_its_missing_fields():
    rows = tech_rows(["2020-01-01 08:00", "2020-01-01 10:00"], watercut=[20, np.nan])
    assert daily_mode(rows, "2020-01-01", "2020-01-02").water_fraction.isna().all()


def test_conflicting_duplicate_timestamp_fails():
    rows = tech_rows(["2020-01-01", "2020-01-01"], q=[100, 200])
    with pytest.raises(ValueError, match="Conflicting"):
        daily_mode(rows, "2020-01-01", "2020-01-02")


@pytest.mark.parametrize("start,end", [("2020-01-03", "2020-01-01"), ("2020-01-01 12:00", "2020-01-03")])
def test_invalid_calendar_window(start, end):
    with pytest.raises(ValueError, match="midnight"):
        daily_mode(tech_rows(["2020-01-01"]), start, end)


def test_grid_and_target_boundaries():
    grid = segment_grid(sections())
    assert grid.end_m.tolist() == [10, 20, 25]
    vtd = inspections([-1, 0, 9.9, 10, 25, 25.01], [1, 7, 6, 5, 4, 1])
    targets, audit = observed_targets(vtd, sections(), "2018-09-23")
    assert targets.thickness_mm.tolist() == [6, 5, 4]
    assert audit["invalid_geometry"] == 2
    exact, _ = observed_targets(inspections([20], [6]), sections(20), "2018-09-23")
    assert exact.segment_id.item() == 1


def test_target_policy_handles_2018_and_2022_types():
    vtd = inspections([0, 1, 10, 11], [7, 2, 6, 1])
    vtd["Feature Description"] = ["потеря металла", "аномалия поперечного шва", "Коррозионный кластер", "Коррозия"]
    vtd["Defect Type"] = ["внутр.", "внутр.", "Внутренний", "Внешний"]
    targets, _ = observed_targets(vtd, sections(), "2018-09-23")
    assert targets.thickness_mm.tolist() == [7, 6]
    all_targets, _ = observed_targets(vtd, sections(), "2018-09-23", "all-residual")
    assert all_targets.thickness_mm.tolist() == [2, 1]


def test_terrain_slope_units_direction_and_coverage():
    grid = segment_grid(sections())
    profile = pd.DataFrame({"id": "1", "total_distance": [0, 10, 25], "z": [0, 1, 2.5]})
    forward = terrain_features(grid, profile, "increasing")
    backward = terrain_features(grid, profile, "decreasing")
    np.testing.assert_allclose(forward.slope, 0.1)
    np.testing.assert_allclose(backward.slope, -0.1)
    assert forward.basin_depth_m.eq(0).all()
    profile.total_distance += 5
    assert np.isnan(terrain_features(grid, profile, "increasing").slope.iloc[0])


def test_target_campaign_and_forbidden_fields_cannot_change_features():
    old = inspections([0, 10], [7, 6])
    new = inspections([0, 10], [5, 4], "2022-07-06")
    sources = {"sections": sections(), "vtd": pd.concat([old, new], ignore_index=True),
               "tech": tech_rows(["2019-01-01", "2024-01-01"])}
    expected, _ = build_features(sources, "2018-09-23", "2022-07-06")
    sources["vtd"].loc[2:, "Residual Wall Thickness"] = 9999
    sources["tech"]["pit_corrosion_intensity"] = 9999
    sources["tech"].loc[1, "q"] = 9999
    actual, _ = build_features(sources, "2018-09-23", "2022-07-06")
    assert_frame_equal(expected, actual)
    for columns in feature_blocks(actual).values():
        assert not {"id", "segment_id", "center_m", "thickness_mm", "pit_corrosion_intensity"} & set(columns)


def test_folds_are_disjoint_and_buffered():
    frame = segment_grid(sections(1000))
    assigned = []
    for _, train, test in spatial_folds(frame, 5, 50):
        assigned.extend(test)
        distances = abs(frame.center_m.to_numpy()[train, None] - frame.center_m.to_numpy()[test])
        assert distances.min() > 50
        assert not set(train) & set(test)
    assert sorted(assigned) == list(range(len(frame)))


def test_imputation_is_fitted_on_training_rows_only():
    model = ridge_model()
    model.fit(pd.DataFrame({"x": [1, np.nan, 3], "empty": [np.nan] * 3}), [6, 5, 4])
    np.testing.assert_allclose(model.named_steps["impute"].statistics_, [2, 0])
    prediction = model.predict(pd.DataFrame({"x": [9999], "empty": [2]}))
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(model.named_steps["impute"].statistics_, [2, 0])


def test_metrics_report_sign_and_undefined_correlation():
    assert regression_metrics([1, 2, 3], [3, 2, 1])["pearson_r"] == pytest.approx(-1)
    assert regression_metrics([1, 2, 3], [2, 2, 2])["pearson_squared"] is None
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([1, np.nan], [1, 2])


def test_ablation_and_inference_use_all_keys_once():
    frame = segment_grid(sections(1000))
    frame["previous_thickness_mm"] = 6 + np.sin(frame.center_m / 100)
    frame["thickness_mm"] = frame.previous_thickness_mm - 0.2
    blocks = {"mode": ["previous_thickness_mm"]}
    predictions = evaluate_ablation(frame, blocks, 5, 50, 10)
    assert len(predictions) == 200
    assert not predictions.duplicated(["model", "id", "segment_id"]).any()
    model = ridge_model().fit(frame[blocks["mode"]], frame.thickness_mm)
    bundle = {"models": {"mode": model}, "features": blocks}
    assert len(predict_models(bundle, frame)) == 100
    frame.loc[0, "previous_thickness_mm"] = np.nan
    with pytest.raises(ValueError, match="prior thickness"):
        predict_models(bundle, frame)


@pytest.mark.parametrize("value", [None, "1.2", "1e9", ""])
def test_invalid_identifiers(value):
    with pytest.raises(ValueError, match="identifiers"):
        canonical_ids(pd.Series([value]))


def test_legacy_float_identifier():
    assert canonical_ids(pd.Series([1250009490.0])).item() == "1250009490"


def test_serialized_models_reproduce_inference(tmp_path):
    import joblib

    frame = segment_grid(sections(100))
    frame["previous_thickness_mm"] = np.linspace(4, 8, len(frame))
    columns = ["previous_thickness_mm"]
    model = ridge_model().fit(frame[columns], frame.previous_thickness_mm - 0.2)
    bundle = {"models": {"mode": model}, "features": {"mode": columns}}
    expected = predict_models(bundle, frame)
    path = tmp_path / "models.joblib"
    joblib.dump(bundle, path)
    assert_frame_equal(expected, predict_models(joblib.load(path), frame))


@pytest.mark.parametrize("fail_write", [False, True])
def test_complete_run_publication_and_failed_write_cleanup(tmp_path, monkeypatch, fail_write):
    from pipetwin import pipeline

    distance = np.arange(0, 1000, 10)
    thickness = 6 + np.sin(distance / 100)
    sources = {
        "sections": sections(1000),
        "vtd": pd.concat([inspections(distance, thickness),
                          inspections(distance, thickness - 0.2, "2022-07-06")]),
        "tech": tech_rows(["2019-01-01"]),
    }
    monkeypatch.setattr(pipeline, "read_sources", lambda *args, **kwargs: sources)
    for name in ["pot_south.csv", "vtd_south_train.csv",
                 "tech_mode_for_embedings_with_length_with_temperatures_ids16.csv"]:
        (tmp_path / name).write_text("fixture", encoding="utf-8")
    output = tmp_path / "result"
    if fail_write:
        def broken_dump(*args):
            raise OSError("Simulated interrupted model write")
        monkeypatch.setattr(pipeline.joblib, "dump", broken_dump)
        with pytest.raises(OSError, match="interrupted"):
            pipeline.run_experiment(tmp_path, output)
        assert not output.exists()
        assert not list(tmp_path.glob(".e02-*"))
    else:
        report = pipeline.run_experiment(tmp_path, output)
        assert report["scored_segments"] == 100
        assert {p.name for p in output.iterdir()} == {
            "metrics.csv", "oof_predictions.csv", "report.json", "models.joblib"
        }
        with pytest.raises(ValueError, match="already exists"):
            pipeline.run_experiment(tmp_path, output)
