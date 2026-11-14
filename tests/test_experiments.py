import numpy as np
import pandas as pd
import pytest

from pipetwin import experiments, pipeline
from pipetwin.uncertainty import calibrate_interval, interval_metrics, predict_interval
from pipetwin.validation import blocked_folds


def test_nested_stack_calibration_excludes_outer_validation_and_buffer(monkeypatch):
    frame = pd.DataFrame({"id": "1", "main_id": "m", "segment_id": np.arange(240),
                          "chain_position_m": np.arange(240) * 10, "nominal_mm": 8.0,
                          "observed": True, "y_mm": 3 + np.arange(240) / 1000})
    config = {"seed": 42, "folds": 4, "spatial_buffer_m": 20, "history_radius_m": 10,
              "ensemble_candidates": ["a", "b"]}

    class FixedModel:
        def predict(self, held):
            return 4 + held.chain_position_m.to_numpy() / 10000

    monkeypatch.setattr(experiments, "fit_models", lambda frame, names, scenario, seed: {name: FixedModel() for name in names})
    seen = []

    def record_weights(matrix, y):
        seen.append(np.asarray(y))
        return np.array([0.5, 0.5])

    monkeypatch.setattr(experiments, "fit_stack_weights", record_weights)
    result = experiments.cross_validate(frame, ["a", "b"], "between", config)
    folds = blocked_folds(frame, 4, 20)
    assert len(seen) == len(folds) + 1
    for values, fold in zip(seen[:-1], folds):
        np.testing.assert_array_equal(values, frame.iloc[fold.train].y_mm.to_numpy())
        assert not set(values) & set(frame.iloc[fold.valid].y_mm)
    np.testing.assert_array_equal(seen[-1], frame.y_mm)
    assert all(np.isfinite(values).all() for values in result.predictions.values())
    assert set(result.folds) == {0, 1, 2, 3}


def test_physical_prediction_envelope_is_shared():
    class ExtremeModel:
        def predict(self, frame):
            return np.array([-1, 9, 3])

    frame = pd.DataFrame({"nominal_mm": [8, 6, 4]})
    np.testing.assert_array_equal(experiments.checked_prediction(ExtremeModel(), frame), [0, 6, 3])


def test_intervals_preserve_order_and_measure_transfer():
    calibration_y = np.linspace(4, 7, 100)
    quantiles = calibrate_interval(calibration_y, np.full(100, 5.5))
    lower, upper = predict_interval(np.full(100, 5.5), quantiles, np.full(100, 8))
    assert (lower <= upper).all()
    assert interval_metrics(calibration_y, lower, upper)["coverage_80"] == pytest.approx(0.8)
    assert interval_metrics(calibration_y - 10, lower, upper)["coverage_80"] == 0


def test_failed_run_is_never_published(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")

    def interrupted(config, output):
        (output / "partial.csv").write_text("partial", encoding="utf-8")
        raise OSError("simulated interrupted run")

    monkeypatch.setattr(pipeline, "_run_benchmark", interrupted)
    output = tmp_path / "run"
    with pytest.raises(OSError, match="interrupted"):
        pipeline.run_benchmark(config, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".pipetwin-*"))


def test_json_has_null_for_undefined_metrics(tmp_path):
    import json
    path = tmp_path / "metrics.json"
    pipeline.write_json(path, {"r": np.nan, "weights": np.array([0.25, 0.75])})
    assert json.loads(path.read_text()) == {"r": None, "weights": [0.25, 0.75]}
