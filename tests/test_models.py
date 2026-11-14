import numpy as np
import pandas as pd
import pytest

from pipetwin.features import attach_labels, build_snapshot, feature_columns
from pipetwin.geometry import build_grid
from pipetwin.models import MODEL_NAMES, make_model
from pipetwin.models.ensemble import fit_stack_weights
from pipetwin.sources import read_config
from pipetwin.targets import build_targets
from pipetwin.validation import blocked_folds
from conftest import ROOT


@pytest.fixture(scope="module", params=["between", "cold"])
def model_data(request, sources, contract):
    scenario = request.param
    config = read_config(ROOT / "configs/forecast.yaml")
    grid = build_grid(sources.passport)
    targets, _ = build_targets(sources, grid, contract)
    spec = config["development" if scenario == "between" else "cold_train"]
    frame = attach_labels(build_snapshot(sources, grid, targets, spec, config), targets)
    fold = blocked_folds(frame)[0]
    train = frame.iloc[fold.train].loc[lambda table: table.observed].copy()
    valid = frame.iloc[fold.valid].copy()
    return scenario, train, valid


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_real_fold_finite_and_no_target_column_access(name, model_data):
    scenario, train, valid = model_data
    model = make_model(name, scenario)
    model.fit(train, train.y_mm)
    prediction = model.predict(valid)
    assert prediction.shape == (len(valid),)
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    changed_labels = valid.assign(y_mm=10000, observed=False, recorded_features=10000)
    np.testing.assert_allclose(model.predict(changed_labels), prediction)


def test_signed_observation_change_and_nonnegative_physical_loss():
    frame = pd.DataFrame({"anchor_mm": [8.0] * 20, "anchor_observed": [1] * 20,
                          "horizon_years": [2.0] * 20, "active_years_observed": [1.0] * 20,
                          "water_exposure_years": [0.3] * 20, "lowpoint_depth_m": [0.0] * 20})
    observations = np.full(20, 8.4)
    signed = make_model("global_rate", "between").fit(frame, observations)
    physical = make_model("physics", "between").fit(frame, observations)
    np.testing.assert_allclose(signed.predict(frame), 8.4)
    np.testing.assert_allclose(physical.predict(frame), 8.0)


def test_imputer_learns_training_statistics_only(model_data):
    scenario, train, valid = model_data
    model = make_model("ridge", scenario).fit(train, train.y_mm)
    actual = model.estimator.named_steps["simpleimputer"].statistics_.copy()
    expected = train[feature_columns(scenario)].median().fillna(0).to_numpy()
    np.testing.assert_allclose(actual, expected)
    changed = valid.copy()
    changed[feature_columns(scenario)] = 1e9
    model.predict(changed)
    np.testing.assert_array_equal(actual, model.estimator.named_steps["simpleimputer"].statistics_)


def test_convex_stack_and_constant_models():
    y = np.linspace(3, 8, 100)
    matrix = np.column_stack([y, 11 - y, np.full(100, 6.0)])
    weights = fit_stack_weights(matrix, y)
    assert weights.sum() == pytest.approx(1)
    assert (weights >= 0).all() and weights[0] > 0.9
    assert ((matrix @ weights) >= matrix.min(axis=1) - 1e-10).all()
    assert ((matrix @ weights) <= matrix.max(axis=1) + 1e-10).all()
    np.testing.assert_allclose(weights, fit_stack_weights(matrix[::-1], y[::-1]), atol=1e-6)
    with pytest.raises(ValueError, match="10 held-out"):
        fit_stack_weights(matrix[:3], y[:3])
