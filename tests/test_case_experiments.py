from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from pipetwin.case_data import model_views, prepare_case
from pipetwin.case_experiments import fit_case_model, predict_models, run_case, validate_package
from pipetwin.quality import load_case_sources
from pipetwin.sources import read_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def package():
    clean = load_case_sources(ROOT / "data/raw", "case2", ROOT / "contracts", ROOT / "configs/data_contract.yaml")
    result, _ = prepare_case(clean, read_config(ROOT / "configs/data_contract.yaml"), read_config(ROOT / "configs/forecast.yaml"))
    result["contract_sha256"] = {"fixture": "in-memory-cleaned-real-data"}
    return result


@pytest.fixture
def config():
    return {"seed": 42, "models": ["global_rate", "ridge"], "agent2_options": {"missing_policy": "observed-only"},
            "fixed_blends": {"blend": {"global_rate": .5, "ridge": .5}}, "include_matching": False,
            "selection": "positive_r_then_pearson_squared_then_mae"}


def test_package_rejects_future_labels_and_overlapping_or_missing_folds(package):
    validate_package(package)
    changed = deepcopy(package)
    changed["train_frame"].loc[changed["train_frame"].observed, "label_available_at"] = pd.Timestamp("2022-07-06")
    with pytest.raises(ValueError, match="Unavailable"):
        validate_package(changed)
    changed = deepcopy(package)
    changed["folds"][0]["train"].append(changed["folds"][0]["valid"][0])
    with pytest.raises(ValueError, match="overlap"):
        validate_package(changed)
    changed = deepcopy(package)
    changed["folds"].pop()
    with pytest.raises(ValueError, match="exactly one"):
        validate_package(changed)


def test_model_and_fallback_do_not_depend_on_held_labels(package, config):
    first_train, first_valid, first_fallback = model_views(package, 0)
    changed = deepcopy(package)
    changed["train_frame"].loc[changed["folds"][0]["valid"], "y_mm"] = .01
    train, valid, fallback = model_views(changed, 0)
    for name in ["ridge", "agent2_positive_rate"]:
        a, _ = fit_case_model(name, package, first_train, first_fallback, config)
        b, _ = fit_case_model(name, changed, train, fallback, config)
        np.testing.assert_allclose(a.predict(first_valid), b.predict(valid), rtol=0, atol=1e-12)


def test_query_labels_and_other_case_cannot_affect_prediction(package, config):
    train, query, fallback = model_views(package)
    model, _ = fit_case_model("ridge", package, train, fallback, config)
    raw = model.predict(query)
    contaminated = query.assign(y_mm=999, observed=True, label_available_at=pd.Timestamp("2030-01-01"), fallback_mm=999)
    np.testing.assert_array_equal(raw, model.predict(contaminated))
    with pytest.raises(ValueError, match="different case"):
        model.predict(query.assign(case_id="case1"))
    with pytest.raises(ValueError, match="cutoff"):
        model.predict(query.assign(target_date=pd.Timestamp("2022-07-06")))


def test_warm_calibration_excludes_nominal_placeholders_and_uses_shared_fallback(config):
    frame = pd.DataFrame({"case_id": "case1", "scenario": "between", "observed": True,
                          "target_date": pd.Timestamp("2021-01-01"), "label_available_at": pd.Timestamp("2021-01-01"),
                          "anchor_observed": [True, True, False], "anchor_mm": [7., 7., 8.],
                          "y_mm": [6.8, 6.8, 1.], "horizon_years": 1., "nominal_mm": 8.})
    package = {"case_id": "case1", "scenario": "between", "cutoff_exclusive": "2022-01-01"}
    model, info = fit_case_model("global_rate", package, frame, {"fraction": .8}, config)
    assert info["diagnostics"]["fit_rows"] == 2
    assert model.estimator.rate_ == pytest.approx(.2)
    np.testing.assert_allclose(model.predict(frame), [6.8, 6.8, 6.4])


def test_complete_run_shared_keys_metrics_fixed_blend_and_reload(package, config, tmp_path):
    output = tmp_path / "run"
    metrics, selection, selected = run_case(package, config, output)
    table = pd.read_csv(output / "oof_predictions.csv.gz")
    assert len(table) == 580 * 3
    assert not table.duplicated(["sample_id", "model"]).any()
    assert table.groupby("model")["observed"].sum().eq(372).all()
    assert metrics.n.eq(372).all() and set(metrics.output) == {"raw", "bounded"}
    assert selected.id.eq("1250009583").all() and len(selected) == 1475
    assert selected.thickness_pred.between(0, 4.5).all()
    bundle = joblib.load(output / "models.joblib")
    _, query, _ = model_views(package)
    raw, bounded = predict_models(bundle["models"], query, bundle["fixed_blends"])
    np.testing.assert_allclose(bounded["blend"], .5 * bounded["ridge"] + .5 * bounded["global_rate"])
    np.testing.assert_allclose(selected.thickness_pred, bounded[selection["model"]])
    assert bundle["cutoff_exclusive"] == package["cutoff_exclusive"]
    assert set(raw) == set(bounded) == {"ridge", "global_rate", "blend"}
