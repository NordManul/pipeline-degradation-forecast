from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipetwin.case_data import build_exposure, canonical_snapshot, model_views, prepare_case
from pipetwin.geometry import build_grid
from pipetwin.models.tabular import TabularModel
from pipetwin.physical_features import daily_mode
from pipetwin.quality import clean_sources, read_raw_sources
from pipetwin.sources import read_config
from pipetwin.targets import build_targets


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared():
    raw, manifest = read_raw_sources(ROOT / "data/raw")
    contract = read_config(ROOT / "contracts/cases.yaml")
    target = read_config(ROOT / "configs/data_contract.yaml")
    config = read_config(ROOT / "configs/forecast.yaml")
    clean = clean_sources(raw, "case2", contract, target, manifest)
    package, _ = prepare_case(clean, target, config)
    return raw, clean, package, target, config


def test_real_case2_has_only_2018_labels_and_all_requested_cells(prepared):
    _, clean, package, _, _ = prepared
    train, predict = package["train_frame"], package["predict_frame"]
    assert train.observed.sum() == 372
    assert train.loc[train.observed, "target_date"].eq(pd.Timestamp("2018-09-23")).all()
    assert train.loc[train.observed, "label_available_at"].lt(clean.cutoff).all()
    assert len(predict) == 1475 and predict.id.eq("1250009583").all()
    assert not predict.anchor_observed.any()
    assert "y_mm" not in predict and "label_available_at" not in predict
    assert package["exposure"].day.lt(clean.cutoff).all()
    assert not package["exposure"].duplicated(["exposure_id", "day"]).any()
    assert not set(package["feature_columns"]) & {"anchor_mm", "anchor_observed", "y_mm", "label_available_at"}


def test_fallback_does_not_use_validation_labels(prepared):
    package = prepared[2]
    first_train, first_valid, first_info = model_views(package, 0)
    altered = dict(package)
    altered["train_frame"] = package["train_frame"].copy()
    altered["train_frame"].loc[package["folds"][0]["valid"], "y_mm"] = 0.01
    train, valid, info = model_views(altered, 0)
    assert info == first_info
    np.testing.assert_array_equal(train.fallback_mm, first_train.fallback_mm)
    np.testing.assert_array_equal(valid.fallback_mm, first_valid.fallback_mm)


def test_fold_buffer_crosses_simple_section_boundaries(prepared):
    package = prepared[2]
    frame = package["train_frame"]
    assignments = np.full(len(frame), -1)
    for fold in package["folds"]:
        train = frame.iloc[fold["train"]].chain_position_m.to_numpy()
        valid = frame.iloc[fold["valid"]].chain_position_m.to_numpy()
        assert np.min(np.abs(train[:, None] - valid[None, :])) >= package["config"]["spatial_buffer_m"]
        assignments[fold["valid"]] = fold["number"]
    assert (assignments >= 0).all()


def test_late_arriving_old_ili_row_cannot_change_historical_anchor(prepared):
    _, clean, _, target, config = prepared
    grid = build_grid(clean.sources.passport)
    targets, _ = build_targets(clean.sources, grid, target)
    spec = {"ids": ["1250022871"], "anchor_date": "2018-09-23", "target_date": "2019-01-01"}
    first = canonical_snapshot(clean, grid, targets, spec, config, target, labelled=False)
    row = clean.sources.inspection.loc[clean.sources.inspection.id.eq("1250022871") & clean.sources.inspection.thickness_mm.notna()].iloc[[0]].copy()
    row["row_id"], row["thickness_mm"], row["available_at"] = 999999, 0.1, pd.Timestamp("2020-01-01")
    altered_sources = replace(clean.sources, inspection=pd.concat([clean.sources.inspection, row], ignore_index=True))
    altered = replace(clean, sources=altered_sources)
    altered_targets, _ = build_targets(altered.sources, grid, target)
    second = canonical_snapshot(altered, grid, altered_targets, spec, config, target, labelled=False)
    pd.testing.assert_frame_equal(first, second)


def test_late_regime_release_is_absent_from_historical_features_and_exposure(prepared):
    _, clean, package, target, config = prepared
    regime = clean.sources.regime
    row = regime.loc[regime.id.eq("1250022871")].iloc[[0]].copy()
    row["date"] = row["calc_date"] = pd.Timestamp("2018-09-22 12:00:00")
    row["available_at"] = pd.Timestamp("2019-01-01")
    row["q"] = 999999.0
    modified = replace(clean, sources=replace(clean.sources, regime=pd.concat([regime, row], ignore_index=True)))
    changed, _ = prepare_case(modified, target, config)
    pd.testing.assert_frame_equal(package["train_frame"], changed["train_frame"])
    train_history_ids = set(package["train_frame"].exposure_id)
    first = package["exposure"].loc[package["exposure"].exposure_id.isin(train_history_ids)].reset_index(drop=True)
    second = changed["exposure"].loc[changed["exposure"].exposure_id.isin(train_history_ids)].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


def test_histories_for_same_section_different_windows_do_not_mix(prepared):
    clean, package = prepared[1:3]
    one = package["train_frame"].iloc[[0]].copy()
    two = one.copy()
    two["window_end"] = pd.Timestamp("2020-01-01")
    two["horizon_years"] = (two.window_end - two.window_start).dt.days / 365.25
    two["exposure_id"] = "second-window"
    exposure = build_exposure(clean.sources.regime, [one, two], clean.cutoff)
    durations = exposure.groupby("exposure_id").dt_years.sum()
    assert durations.loc[one.exposure_id.iloc[0]] == pytest.approx(one.horizon_years.iloc[0])
    assert durations.loc["second-window"] == pytest.approx(two.horizon_years.iloc[0])
    invalid = two.assign(window_end=pd.Timestamp("2023-01-01"))
    with pytest.raises(ValueError, match="cutoff"):
        build_exposure(clean.sources.regime, [invalid], clean.cutoff)


def test_negative_celsius_is_preserved_in_declared_policy(prepared):
    regime = prepared[1].sources.regime.iloc[[0]].copy()
    regime["t_start"], regime["t_end"] = -4.0, -2.0
    day = regime.calc_date.iloc[0].normalize()
    measured = daily_mode(regime, day, day + pd.Timedelta(days=1), temperature_policy="observed_endpoints")
    assert measured.temperature_c.iloc[0] == -3.0
    legacy = daily_mode(regime, day, day + pd.Timedelta(days=1))
    assert pd.isna(legacy.temperature_c.iloc[0])


def test_real_future_poison_does_not_change_case2_features_or_fitted_prediction(prepared):
    raw, clean, package, target, config = prepared
    poisoned = deepcopy(raw)
    for source, date_column, column in [("inspection", "date", "Residual Wall Thickness"),
                                       ("regime", "calc_date", "q")]:
        mask = pd.to_datetime(poisoned[source][date_column]).ge(clean.cutoff)
        poisoned[source][column] = poisoned[source][column].astype(object)
        poisoned[source].loc[mask, column] = "INVALID-FUTURE"
    changed_clean = clean_sources(poisoned, "case2", clean.contract, target)
    changed, _ = prepare_case(changed_clean, target, config)
    pd.testing.assert_frame_equal(package["train_frame"], changed["train_frame"])
    pd.testing.assert_frame_equal(package["predict_frame"], changed["predict_frame"])
    pd.testing.assert_frame_equal(package["exposure"], changed["exposure"])
    predictions = []
    for data in [package, changed]:
        train, future, _ = model_views(data)
        observed = train.loc[train.observed]
        model = TabularModel("cold", "ridge_direct").fit(observed, observed.y_mm)
        predictions.append(model.predict(future))
    np.testing.assert_array_equal(*predictions)
