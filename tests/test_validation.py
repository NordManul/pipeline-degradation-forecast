"""Tests of information boundaries and validation behaviour."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pipetwin.features import attach_labels, build_snapshot, feature_columns
from pipetwin.geometry import build_grid
from pipetwin.sources import read_config
from pipetwin.targets import build_targets
from pipetwin.validation import blocked_folds, regression_metrics, score_predictions
from conftest import ROOT


@pytest.fixture(scope="module")
def snapshots(sources, contract):
    config = read_config(ROOT / "configs/forecast.yaml")
    grid = build_grid(sources.passport)
    targets, _ = build_targets(sources, grid, contract)
    return config, grid, targets


def test_target_and_future_regime_cannot_change_features(sources, snapshots):
    config, grid, targets = snapshots
    spec = config["development"]
    before = build_snapshot(sources, grid, targets, spec, config)
    changed_targets = targets.copy()
    changed_targets.loc[changed_targets.date.ge(spec["target_date"]), "thickness_mm"] = 1000
    changed_regime = sources.regime.copy()
    changed_regime.loc[changed_regime.date.ge(spec["target_date"]), ["q", "water_fraction", "t_mix"]] = 1e9
    after = build_snapshot(replace(sources, regime=changed_regime), grid, changed_targets, spec, config)
    pd.testing.assert_frame_equal(before, after)
    assert "y_mm" not in before
    assert not set(feature_columns("between")) & {"id", "chain_position_m", "y_mm", "nominal_observed_mm"}


def test_cold_start_has_no_inspection_inputs(sources, snapshots):
    config, grid, targets = snapshots
    date = "2023-08-12"
    ids = sorted(targets.loc[targets.date.eq(date), "id"].unique())
    spec = {"ids": ids, "target_date": date}
    before = build_snapshot(sources, grid, targets, spec, config)
    after = build_snapshot(sources, grid, targets.iloc[:0], spec, config)
    pd.testing.assert_frame_equal(before, after)
    assert not any("anchor" in value or "neighbor" in value for value in feature_columns("cold"))
    labelled = attach_labels(before, targets)
    assert labelled.observed.sum() == 94
    assert labelled.loc[~labelled.observed, "y_mm"].isna().all()


def test_spatial_folds_cover_once_and_exclude_buffer():
    frame = pd.DataFrame({"main_id": ["m"] * 101, "chain_position_m": np.arange(101) * 10.0})
    folds = blocked_folds(frame, n_splits=4, buffer_m=100)
    assert sorted(np.concatenate([fold.valid for fold in folds])) == list(range(101))
    for fold in folds:
        distances = np.abs(frame.iloc[fold.train].chain_position_m.to_numpy()[:, None] - frame.iloc[fold.valid].chain_position_m.to_numpy())
        assert distances.min() > 100
        assert not set(fold.train) & set(fold.valid)


def test_signed_correlation_constants_and_identical_metric_mask():
    assert regression_metrics([1, 2, 3], [3, 2, 1])["pearson_r"] == pytest.approx(-1)
    assert np.isnan(regression_metrics([1, 2, 3], [2, 2, 2])["pearson_squared"])
    assert regression_metrics(np.arange(20), np.ones(20))["recall_thin_10pct"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="non-finite"):
        regression_metrics([1, 2], [1, np.nan])
    frame = pd.DataFrame({"y_mm": [1, np.nan, 3], "observed": [True, False, True]})
    result = score_predictions(frame, {"a": [1, 4, 3], "b": [3, 4, 1]}, "development")
    assert result.n.tolist() == [2, 2]
    assert result.pearson_r.tolist() == pytest.approx([1, -1])
