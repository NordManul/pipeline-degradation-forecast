"""Research gate cannot turn inverse correlation or missing folds into a win."""

import pandas as pd
import pytest

from pipetwin.agent2_physics.case2_search import assess_candidates, nominal_identifiability, BASE_MODELS
from pipetwin.agent2_physics.shared_comparison import experiment_package


@pytest.fixture
def scores():
    rows = [{"case_id": "case2", "output": "bounded", "missing_policy": "observed-only", "n": 40,
             "variant": "references" if name != "agent2_hybrid_residual" else "terrain-symmetric",
             "model": name, "mae_mm": mae, "pearson_r": r, "pearson_squared": r * r}
            for name, mae, r in [("global_rate", 1.0, 0.1), ("gam", 1.1, 0.2),
                                 ("catboost", 1.2, 0.1), ("agent2_hybrid_residual", 0.8, 0.3)]]
    pooled = pd.DataFrame(rows)
    folds = pd.concat([pooled.assign(fold_id=i, n=10) for i in range(4)], ignore_index=True)
    return pooled, folds


def test_auxiliary_gain_never_certifies_target_transfer(scores):
    pooled, folds = scores
    report = assess_candidates(pooled, folds).iloc[0]
    assert report.passes_auxiliary_gate
    assert not report.target_transfer_validated
    assert report.mae_improvement_fraction == pytest.approx(0.2)


@pytest.mark.parametrize("failure", ["negative_r", "unstable_folds", "tiny_gain", "missing_fold", "count_mismatch"])
def test_research_gate_rejects_misleading_evidence(scores, failure):
    pooled, folds = scores
    candidate = pooled.model.eq("agent2_hybrid_residual")
    held = folds.model.eq("agent2_hybrid_residual")
    if failure == "negative_r":
        pooled.loc[candidate, ["pearson_r", "pearson_squared"]] = [-0.9, 0.81]
    elif failure == "unstable_folds":
        folds.loc[held & folds.fold_id.lt(2), "pearson_r"] = -0.1
    elif failure == "tiny_gain":
        pooled.loc[candidate, "mae_mm"] = 0.99
    elif failure == "missing_fold":
        folds = folds.loc[~(held & folds.fold_id.eq(0))]
    else:
        folds.loc[held & folds.fold_id.eq(0), "n"] = 9
    if failure in {"missing_fold", "count_mismatch"}:
        with pytest.raises(ValueError):
            assess_candidates(pooled, folds)
    else:
        assert not assess_candidates(pooled, folds).iloc[0].passes_auxiliary_gate


def test_feature_ablation_preserves_shared_data_and_rejects_new_features():
    frame, folds = object(), object()
    package = {"feature_columns": ["slope_50m", "lowpoint_depth_m", "nominal_mm"],
               "train_frame": frame, "folds": folds}
    config = {"models": ["agent2_hybrid_residual"], "physics_feature_columns": ["slope_50m", "lowpoint_depth_m"]}
    subset = experiment_package(package, config)
    assert subset["train_frame"] is frame and subset["folds"] is folds
    assert package["feature_columns"] == ["slope_50m", "lowpoint_depth_m", "nominal_mm"]
    for invalid in [["y_mm"], [], ["slope_50m", "slope_50m"]]:
        with pytest.raises(ValueError):
            experiment_package(package, {**config, "physics_feature_columns": invalid})
    with pytest.raises(ValueError):
        experiment_package(package, {**config, "models": ["catboost"]})


def test_identical_validation_does_not_imply_identical_target_forecasts():
    pairs = [("references", "nominal-relative", name) for name in BASE_MODELS]
    pairs += [("terrain-symmetric", "terrain-symmetric-relative", "agent2_hybrid_residual"),
              ("terrain-scale", "terrain-scale-relative", "agent2_hybrid_scale")]
    oof = pd.DataFrame([{"missing_policy": policy, "model": model, "variant": variant,
                         "sample_id": sample, "raw_prediction_mm": value}
                        for policy in ["observed-only", "train-median"]
                        for a, b, model in pairs for variant in [a, b]
                        for sample, value in [("a", 6.0), ("b", 6.5)]])
    forecast = oof.iloc[::-1].copy()
    forecast.loc[forecast.variant.str.endswith("relative"), "raw_prediction_mm"] -= 1.0
    report = nominal_identifiability(oof, forecast)
    assert report.oof_max_gap_mm.eq(0).all()
    assert report.forecast_max_gap_mm.eq(1).all()
    with pytest.raises(ValueError, match="sample keys"):
        nominal_identifiability(oof.iloc[1:], forecast)
    with pytest.raises(ValueError, match="sample keys"):
        nominal_identifiability(pd.concat([oof, oof.iloc[[0]]]), forecast)
