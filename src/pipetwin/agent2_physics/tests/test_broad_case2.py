"""Check the new response conversions and the cross-main selection rule."""

import numpy as np
import pandas as pd
import pytest

from pipetwin.agent2_physics.broad_case2 import MODES, decode, encode, ranking, training_rows


@pytest.mark.parametrize("mode", MODES)
def test_response_conversion_preserves_mm_and_signed_observed_loss(mode):
    frame = pd.DataFrame({"nominal_mm": [4.5, 8.0, 6.0], "y_mm": [2.7, 8.03, 5.2],
                          "horizon_years": [29.0, 18.0, 11.0]})
    np.testing.assert_allclose(decode(encode(frame, mode), frame, mode), frame.y_mm)


def test_balanced_training_weights_count_only_observed_campaign_rows():
    frame = pd.DataFrame({"main_id": ["A", "A", "A", "B"], "target_date": [1, 1, 1, 2],
                          "sample_id": ["a", "b", "c", "d"], "observed": [True, True, False, True]})
    measured, weights = training_rows(frame, "balanced")
    assert measured.sample_id.tolist() == ["a", "b", "d"]
    assert weights[:2].sum() == pytest.approx(weights[2])


def test_selection_equalizes_mains_and_rejects_negative_constant_and_warning():
    scores = []
    for model, values in {"reference_catboost": [0.1] * 4, "win": [0.2, 0.2, 0.2, 0.6],
                          "warning": [0.5] * 4, "negative": [0.8, -0.8, 0.8, 0.8],
                          "constant": [0.8, np.nan, 0.8, 0.8]}.items():
        for main, value in zip(["A", "A", "A", "B"], values):
            scores.append({"model": model, "held_main": main, "output": "bounded", "pearson_r": value,
                           "mae_mm": 1.0 if model == "reference_catboost" else 0.8,
                           "rmse_mm": 1.0, "recall_thin_10pct": 0.2})
    result = ranking(pd.DataFrame(scores), {"warning"}).set_index("model")
    assert result.loc["win", "mean_main_r"] == pytest.approx(0.4)
    assert result.index[result.passes_gate].tolist() == ["win"]
