import numpy as np
import pandas as pd

from pipetwin.models.advanced import MultiplicativeColdPrior, PairwiseThickness, PooledMatchedRates


def test_rank_calibration_uses_only_inner_block_training_and_preserves_orientation(monkeypatch):
    n = 60
    train = pd.DataFrame({"main_id": "MAIN-2", "chain_position_m": np.arange(n) * 10,
                          "observed": True, "anchor_observed": True, "y_mm": 4. + np.arange(n) / 50.})
    estimator = PairwiseThickness([], buffer_m=100.)
    fit_positions = []

    def fake_fit(frame):
        positions = frame.chain_position_m.to_numpy()
        fit_positions.append(positions)
        return positions,

    def fake_score(positions, frame):
        assert "y_mm" not in frame
        distance = abs(positions[:, None] - frame.chain_position_m.to_numpy()[None, :])
        assert (distance >= 100.).all()
        return -frame.chain_position_m.to_numpy()

    monkeypatch.setattr(estimator, "_fit_ranker", fake_fit)
    monkeypatch.setattr(estimator, "_score", fake_score)
    estimator.fit(train)
    assert len(fit_positions) == 4  # Three inner fits and the outer training fit.
    assert len(fit_positions[-1]) == n and estimator.calibration_rows_ == n
    assert estimator.slope_ == 0  # A reversed ranking cannot be flipped to inflate r².


def test_matched_pool_does_not_count_dense_defects_as_independent_cells():
    day = pd.Timestamp("2022-07-06")
    train = pd.DataFrame({"id": "p", "main_id": "MAIN-2", "segment_id": [0, 1], "chain_position_m": [5., 15.],
                          "observed": True, "anchor_observed": True, "scenario": "between", "nominal_mm": 8.,
                          "anchor_mm": 6., "y_mm": 5.8, "horizon_years": 1., "target_date": day})
    pairs = pd.DataFrame({"id": "p", "earlier_date": pd.Timestamp("2018-09-23"), "later_date": day,
                          "earlier_distance_m": 5. + np.linspace(-2, 2, 30), "later_distance_m": 5.,
                          "earlier_row_id": np.arange(30), "observed_loss_rate_mm_year": .8,
                          "alignment_supported": True, "growth_eligible": True})
    dense = PooledMatchedRates().fit(train, pairs, 10.)
    single = PooledMatchedRates().fit(train, pairs.iloc[[0]], 10.)
    assert dense.matched_cells_ == 1
    np.testing.assert_allclose(dense.predict(train)["pooled_matched"], single.predict(train)["pooled_matched"])


def test_cold_prior_caps_multiplier_before_thickness_clipping(monkeypatch):
    model = MultiplicativeColdPrior.__new__(MultiplicativeColdPrior)
    frame = pd.DataFrame({"nominal_mm": [4.5, 4.5], "terrain": [-1., 1.]})
    monkeypatch.setattr(model, "_values", lambda query: query[["terrain"]].to_numpy())
    monkeypatch.setattr(model, "_loss", lambda query: np.full(len(query), 1.5))

    class Extrapolator:
        def predict(self, x):
            return 100 * x[:, 0]

    model.ridge_ = Extrapolator()
    np.testing.assert_allclose(model.predict(frame), [3.75, 1.5])
