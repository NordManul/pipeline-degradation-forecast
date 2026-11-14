"""Guard the covariance, response intervals and supervised rank calibration."""

import numpy as np
import pandas as pd

from pipetwin.agent2_physics.broad_case2 import Candidate
from pipetwin.agent2_physics.case2_followup_models import PairwiseRank, ProfileKernel, SmoothProfile


def test_profile_kernel_has_consistent_positive_covariance():
    x = np.array([[0.1, -1.0], [0.2, 0.0], [0.7, 0.5]])
    kernel = ProfileKernel(0.3)
    cov = kernel(x)
    np.testing.assert_allclose(kernel.diag(x), np.diag(cov))
    np.testing.assert_allclose(kernel(x[:2], x[2:]), cov[:2, 2:])
    assert np.linalg.eigvalsh(cov).min() > 0


def profile():
    return pd.DataFrame({"position_fraction": [0.1, 0.2, 0.5, 0.9],
                         "altitude_relative_m": [-1.0, 0.1, 0.5, 0.8],
                         "nominal_mm": 4.5, "horizon_years": 20,
                         "id": "a", "target_date": pd.Timestamp("2022-07-01")})


def test_gp_interval_inverts_loss_direction_when_converting_to_thickness():
    frame = profile()
    columns = ["position_fraction", "altitude_relative_m"]
    model = SmoothProfile(Candidate("gp", "gp"), columns, 0.1).fit(
        frame, np.array([0.01, 0.02, 0.01, 0.025]), np.ones(len(frame)))
    lower, upper = model.interval(frame)
    center = model.predict(frame)
    assert np.all(lower < center) and np.all(center < upper)


def test_rank_calibration_predicts_in_mm_with_no_query_labels_or_inspection_groups():
    frame = profile()
    model = PairwiseRank(Candidate("rank", "rank"), ["position_fraction"]).fit(
        frame, np.array([0.01, 0.02, 0.03, 0.04]), np.ones(len(frame)))
    query = frame.drop(columns=["id", "target_date"])
    results = model.predict(query)
    assert np.diff(results).max() <= 0
    assert results[0] > results[-1] > 0
