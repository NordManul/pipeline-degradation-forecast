import numpy as np
import pandas as pd
import pytest

from pipetwin.agent2_physics.broad_case2 import AlignedProfileBlend, training_rows
from pipetwin.agent2_physics.final_selection import combine_submission, robust_training


def test_caps_use_each_training_campaign_and_weights_keep_its_total():
    frame = pd.DataFrame({
        "id": ["a"] * 6, "segment_id": [0, 1, 2, 0, 3, 4], "main_id": ["m"] * 6,
        "target_date": pd.to_datetime(["2000-01-01"] * 3 + ["2001-01-01"] * 3),
        "sample_id": [str(i) for i in range(6)], "observed": True,
        "y_mm": [1., 3., 4., 3., 4., 4.5], "nominal_mm": 5., "horizon_years": 10.,
    })
    rows, weights = training_rows(frame, "balanced")
    capped, cap_weights = robust_training(rows, weights, "cap_5")
    rate = -np.log(frame.y_mm / frame.nominal_mm) / 10
    np.testing.assert_array_equal(cap_weights, weights)
    np.testing.assert_allclose(capped.iloc[:3].max(), rate.iloc[:3].quantile(.95))
    np.testing.assert_allclose(capped.iloc[3:].max(), rate.iloc[3:].quantile(.95))
    for trial in ("mad_weight", "thin_weight", "repeat_weight"):
        _, weighted = robust_training(rows, weights, trial)
        np.testing.assert_allclose(weighted.reshape(2, 3).sum(axis=1), [3., 3.])
    _, repeat = robust_training(rows, weights, "repeat_weight")
    assert repeat[0] < repeat[1] and repeat[3] < repeat[4]


def test_combined_export_replaces_case2_in_release_order(tmp_path):
    release = pd.DataFrame({"id": ["101", "202", "202"], "segment_id": [0, 1, 0],
                            "thickness_pred": [6.12345678, 2., 2.5]})
    query = pd.DataFrame({"id": ["202", "202"], "segment_id": [0, 1], "nominal_mm": 4.5})
    path = tmp_path / "result.csv"
    combine_submission(release, query, np.array([3.2, 3.4]), path)
    saved = pd.read_csv(path, dtype={"id": str})
    pd.testing.assert_frame_equal(saved, release.assign(thickness_pred=[6.12345678, 3.4, 3.2]))
    with pytest.raises(ValueError, match="Template key mismatch"):
        combine_submission(release, query.iloc[:1], np.array([3.2]), tmp_path / "incomplete.csv")
    with pytest.raises(ValueError, match="passport range"):
        combine_submission(release, query, np.array([5., 3.4]), tmp_path / "out-of-bounds.csv")


def test_shape_alignment_keeps_level_and_separates_sections():
    class Profile:
        def __init__(self, column):
            self.column = column

        def predict(self, frame, *, development=False):
            return frame[self.column].to_numpy()

    model = AlignedProfileBlend(Profile("base"), Profile("knn"), ["exposure_id"], .5)
    data = pd.DataFrame({"exposure_id": ["a", "b", "a", "b"],
                         "base": [2., 5., 4., 9.], "knn": [200., 40., 100., 40.]})
    np.testing.assert_allclose(model.predict(data), [3., 6., 3., 8.])
    changed = data.assign(knn=[200., 20., 100., 90.])
    np.testing.assert_allclose(model.predict(changed)[[0, 2]], [3., 3.])
