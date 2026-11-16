import numpy as np
import pandas as pd
import pytest

from analysis.defense_intervals import residual_quantiles, thickness_bounds


def test_calibration_can_assign_equal_weight_to_unequal_mains():
    rows = pd.DataFrame({"main_id": ["A"] * 20 + ["B"], "target_date": ["2022-07-01"] * 21,
                         "y_mm": [3.] * 20 + [7.], "thickness_pred": [3.] * 21})
    np.testing.assert_array_equal(residual_quantiles(rows), [0., 0.])
    np.testing.assert_array_equal(residual_quantiles(rows, balance_mains=True), [0., 4.])
    with pytest.raises(ValueError, match="nonempty"):
        residual_quantiles(rows.iloc[:0])


def test_interval_respects_nominal_and_zero():
    rows = pd.DataFrame({"thickness_pred": [.1, 4.4], "nominal_mm": [4.5, 4.5]})
    lower, upper = thickness_bounds(rows, [-.5, .5])
    np.testing.assert_allclose(lower, [0., 3.9])
    np.testing.assert_allclose(upper, [.6, 4.5])
