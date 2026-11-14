import numpy as np
import pandas as pd

from pipetwin.submission_checks import QuantileGate


def test_quantile_gate_uses_train_median_and_indicates_extrapolation():
    train = pd.DataFrame({"varies": np.arange(101.), "constant": 8., "unknown": np.nan})
    query = pd.DataFrame({"varies": [1., 1000., np.nan], "constant": [4.5, 8., np.nan], "unknown": [2., 1., np.nan]})
    gated, flags = QuantileGate().fit(train).transform(query)
    np.testing.assert_array_equal(gated.varies, [1., 50., 50.])
    np.testing.assert_array_equal(gated.constant, [8., 8., 8.])
    assert gated.unknown.isna().all()
    np.testing.assert_array_equal(flags.varies, [False, True, False])
    np.testing.assert_array_equal(gated.is_out_of_range__constant, [1., 0., 0.])
    np.testing.assert_array_equal(flags.unknown, [True, True, False])
