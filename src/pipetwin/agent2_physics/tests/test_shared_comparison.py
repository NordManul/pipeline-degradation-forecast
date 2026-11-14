"""Check that the reporting wrapper cannot silently change shared keys/masks."""

import numpy as np
import pandas as pd
import pytest

from pipetwin.agent2_physics.shared_comparison import verify_shared_predictions


@pytest.fixture
def shared_table():
    frame = pd.DataFrame({"case_id": "case1", "sample_id": ["a", "b", "c", "d"],
                          "id": "123", "segment_id": range(4), "target_date": pd.Timestamp("2020-01-01"),
                          "observed": [True, False, True, True], "y_mm": [7.0, np.nan, 6.0, 5.0]})
    package = {"train_frame": frame, "predict_frame": frame.drop(columns=["observed", "y_mm"]),
               "folds": [{"number": i, "valid": [i]} for i in range(4)]}
    table = frame.assign(fold_id=range(4), model="model", raw_prediction_mm=6.0, prediction_mm=6.0)
    return package, table


def test_saved_shared_outputs_may_reorder_but_not_remove_keys(shared_table):
    package, table = shared_table
    verify_shared_predictions(package, table.iloc[::-1])
    verify_shared_predictions(package, table.assign(fold_id=-1).drop(columns=["observed", "y_mm"]), forecast=True)


@pytest.mark.parametrize("change", ["mask", "fold", "key", "id", "date", "missing", "duplicate", "nonfinite"])
def test_inconsistent_saved_outputs_are_rejected(shared_table, change):
    package, table = shared_table
    if change == "mask":
        table.loc[1, "observed"] = True
    elif change == "fold":
        table.loc[1, "fold_id"] = 0
    elif change == "key":
        table.loc[1, "sample_id"] = "other"
    elif change == "id":
        table.loc[1, "id"] = "other"
    elif change == "date":
        table.loc[1, "target_date"] = pd.Timestamp("2021-01-01")
    elif change == "missing":
        table = table.iloc[:-1]
    elif change == "duplicate":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    else:
        table.loc[1, "prediction_mm"] = np.nan
    with pytest.raises(ValueError):
        verify_shared_predictions(package, table)
