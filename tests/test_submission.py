import numpy as np
import pandas as pd
import pytest

from pipetwin.submission import export_submission, rank_segments, validate_submission


def predictions():
    return pd.DataFrame({"id": [1250009490.0, 1250009490.0], "segment_id": [0, 1], "thickness_pred": [7.12, 6.345]})


def test_template_exact_keys_order_and_csv_roundtrip(tmp_path):
    data = predictions()
    template = data[["id", "segment_id"]].iloc[::-1]
    output = tmp_path / "result.csv"
    result = export_submission(data, template, output)
    assert result.columns.tolist() == ["id", "segment_id", "thickness_pred"]
    assert result.segment_id.tolist() == [1, 0]
    assert output.read_text().splitlines()[0] == "id,segment_id,thickness_pred"
    assert "1250009490,1,6.34500000" in output.read_text()
    loaded = pd.read_csv(output)
    np.testing.assert_allclose(loaded.thickness_pred, [6.345, 7.12])
    with pytest.raises(ValueError, match="exists"):
        export_submission(data, template, output)


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate", "nonfinite", "negative", "fractional_segment"])
def test_invalid_submission_fails_instead_of_silently_fixing(problem):
    data = predictions()
    template = data[["id", "segment_id"]].copy()
    if problem == "missing":
        data = data.iloc[:1]
    elif problem == "extra":
        data = pd.concat([data, data.iloc[:1].assign(segment_id=2)], ignore_index=True)
    elif problem == "duplicate":
        data = pd.concat([data, data.iloc[:1]], ignore_index=True)
    elif problem == "nonfinite":
        data.loc[0, "thickness_pred"] = np.nan
    elif problem == "negative":
        data.loc[0, "thickness_pred"] = -1
    else:
        data["segment_id"] = [0.5, 1.0]
    with pytest.raises(ValueError):
        validate_submission(data, template)
def test_rank_uses_a_rounded_up_decile_and_stable_ties():
    frame = pd.DataFrame({"id": ["101"] * 12, "segment_id": list(reversed(range(12)))})
    scores = [6.] * 9 + [2.] * 3
    ranked = rank_segments(frame, scores, "forest")
    assert ranked.segment_id.tolist() == [0, 1]
    assert ranked["rank"].tolist() == [1, 2]
    assert ranked.ranking_model.tolist() == ["forest"] * 2
