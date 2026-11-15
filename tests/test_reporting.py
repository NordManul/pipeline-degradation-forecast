import json
import re

import numpy as np
import pandas as pd
import pytest

from pipetwin.reporting import FIGURES, PROFILE_COLUMNS, observation_table, profile_table, write_report


def _frame(section, anchor):
    frame = pd.DataFrame({name: [1., 2.] for name in PROFILE_COLUMNS})
    return frame.assign(id=section, segment_id=[0, 1], target_date=pd.Timestamp("2023-01-01"),
                        anchor_observed=[anchor, False], anchor_mm=[6., 8.], nominal_mm=8.,
                        thickness_pred=[5., 6.], main_id="MAIN-X", y_mm=[5.2, 5.8], observed=True)


def test_report_masks_missing_anchors_and_joins_priority_by_key():
    warm, cold = _frame("warm", True), _frame("cold", False)
    priority = pd.DataFrame({"id": ["cold", "warm"], "segment_id": [1, 0],
                             "rank": [1, 1], "ranking_thickness_mm": [4.2, 5.],
                             "ranking_model": ["forest", "blend"]})
    rows = profile_table(warm, cold, priority)
    assert len(rows) == 4
    assert rows.loc[~rows.anchor_observed, "anchor_mm"].isna().all()
    cold_ranked = rows.loc[rows.id.eq("cold") & rows.segment_id.eq(1)].iloc[0]
    assert cold_ranked.thickness_pred == 6
    assert cold_ranked.ranking_thickness_mm == 4.2
    with pytest.raises(ValueError, match="unique"):
        profile_table(warm, warm, priority)


def test_report_preserves_campaign_keys_and_rejects_unobserved_pairs():
    warm, late, cold = _frame("warm", True), _frame("late", True), _frame("cold", False)
    second = cold.assign(target_date=pd.Timestamp("2024-01-01"))
    rows = observation_table(warm, late, pd.concat([cold, second], ignore_index=True))
    assert len(rows) == 8
    assert rows.loc[rows.case.eq("case2"), "anchor_mm"].isna().all()
    with pytest.raises(ValueError, match="Duplicate"):
        observation_table(warm, late, pd.concat([cold, cold], ignore_index=True))
    with pytest.raises(ValueError, match="observed"):
        observation_table(warm.assign(observed=False), late, cold)
    with pytest.raises(ValueError, match="finite"):
        observation_table(warm.assign(y_mm=np.nan), late, cold)


def test_portable_report_preserves_download_bytes_and_embeds_strict_json(tmp_path):
    warm, cold = _frame("warm", True), _frame("cold", False)
    priority = pd.DataFrame({"id": ["warm", "cold"], "segment_id": [0, 0], "rank": [1, 1],
                             "ranking_thickness_mm": [5., 4.], "ranking_model": ["direct_blend", "random_forest"]})
    metrics = pd.DataFrame({"case": ["case1", "case1"], "split": ["spatial_2022", "late_2023"],
                            "model": ["direct_blend", "direct_blend"], "pearson_squared": [.3, .2]})
    summary = {"seed": 42, "case1": {"cells": 2}, "case2": {"cells": 2},
               "case2_macro_validation": [{"model": "xgboost_knn", "seed": 42, "pearson_squared": .1}],
               "test": "</script><script>bad()</script>"}
    original = b"id,segment_id,thickness_pred\r\nwarm,0,5.00000000\r\n"
    (tmp_path / "result.csv").write_bytes(original)
    for name in ("metrics.csv", "priority_segments.csv", "summary.json", "cleaning.csv"):
        (tmp_path / name).write_text("test\n", encoding="utf-8")
    for name in FIGURES:
        (tmp_path / (name + ".png")).write_bytes(b"image-for-serialization-test")
    write_report(tmp_path, warm, cold, warm, _frame("late", True), cold, priority, metrics, summary)
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    embedded = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', html, re.S).group(1)
    data = json.loads(embedded, parse_constant=lambda value: pytest.fail("Non-JSON value: " + value))
    assert data["downloads"]["result.csv"].encode("utf-8") == original
    assert data["summary"]["test"] == summary["test"]
    assert "</script>" not in embedded
    assert sum(row["anchor_mm"] is None for row in data["profiles"]) == 3
    assert html.count('src="data:image/png;base64,') == len(FIGURES)
    assert "@@" not in html
