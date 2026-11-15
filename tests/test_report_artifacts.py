"""The published report must describe the actual submission and holdout rows."""

from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from pipetwin.validation import regression_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_published_profile_and_embedded_download_match_submission():
    report = ROOT / "reports"
    result = pd.read_csv(ROOT / "result.csv", dtype={"id": str})
    profiles = pd.read_csv(report / "profile_data.csv", dtype={"id": str})
    aligned = result.merge(profiles, on=["id", "segment_id"], validate="one_to_one")
    assert len(aligned) == len(result) == len(profiles) == 2000
    np.testing.assert_allclose(aligned.thickness_pred_x, aligned.thickness_pred_y, rtol=0, atol=5.01e-9)
    assert profiles.loc[~profiles.anchor_observed, "anchor_mm"].isna().all()
    priority = pd.read_csv(report / "priority_segments.csv", dtype={"id": str})
    selected = profiles.loc[profiles["rank"].notna()]
    assert set(zip(priority.id, priority.segment_id)) == set(zip(selected.id, selected.segment_id))
    html = (report / "dashboard.html").read_text(encoding="utf-8")
    encoded = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', html, re.S).group(1)
    data = json.loads(encoded)
    downloaded = data["downloads"]["result.csv"].encode("utf-8")
    assert downloaded == (ROOT / "result.csv").read_bytes()
    assert sha256(downloaded).hexdigest() == data["summary"]["result_sha256"]
    assert data["summary"]["closed_validation"]["result_sha256"] == sha256(downloaded).hexdigest()
    assert set(data["routes"]) == set(profiles.id)
    for section, rows in profiles.groupby("id"):
        route = np.array(data["routes"][section], dtype=float)
        assert route.shape == (len(rows), 3)
        assert np.isfinite(route).all()
        assert np.linalg.norm(np.ptp(route[:, :2], axis=0)) > 100
        assert np.ptp(route[:, 2]) > 1
    for name in ("metrics.csv", "priority_segments.csv", "profile_data.csv", "validation_predictions.csv"):
        assert data["downloads"][name].encode("utf-8") == (report / name).read_bytes()


def test_published_metrics_can_be_recomputed_from_holdout_predictions():
    report = ROOT / "reports"
    rows = pd.read_csv(report / "validation_predictions.csv", dtype={"id": str}, parse_dates=["target_date"])
    metrics = pd.read_csv(report / "metrics.csv", parse_dates=["campaign"])
    assert rows.groupby(["case", "split"]).size().to_dict() == {
        ("case1", "spatial_2022"): 508, ("case1", "late_2023"): 53, ("case2", "holdout_main"): 1080,
    }
    assert not rows.duplicated(["case", "split", "id", "target_date", "segment_id"]).any()
    for record in metrics.loc[metrics.seed.eq(42) & ~metrics.split.eq("donor_expansion")].itertuples():
        part = rows.loc[rows.case.eq(record.case)]
        if record.case == "case1":
            if record.split.startswith("spatial_fold_"):
                part = part.loc[part.split.eq("spatial_2022") & part.fold.eq(int(record.split.rsplit("_", 1)[-1]))]
            else:
                part = part.loc[part.split.eq(record.split)]
        else:
            part = part.loc[part.main_id.eq(record.held_main) & part.target_date.eq(record.campaign)]
        column = "thickness_pred" if record.model in {"direct_blend", "xgboost_knn"} else record.model
        recomputed = regression_metrics(part.y_mm, part[column])
        for name in ("n", "pearson_r", "pearson_squared", "mae_mm", "rmse_mm", "bias_mm", "recall_thin_10pct"):
            assert recomputed[name] == pytest.approx(getattr(record, name), abs=2e-10), (record.split, record.model, name)
