"""Portable report with embedded data, images and auditable plotting tables."""

import base64
from html import escape
from importlib.resources import files
import json

import numpy as np
import pandas as pd


FIGURES = ("profiles", "validation", "predicted_observed", "dependencies", "residuals", "stability")
PROFILE_COLUMNS = [
    "id", "segment_id", "target_date", "start_m", "end_m", "center_m", "nominal_mm",
    "thickness_pred", "position_fraction", "altitude_relative_m", "anchor_observed",
    "anchor_mm", "regime_coverage", "water_fraction_mean", "temperature_mean_c",
]
OBSERVATION_COLUMNS = [
    "id", "segment_id", "main_id", "target_date", "center_m", "nominal_mm",
    "y_mm", "thickness_pred", "anchor_observed", "anchor_mm", "altitude_relative_m",
    "position_fraction", "fold", "persistence", "annual_loss_blend",
    "catboost_weighted", "xgboost", "random_forest",
]


def profile_table(warm, cold, priority):
    rows = pd.concat([warm[PROFILE_COLUMNS].assign(case="case1"),
                      cold[PROFILE_COLUMNS].assign(case="case2")], ignore_index=True)
    if rows.duplicated(["id", "segment_id"]).any():
        raise ValueError("Report profiles require unique submission cells")
    rows.loc[~rows.anchor_observed, "anchor_mm"] = np.nan
    return rows.merge(priority, on=["id", "segment_id"], how="left", validate="one_to_one")


def observation_table(warm, late, cold):
    records = []
    for frame, case, split in [(warm, "case1", "spatial_2022"), (late, "case1", "late_2023"),
                               (cold, "case2", "holdout_main")]:
        if not frame.observed.all() or not np.isfinite(frame[["y_mm", "thickness_pred"]]).all().all():
            raise ValueError("Validation report accepts observed, finite holdout pairs only")
        part = frame[[name for name in OBSERVATION_COLUMNS if name in frame]].copy()
        part.loc[~part.anchor_observed, "anchor_mm"] = np.nan
        records.extend(part.assign(case=case, split=split).to_dict("records"))
    result = pd.DataFrame(records)
    if result.duplicated(["case", "split", "id", "target_date", "segment_id"]).any():
        raise ValueError("Duplicate holdout predictions")
    return result


def _records(frame):
    # pandas maps NaN/NaT to strict JSON null, including optional anchors.
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=12))


def _overview(output, metrics, summary):
    warm = metrics.loc[metrics.case.eq("case1") & metrics.model.eq("direct_blend")].set_index("split")
    cold = next(row for row in summary["case2_macro_validation"]
                if row["model"] == "xgboost_knn" and row["seed"] == summary["seed"])
    values = [f'{warm.loc["spatial_2022", "pearson_squared"]:.3f}',
              f'{warm.loc["late_2023", "pearson_squared"]:.3f}', f'{cold["pearson_squared"]:.3f}',
              f'{summary["case1"]["cells"] + summary["case2"]["cells"]:,}'.replace(",", " ")]
    labels = ["Кейс 1 · история", "Кейс 1 · позднее ВТД", "Кейс 2 · перенос", "Прогнозов в result.csv"]
    cards = []
    for i, (value, label) in enumerate(zip(values, labels)):
        x = 48 + i*286
        cards.append(f'<text x="{x}" y="195" fill="#85e0d0" font-size="42" font-weight="700">{escape(value)}</text>'
                     f'<text x="{x}" y="225" fill="#bccbd8" font-size="16">{escape(label)}</text>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="300" viewBox="0 0 1200 300">'
           '<rect width="1200" height="300" rx="20" fill="#132b3c"/>'
           '<g font-family="Segoe UI,DejaVu Sans,sans-serif">'
           '<text x="48" y="65" font-size="38" font-weight="700" fill="white">PipeTwin</text>'
           '<text x="48" y="104" font-size="20" fill="#bccbd8">Прогноз остаточной толщины · две модели · единый воспроизводимый запуск</text>'
           + "".join(cards)
           + '<text x="48" y="272" fill="#93a7b8" font-size="14">Pearson² на открытых контрольных ВТД. Качество на скрытых целевых участках ещё не измерено.</text></g></svg>')
    (output / "overview.svg").write_text(svg + "\n", encoding="utf-8")


def write_report(output, warm, cold, warm_oof, late_oof, cold_oof, priority, metrics, summary):
    profiles = profile_table(warm, cold, priority)
    observed = observation_table(warm_oof, late_oof, cold_oof)
    profiles.to_csv(output / "profile_data.csv", index=False, float_format="%.12g", lineterminator="\n")
    observed.to_csv(output / "validation_predictions.csv", index=False, float_format="%.12g", lineterminator="\n")
    data = {"profiles": _records(profiles), "metrics": _records(metrics), "summary": summary,
            "downloads": {name: (output / name).read_bytes().decode("utf-8")
                          for name in ("result.csv", "metrics.csv", "priority_segments.csv",
                                       "profile_data.csv", "validation_predictions.csv", "summary.json", "cleaning.csv")}}
    template = files("pipetwin").joinpath("templates/dashboard.html").read_text(encoding="utf-8")
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = template.replace("@@DATA@@", encoded)
    for name in FIGURES:
        image = base64.b64encode((output / f"{name}.png").read_bytes()).decode("ascii")
        html = html.replace(f"@@{name.upper()}@@", f"data:image/png;base64,{image}")
    if "@@" in html:
        raise ValueError("An unresolved report template marker remains")
    (output / "dashboard.html").write_text(html, encoding="utf-8")
    _overview(output, metrics, summary)
