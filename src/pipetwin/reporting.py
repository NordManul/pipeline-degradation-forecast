"""Portable report with embedded data and auditable plotting tables."""

from html import escape
from importlib.resources import files
import json

import numpy as np
import pandas as pd


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


def route_coordinates(profiles, survey):
    """Project the supplied surveyed route to local metres, without extrapolation."""
    result = {}
    if survey is None:
        return result
    ids = pd.to_numeric(survey.id, errors="coerce")
    for section, rows in profiles.groupby("id", sort=False):
        numeric_id = pd.to_numeric(section, errors="coerce")
        points = survey.loc[ids.eq(numeric_id), ["total_distance", "lon", "lat", "z"]]
        points = points.apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
        points = points.loc[np.isfinite(points).all(axis=1) & points.total_distance.ge(0)
                            & points.total_distance.le(float(rows.end_m.max()) + .1)]
        points = points.loc[~points.total_distance.duplicated(keep=False)].sort_values("total_distance")
        if len(points) < 2:
            continue
        local = points[["lon", "lat", "z"]].to_numpy()
        origin = local[0].copy()
        local -= origin
        local[:, 0] *= 111_195*np.cos(np.deg2rad(origin[1]))
        local[:, 1] *= 111_195
        distance = points.total_distance.to_numpy()
        centre = rows.sort_values("segment_id").center_m.to_numpy()
        coords = np.column_stack([np.interp(centre, distance, column, left=np.nan, right=np.nan)
                                  for column in local.T])
        result[section] = [[round(float(value), 2) for value in point] if np.isfinite(point).all() else None
                           for point in coords]
    return result


def _overview(output, profiles, routes):
    cards = []
    for index, (_, rows) in enumerate(profiles.groupby("case", sort=True)):
        left = 805 if index == 0 else 999
        section = rows.id.iloc[0]
        hue = "#73dfc5" if index == 0 else "#ffd28c"
        cards.append(f'<rect x="{left}" y="35" width="176" height="246" rx="12" fill="#102027" stroke="#294047"/>'
                     f'<text x="{left + 15}" y="63" font-size="15" fill="{hue}">Кейс {index + 1}</text>'
                     f'<text x="{left + 15}" y="259" font-size="13" fill="#b0c3c6">{escape(str(section))}</text>')
        route = routes.get(section, [])
        points = np.array([point if point is not None else [np.nan] * 3 for point in route])
        if not len(points) or not np.isfinite(points).any():
            continue
        xy = points[:, :2] @ np.array([[.82, .57], [-.57, .82]])
        xy -= (np.nanmin(xy, axis=0) + np.nanmax(xy, axis=0)) / 2
        xy *= min(130 / max(1, np.nanmax(np.abs(xy[:, 0])) * 2),
                  134 / max(1, np.nanmax(np.abs(xy[:, 1])) * 2))
        xy[:, 0] += left + 88
        xy[:, 1] = 152 - xy[:, 1]
        commands = []
        connected = False
        for x, y in xy:
            if np.isfinite([x, y]).all():
                commands.append(f'{"L" if connected else "M"}{x:.1f},{y:.1f}')
                connected = True
            else:
                connected = False
        cards.append(f'<path d="{" ".join(commands)}" fill="none" stroke="{hue}" stroke-width="3.2" '
                     'stroke-linecap="round" stroke-linejoin="round"/>')
    cells = f"{len(profiles):,}".replace(",", " ")
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="316" viewBox="0 0 1200 316">'
           '<title>PipeTwin. Прогноз остаточной толщины на двух участках</title>'
           '<rect width="1200" height="316" rx="20" fill="#091319"/>'
           '<g font-family="Segoe UI,DejaVu Sans,sans-serif">'
           '<circle cx="52" cy="51" r="8" fill="none" stroke="#73dfc5" stroke-width="3"/>'
           '<text x="70" y="59" font-size="22" font-weight="700" fill="#edf4f2">PipeTwin</text>'
           '<text x="42" y="129" font-size="48" font-weight="700" fill="#edf4f2">Толщина стенки</text>'
           '<text x="42" y="185" font-size="48" font-weight="700" fill="#edf4f2">вдоль трубопровода</text>'
           '<path d="M42 217H748" stroke="#294047"/>'
           f'<text x="42" y="253" font-size="18" fill="#b0c3c6">{cells} ячеек · две трассы · профиль и приоритеты</text>'
           + "".join(cards) + '</g></svg>')
    (output / "overview.svg").write_text(svg + "\n", encoding="utf-8")


def write_report(output, warm, cold, warm_oof, late_oof, cold_oof, priority, metrics, summary, survey=None):
    profiles = profile_table(warm, cold, priority)
    observed = observation_table(warm_oof, late_oof, cold_oof)
    profiles.to_csv(output / "profile_data.csv", index=False, float_format="%.12g", lineterminator="\n")
    observed.to_csv(output / "validation_predictions.csv", index=False, float_format="%.12g", lineterminator="\n")
    routes = route_coordinates(profiles, survey)
    data = {"profiles": _records(profiles), "metrics": _records(metrics), "summary": summary,
            "routes": routes,
            "observations": _records(observed[["case", "split", "main_id", "y_mm", "thickness_pred"]]),
            "downloads": {name: (output / name).read_bytes().decode("utf-8")
                          for name in ("result.csv", "metrics.csv", "priority_segments.csv",
                                       "profile_data.csv", "validation_predictions.csv", "summary.json", "cleaning.csv")}}
    assets = files("pipetwin").joinpath("templates")
    template = assets.joinpath("dashboard.html").read_text(encoding="utf-8")
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = (template.replace("@@STYLES@@", assets.joinpath("dashboard.css").read_text(encoding="utf-8"))
            .replace("@@SCRIPT@@", assets.joinpath("dashboard.js").read_text(encoding="utf-8"))
            .replace("@@DATA@@", encoded))
    if "@@" in html:
        raise ValueError("An unresolved report template marker remains")
    (output / "dashboard.html").write_text(html, encoding="utf-8")
    _overview(output, profiles, routes)
