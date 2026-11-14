"""Validate canonical model inputs; no raw-data cleaning or target construction."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


DAYS_PER_YEAR = 365.25
RATE_COLUMNS = ("water_fraction", "temperature_c", "velocity_m_s", "fill_coefficient")
KEY_COLUMNS = ("sample_id", "id", "main_id", "exposure_id")
FORBIDDEN_FEATURES = {
    *KEY_COLUMNS, "segment_id", "scenario", "target_date", "label_available_at",
    "window_start", "window_end", "anchor_date", "y_mm", "thickness_mm",
    "residual_wall_thickness", "nominal_observed_mm", "recorded_features",
    "coverage_status", "observed", "prediction_mm", "fold_id", "case_id",
}


def timestamp(value):
    """Naive timestamps mean Moscow local time, as specified by the contract."""
    try:
        result = pd.Timestamp(value)
        if pd.isna(result):
            raise ValueError("Missing timestamp")
        if result.tzinfo is not None:
            result = result.tz_convert("Europe/Moscow").tz_localize(None)
        return result
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid timestamp: {value!r}") from error


def _dates(values, name, missing=False):
    try:
        result = pd.to_datetime(values, errors="raise")
        if result.dt.tz is not None:
            result = result.dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"Invalid or mixed-timezone dates in {name}") from error
    if not missing and result.isna().any():
        raise ValueError(f"Missing dates in {name}")
    return result


def validate_features(columns, scenario):
    if isinstance(columns, str):
        raise ValueError("feature_columns must be an explicit sequence of names")
    columns = list(columns)
    if any(not isinstance(c, str) or not c.strip() for c in columns) or len(set(columns)) != len(columns):
        raise ValueError("Feature names must be unique nonempty strings")
    for column in columns:
        name = column.lower()
        if name in FORBIDDEN_FEATURES or name.startswith(("target_", "label_")):
            raise ValueError(f"Forbidden target/metadata feature: {column}")
        if scenario == "cold" and name.startswith(("anchor", "previous", "neighbor", "history", "vtd", "ili")):
            raise ValueError(f"Cold model cannot use inspection-history feature: {column}")
    return columns


def _required(frame, columns, table):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{table} missing columns: {sorted(missing)}")
    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate column names in {table}")


def _strings(values, name):
    if not values.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError(f"{name} must contain nonempty strings")


def _numbers(values, name, positive=False):
    try:
        result = values.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric {name}") from error
    if not np.isfinite(result).all() or (positive and (result <= 0).any()):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _boolean(values, name):
    if values.isna().any() or not values.isin([True, False, 0, 1]).all():
        raise ValueError(f"{name} must contain boolean flags")
    return values.astype(bool).to_numpy()


@dataclass
class Inputs:
    frame: pd.DataFrame
    exposure: pd.DataFrame
    eligible: np.ndarray
    initial: np.ndarray
    fallback: np.ndarray
    years: np.ndarray
    y: np.ndarray | None


def validate_inputs(frame, exposure, scenario, cutoff, *, y=None):
    """Validate only referenced histories, leaving unrelated validation data unused."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a nonempty DataFrame")
    columns = [*KEY_COLUMNS, "segment_id", "scenario", "target_date", "window_start",
               "window_end", "horizon_years", "nominal_mm"]
    if scenario == "between":
        columns += ["anchor_mm", "anchor_observed", "anchor_date", "fallback_mm"]
    if y is not None:
        columns += ["label_available_at"]
    _required(frame, columns, "frame")
    frame = frame.copy().reset_index(drop=True)
    for column in KEY_COLUMNS:
        _strings(frame[column], column)
    if frame.sample_id.duplicated().any():
        raise ValueError("sample_id must be unique")
    segments = _numbers(frame.segment_id, "segment_id")
    if ((segments < 0) | (segments != np.floor(segments))).any():
        raise ValueError("segment_id must be a nonnegative integer")
    if not frame.scenario.eq(scenario).all():
        raise ValueError("Frame scenario differs from model scenario")
    for column in ["window_start", "window_end", "target_date"]:
        frame[column] = _dates(frame[column], column)
    start, end = frame.window_start, frame.window_end
    if not (start.eq(start.dt.normalize()) & end.eq(end.dt.normalize())).all():
        raise ValueError("Daily windows must start and end at local midnight")
    if not (start.lt(end) & end.eq(frame.target_date) & end.le(cutoff)).all():
        raise ValueError("Require window_start < window_end = target_date <= cutoff")
    years = _numbers(frame.horizon_years, "horizon_years", positive=True)
    expected = (end - start).dt.days.to_numpy() / DAYS_PER_YEAR
    if not np.allclose(years, expected, rtol=1e-12, atol=1e-12):
        raise ValueError("horizon_years disagrees with calendar window")
    nominal = _numbers(frame.nominal_mm, "nominal_mm", positive=True)
    eligible = np.ones(len(frame), dtype=bool)
    initial, fallback = nominal.copy(), np.full(len(frame), np.nan)
    if scenario == "between":
        frame["anchor_date"] = _dates(frame.anchor_date, "anchor_date")
        if not frame.anchor_date.eq(start).all():
            raise ValueError("anchor_date must equal window_start and precede target")
        eligible = _boolean(frame.anchor_observed, "anchor_observed")
        initial[eligible] = _numbers(frame.loc[eligible, "anchor_mm"], "observed anchor_mm", positive=True)
        fallback[~eligible] = _numbers(frame.loc[~eligible, "fallback_mm"], "fallback_mm")
    target = None
    if y is not None:
        target = np.asarray(y, dtype=float)
        if target.shape != (len(frame),) or not (np.isfinite(target) & (target > 0)).all():
            raise ValueError("y_mm must be positional finite positive labels")
        available = _dates(frame.label_available_at, "label_available_at")
        if not (end.lt(cutoff) & available.lt(cutoff) & available.ge(end)).all():
            raise ValueError("Training label and its availability must precede cutoff; availability cannot precede target")
        if not eligible.any():
            raise ValueError("No observed anchors available for between calibration")

    _required(exposure, ["exposure_id", "id", "day", "available_at", "observed",
                         "dt_years", *RATE_COLUMNS], "exposure")
    history = exposure.loc[exposure.exposure_id.isin(frame.exposure_id)].copy()
    history = history.reset_index(drop=True)
    if set(frame.exposure_id) != set(history.exposure_id):
        raise ValueError("Missing exposure_id referenced by frame")
    for column in ["exposure_id", "id"]:
        _strings(history[column], column)
    history["day"] = _dates(history.day, "day")
    history["available_at"] = _dates(history.available_at, "available_at", missing=True)
    history["observed"] = _boolean(history.observed, "observed")
    if history.duplicated(["exposure_id", "day"]).any():
        raise ValueError("Duplicate exposure_id/day would double-count exposure")
    duration = _numbers(history.dt_years, "dt_years", positive=True)
    if not np.allclose(duration, 1 / DAYS_PER_YEAR, rtol=1e-12, atol=1e-12):
        raise ValueError("dt_years must represent exactly one calendar day")
    for column in RATE_COLUMNS:
        try:
            values = history[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Non-numeric exposure channel {column}") from error
        if np.isinf(values).any():
            raise ValueError(f"Infinite exposure channel {column}")
        if column in ("water_fraction", "fill_coefficient") and ((values < 0) | (values > 1)).any():
            raise ValueError(f"Exposure fraction {column} outside [0, 1]")
        if column == "velocity_m_s" and (values < 0).any():
            raise ValueError("Negative liquid velocity")
        if history.loc[~history.observed, column].notna().any():
            raise ValueError("Unknown days must retain missing physical channels")
    if history.loc[~history.observed, "available_at"].notna().any():
        raise ValueError("Unknown days must have available_at=NaT")
    definitions = frame[["exposure_id", "id", "window_start", "window_end"]].drop_duplicates()
    if definitions.exposure_id.duplicated().any():
        raise ValueError("One exposure_id cannot identify different sections or windows")
    groups = {key: part for key, part in history.groupby("exposure_id", sort=False)}
    for row in definitions.itertuples(index=False):
        part = groups[row.exposure_id]
        days = pd.date_range(row.window_start, row.window_end, inclusive="left", freq="D")
        if not part.id.eq(row.id).all() or not pd.DatetimeIndex(part.day.sort_values()).equals(days):
            raise ValueError("Exposure section or calendar days disagree with referenced window")
        observed = part.loc[part.observed]
        if not (observed.available_at.notna() & observed.available_at.ge(observed.day)
                & observed.available_at.lt(row.window_end) & observed.available_at.lt(cutoff)).all():
            raise ValueError("Observed exposure availability must be before window_end and cutoff and not before its day")
    history = history.sort_values(["exposure_id", "day"]).reset_index(drop=True)
    return Inputs(frame, history, eligible, initial, fallback, years, target)


def feature_matrix(frame, columns):
    _required(frame, columns, "feature frame")
    matrix = frame[columns].to_numpy(dtype=float)
    if np.isinf(matrix).any():
        raise ValueError("Infinite ML feature; cleaner must mask invalid channels")
    return matrix


def integrate_loss(exposure, daily_rates, missing_policy):
    """Pure rate (mm/year) times duration (years), grouped by exposure_id."""
    rates = np.asarray(daily_rates, dtype=float)
    if rates.shape != (len(exposure),) or not (np.isfinite(rates) & (rates >= 0)).all():
        raise ValueError("Daily rates must be finite nonnegative values matching exposure")
    if missing_policy not in {"observed-only", "train-median"}:
        raise ValueError("Unknown missing_policy")
    duration = exposure.dt_years.to_numpy(dtype=float)
    if missing_policy == "observed-only":
        duration = duration * exposure.observed.to_numpy(dtype=bool)
    return pd.Series(rates * duration).groupby(exposure.exposure_id.to_numpy(), sort=False).sum()
