"""Strict template-key validation and the three-column competition export."""

from pathlib import Path

import numpy as np
import pandas as pd

from .sources import canonical_ids


KEYS = ["id", "segment_id"]


def normalize_keys(frame):
    frame = frame.copy()
    if not set(KEYS).issubset(frame):
        raise ValueError("Expected id and segment_id columns")
    frame["id"] = canonical_ids(frame.id)
    segment = pd.to_numeric(frame.segment_id, errors="raise")
    if not np.isfinite(segment).all() or (segment < 0).any() or not np.equal(segment, np.floor(segment)).all():
        raise ValueError("Segment IDs must be finite nonnegative integers")
    frame["segment_id"] = segment.astype("int64")
    if frame.empty or frame.duplicated(KEYS).any():
        raise ValueError("Keys must be nonempty and unique")
    return frame


def validate_submission(predictions, template):
    predictions, template = normalize_keys(predictions), normalize_keys(template)
    if "thickness_pred" not in predictions:
        raise ValueError("Missing thickness_pred")
    values = pd.to_numeric(predictions.thickness_pred, errors="raise")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Thickness predictions must be finite nonnegative millimetres")
    predictions["thickness_pred"] = values
    expected = set(map(tuple, template[KEYS].to_numpy()))
    actual = set(map(tuple, predictions[KEYS].to_numpy()))
    if expected != actual:
        raise ValueError(f"Template key mismatch: missing={len(expected - actual)}, extra={len(actual - expected)}")
    return template[KEYS].merge(predictions[KEYS + ["thickness_pred"]], on=KEYS, how="left", sort=False, validate="one_to_one")


def export_submission(predictions, template, output):
    result = validate_submission(predictions, template)
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists; choose a new submission path")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, float_format="%.8f")
    return result


def assemble_cases(case1_path, case2_path, data_dir, output, template_path=None):
    """Combine the selected strict case 1 and retrospective case 2 forecasts."""
    from .geometry import build_grid
    from .quality import load_case_sources

    frames, grids = [], []
    for name, path, column in [("case1", case1_path, "combined_context"), ("case2", case2_path, "thickness_pred")]:
        cleaned = load_case_sources(data_dir, name)
        spec = cleaned.contract["cases"][name]
        grid = build_grid(cleaned.sources.passport, cleaned.contract["geometry"]["segment_length_m"])
        grids.append(grid.loc[grid.id.eq(spec["id"]), KEYS])
        frame = pd.read_csv(path, dtype={"id": str})
        if not frame.id.eq(spec["id"]).all() or not pd.to_datetime(frame.target_date).eq(cleaned.cutoff).all():
            raise ValueError(f"Wrong section or date in {name} forecast")
        if not frame[column].between(0, spec["nominal_mm"]).all():
            raise ValueError(f"Thickness outside audited bounds in {name}")
        frames.append(frame[KEYS + [column]].rename(columns={column: "thickness_pred"}))
    predictions = pd.concat(frames, ignore_index=True)
    passport_keys = pd.concat(grids, ignore_index=True)
    validate_submission(predictions, passport_keys)
    template = pd.read_csv(template_path, dtype={"id": str}) if template_path else passport_keys
    return export_submission(predictions, template, output)
