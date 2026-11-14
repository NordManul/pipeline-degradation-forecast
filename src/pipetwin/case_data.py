"""Canonical case frames, historical exposure windows and shared fold inputs."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

from .features import attach_labels, build_snapshot, feature_columns
from .geometry import build_grid
from .physical_features import daily_mode
from .quality import clean_sources, read_raw_sources
from .sources import read_config
from .targets import build_targets
from .validation import blocked_folds


EXPOSURE_COLUMNS = ["water_fraction", "temperature_c", "velocity_m_s", "fill_coefficient"]


def _prior_targets(cleaned, grid, target_contract, as_of):
    inspection = cleaned.sources.inspection
    known = inspection.loc[inspection.date.lt(as_of) & inspection.available_at.lt(as_of)].copy()
    return build_targets(replace(cleaned.sources, inspection=known), grid, target_contract)[0]


def canonical_snapshot(cleaned, grid, all_targets, spec, config, target_contract, *, labelled):
    target = pd.Timestamp(spec["target_date"])
    if target > cleaned.cutoff or (labelled and target >= cleaned.cutoff):
        raise ValueError("Requested snapshot violates the official case cutoff")
    # Reaggregate OLD labels using only records available before this sample's T.
    # Filtering a fully aggregated future minimum afterwards would erase known
    # past observations when one late-arriving record changes the minimum.
    history = _prior_targets(cleaned, grid, target_contract, target)
    frame = build_snapshot(cleaned.sources, grid, history, spec, config)
    frame["window_start"] = frame.anchor_date if "anchor_date" in spec else frame.commission_date
    frame["window_end"] = target
    frame["anchor_observed"] = frame.anchor_observed.astype(bool)
    date_text = target.strftime("%Y-%m-%d")
    start_text = pd.to_datetime(frame.window_start).dt.strftime("%Y-%m-%d")
    frame["exposure_id"] = frame.id + "|" + start_text + "|" + date_text
    frame["sample_id"] = cleaned.case_id + "|" + frame.scenario + "|" + frame.exposure_id + "|" + frame.segment_id.astype(str)
    frame["case_id"] = cleaned.case_id
    if labelled:
        frame = attach_labels(frame, all_targets)
        if (frame.loc[frame.observed, "label_available_at"] >= cleaned.cutoff).any():
            raise ValueError("Training labels were not available at case cutoff")
    if frame.sample_id.duplicated().any():
        raise ValueError("Canonical sample IDs must be unique")
    return frame


def build_exposure(regime, frames, cutoff, temperature_policy="observed_endpoints"):
    """One calendar per distinct section/window; overlapping windows stay separate."""
    windows = pd.concat([f[["exposure_id", "id", "window_start", "window_end", "horizon_years"]] for f in frames], ignore_index=True)
    windows = windows.drop_duplicates()
    if windows.exposure_id.duplicated().any():
        raise ValueError("Exposure key refers to inconsistent windows")
    pieces = []
    for row in windows.itertuples(index=False):
        start, end = pd.Timestamp(row.window_start), pd.Timestamp(row.window_end)
        if not start < end <= cutoff:
            raise ValueError("Exposure window violates temporal order or case cutoff")
        if not np.isclose((end - start).days / 365.25, row.horizon_years):
            raise ValueError("Calendar and sample horizon disagree")
        history = daily_mode(regime.loc[regime.id.eq(row.id)], start, end, temperature_policy=temperature_policy)
        history = history[["day", "available_at", *EXPOSURE_COLUMNS]].copy()
        history["observed"] = True
        calendar = pd.DataFrame({"day": pd.date_range(start, end - pd.Timedelta(days=1), freq="D")})
        calendar = calendar.merge(history, on="day", how="left", validate="one_to_one")
        calendar["observed"] = calendar.observed.eq(True)
        calendar["id"], calendar["exposure_id"] = row.id, row.exposure_id
        calendar["dt_years"] = 1 / 365.25
        pieces.append(calendar)
    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(["exposure_id", "day"]).any() or result.day.ge(cutoff).any():
        raise ValueError("Duplicate or future exposure days")
    return result[["exposure_id", "id", "day", "available_at", "observed", "dt_years", *EXPOSURE_COLUMNS]]


def _fallback_fraction(train):
    measured = train.loc[train.observed]
    if measured.empty:
        raise ValueError("No observed labels for a fallback fit")
    return float(np.clip(np.median(measured.y_mm / measured.nominal_mm), 0, 1))


def model_views(package, fold_number=None):
    """Fit the shared fallback on TRAIN ONLY, never on the held-out fold."""
    frame = package["train_frame"]
    if fold_number is None:
        train, valid = frame.copy(), package["predict_frame"].copy()
    else:
        fold = next(f for f in package["folds"] if f["number"] == fold_number)
        train, valid = frame.iloc[fold["train"]].copy(), frame.iloc[fold["valid"]].copy()
    fraction = _fallback_fraction(train)
    for part in [train, valid]:
        part["fallback_mm"] = part.nominal_mm * fraction
    return train, valid, {"method": "train_median_residual_fraction", "fraction": fraction}


def prepare_case(cleaned, target_contract, config):
    case = cleaned.contract["cases"][cleaned.case_id]
    config = {**config, "temperature_policy": "observed_endpoints"}
    grid = build_grid(cleaned.sources.passport, target_contract["segment_length_m"])
    targets, quarantine = build_targets(cleaned.sources, grid, target_contract)
    if case["scenario"] == "between":
        training = {"ids": config["development"]["ids"], "anchor_date": "2018-09-23", "target_date": "2022-07-06"}
    else:
        training = {"ids": config["cold_train"]["ids"], "target_date": "2018-09-23"}
    prediction = {"ids": [case["id"]], "target_date": case["target_date"]}
    if case["anchor_date"]:
        prediction["anchor_date"] = case["anchor_date"]
    train = canonical_snapshot(cleaned, grid, targets, training, config, target_contract, labelled=True)
    requested = canonical_snapshot(cleaned, grid, targets, prediction, config, target_contract, labelled=False)
    if len(requested) != case["expected_provisional_segments"]:
        raise ValueError("Requested grid disagrees with the case contract")
    exposure = build_exposure(cleaned.sources.regime, [train, requested], cleaned.cutoff)
    folds = blocked_folds(train, config["folds"], config["spatial_buffer_m"])
    summary = {
        "case_id": cleaned.case_id, "contract_version": cleaned.contract["contract_version"],
        "id": case["id"], "scenario": case["scenario"], "target_date": case["target_date"],
        "cutoff_exclusive": case["cutoff_exclusive"], "training_date": training["target_date"],
        "training_grid_cells": len(train), "training_observed_labels": int(train.observed.sum()),
        "prediction_cells": len(requested), "prediction_observed_anchors": int(requested.anchor_observed.sum()),
        "eligible_label_count_all_campaigns": int(targets.observed.sum()),
        "regime_rows_after_cleaning": len(cleaned.sources.regime), "inspection_rows_after_cleaning": len(cleaned.sources.inspection),
        "exposure_histories": exposure.exposure_id.nunique(), "exposure_days": len(exposure),
        "observed_exposure_days": int(exposure.observed.sum()), "audit_events": len(cleaned.audit),
        "evaluation": "auxiliary_spatial_not_independent_temporal_or_pipeline_backtest",
        "fallback": "median(y/nominal) fitted separately on each fold train; all train only for final inference",
        "temperature_policy": config["temperature_policy"], "grid_status": "provisional_pending_official_template",
        "label_policy": target_contract["target_policy"],
    }
    package = {
        **summary, "summary": summary, "spec": case, "config": config,
        "train_frame": train, "predict_frame": requested, "exposure": exposure,
        "eligible_targets": targets, "feature_columns": feature_columns(case["scenario"]),
        "folds": [{"number": f.number, "train": f.train.tolist(), "valid": f.valid.tolist(),
                   "left_m": f.left_m, "right_m": f.right_m} for f in folds],
        "input_manifest": cleaned.sources.manifest,
    }
    return package, quarantine


def write_prepared_case(package, cleaned, quarantine, directory):
    from .pipeline import write_json
    directory = Path(directory)
    directory.mkdir()
    joblib.dump(package, directory / "prepared.joblib", compress=3)
    package["train_frame"].to_csv(directory / "train_frame.csv.gz", index=False)
    package["predict_frame"].to_csv(directory / "predict_frame.csv.gz", index=False)
    package["exposure"].to_csv(directory / "exposure.csv.gz", index=False)
    cleaned.audit.to_csv(directory / "cleaning_audit.csv.gz", index=False)
    quarantine.to_csv(directory / "target_quarantine.csv.gz", index=False)
    cleaned.audit.groupby(["source_file", "rule_id", "action"], dropna=False).size().rename("events").reset_index().to_csv(directory / "cleaning_summary.csv", index=False)
    write_json(directory / "summary.json", package["summary"])
    write_json(directory / "folds.json", package["folds"])
    write_json(directory / "input_manifest.json", package["input_manifest"])
    # Ready-to-use fold files carry only train-fitted fallback values.
    (directory / "folds").mkdir()
    for fold in package["folds"]:
        train, valid, info = model_views(package, fold["number"])
        train.to_csv(directory / "folds" / f"fold_{fold['number']}_train.csv.gz", index=False)
        valid.to_csv(directory / "folds" / f"fold_{fold['number']}_valid.csv.gz", index=False)
    train, requested, fallback = model_views(package)
    train.to_csv(directory / "final_train.csv.gz", index=False)
    requested.to_csv(directory / "final_predict.csv.gz", index=False)
    write_json(directory / "fallback.json", fallback)


def prepare_cases(data_dir, output_dir, contracts_dir="contracts", config_path="configs/forecast.yaml"):
    """Publish both independently cleaned cases atomically for all model owners."""
    from .pipeline import write_json
    output, contracts = Path(output_dir).resolve(), Path(contracts_dir)
    if output.exists():
        raise ValueError("Preparation output already exists; use a new run directory")
    contract = read_config(contracts / "cases.yaml")
    config = read_config(config_path)
    target_contract = read_config(config["contract"])
    raw, manifest = read_raw_sources(data_dir)
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(contracts.glob("*.yaml"))}
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with tempfile.TemporaryDirectory(prefix=".case-preparation-", dir=output.parent) as temp:
        staging = Path(temp) / "prepared"
        staging.mkdir()
        for name in contract["cases"]:
            cleaned = clean_sources(raw, name, contract, target_contract, manifest)
            package, quarantine = prepare_case(cleaned, target_contract, config)
            package["contract_sha256"] = hashes
            write_prepared_case(package, cleaned, quarantine, staging / name)
            summaries.append(package["summary"])
            print(f"{name}: {package['summary']['training_observed_labels']} labels, {package['summary']['prediction_cells']} forecast cells", flush=True)
        write_json(staging / "summary.json", {"contract_version": contract["contract_version"], "contract_sha256": hashes,
                                               "cases": summaries, "completed": True})
        staging.rename(output)
    return summaries
