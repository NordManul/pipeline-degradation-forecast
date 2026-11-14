"""Reproduce the two-case submission and its two validation figures."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .case1 import FINAL_MODEL as CASE1_MODEL
from .case1 import fit_case1, prepare_case1, score_late
from .case2 import FINAL_MODEL as CASE2_MODEL
from .case2 import fit_case2, prepare_transfer
from .plots import plot_profiles, plot_validation
from .quality import read_raw_sources
from .sources import read_config
from .submission import KEYS, export_submission, validate_submission


def _forecast(frame, values, spec):
    if not frame.id.eq(spec["id"]).all() or not frame.target_date.eq(pd.Timestamp(spec["target_date"])).all():
        raise ValueError("Forecast belongs to the wrong section or target date")
    if not np.allclose(frame.nominal_mm, spec["nominal_mm"], rtol=0, atol=1e-8):
        raise ValueError("The target nominal thickness differs from the case contract")
    if not np.isfinite(values).all() or ((values < 0) | (values > frame.nominal_mm)).any():
        raise ValueError("Forecast thickness falls outside audited bounds")
    return frame.assign(thickness_pred=values)


def _cleaning_summary(audits):
    grouped = []
    for scope, audit in audits.items():
        grouped.append(audit.groupby(["source_file", "rule_id", "action", "reason"], dropna=False)
                       .size().rename("events").reset_index().assign(scope=scope))
    return pd.concat(grouped, ignore_index=True)


def run(data_dir, output, template_path=None):
    """Build both cases from source, validate submission keys and publish atomically."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    cases = read_config("contracts/cases.yaml")
    config = read_config("configs/forecast.yaml")
    target_contract = read_config("configs/data_contract.yaml")
    policy = read_config("contracts/case2/retrospective.yaml")
    raw, manifest = read_raw_sources(data_dir)
    warm_train, warm, late, strict, grid = prepare_case1(raw, manifest, cases, target_contract, config)
    cold_train, cold, strict_audit, donor_audit = prepare_transfer(raw, manifest, cases, target_contract, policy, config)
    with threadpool_limits(limits=1):
        warm_values, late_values, historic = fit_case1(warm_train, warm, late, config, cases["cases"]["case1"]["id"])
        cold_values, transfer = fit_case2(cold_train, cold, config["seed"])
    late_scores, late_audit = score_late(
        late, late_values, raw, strict, grid, target_contract, config, config["late"]["label_pool_cutoff_exclusive"]
    )
    warm_scores = pd.concat([historic, late_scores], ignore_index=True)
    warm = _forecast(warm, warm_values, cases["cases"]["case1"])
    cold = _forecast(cold, cold_values, cases["cases"]["case2"])
    columns = KEYS + ["thickness_pred"]
    predictions = pd.concat([warm[columns], cold[columns]], ignore_index=True)
    expected_keys = pd.concat([warm[KEYS], cold[KEYS]], ignore_index=True)
    template = pd.read_csv(template_path, dtype={"id": str}) if template_path else expected_keys
    validate_submission(predictions, expected_keys)
    metrics = pd.concat([warm_scores.assign(case="case1"), transfer.assign(case="case2")], ignore_index=True)
    metrics = metrics[["case", "split", "model", "held_main", "campaign", "n", "pearson_r", "pearson_squared",
                       "mae_mm", "rmse_mm", "bias_mm", "recall_thin_10pct"]]
    audits = {"case1": strict.audit, "case1_late_labels": late_audit, "case2": strict_audit, "case2_donors": donor_audit}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pipetwin-", dir=output.parent) as temp:
        staged = Path(temp) / "run"
        staged.mkdir()
        result = staged / "result.csv"
        export_submission(predictions, template, result)
        metrics.to_csv(staged / "metrics.csv", index=False, float_format="%.12g")
        _cleaning_summary(audits).to_csv(staged / "cleaning.csv", index=False)
        plot_profiles(staged, warm, cold)
        plot_validation(staged, warm_scores, transfer)
        summary = {
            "result_sha256": sha256(result.read_bytes()).hexdigest(),
            "template": "organizer_file" if template_path else "passport_grid",
            "seed": config["seed"],
            "case1": {"model": CASE1_MODEL, "train_labels": int(warm_train.observed.sum()), "cells": len(warm),
                      "cells_with_anchor": int(warm.anchor_observed.sum())},
            "case2": {"model": CASE2_MODEL, "train_labels": len(cold_train), "cells": len(cold),
                      "training_mains": sorted(cold_train.main_id.unique()), "tech_coverage": float(cold.regime_coverage.iloc[0])},
            "sources": manifest,
        }
        (staged / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staged.rename(output)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True, help="New directory for the submission and evaluation")
    parser.add_argument("--template", type=Path, help="Organizer CSV with id and segment_id in the required order")
    args = parser.parse_args(argv)
    summary = run(args.data_dir, args.output, args.template)
    print(f"{summary['case1']['cells'] + summary['case2']['cells']:,} rows → {args.output / 'result.csv'}")


if __name__ == "__main__":
    main()
