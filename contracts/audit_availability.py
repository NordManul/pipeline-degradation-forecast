"""Reproduce case availability counts; this is not the new production cleaner.

Run from any directory with the project's Python environment. All raw files are
read-only; the only output is contracts/data-availability.json. Existing target
validity rules are applied, then unresolved pre-commission epochs are excluded.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipetwin.geometry import build_grid
from pipetwin.sources import load_sources, read_config
from pipetwin.targets import build_targets


def main():
    root = Path(__file__).resolve().parents[1]
    specification = read_config(root / "contracts/cases.yaml")
    target_contract = read_config(root / "configs/data_contract.yaml")
    sources = load_sources(root / "data/raw", target_contract)
    grid = build_grid(sources.passport, specification["geometry"]["segment_length_m"])
    targets, target_quarantine = build_targets(sources, grid, target_contract)
    report = {
        "contract_version": specification["contract_version"],
        "audit_date": specification["updated_at"],
        "scope": "availability_and_existing_target_rules_not_complete_new_quality_pipeline",
        "source_manifest": sources.manifest,
        "existing_target_quarantine_counts": target_quarantine.exclusion_reason.value_counts().to_dict(),
        "unresolved_epoch_policy": "exclude_pre_commission_targets",
        "cases": {},
    }
    for name, case in specification["cases"].items():
        cutoff = pd.Timestamp(case["cutoff_exclusive"])
        passport = sources.passport.set_index("id").loc[case["id"]]
        section_grid = grid.loc[grid.id.eq(case["id"])]
        if not np.isclose(passport.length_m, case["length_m"], atol=1e-6, rtol=0):
            raise ValueError(f"{name}: passport length differs from the contract")
        if len(section_grid) != case["expected_provisional_segments"]:
            raise ValueError(f"{name}: provisional grid differs from the contract")
        if not np.isclose(section_grid.iloc[-1].segment_length_m, case["final_segment_length_m"], atol=1e-6, rtol=0):
            raise ValueError(f"{name}: terminal segment length differs from the contract")
        eligible = targets.loc[targets.date.lt(cutoff) & targets.observed & ~targets.pre_commission]
        dates = sorted(eligible.date.dt.strftime("%Y-%m-%d").unique().tolist())
        if dates != case["eligible_known_label_dates"]:
            raise ValueError(f"{name}: eligible label dates differ from the contract: {dates}")
        campaign_counts = eligible.groupby(["main_id", "date"]).size().rename("observed_cells").reset_index()
        campaign_counts["date"] = campaign_counts.date.dt.strftime("%Y-%m-%d")
        own_inspection = sources.inspection.loc[sources.inspection.id.eq(case["id"])]
        own_tech = sources.regime.loc[sources.regime.id.eq(case["id"]) & sources.regime.date.lt(cutoff)]
        target_present = bool(own_inspection.date.eq(pd.Timestamp(case["target_date"])).any())
        if target_present != case["official_target_labels_present_in_current_csv"]:
            raise ValueError(f"{name}: presence of target labels changed; re-audit before use")
        if case["anchor_date"] is not None:
            prior = own_inspection.loc[own_inspection.date.lt(cutoff), "date"].max()
            if prior != pd.Timestamp(case["anchor_date"]):
                raise ValueError(f"{name}: latest known anchor differs from the contract")
        else:
            if own_inspection.date.lt(cutoff).any():
                raise ValueError(f"{name}: cold scenario now has a prior inspection")
        report["cases"][name] = {
            "id": case["id"], "cutoff_exclusive": case["cutoff_exclusive"],
            "provisional_grid_cells": len(section_grid),
            "eligible_observed_segment_labels": len(eligible),
            "eligible_main_campaigns": campaign_counts.to_dict("records"),
            "target_labels_present": target_present,
            "own_inspection_raw_rows_by_date": {
                date.strftime("%Y-%m-%d"): int(count)
                for date, count in own_inspection.groupby("date").size().items()
            },
            "own_regime_temporally_eligible_raw_rows": len(own_tech),
            "own_regime_first_timestamp": own_tech.date.min().isoformat(),
            "own_regime_last_timestamp": own_tech.date.max().isoformat(),
            "future_inspection_raw_rows_excluded_all_ids": {
                date.strftime("%Y-%m-%d"): int(count)
                for date, count in sources.inspection.loc[sources.inspection.date.ge(cutoff)].groupby("date").size().items()
            },
            "quality_note": "New regime/channel/coupon cleaning is specified in quality.yaml and is not implemented by this audit.",
        }
    output = root / "contracts/data-availability.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for name, case in report["cases"].items():
        print(f"{name}: {case['provisional_grid_cells']} grid cells, {case['eligible_observed_segment_labels']} eligible observed labels")


if __name__ == "__main__":
    main()
