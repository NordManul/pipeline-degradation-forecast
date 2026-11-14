"""Diagnostic only: does MAIN-2 2022 help cold transfer to MAIN-3 2023?

This is NOT an official case2 model or a measurement of its hidden accuracy.
Both training sets precede the proxy target. The whole MAIN-3 is held out.
The model list/options are fixed; no tuning, new folds or scoring code.
"""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from pipetwin.case_data import canonical_snapshot, build_exposure
from pipetwin.case_experiments import fit_case_model, predict_models, metric_table, prediction_table
from pipetwin.features import feature_columns
from pipetwin.geometry import build_grid
from pipetwin.pipeline import write_json
from pipetwin.quality import read_raw_sources, clean_sources
from pipetwin.sources import read_config
from pipetwin.targets import build_targets

from .shared_comparison import comparison_config


MODELS = ["persistence", "global_rate", "gam", "catboost",
          "agent2_constant_rate", "agent2_hybrid_residual"]
PROXY_DATE = pd.Timestamp("2023-08-12")


def run(output, missing_policy="observed-only", physics_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    raw, manifest = read_raw_sources("data/raw")
    contract = deepcopy(read_config("contracts/cases.yaml"))
    target_contract = read_config("configs/data_contract.yaml")
    config = read_config("configs/forecast.yaml")
    config["temperature_policy"] = "observed_endpoints"
    context = "diagnostic_cold_transfer"
    # The extra day loads the proxy's labels for SCORING. Training and every
    # feature window still end strictly before the proxy observation date.
    contract["cases"][context] = {"id": "1250009583", "target_date": "2023-08-13",
                                  "cutoff_exclusive": "2023-08-13"}
    cleaned = clean_sources(raw, context, contract, target_contract, manifest)
    grid = build_grid(cleaned.sources.passport, target_contract["segment_length_m"])
    targets, _ = build_targets(cleaned.sources, grid, target_contract)
    snapshots = []
    for ids, date in [(config["cold_train"]["ids"], "2018-09-23"),
                      (config["cold_train"]["ids"], "2022-07-06"),
                      (config["cold_test"]["ids"], "2023-08-12")]:
        snapshots.append(canonical_snapshot(cleaned, grid, targets,
                         {"ids": ids, "target_date": date}, config, target_contract, labelled=True))
    first, later, valid = snapshots
    exposure = build_exposure(cleaned.sources.regime, snapshots, PROXY_DATE)
    package = {"case_id": context, "scenario": "cold", "cutoff_exclusive": PROXY_DATE,
               "exposure": exposure, "feature_columns": feature_columns("cold")}
    names = [n for n in MODELS if n.startswith("agent2_")] if physics_only else list(MODELS)
    config = comparison_config(missing_policy, "train-bounded")
    config["models"] = names
    write_json(output / "frozen_protocol.json", {
        "purpose": __doc__, "models": names, "config": config,
        "followup_coverage_diagnostic": bool(physics_only),
        "train_variants": {"2018_only": ["2018-09-23"], "2018_plus_2022": ["2018-09-23", "2022-07-06"]},
        "holdout": {"main_id": "MAIN-3", "target_date": PROXY_DATE},
        "evaluation": "exploratory_single_main_proxy_not_case2_accuracy",
        "official_case2_modified": False, "input_manifest": manifest,
    })
    scores, predictions, audit = [], [], {}
    with threadpool_limits(limits=1):
        for variant, train in [("2018_only", first), ("2018_plus_2022", pd.concat([first, later], ignore_index=True))]:
            measured = train.loc[train.observed]
            assert set(train.main_id) == {"MAIN-2"} and set(valid.main_id) == {"MAIN-3"}
            assert set(train.id).isdisjoint(valid.id)
            assert measured.label_available_at.lt(PROXY_DATE).all()
            assert not measured.id.eq("1250009583").any()
            assert not measured.target_date.ge(PROXY_DATE).any()
            fallback = {"fraction": float(np.clip(np.median(measured.y_mm / measured.nominal_mm), 0, 1))}
            raw_predictions, bounded_predictions, fits = {}, {}, []
            # Fit/score one model at a time to limit memory on the shared laptop.
            for name in names:
                print(f"Proxy MAIN-3: {variant}, {name}", flush=True)
                model, info = fit_case_model(name, package, train, fallback, config)
                raw_pred, bounded_pred = predict_models({name: model}, valid, {}, development=True)
                raw_predictions.update(raw_pred)
                bounded_predictions.update(bounded_pred)
                fits.append(info)
                del model
            scores.append(metric_table(valid, raw_predictions, bounded_predictions,
                                       split="whole_main3_proxy").assign(train_variant=variant))
            predictions.append(prediction_table(valid, raw_predictions, bounded_predictions, -1).assign(train_variant=variant))
            audit[variant] = {"observed_train_labels": len(measured), "train_main": "MAIN-2",
                              "train_sample_ids": measured.sample_id.tolist(), "fits": fits}
    pd.concat(scores, ignore_index=True).to_csv(output / "proxy_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "proxy_predictions.csv.gz", index=False)
    write_json(output / "fit_audit.json", audit)
    write_json(output / "summary.json", {"completed": True, "held_main": "MAIN-3", "held_labels": int(valid.observed.sum()),
               "held_grid_cells": len(valid), "models_per_variant": len(names),
               "target_case2_scored": False, "official_case2_modified": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--missing-policy", choices=["observed-only", "train-median"], default="observed-only")
    parser.add_argument("--physics-only", action="store_true")
    args = parser.parse_args()
    run(args.output, args.missing_policy, args.physics_only)
