"""Read-only inventory of the information available for case2 diagnosis.

Later campaigns are inventoried, never fitted as an official case2 model.
All labels and historical features come from the owner's shared pipeline.
"""

import argparse
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipetwin.case_data import canonical_snapshot
from pipetwin.features import feature_columns
from pipetwin.geometry import build_grid
from pipetwin.pipeline import write_json
from pipetwin.quality import read_raw_sources, clean_sources
from pipetwin.sources import read_config
from pipetwin.targets import build_targets


def run(prepared, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    raw, manifest = read_raw_sources("data/raw")
    contract = deepcopy(read_config("contracts/cases.yaml"))
    target_contract = read_config("configs/data_contract.yaml")
    config = {**read_config("configs/forecast.yaml"), "temperature_policy": "observed_endpoints"}
    context = "diagnostic_inventory"
    contract["cases"][context] = {"id": "1250009583", "target_date": "2024-12-01",
                                  "cutoff_exclusive": "2024-12-01"}
    cleaned = clean_sources(raw, context, contract, target_contract, manifest)
    grid = build_grid(cleaned.sources.passport, target_contract["segment_length_m"])
    targets, _ = build_targets(cleaned.sources, grid, target_contract)
    observed = targets.loc[targets.observed]
    campaigns = observed.groupby(["main_id", "date"]).agg(
        labels=("thickness_mm", "size"), sections=("id", "nunique"), nominal_min=("nominal_mm", "min"),
        nominal_max=("nominal_mm", "max"), mean_mm=("thickness_mm", "mean"), std_mm=("thickness_mm", "std")).reset_index()
    campaigns.to_csv(output / "all_valid_campaigns.csv", index=False)
    frames = [canonical_snapshot(cleaned, grid, targets,
              {"ids": group.id.unique().tolist(), "target_date": date}, config, target_contract, labelled=True)
              for (_, date), group in observed.groupby(["main_id", "date"])]
    cold = pd.concat(frames, ignore_index=True)
    cold.to_csv(output / "diagnostic_cold_frames_all_dates.csv.gz", index=False)
    support = cold.loc[cold.observed].groupby(["main_id", "target_date"]).agg(
        n=("y_mm", "size"), nominal=("nominal_mm", "first"), age=("age_years", "first"),
        water_min=("water_fraction_mean", "min"), water_max=("water_fraction_mean", "max"),
        coverage=("regime_coverage", "mean")).reset_index()
    support.to_csv(output / "campaign_support.csv", index=False)
    official = joblib.load(Path(prepared) / "case2/prepared.joblib")
    train = official["train_frame"].loc[lambda d: d.observed]
    query = official["predict_frame"]
    columns = feature_columns("cold")
    later_source_query = canonical_snapshot(cleaned, grid, targets,
                         {"ids": ["1250009583"], "target_date": "2022-07-01"}, config, target_contract, labelled=False)
    np.testing.assert_allclose(later_source_query[columns], query[columns], rtol=0, atol=0, equal_nan=True)
    group_mean = train.groupby("id").y_mm.transform("mean")
    # For predictors constant within each simple section, the in-sample optimum
    # is its group mean. This is an information diagnostic, not CV performance.
    oracle_r = float(np.sqrt(np.var(group_mean) / np.var(train.y_mm)))
    repeated = cold.loc[cold.observed & cold.target_date.eq("2018-09-23"), ["id", "segment_id", "y_mm"]].merge(
        cold.loc[cold.observed & cold.target_date.eq("2022-07-06"), ["id", "segment_id", "y_mm"]],
        on=["id", "segment_id"], suffixes=("_2018", "_2022"))
    write_json(output / "diagnostic_inventory.json", {
        "scope": __doc__, "official_case2_unchanged": True, "source_manifest": manifest,
        "all_date_valid_labels": len(observed), "independent_mains": int(observed.main_id.nunique()),
        "target_case2_has_labels": bool(observed.id.eq("1250009583").any()),
        "query_varying_features": [c for c in columns if query[c].nunique(dropna=False) > 1],
        "query_exposure_histories": int(query.exposure_id.nunique()),
        "query_regime_coverage": float(query.regime_coverage.iloc[0]),
        "query_features_unchanged_with_later_sources": True,
        "training_section_constant_insample_pearson_upper_bound": oracle_r,
        "repeated_observed_cells_2018_2022": len(repeated),
        "increases_gt_0_05_mm": int((repeated.y_mm_2022 - repeated.y_mm_2018 > 0.05).sum()),
        "note": "An increase alone does not establish measurement error, repair or corrosion reversal",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.prepared, args.output)
