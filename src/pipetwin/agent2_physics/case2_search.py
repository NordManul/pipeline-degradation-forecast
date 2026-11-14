"""Fixed case-2 hypotheses, evaluated only by the shared case evaluator.

See CASE2_SEARCH_PROTOCOL.md before interpreting a provisional winner.
No target labels, custom scoring metric, fold generator or blend is used here.
"""

import argparse
from pathlib import Path

import pandas as pd

from pipetwin.pipeline import write_json

from .shared_comparison import CONTROLS, comparison_config, run


LEARNED_CONTROLS = ["global_rate", "gam", "catboost"]
TERRAIN = ["slope_50m", "lowpoint_depth_m"]
BASE_MODELS = ["agent2_constant_rate", "agent2_positive_rate", "agent2_hybrid_residual"]


def search_configs():
    """Frozen small hypothesis set, without data-dependent configuration."""
    configs = {}
    for policy in ["observed-only", "train-median"]:
        for variant in ["references", "nominal-relative", "terrain-signed", "terrain-symmetric",
                        "terrain-symmetric-relative", "terrain-scale", "terrain-scale-relative"]:
            config = comparison_config(policy, "train-bounded")
            if variant == "references":
                config["models"] = CONTROLS + BASE_MODELS
            elif variant == "nominal-relative":
                config["models"] = list(BASE_MODELS)
            if variant.startswith("terrain-"):
                config["physics_feature_columns"] = list(TERRAIN)
            if "symmetric" in variant:
                config["agent2_options"]["residual_feature_map"] = "terrain-symmetric"
            if "scale" in variant:
                config["models"] = ["agent2_hybrid_scale"]
            if variant.endswith("relative"):
                config["agent2_options"]["cold_nominal_scaling"] = "train-relative"
            configs[f"{policy}/{variant}"] = config
    return configs


def assess_candidates(metrics, fold_metrics):
    """Apply the predeclared gate to metrics already computed by the owner."""
    if set(metrics.case_id) != {"case2"} or set(fold_metrics.case_id) != {"case2"}:
        raise ValueError("This research gate is only for case2")
    bounded = metrics.loc[metrics.output.eq("bounded")]
    folds = fold_metrics.loc[fold_metrics.output.eq("bounded")]
    records = []
    for policy, group in bounded.groupby("missing_policy", sort=False):
        controls = group.loc[group.variant.eq("references") & group.model.isin(LEARNED_CONTROLS)]
        if len(controls) != 3 or set(controls.model) != set(LEARNED_CONTROLS):
            raise ValueError("All three learned controls are required")
        best = controls.loc[controls.mae_mm.idxmin()]
        control_folds = folds.loc[folds.missing_policy.eq(policy) & folds.variant.eq("references")
                                 & folds.model.eq(best.model)].set_index("fold_id")
        if sorted(control_folds.index.tolist()) != [0, 1, 2, 3]:
            raise ValueError("Exactly four shared control folds are required")
        for _, row in group.loc[group.model.str.startswith("agent2_")].iterrows():
            parts = folds.loc[folds.missing_policy.eq(policy) & folds.variant.eq(row.variant)
                              & folds.model.eq(row.model)].set_index("fold_id")
            if sorted(parts.index.tolist()) != [0, 1, 2, 3]:
                raise ValueError("Exactly four shared candidate folds are required")
            parts = parts.loc[control_folds.index]
            if not parts.n.equals(control_folds.n) or int(parts.n.sum()) != int(row.n):
                raise ValueError("Common scoring counts differ")
            improvement = 1.0 - row.mae_mm / best.mae_mm
            positive = int(parts.pearson_r.gt(0).sum())
            wins = int(parts.mae_mm.lt(control_folds.mae_mm).sum())
            passes = (row.pearson_r > max(0.0, controls.pearson_r.max()) and improvement >= 0.05
                      and positive >= 3 and wins >= 3)
            records.append({"missing_policy": policy, "variant": row.variant, "model": row.model,
                            "pearson_r": row.pearson_r, "pearson_squared": row.pearson_squared,
                            "mae_mm": row.mae_mm, "best_mae_control": best.model,
                            "control_mae_mm": best.mae_mm, "best_control_r": controls.pearson_r.max(),
                            "mae_improvement_fraction": improvement, "positive_r_folds": positive,
                            "mae_winning_folds": wins, "passes_auxiliary_gate": bool(passes),
                            "target_transfer_validated": False})
    return pd.DataFrame(records)


def nominal_identifiability(oof, forecast):
    """Compare paired hypotheses by key, without using any target labels."""
    pairs = [("references", "nominal-relative", name) for name in BASE_MODELS]
    pairs += [("terrain-symmetric", "terrain-symmetric-relative", "agent2_hybrid_residual"),
              ("terrain-scale", "terrain-scale-relative", "agent2_hybrid_scale")]
    records = []
    for policy in ["observed-only", "train-median"]:
        for absolute, relative, model in pairs:
            record = {"missing_policy": policy, "absolute": absolute, "relative": relative, "model": model}
            for kind, table in [("oof", oof), ("forecast", forecast)]:
                chosen = table.loc[table.missing_policy.eq(policy) & table.model.eq(model)]
                a = chosen.loc[chosen.variant.eq(absolute)].set_index("sample_id").sort_index()
                b = chosen.loc[chosen.variant.eq(relative)].set_index("sample_id").sort_index()
                if a.empty or not a.index.is_unique or not b.index.is_unique or not a.index.equals(b.index):
                    raise ValueError("Nominal hypotheses require identical complete sample keys")
                gap = (a.raw_prediction_mm - b.raw_prediction_mm).abs()
                if not gap.notna().all():
                    raise ValueError("Missing paired predictions")
                record[f"{kind}_max_gap_mm"] = float(gap.max())
                record[f"{kind}_mean_gap_mm"] = float(gap.mean())
            records.append(record)
    return pd.DataFrame(records)


def write_assessment(output):
    """Summarize saved shared scores; never refit or change the candidates."""
    output = Path(output)
    assessment = assess_candidates(pd.read_csv(output / "metrics.csv"), pd.read_csv(output / "metrics_by_fold.csv"))
    assessment.to_csv(output / "candidate_assessment.csv", index=False)
    pairs = nominal_identifiability(pd.read_csv(output / "oof_predictions.csv.gz"),
                                    pd.read_csv(output / "forecast_predictions.csv.gz"))
    pairs.to_csv(output / "nominal_identifiability.csv", index=False)
    write_json(output / "research_conclusion.json", {
        "protocol": "CASE2_SEARCH_PROTOCOL.md", "primary_missing_policy": "observed-only",
        "auxiliary_gate_passes": int(assessment.passes_auxiliary_gate.sum()),
        "nominal_hypotheses_oof_identical": bool(pairs.oof_max_gap_mm.eq(0).all()),
        "target_transfer_validated": False, "independent_holdout": False,
        "reason": "Only shared MAIN-2 2018 spatial folds; MAIN-4 target has no labels",
    })
    return assessment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    provenance = run(args.prepared, args.output, configs=search_configs(), case_ids=["case2"])
    assessment = write_assessment(args.output)
    print(f"Completed {len(provenance['runs'])} runs; auxiliary gate passes: "
          f"{assessment.passes_auxiliary_gate.sum()}; target transfer remains unvalidated.")


if __name__ == "__main__":
    main()
