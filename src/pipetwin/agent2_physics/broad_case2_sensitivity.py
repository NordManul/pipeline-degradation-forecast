"""Sensitivity check for the two Random Forest candidates after the search."""

import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from pipetwin.case_experiments import prediction_table, range_diagnostics
from pipetwin.pipeline import write_json
from pipetwin.quality import read_raw_sources

from .broad_case2 import Candidate, evaluate, fit, predict_candidate, ranking
from .broad_case2_features import add_features
from .retrospective_case2 import compare_query_inputs, prepare
from .shared_comparison import verify_shared_predictions


def run(output, strict_reference):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    specs = [Candidate("reference_catboost", "catboost")]
    specs += [Candidate(f"forest_{response}_{pool}_seed{seed}", "forest", "position", response, pool, seed=seed)
              for response, pool, seed in product(["fraction_rate", "log_rate"],
                                                  ["all", "first", "last", "balanced"], [7, 42, 123])]
    write_json(output / "protocol.json", {"purpose": __doc__, "configurations": [asdict(s) for s in specs],
               "status": "post_search_sensitivity_same_mains"})
    raw, manifest = read_raw_sources("data/raw")
    package = prepare(raw, manifest)
    del raw
    compare_query_inputs(package, joblib.load(strict_reference))
    package["train_frame"], groups = add_features(package["train_frame"], package["exposure"])
    package["predict_frame"], _ = add_features(package["predict_frame"], package["exposure"])
    with threadpool_limits(limits=1):
        metrics, _, timings = evaluate(specs, package, groups, blends={})
    metrics.to_csv(output / "metrics.csv", index=False)
    board = ranking(metrics, {t["model"] for t in timings if t["convergence_warning"]})
    board.to_csv(output / "leaderboard.csv", index=False)
    # Preserve the existing seed and save both balanced responses. Do not
    # replace the fixed seed by whichever run scored highest.
    finalists = [s for s in specs if s.pool == "balanced" and s.seed == 42]
    query = package["predict_frame"]
    raw, bounded = {}, {}
    with threadpool_limits(limits=1):
        for spec in finalists:
            model = fit(spec, package, package["train_frame"], groups)
            r, b = predict_candidate(spec.name, model, query)
            path = output / f"{spec.name}.joblib"
            joblib.dump(model, path, compress=3)
            rr, bb = predict_candidate(spec.name, joblib.load(path), query)
            np.testing.assert_allclose(rr[spec.name], r[spec.name], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(bb[spec.name], b[spec.name], rtol=1e-12, atol=1e-12)
            raw.update(r)
            bounded.update(b)
    table = prediction_table(query, raw, bounded, -1)
    verify_shared_predictions(package, table, forecast=True)
    table.to_csv(output / "candidate_forecasts.csv.gz", index=False)
    candidate = table.loc[table.model.eq("forest_log_rate_balanced_seed42"),
                          ["id", "segment_id", "prediction_mm"]]
    candidate = candidate.rename(columns={"prediction_mm": "thickness_pred"}).sort_values("segment_id")
    candidate.to_csv(output / "case2_candidate.csv", index=False)
    range_diagnostics(query, raw, bounded).to_csv(output / "forecast_ranges.csv", index=False)
    query.to_csv(output / "query_features.csv.gz", index=False)
    write_json(output / "summary.json", {"completed": True, "target_vtd_available": False,
               "fits_for_evaluation": len(timings), "replayed_models": [s.name for s in finalists]})
    print(board.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strict-reference", required=True, type=Path)
    args = parser.parse_args()
    run(args.output, args.strict_reference)
