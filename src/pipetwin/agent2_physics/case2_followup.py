"""Fixed comparison of the proposed case 2 methods on held-out whole mains."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import importlib.metadata
from itertools import product
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from pipetwin.case_experiments import prediction_table, range_diagnostics
from pipetwin.pipeline import write_json
from pipetwin.quality import read_raw_sources

from .broad_case2 import Candidate, TabularCandidate, encode, fit, predict_candidate, ranking, training_rows
from .broad_case2_features import add_features
from .case2_followup_models import (PairwiseRank, PooledZones, SmoothProfile, boosted_estimator,
                                   continued_boosting, fit_correction)
from .retrospective_case2 import campaign_metrics, compare_query_inputs, prepare, validate_group_split
from .shared_comparison import verify_shared_predictions


SEEDS = (7, 42, 123)
RATES = ("log_rate", "mm_rate", "fraction_rate")
MIX = {"forest": 0.4, "xgb7": 0.3, "catboost_rmse": 0.3}


def make_specs():
    trials = [Candidate(f"{kind}_{response}_{seed}", kind, "position", response, "balanced", seed=seed)
              for kind, response, seed in product(
                  ["forest", "xgb7", "xgb15", "catboost_rmse"], RATES, SEEDS)]
    trials += [Candidate("reference_catboost", "catboost")]
    trials += [Candidate(f"{kind}_{block}", kind, block, "log_rate", "balanced")
               for kind, block in product(["pooled", "rank"], ["position", "canonical"])]
    trials += [Candidate(f"{kind}_log_rate_{seed}", kind, "position", "log_rate", "balanced", seed=seed)
               for kind, seed in product(["continued_catboost", "additive_physics"], SEEDS)]
    trials += [Candidate(kind, kind, "position", "log_rate", "balanced")
               for kind in ["gp_short", "gp_long"]]
    trials += [Candidate("physics_prior", "agent2_positive_rate")]
    return trials


def fit_trial(spec, package, frame, groups):
    kind = spec.learner
    if kind in {"forest", "catboost", "agent2_positive_rate"}:
        return fit(spec, package, frame, groups)
    train, weights = training_rows(frame, spec.pool)
    columns = groups[spec.features]
    y = encode(train, spec.response)
    if kind.startswith("gp"):
        return SmoothProfile(spec, columns, 0.1 if kind == "gp_short" else 0.3).fit(train, y, weights)
    if kind == "pooled":
        return PooledZones(spec, columns).fit(train, y, weights)
    if kind == "rank":
        return PairwiseRank(spec, columns).fit(train, y, weights)
    if kind == "continued_catboost":
        return continued_boosting(spec, train, columns, y, weights)
    if kind == "additive_physics":
        prior = fit(Candidate("physics_prior", "agent2_positive_rate"), package, frame, groups)
        return fit_correction(prior, train, columns, spec.seed, weights)
    model = boosted_estimator(kind, spec.seed).fit(train[columns], y, sample_weight=weights)
    return TabularCandidate(spec, columns, model)


def diagnostics(frame, bounded, fitted):
    rows = []
    intervals = {name: model.interval(frame) for name, model in fitted.items() if hasattr(model, "interval")}
    for (main, date), indices in frame.groupby(["main_id", "target_date"]).indices.items():
        measured = indices[frame.observed.to_numpy()[indices]]
        y = frame.y_mm.to_numpy()[measured]
        nominal = frame.nominal_mm.to_numpy()
        for name, prediction in bounded.items():
            r = spearmanr(y, prediction[measured]).statistic if np.ptp(prediction[measured]) > 0 else np.nan
            # Slope changes are assessed within one section and one campaign.
            profile = frame.iloc[indices][["id", "segment_id"]].assign(prediction=prediction[indices])
            step = profile.sort_values(["id", "segment_id"]).groupby("id").prediction.diff().abs()
            row = {"held_main": main, "target_date": date, "model": name, "spearman": r,
                   "adjacent_mean_abs_step_mm": float(step.mean())}
            if name in intervals:
                lo, hi = (np.clip(x, 0, nominal)[measured] for x in intervals[name])
                row["p10_p90_coverage"] = float(np.mean((y >= lo) & (y <= hi)))
                row["p10_p90_mean_width_mm"] = float(np.mean(hi - lo))
            rows.append(row)
    return pd.DataFrame(rows)


def run(output, strict_reference):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw, manifest = read_raw_sources("data/raw")
    package = prepare(raw, manifest)
    del raw
    query_check = compare_query_inputs(package, joblib.load(strict_reference))
    for key in ["train_frame", "predict_frame"]:
        package[key], groups = add_features(package[key], package["exposure"])
    train, query = package["train_frame"], package["predict_frame"]
    specs = make_specs()
    lookup = {s.name: s for s in specs}
    controls = ["forest_log_rate_42", "reference_catboost"]
    with tempfile.TemporaryDirectory(prefix=".case2-followup-", dir=output.parent) as directory:
        staging = Path(directory) / "run"
        staging.mkdir()
        write_json(staging / "protocol.json", {"configs": [asdict(s) for s in specs], "mix": MIX,
                   "sources": manifest, "query_check": query_check,
                   "code_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ["case2_followup.py", "case2_followup_models.py"]},
                   "status": "exploratory_same_two_held_mains",
                   "versions": {p: importlib.metadata.version(p) for p in ["xgboost", "catboost", "scikit-learn"]},
                   "train_histories": int(train.exposure_id.nunique()),
                   "held_train_histories": [int(train.iloc[f["train"]].exposure_id.nunique()) for f in package["folds"]],
                   "state_date": package["target_state_date"]})
        metrics, extra, oof = [], [], []
        with threadpool_limits(limits=1):
            for fold in package["folds"]:
                tr, va = train.iloc[fold["train"]], train.iloc[fold["valid"]]
                validate_group_split(tr, va)
                raw, bounded, interval_models = {}, {}, {}
                for spec in specs:
                    model = fit_trial(spec, package, tr, groups)
                    r, b = predict_candidate(spec.name, model, va, development=True)
                    raw.update(r)
                    bounded.update(b)
                    if hasattr(model, "interval"):
                        interval_models[spec.name] = model
                for response, seed in product(RATES, SEEDS):
                    name = f"mix_{response}_{seed}"
                    for values in [raw, bounded]:
                        values[name] = sum(weight * values[f"{kind}_{response}_{seed}"] for kind, weight in MIX.items())
                metrics.append(campaign_metrics(va, raw, bounded, fold["number"]))
                extra.append(diagnostics(va, bounded, interval_models))
                oof.append(prediction_table(va, raw, bounded, fold["number"]))
                print(f"Held {va.main_id.iloc[0]}: {len(raw)} configurations", flush=True)
            scores = pd.concat(metrics, ignore_index=True)
            board = ranking(scores, set())
            predictions = pd.concat(oof, ignore_index=True)
            verify_shared_predictions(package, predictions)
            scores.to_csv(staging / "metrics.csv", index=False)
            pd.concat(extra, ignore_index=True).to_csv(staging / "diagnostics.csv", index=False)
            board.to_csv(staging / "leaderboard.csv", index=False)
            predictions.to_csv(staging / "oof_predictions.csv.gz", index=False)
            # Keep the highest r joint improvement over the current RF at the
            # original seed. Other seeds measure sensitivity only.
            ref = board.set_index("model").loc[controls[0]]
            fixed_seed = board.loc[~board.model.str.endswith(("_7", "_123"))]
            eligible = fixed_seed.loc[
                fixed_seed.positive_campaigns.eq(fixed_seed.campaigns)
                & fixed_seed.mean_main_r.gt(ref.mean_main_r)
                & fixed_seed.mean_main_mae.lt(ref.mean_main_mae)
                & fixed_seed.mean_main_recall.ge(ref.mean_main_recall)
            ].sort_values(["mean_main_r", "mean_main_mae"], ascending=[False, True])
            selected = controls[0] if eligible.empty else eligible.iloc[0].model
            correlation_leader = fixed_seed.sort_values("mean_main_r", ascending=False).iloc[0].model
            write_json(staging / "selection.json", {"candidate": selected, "rf_reference": controls[0],
                       "correlation_leader_at_seed_42": correlation_leader,
                       "joint_improvements": eligible.model.tolist(), "target_accuracy_measured": False})
            (staging / "models").mkdir()
            final_raw, final_bounded = {}, {}
            for name in dict.fromkeys([*controls, selected, correlation_leader, "mix_log_rate_42"]):
                if name.startswith("mix_"):
                    suffix = name.removeprefix("mix_")
                    parts = {k: fit_trial(lookup[f"{k}_{suffix}"], package, train, groups) for k in MIX}
                    models = {name: parts}
                else:
                    models = {name: fit_trial(lookup[name], package, train, groups)}
                path = staging / "models" / f"{name}.joblib"
                joblib.dump(models[name], path, compress=3)
                replay = joblib.load(path)
                if name.startswith("mix_"):
                    r, b = mix_prediction(models[name], query), mix_prediction(models[name], query, bounded=True)
                    np.testing.assert_allclose(r, mix_prediction(replay, query), rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(b, mix_prediction(replay, query, bounded=True), rtol=1e-12, atol=1e-12)
                else:
                    original = predict_candidate(name, models[name], query)
                    restored = predict_candidate(name, replay, query)
                    r, b = [item[name] for item in original]
                    for before, after in zip(original, restored):
                        np.testing.assert_allclose(before[name], after[name], rtol=1e-12, atol=1e-12)
                final_raw[name], final_bounded[name] = r, b
            forecast = prediction_table(query, final_raw, final_bounded, -1)
            verify_shared_predictions(package, forecast, forecast=True)
            forecast.to_csv(staging / "forecasts.csv.gz", index=False)
            for name, filename in [(selected, "case2_candidate.csv"),
                                   (correlation_leader, "case2_correlation_candidate.csv")]:
                candidate = forecast.loc[forecast.model.eq(name), ["id", "segment_id", "prediction_mm"]]
                candidate.rename(columns={"prediction_mm": "thickness_pred"}).to_csv(staging / filename, index=False)
            query.to_csv(staging / "query_features.csv.gz", index=False)
            range_diagnostics(query, final_raw, final_bounded).to_csv(staging / "forecast_ranges.csv", index=False)
            write_json(staging / "summary.json", {"completed": True, "configurations": len(board),
                       "fits": len(specs) * len(package["folds"]), "forecast_cells": len(query),
                       "candidate": selected, "hidden_target_available": False})
        staging.rename(output)
    print(board.sort_values("mean_main_r", ascending=False).head(12).to_string(index=False))


def mix_prediction(models, frame, *, bounded=False):
    combined = np.zeros(len(frame))
    for key, model in models.items():
        raw, clipped = predict_candidate(key, model, frame)
        combined += MIX[key] * (clipped if bounded else raw)[key]
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.output, args.strict_reference)
