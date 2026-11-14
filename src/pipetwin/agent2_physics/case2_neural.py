"""Evaluate compact spatial sequence models on whole-main case 2 holdouts."""

import argparse
from hashlib import sha256
from itertools import product
from pathlib import Path
import tempfile
from time import perf_counter

import torch

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from pipetwin.case_experiments import prediction_table, range_diagnostics
from pipetwin.pipeline import write_json
from pipetwin.quality import read_raw_sources

from .broad_case2 import TabularCandidate, encode, predict_candidate, training_rows
from .case2_followup import fit_trial
from .case2_followup_models import boosted_estimator
from .case2_neural_models import NeuralProfile, STATIC, attach_windows, window_columns
from .case2_release_checks import SEEDS, aggregate, spec
from .retrospective_case2 import campaign_metrics, compare_query_inputs, prepare, validate_group_split
from .shared_comparison import verify_shared_predictions


RADII = [100, 250]
ARCHITECTURES = ["mlp", "cnn", "transformer"]
XGB_WEIGHT = 0.75


def fit_models(package, frame, seed):
    groups = {"position": STATIC}
    for kind, name in [("forest", "rf"), ("xgb15", "xgb")]:
        yield name, fit_trial(spec(kind, seed), package, frame, groups)
    measured, weights = training_rows(frame, "balanced")
    y = encode(measured, "log_rate")
    for radius in RADII:
        columns = STATIC + window_columns(radius)
        estimator = boosted_estimator("xgb15", seed).fit(measured[columns], y, sample_weight=weights)
        yield f"xgb_window{radius}", TabularCandidate(spec("xgb15", seed), columns, estimator)
        for architecture in ARCHITECTURES:
            yield f"{architecture}{radius}", NeuralProfile(architecture, radius, seed).fit(frame)


def predictions_for(package, train, query, seed):
    raw, bounded = {}, {}
    for name, model in fit_models(package, train, seed):
        prediction, clipped = predict_candidate(name, model, query)
        raw.update(prediction)
        bounded.update(clipped)
    for name in list(raw):
        if name.startswith(tuple(ARCHITECTURES)):
            for values in [raw, bounded]:
                values["mix_" + name] = XGB_WEIGHT * values["xgb"] + (1 - XGB_WEIGHT) * values[name]
    return raw, bounded


def summarize(metrics):
    summaries = []
    for seed, rows in metrics.groupby("seed"):
        scores = aggregate(rows)
        bounded = rows.loc[rows.output.eq("bounded")]
        mains = bounded.groupby(["model", "held_main"]).pearson_squared.mean().unstack()
        minimum = bounded.groupby("model").pearson_r.min()
        row_count = bounded.groupby("model").size()
        positive = bounded.groupby("model").pearson_r.agg(lambda values: values.gt(0).sum())
        summary = scores.assign(seed=seed, min_campaign_r=minimum, positive_campaigns=positive,
                                campaigns=row_count, main2_r2=mains["MAIN-2"], main3_r2=mains["MAIN-3"])
        summary["r2_gain_vs_xgb"] = summary.pearson_squared - scores.loc["xgb", "pearson_squared"]
        summaries.append(summary.reset_index())
    board = pd.concat(summaries, ignore_index=True)
    wins = board.assign(win=board.r2_gain_vs_xgb.gt(0)).groupby("model").win.sum()
    board["improved_seeds"] = board.model.map(wins)
    fixed = board.loc[board.seed.eq(42)].set_index("model")
    ref = fixed.loc["xgb"]
    passing = fixed.loc[
        fixed.positive_campaigns.eq(4)
        & fixed.main2_r2.gt(ref.main2_r2)
        & fixed.main3_r2.gt(ref.main3_r2)
        & fixed.improved_seeds.ge(2)
        & fixed.mae_mm.le(ref.mae_mm + 0.02)
    ].sort_values("pearson_squared", ascending=False)
    return board, passing.index.tolist()


def run(output, strict_reference):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    raw_sources, manifest = read_raw_sources("data/raw")
    package = prepare(raw_sources, manifest)
    del raw_sources
    check = compare_query_inputs(package, joblib.load(strict_reference))
    train = attach_windows(package["train_frame"], RADII)
    query = attach_windows(package["predict_frame"], RADII)
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix=".case2-neural-", dir=output.parent) as directory:
        staging = Path(directory) / "run"
        staging.mkdir()
        code = [Path(__file__), Path(__file__).with_name("case2_neural_models.py"),
                Path(__file__).with_name("CASE2_NEURAL_PROTOCOL.md")]
        write_json(staging / "protocol.json", {"seeds": SEEDS, "radii_m": RADII, "architectures": ARCHITECTURES,
                   "fixed_blend_xgb_weight": XGB_WEIGHT, "input_manifest": manifest, "strict_query_match": check,
                   "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in code},
                   "torch_version": torch.__version__, "training_rows": int(train.observed.sum())})
        metrics, predictions = [], []
        with threadpool_limits(limits=1):
            for fold, seed in product(package["folds"], SEEDS):
                number = fold["number"]
                fit_frame, valid = train.iloc[fold["train"]], train.iloc[fold["valid"]]
                validate_group_split(fit_frame, valid)
                raw, bounded = predictions_for(package, fit_frame, valid, seed)
                metrics.append(campaign_metrics(valid, raw, bounded, number).assign(seed=seed, protocol="whole_main"))
                predictions.append(prediction_table(valid, raw, bounded, number).assign(seed=seed))
                print(f"Held {valid.main_id.iloc[0]}, seed {seed}: {len(raw)} models", flush=True)
            metrics = pd.concat(metrics, ignore_index=True)
            oof = pd.concat(predictions, ignore_index=True)
            for _, part in oof.groupby("seed"):
                verify_shared_predictions(package, part)
            board, passing = summarize(metrics)
            fixed = board.loc[board.seed.eq(42)]
            best_neural = fixed.loc[fixed.model.str.startswith(tuple(ARCHITECTURES))].sort_values(
                "pearson_squared", ascending=False).model.iloc[0]
            winner = passing[0] if passing else "xgb"
            names = list(dict.fromkeys(["xgb", best_neural, winner]))
            required = set(names)
            required.add(winner.removeprefix("mix_"))
            models = {}
            for name, model in fit_models(package, train, 42):
                if name in required:
                    models[name] = model
            final_raw, final_bounded = {}, {}
            for name in names:
                if name.startswith("mix_"):
                    mix = {key: models[key] for key in ["xgb", name.removeprefix("mix_")]}
                    candidate = mix
                else:
                    candidate = models[name]
                path = staging / (name + ".joblib")
                joblib.dump(candidate, path, compress=3)
                r, b = replay_candidate(name, candidate, query)
                rr, bb = replay_candidate(name, joblib.load(path), query)
                np.testing.assert_allclose(r, rr, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(b, bb, atol=1e-12, rtol=1e-12)
                final_raw[name], final_bounded[name] = r, b
            forecast = prediction_table(query, final_raw, final_bounded, -1)
            verify_shared_predictions(package, forecast, forecast=True)
            forecast.to_csv(staging / "forecast.csv.gz", index=False)
            selected = forecast.loc[forecast.model.eq(winner), ["id", "segment_id", "prediction_mm"]]
            selected.rename(columns={"prediction_mm": "thickness_pred"}).to_csv(staging / "case2_candidate.csv", index=False)
            query.to_csv(staging / "query_features.csv.gz", index=False)
            range_diagnostics(query, final_raw, final_bounded).to_csv(staging / "forecast_ranges.csv", index=False)
        metrics.to_csv(staging / "metrics_by_campaign.csv", index=False)
        board.to_csv(staging / "leaderboard_by_seed.csv", index=False)
        oof.to_csv(staging / "whole_main_predictions.csv.gz", index=False)
        write_json(staging / "summary.json", {"recommended_candidate": winner, "passes_replacement_gate": passing,
                   "best_standalone_neural_at_42": best_neural, "query_cells": len(query),
                   "elapsed_seconds": perf_counter() - started, "models_per_seed": len(board) // len(SEEDS),
                   "official_target_metrics_available": False})
        staging.rename(output)
    print(fixed.sort_values("pearson_squared", ascending=False).round(5).to_string(index=False))


def replay_candidate(name, model, query):
    if not isinstance(model, dict):
        raw, bounded = predict_candidate(name, model, query)
        return raw[name], bounded[name]
    neural_name = name.removeprefix("mix_")
    raw_base, bound_base = predict_candidate("xgb", model["xgb"], query)
    raw_neural, bound_neural = predict_candidate(neural_name, model[neural_name], query)
    return tuple(XGB_WEIGHT * base["xgb"] + (1 - XGB_WEIGHT) * neural[neural_name]
                 for base, neural in [(raw_base, raw_neural), (bound_base, bound_neural)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.output, args.strict_reference)
