"""Validate fixed RF/XGB candidates and export reproducible case 2 files."""

import argparse
from hashlib import sha256
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import LeaveOneGroupOut
from threadpoolctl import threadpool_limits

from pipetwin.case_experiments import prediction_table
from pipetwin.pipeline import write_json
from pipetwin.quality import read_raw_sources

from .broad_case2 import Candidate, TabularCandidate, decode, encode, fit, predict_candidate, training_rows
from .broad_case2_features import add_features
from .case2_followup import fit_trial
from .case2_followup_models import boosted_estimator
from .retrospective_case2 import campaign_metrics, compare_query_inputs, prepare, validate_group_split
from .shared_comparison import verify_shared_predictions


SEEDS = (42, 123, 2024)
MIXTURES = {"rf50_xgb50": 0.5, "rf60_xgb40": 0.6}


def spec(kind, seed):
    return Candidate(kind, kind, "position", "log_rate", "balanced", seed=seed)


def held_campaigns(frame, *, purge):
    """Keep date IDs distinct and audit repeated observed cells across dates."""
    groups = frame.main_id + "|" + frame.target_date.dt.strftime("%Y-%m-%d")
    for number, (a, b) in enumerate(LeaveOneGroupOut().split(frame, groups=groups)):
        train, valid = frame.iloc[a], frame.iloc[b]
        valid_keys = pd.MultiIndex.from_frame(valid.loc[valid.observed, ["id", "segment_id"]])
        matches = pd.MultiIndex.from_frame(train[["id", "segment_id"]]).isin(valid_keys)
        removed = int((matches & train.observed.to_numpy()).sum())
        if purge:
            train = train.loc[~matches]
        if not train.observed.any() or not valid.observed.any():
            raise ValueError("Campaign holdout needs observed training and validation labels")
        yield number, train, valid, removed


def quantile_trial(train, groups):
    rows, weights = training_rows(train, "balanced")
    candidate = spec("xgb15", 42)
    columns = groups["position"]
    # Thickness decreases monotonically with log-rate. The 90th percentile
    # of rate corresponds to the 10th percentile of thickness.
    model = boosted_estimator("xgb15", 42).set_params(objective="reg:quantileerror", quantile_alpha=0.9)
    model.fit(rows[columns], encode(rows, candidate.response), sample_weight=weights)
    return TabularCandidate(candidate, columns, model)


def isotonic_trial(package, train, valid, groups, fitted):
    rows, weights = training_rows(train, "balanced")
    # Use internal section holdouts so a later inspection of the same segment
    # cannot calibrate a score previously fitted on that segment.
    oof_rate = np.full(len(rows), np.nan)
    for a, b in LeaveOneGroupOut().split(rows, groups=rows.id):
        inner = fit_trial(spec("xgb15", 42), package, rows.iloc[a], groups)
        oof_rate[b] = inner.estimator.predict(rows.iloc[b][inner.columns])
    if not np.isfinite(oof_rate).all():
        raise ValueError("Incomplete training-only calibration predictions")
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(
        oof_rate, encode(rows, "log_rate"), sample_weight=weights)
    rates = fitted.estimator.predict(valid[fitted.columns])
    return decode(isotonic.predict(rates), valid, "log_rate")


def evaluate(package, groups):
    frame = package["train_frame"]
    metrics, predictions, audit = [], [], []
    for fold in package["folds"]:
        number = fold["number"]
        train, valid = frame.iloc[fold["train"]], frame.iloc[fold["valid"]]
        validate_group_split(train, valid)
        print(f"Whole-main checks: {valid.main_id.iloc[0]}", flush=True)
        raw, bounded = {}, {}
        for seed in SEEDS:
            for kind, tag in [("forest", "rf"), ("xgb15", "xgb")]:
                model = fit_trial(spec(kind, seed), package, train, groups)
                name = f"{tag}_{seed}"
                r, b = predict_candidate(name, model, valid)
                raw.update(r)
                bounded.update(b)
                if name == "xgb_42":
                    calibrated = isotonic_trial(package, train, valid, groups, model)
                    raw["xgb_isotonic"] = calibrated
                    bounded["xgb_isotonic"] = np.clip(calibrated, 0, valid.nominal_mm.to_numpy())
        for name, fraction in MIXTURES.items():
            for values in [raw, bounded]:
                values[name] = fraction * values["rf_42"] + (1 - fraction) * values["xgb_42"]
        fitted = {"xgb_quantile10": quantile_trial(train, groups),
                  "nominal": fit(Candidate("reference_persistence", "persistence"), package, train, groups),
                  "global_rate": fit(Candidate("reference_global_rate", "global_rate"), package, train, groups)}
        for name, model in fitted.items():
            r, b = predict_candidate(name, model, valid)
            raw.update(r)
            bounded.update(b)
        metrics.append(campaign_metrics(valid, raw, bounded, number).assign(protocol="whole_main"))
        predictions.append(prediction_table(valid, raw, bounded, number))
    oof = pd.concat(predictions, ignore_index=True)
    verify_shared_predictions(package, oof)
    for purge in [False, True]:
        protocol = "campaign_purged" if purge else "campaign"
        for number, train, valid, repeated in held_campaigns(frame, purge=purge):
            raw, bounded = {}, {}
            for kind, name in [("forest", "rf_42"), ("xgb15", "xgb_42")]:
                model = fit_trial(spec(kind, 42), package, train, groups)
                r, b = predict_candidate(name, model, valid)
                raw.update(r)
                bounded.update(b)
            scores = campaign_metrics(valid, raw, bounded, number).assign(protocol=protocol)
            metrics.append(scores)
            audit.append({"protocol": protocol, "held_main": valid.main_id.iloc[0],
                          "held_date": valid.target_date.iloc[0], "train_labels": int(train.observed.sum()),
                          "held_labels": int(valid.observed.sum()), "shared_labels_before_purge": repeated,
                          "shared_labels_in_fit": 0 if purge else repeated})
    return pd.concat(metrics, ignore_index=True), oof, pd.DataFrame(audit)


def aggregate(scores):
    metrics = ["pearson_r", "pearson_squared", "mae_mm", "rmse_mm", "recall_thin_10pct"]
    campaign = scores.loc[scores.output.eq("bounded") & scores.protocol.eq("whole_main")]
    grouped = campaign.groupby(["model", "held_main"])[metrics].mean()
    return grouped.groupby("model").mean()


def seed_summary(averages):
    rows = []
    for seed in SEEDS:
        rf, xgb = (float(averages.loc[f"{model}_{seed}", "pearson_squared"]) for model in ["rf", "xgb"])
        rows.append({"seed": seed, "rf_r2": rf, "xgb_r2": xgb, "delta_r2": xgb - rf})
    table = pd.DataFrame(rows)
    values = table.drop(columns="seed")
    mean, std = values.mean(), values.std(ddof=1)
    passed = bool(mean.delta_r2 > 0 and std.delta_r2 < mean.delta_r2 / 2)
    return table, {"mean": mean.to_dict(), "sample_std": std.to_dict(), "passes": passed}


def export_and_verify(query, original, output):
    paths = {"rf": "forest_log_rate_42.joblib", "xgb": "xgb15_log_rate_42.joblib"}
    profiles = {}
    # The new run audits the exact release candidates already frozen at a27fc83.
    for name, path in paths.items():
        model = joblib.load(original / "models" / path)
        raw, bounded = predict_candidate(name, model, query)
        frame = query[["id", "segment_id"]].assign(thickness_pred=bounded[name])
        profiles[name] = frame
        if len(frame) != 1475 or not np.array_equal(frame.segment_id, np.arange(1475)):
            raise ValueError("Unexpected case 2 export grid")
        filename = "result.csv" if name == "xgb" else "rf_full_profile.csv"
        path = output / filename
        frame.to_csv(path, index=False, float_format="%.9f")
        replay = pd.read_csv(path, dtype={"id": str})
        pd.testing.assert_frame_equal(replay[["id", "segment_id"]], frame[["id", "segment_id"]])
        np.testing.assert_allclose(replay.thickness_pred, frame.thickness_pred, rtol=0, atol=5.1e-10)
        old_file = "case2_correlation_candidate.csv" if name == "xgb" else "case2_candidate.csv"
        old = pd.read_csv(original / old_file, dtype={"id": str})
        pd.testing.assert_frame_equal(old[["id", "segment_id"]], frame[["id", "segment_id"]])
        np.testing.assert_allclose(old.thickness_pred, bounded[name], rtol=1e-12, atol=1e-12)
        if name == "rf":
            count = int(np.ceil(0.1 * len(frame)))
            priority = frame.sort_values(["thickness_pred", "segment_id"]).head(count)
            priority_path = output / "dangerous_segments.csv"
            priority.to_csv(priority_path, index=False, float_format="%.9f")
            reloaded = pd.read_csv(priority_path, dtype={"id": str})
            pd.testing.assert_frame_equal(reloaded[["id", "segment_id"]],
                                          priority[["id", "segment_id"]].reset_index(drop=True))
            np.testing.assert_allclose(reloaded.thickness_pred, priority.thickness_pred, rtol=0, atol=5.1e-10)
        else:
            nominal = query.nominal_mm.to_numpy()
            below, above = raw[name] < 0.5 * nominal, raw[name] > 1.05 * nominal
            shift = {"min_mm": float(raw[name].min()), "max_mm": float(raw[name].max()),
                     "below_half_nominal": int(below.sum()), "above_1p05_nominal": int(above.sum()),
                     "changed_by_shared_clipping": int(np.count_nonzero(raw[name] != bounded[name])),
                     "max_csv_replay_error_mm": float(np.max(np.abs(replay.thickness_pred - bounded[name])))}
    for name, fraction in MIXTURES.items():
        mixture = fraction * profiles["rf"].thickness_pred + (1 - fraction) * profiles["xgb"].thickness_pred
        query[["id", "segment_id"]].assign(thickness_pred=mixture).to_csv(
            output / f"{name}.csv", index=False, float_format="%.9f")
    return shift


def run(output, strict_reference, original):
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
    with tempfile.TemporaryDirectory(prefix=".case2-check-", dir=output.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        write_json(staging / "protocol.json", {"seeds": SEEDS, "mixtures_rf_weight": MIXTURES,
                   "fit_feature_names": groups["position"], "query_invariance": query_check,
                   "source_manifest": manifest, "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                   "pool": "equal_campaign_weight", "seed_gate": "sample_sd_delta < mean_delta / 2"})
        with threadpool_limits(limits=1):
            scores, predictions, audit = evaluate(package, groups)
            shift = export_and_verify(package["predict_frame"], original, staging)
        scores.to_csv(staging / "metrics_by_campaign.csv", index=False)
        predictions.to_csv(staging / "whole_main_predictions.csv.gz", index=False)
        audit.to_csv(staging / "campaign_split_audit.csv", index=False)
        averages = aggregate(scores)
        averages.to_csv(staging / "whole_main_summary.csv")
        seed_table, seed_gate = seed_summary(averages)
        seed_table.to_csv(staging / "seed_comparison.csv", index=False)
        win_counts = {}
        for protocol in scores.protocol.unique():
            rows = scores.loc[scores.protocol.eq(protocol) & scores.output.eq("bounded")]
            r2 = rows.pivot(index=["held_main", "target_date"], columns="model", values="pearson_squared")
            win_counts[protocol] = int(r2.xgb_42.gt(r2.rf_42).sum())
        write_json(staging / "summary.json", {"seed_gate": seed_gate, "xgb_wins_out_of_four": win_counts,
                   "main4_shift_and_replay": shift, "hidden_main4_vtd_available": False})
        staging.rename(output)
    print(averages.round(6).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-reference", required=True, type=Path)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.output, args.strict_reference, args.original)
