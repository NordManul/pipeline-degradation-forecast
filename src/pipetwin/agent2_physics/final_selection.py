"""Compare case 2 candidates with the pinned main release and build a combined CSV."""

import argparse
from dataclasses import replace
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from pipetwin.case2_data import prepare_transfer
from pipetwin.case2_transfer import ColdTabular, TABULAR, weights_by_campaign
from pipetwin.pipeline import write_json
from pipetwin.quality import read_raw_sources
from pipetwin.sources import read_config
from pipetwin.submission import KEYS, export_submission, validate_submission

from .broad_case2 import AlignedProfileBlend, Candidate, TabularCandidate, encode, fit, training_rows
from .case2_followup import fit_trial
from .case2_followup_models import boosted_estimator
from .case2_release_checks import SEEDS, spec
from .case2_transfer_shape import POSITION, REFERENCE, from_models, score_summary
from .retrospective_case2 import campaign_metrics, compare_query_inputs, validate_group_split


MAIN_COMMIT = "e84586b"
MAIN_RESULT_SHA256 = "ce83e210e43f72ba5acee9a152c2c78cea62f20b072f0ab85688370f01f4dc51"
MAIN_METRICS_SHA256 = "806063f51da414af44cf30402155f32d1121ae0c7f641474f56c02297eeca8a8"
CATBOOST = "catboost_main"
KNN = "knn_direct"
TREE_COUNTS = {"trees80": 80, "trees240": 240, "trees320": 320}
ROBUST_TRIALS = {"cap_1": .01, "cap_5": .05, "mad_weight": None, "thin_weight": None, "repeat_weight": None}
PROFILE_GROUPS = {"section": ["exposure_id"], "campaign": ["main_id", "target_date"]}
PROFILE_MIXES = {f"knn_{scope}{int(weight * 100)}": (columns, weight)
                 for (scope, columns), weight in product(PROFILE_GROUPS.items(), [.25, .5])}


def robust_training(rows, weights, trial):
    response = pd.Series(encode(rows, "log_rate"), index=rows.index)
    campaign = [rows.main_id, rows.target_date]
    by_campaign = response.groupby(campaign)
    q = ROBUST_TRIALS[trial]
    if q is not None:
        return response.clip(by_campaign.transform("quantile", q=q),
                             by_campaign.transform("quantile", q=1 - q)), weights
    if trial == "mad_weight":
        distance = (response - by_campaign.transform("median")).abs()
        mad = distance.groupby(campaign).transform("median")
        ratio = (distance / (2.5 * 1.4826 * mad.clip(lower=1e-6))).clip(lower=1.)
        relative = 1. / ratio
    elif trial == "thin_weight":
        fraction = rows.y_mm / rows.nominal_mm
        threshold = fraction.groupby(campaign).transform("quantile", q=.1)
        relative = 1. + fraction.le(threshold).astype(float)
    else:
        repeats = rows.groupby(KEYS).sample_id.transform("size")
        relative = 1. / repeats
    return response, weights * (relative / relative.groupby(campaign).transform("mean")).to_numpy()


def fit_models(frame, seed, requested=None):
    fixed = {REFERENCE, "forest", KNN, *PROFILE_MIXES}
    modified = {*ROBUST_TRIALS, *TREE_COUNTS, "monotone_height"}
    selected = fixed | modified if requested is None else requested
    mixes = selected & PROFILE_MIXES.keys()
    if REFERENCE in selected or mixes:
        base = fit_trial(spec(REFERENCE, seed), {}, frame, {"position": POSITION})
        if REFERENCE in selected:
            yield REFERENCE, base
    if "forest" in selected:
        yield "forest", fit_trial(spec("forest", seed), {}, frame, {"position": POSITION})
    if KNN in selected or mixes:
        knn = fit(Candidate(KNN, "knn", "position_only", "direct"), {}, frame,
                  {"position_only": ["position_fraction"]})
        if KNN in selected:
            yield KNN, knn
        for trial, (columns, weight) in PROFILE_MIXES.items():
            if trial in mixes:
                yield trial, AlignedProfileBlend(base, knn, columns, weight)

    rows, weights = training_rows(frame, "balanced")
    for trial in sorted(selected & modified):
        model = boosted_estimator(REFERENCE, seed)
        target = encode(rows, "log_rate")
        training_weights = weights
        if trial in ROBUST_TRIALS:
            target, training_weights = robust_training(rows, weights, trial)
        elif trial in TREE_COUNTS:
            model.set_params(n_estimators=TREE_COUNTS[trial])
        else:
            model.set_params(monotone_constraints=(0, -1))
        model.fit(rows[POSITION], target, sample_weight=training_weights)
        yield trial, TabularCandidate(replace(spec(REFERENCE, seed), name=trial), POSITION, model)


def check_release(main_dir, manifest):
    result = main_dir / "result.csv"
    metrics = main_dir / "reports/metrics.csv"
    for path, expected in [(result, MAIN_RESULT_SHA256), (metrics, MAIN_METRICS_SHA256)]:
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{path.name} differs from the pinned main release {MAIN_COMMIT}")
    if json.loads((main_dir / "reports/summary.json").read_text(encoding="utf-8"))["sources"] != manifest:
        raise ValueError("Raw source manifest differs from main")
    return pd.read_csv(result, dtype={"id": str}), pd.read_csv(metrics)


def check_baseline(metrics, release):
    keys = ["held_main", "target_date"]
    values = ["n", "pearson_r", "pearson_squared", "mae_mm", "rmse_mm", "bias_mm", "recall_thin_10pct"]
    expected = release.loc[release.model.eq("catboost_weighted")].rename(columns={"campaign": "target_date"})
    expected["target_date"] = pd.to_datetime(expected.target_date)
    actual = metrics.loc[metrics.model.eq(CATBOOST) & metrics.output.eq("bounded") & metrics.seed.eq(42)]
    pd.testing.assert_frame_equal(actual.set_index(keys)[values].sort_index(),
                                  expected.set_index(keys)[values].sort_index(),
                                  check_exact=False, atol=5e-12, rtol=1e-11)


def combine_submission(release, query, values, output):
    case2 = query[KEYS].assign(thickness_pred=values)
    if not case2.thickness_pred.between(0, query.nominal_mm).all():
        raise ValueError("Case 2 thickness outside the passport range")
    case2_keys = release.loc[release.id.isin(case2.id), KEYS]
    validate_submission(case2, case2_keys)
    case1 = release.loc[~release.id.isin(case2.id)]
    combined = pd.concat([case1, case2], ignore_index=True)
    return export_submission(combined, release[KEYS], output)


def run(main_dir, strict_reference, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw, manifest = read_raw_sources("data/raw")
    release, release_metrics = check_release(main_dir, manifest)
    paths = ["contracts/cases.yaml", "configs/data_contract.yaml", "contracts/case2/retrospective.yaml",
             "configs/forecast.yaml"]
    cases, target, transfer, config = (read_config(p) for p in paths)
    train, query, exposure, _, _ = prepare_transfer(raw, manifest, cases, target, transfer, config)
    del raw
    train = train.sort_values(["main_id", "target_date", "id", "segment_id"]).reset_index(drop=True)
    audit = compare_query_inputs({"input_manifest": manifest, "predict_frame": query, "exposure": exposure},
                                 joblib.load(strict_reference))
    del exposure
    reports = []
    with threadpool_limits(limits=1):
        for (number, (_, valid)), seed in product(enumerate(train.groupby("main_id", sort=True)), SEEDS):
            fit = train.loc[~train.main_id.eq(valid.main_id.iloc[0])]
            validate_group_split(fit, valid)
            raw, bounded = from_models(dict(fit_models(fit, seed)), valid, smooth=False)
            if seed == 42:
                rows = fit.loc[fit.observed]
                control = ColdTabular(*TABULAR["catboost_reference_rate"]).fit(rows, weights_by_campaign(rows))
                raw[CATBOOST] = control.predict(valid)
                bounded[CATBOOST] = np.clip(raw[CATBOOST], 0, valid.nominal_mm.to_numpy())
            reports.append(campaign_metrics(valid, raw, bounded, number).assign(seed=seed))
        metrics = pd.concat(reports, ignore_index=True)
        check_baseline(metrics, release_metrics)
        by_seed, board = score_summary(metrics)
        candidates = board.loc[board.passes & ~board.model.isin([CATBOOST, "forest", KNN])]
        winner = candidates.iloc[0].model if len(candidates) else REFERENCE
        models = dict(fit_models(train, 42, requested={winner, "forest"}))
        _, predictions = from_models(models, query, smooth=False)

    with tempfile.TemporaryDirectory(prefix=".final-selection-", dir=output.parent) as directory:
        stage = Path(directory) / "run"
        stage.mkdir()
        joblib.dump(models, stage / "case2_models.joblib", compress=3)
        _, replay = from_models(joblib.load(stage / "case2_models.joblib"), query, smooth=False)
        for name, values in predictions.items():
            np.testing.assert_allclose(replay[name], values, atol=1e-12, rtol=1e-12)
        combine_submission(release, query, predictions[winner], stage / "result.csv")
        result = pd.read_csv(stage / "result.csv", dtype={"id": str})
        pd.testing.assert_frame_equal(release[KEYS], result[KEYS])
        case1 = ~release.id.isin(query.id)
        pd.testing.assert_series_equal(result.loc[case1, "thickness_pred"], release.loc[case1, "thickness_pred"])
        cold = result.loc[~case1, KEYS].merge(query[KEYS].assign(pred=predictions[winner]), on=KEYS)
        np.testing.assert_allclose(result.loc[~case1, "thickness_pred"], cold.pred, atol=5.01e-9, rtol=0)
        priority = query[KEYS].assign(thickness_pred=predictions["forest"])
        count = int(np.ceil(.1 * len(query)))
        priority.sort_values(["thickness_pred", "segment_id"]).head(count).to_csv(
            stage / "dangerous_segments.csv", index=False)
        metrics.to_csv(stage / "case2_metrics.csv", index=False)
        board.to_csv(stage / "case2_leaderboard.csv", index=False)
        by_seed.to_csv(stage / "case2_by_seed.csv", index=False)
        write_json(stage / "summary.json", {
            "main_commit": MAIN_COMMIT, "case1": "direct_blend", "case2": winner,
            "priority_model": "forest", "case1_cells": int(case1.sum()), "case2_cells": len(query),
            "priority_cells": count, "donor_labels": int(train.observed.sum()),
            "case2_validation_campaigns": int(train.groupby(["main_id", "target_date"]).ngroups),
            "query_invariance": audit, "main_catboost_metrics_reproduced": True,
            "result_sha256": sha256((stage / "result.csv").read_bytes()).hexdigest(),
        })
        stage.rename(output)
    print(f"{winner}: {len(query)} case 2 cells; combined export under {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-dir", type=Path, required=True, help="Clean archive of the pinned main release")
    parser.add_argument("--strict-reference", type=Path, required=True, help="Strict case 2 package from prepare-cases")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the combined result")
    args = parser.parse_args()
    run(args.main_dir, args.strict_reference, args.output)


if __name__ == "__main__":
    main()
