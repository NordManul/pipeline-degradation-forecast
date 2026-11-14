"""Compare five final model families without tuning on late or on the targets."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import os
from pathlib import Path
import tempfile
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .advanced_checks import late_labels, score, warm_data, warm_scores
from .case1_heuristics import FEATURE_SETS
from .case1_measurement import STABLE
from .case2_data import prepare_transfer
from .case2_transfer import TABULAR, ColdTabular, ranking, weights_by_campaign
from .case_data import build_exposure, model_views
from .case_experiments import KEY_COLUMNS, query_view
from .models.advanced import anchored, direct_components, warm_bounds
from .pipeline import write_json
from .quality import clean_sources, read_raw_sources
from .sources import read_config


ARCHITECTURES = ["lstm", "cnn", "transformer"]
SEEDS = [42, 73, 101]
EPOCHS = 48
TREE_PARAMS = {"n_estimators": 320, "max_depth": 3, "learning_rate": .035, "reg_lambda": 10.,
               "subsample": .85, "colsample_bytree": .85, "random_state": 42, "n_jobs": 1}
TREES = ["lightgbm", "xgboost"]
CANDIDATES = [*TREES, *ARCHITECTURES]


def tree_estimator(name):
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**TREE_PARAMS, objective="regression", num_leaves=8, min_child_samples=12,
                            subsample_freq=1, deterministic=True, force_col_wise=True, verbosity=-1)
    from xgboost import XGBRegressor
    return XGBRegressor(**TREE_PARAMS, objective="reg:squarederror", min_child_weight=8,
                        tree_method="hist", verbosity=0)


def make_inputs(grid, exposure, columns, *, warm):
    from .models.sequence import BETWEEN_PROFILE, COMMON_PROFILE, spatial_windows, tech_windows

    local = COMMON_PROFILE + (BETWEEN_PROFILE if warm else [])
    clean = query_view(grid)
    values = (clean[columns].to_numpy(dtype=float), spatial_windows(clean, local), tech_windows(exposure, clean))
    return {"keys": pd.Index(clean.sample_id), "values": values, "columns": columns}


def datasets(inputs, frame):
    index = inputs["keys"]
    keys = index.get_indexer(frame.sample_id)
    if (keys < 0).any():
        raise ValueError("Unknown segment keys in sequence inputs")
    return tuple(x[keys] for x in inputs["values"])


def train_candidates(fit, query, inputs, weights, tag, bound, *, warm):
    from .models.sequence import fit_predict

    columns = inputs["columns"]
    if warm:
        y = fit.y_mm.to_numpy()
        scale = None
    else:
        scale = query.horizon_years.to_numpy()
        y = (1 - fit.y_mm / fit.nominal_mm).to_numpy() / fit.horizon_years.to_numpy()

    def thickness(p):
        return p if scale is None else (1 - scale * p) * query.nominal_mm.to_numpy()

    predictions, raw, metadata = {}, {}, []
    for name in TREES:
        t = perf_counter()
        fitted = tree_estimator(name).fit(fit[columns], y, sample_weight=weights)
        pred = thickness(fitted.predict(query[columns]))
        metadata.append({"scope": tag, "model": name, "seed": 42, "seconds": perf_counter() - t, "train_rows": len(y)})
        raw[name], predictions[name] = pred, bound(pred)
    x, validation = datasets(inputs, fit), datasets(inputs, query)
    for name in ARCHITECTURES:
        individual = []
        for seed in SEEDS:
            t = perf_counter()
            result, parameters = fit_predict(name, x, validation, y, weights, seed=seed, epochs=EPOCHS)
            pred = thickness(result)
            individual.append(pred)
            predictions[f"{name}_s{seed}"] = bound(pred)
            metadata.append({"scope": tag, "model": name, "seed": seed, "parameters": parameters,
                             "seconds": perf_counter() - t, "train_rows": len(y)})
        raw[name] = np.mean(individual, axis=0)
        predictions[name] = bound(raw[name])
    if any(not np.isfinite(p).all() for p in [*predictions.values(), *raw.values()]):
        raise ValueError("Nonfinite final-round predictions")
    return predictions, raw, metadata


def warm_predict(train, query, inputs, fallback, tag):
    fit = anchored(train)
    baseline = np.mean([warm_bounds(query, p, fallback) for p in direct_components(fit, query, inputs["columns"])], axis=0)
    candidates, raw, audit = train_candidates(fit, query, inputs, np.ones(len(fit)), tag,
                                              lambda p: warm_bounds(query, p, fallback), warm=True)
    return {"baseline": baseline, **candidates}, {"baseline": baseline, **raw}, audit


def forecast_ranges(query, bounded, raw):
    nominal = query.nominal_mm.to_numpy()
    return pd.DataFrame([{"model": name, "n": len(p), "min_mm": p.min(), "max_mm": p.max(), "std_mm": p.std(),
                          "fraction_at_bounds": ((p == 0) | (p == nominal)).mean(),
                          "fraction_raw_outside_bounds": ((raw[name] < 0) | (raw[name] > nominal)).mean()}
                         for name, p in bounded.items() if name in raw])


def case1(package, raw, clean, target):
    from .models.sequence import SPATIAL_RADIUS_CELLS

    package, late = warm_data(package, clean, target)
    config = package["config"]
    support_m = SPATIAL_RADIUS_CELLS * target["segment_length_m"] + config["history_radius_m"]
    if support_m > config["spatial_buffer_m"]:
        raise ValueError("Spatial network context exceeds the outer fold buffer")
    columns = [*package["feature_columns"], *FEATURE_SETS["combined_context"], *STABLE]
    grid = pd.concat([package["train_frame"], package["predict_frame"], late], ignore_index=True)
    exposure = build_exposure(clean.sources.regime, [grid], clean.cutoff)
    inputs = make_inputs(grid, exposure, columns, warm=True)
    rows, audit = [], []
    for split in package["folds"]:
        fold = split["number"]
        train, held, fallback = model_views(package, fold)
        print(f"Case 1 | fold {fold + 1}/{len(package['folds'])}", flush=True)
        pred, _, info = warm_predict(train, query_view(held), inputs, fallback["fraction"], f"case1_fold_{fold + 1}")
        rows.append(held.assign(fold=fold + 1, **pred))
        audit.extend(info)
    oof = pd.concat(rows, ignore_index=True).sort_values(["id", "segment_id"]).reset_index(drop=True)
    target_rows = oof.loc[oof.id.eq(package["spec"]["id"])]
    scores = warm_scores(oof, list(pred), "historical_all") + warm_scores(target_rows, list(pred), "historical_target")
    for fold, part in target_rows.groupby("fold"):
        scores.extend(warm_scores(part, list(pred), f"target_fold_{fold}"))
        scores.extend(warm_scores(target_rows.loc[target_rows.fold.ne(fold)], list(pred), f"target_omit_{fold}"))
    # Final fit and all late predictions happen before attaching the 2023 labels.
    train, query, fallback = model_views(package)
    print("Case 1 | final fit and late predictions", flush=True)
    full = pd.concat([query, late], ignore_index=True)
    all_pred, all_raw, info = warm_predict(train, full, inputs, fallback["fraction"], "case1_final")
    audit.extend(info)
    held = late_labels(late, raw, clean, target).assign(**{name: p[len(query):] for name, p in all_pred.items()})
    scores.extend(warm_scores(held, list(all_pred), "late"))
    for section, part in held.groupby("id"):
        scores.extend(warm_scores(part, list(all_pred), f"late_{section}"))
    selected = {name: p[:len(query)] for name, p in all_pred.items()}
    forecast = query[KEY_COLUMNS + ["center_m", "nominal_mm"]].assign(**selected)
    return {"case1_metrics": pd.DataFrame(scores),
            "case1_oof": oof[KEY_COLUMNS + ["fold", "observed", "y_mm", *pred]],
            "case1_late": held[KEY_COLUMNS + ["observed", "y_mm", *pred]],
            "case1_forecasts": forecast,
            "case1_ranges": forecast_ranges(query, selected, {name: p[:len(query)] for name, p in all_raw.items()})}, audit


def cold_predict(train, query, inputs, tag):
    weights = weights_by_campaign(train)
    nominal = query.nominal_mm.to_numpy()
    model = ColdTabular(*TABULAR["catboost_reference_rate"]).fit(train, weights)
    baseline = model.predict(query)
    pred, raw, audit = train_candidates(train, query, inputs, weights, tag, lambda p: np.clip(p, 0, nominal), warm=False)
    return {"baseline": np.clip(baseline, 0, nominal), **pred}, {"baseline": baseline, **raw}, audit


def case2(raw, manifest, cases, target, config):
    policy = read_config("contracts/case2/retrospective.yaml")
    all_rows, query, exposure, *_ = prepare_transfer(raw, manifest, cases, target, policy, config)
    inputs = make_inputs(pd.concat([all_rows, query], ignore_index=True), exposure,
                         TABULAR["catboost_reference_rate"][2], warm=False)
    train = all_rows.loc[all_rows.observed].reset_index(drop=True)
    scores, pieces, audit = [], [], []
    for main in sorted(train.main_id.unique()):
        print(f"Case 2 | holding out {main}", flush=True)
        fit, held = train.loc[train.main_id.ne(main)], train.loc[train.main_id.eq(main)]
        pred, _, info = cold_predict(fit, query_view(held), inputs, f"case2_{main}")
        frame = held.assign(**pred)
        pieces.append(frame[KEY_COLUMNS + ["main_id", "observed", "y_mm", *pred]])
        audit.extend(info)
        for date, part in frame.groupby("target_date"):
            scores.extend({"held_main": main, "campaign": date, "model": name, **score(part, part[name].to_numpy())}
                          for name in pred)
    print("Case 2 | final fit", flush=True)
    pred, raw_forecast, info = cold_predict(train, query, inputs, "case2_final")
    audit.extend(info)
    metrics = pd.DataFrame(scores)
    return {"case2_metrics": metrics, "case2_ranking": ranking(metrics), "case2_oof": pd.concat(pieces, ignore_index=True),
            "case2_forecasts": query[KEY_COLUMNS + ["center_m", "nominal_mm"]].assign(**pred),
            "case2_ranges": forecast_ranges(query, pred, raw_forecast)}, audit


def verify_submission(warm, cold, submission):
    keys = ["id", "segment_id"]
    base = pd.concat([warm["case1_forecasts"], cold["case2_forecasts"]], ignore_index=True)
    merged = base.merge(submission, on=keys, validate="one_to_one")
    if len(base) != len(submission) or len(merged) != len(base):
        raise ValueError("The comparison grid does not equal the submission keys")
    np.testing.assert_allclose(merged.baseline, merged.thickness_pred, rtol=0, atol=5.01e-9)


def plot(output, warm, cold):
    from matplotlib import pyplot as plt

    names = ["baseline", *CANDIDATES]
    labels = ["Submitted model", "LightGBM", "XGBoost", "LSTM", "1D-CNN", "Transformer"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    x = np.arange(len(names))
    for ax, table, title in [(axes[0], warm["case1_metrics"].pivot(index="model", columns="scope", values="pearson_r")[["historical_target", "late"]], "Case 1: target OOF and adjacent `late`"),
                             (axes[1], cold["case2_metrics"].groupby(["model", "held_main"]).pearson_r.mean().unstack(), "Case 2: held-out mains")]:
        for index, (column, values) in enumerate(table.loc[names].items()):
            ax.barh(x + (index - .5) * .33, values, .31, label=column, color=["#247f83", "#bd7956"][index])
        ax.set_yticks(x, labels)
        ax.invert_yaxis()
        ax.axvline(0, color="#444444", lw=.75)
        ax.set(title=title, xlabel="Signed Pearson r")
        ax.grid(axis="x", alpha=.22)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", fontsize=8)
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True, help="Frozen case 1 prepared.joblib")
    parser.add_argument("--submission", type=Path, default=Path("reports/2026-11-14-case1-measurement/result.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    import torch
    from .models.sequence import SPATIAL_RADIUS_CELLS, TECH_BIN_DAYS, TECH_BINS

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    package = joblib.load(args.prepared)
    cases = read_config("contracts/cases.yaml")
    target = read_config(package["config"]["contract"])
    raw, manifest = read_raw_sources("data/raw")
    clean = clean_sources(raw, "case1", cases, target, manifest)
    with threadpool_limits(limits=1):
        warm, audit1 = case1(package, raw, clean, target)
        cold, audit2 = case2(raw, manifest, cases, target, package["config"])
    verify_submission(warm, cold, pd.read_csv(args.submission, dtype={"id": str}))

    with tempfile.TemporaryDirectory(prefix=".final-round-", dir=output.parent) as temp:
        stage = Path(temp) / "run"
        stage.mkdir()
        for name, table in {**warm, **cold, "fit_audit": pd.DataFrame(audit1 + audit2)}.items():
            if name.endswith(("_forecasts", "_oof", "_late")) and table.sample_id.duplicated().any():
                raise ValueError(f"Duplicate segment keys in {name}")
            table.to_csv(stage / f"{name}.csv", index=False)
        write_json(stage / "provenance.json", {"warm_spatial_radius_cells": SPATIAL_RADIUS_CELLS,
                   "tech_bins": TECH_BINS, "tech_bin_days": TECH_BIN_DAYS, "torch_threads": 2,
                   "nn_seeds": SEEDS, "epochs": EPOCHS, "tree_params": TREE_PARAMS, "input_manifest": manifest,
                   "prepared_sha256": sha256(args.prepared.read_bytes()).hexdigest(),
                   "verified_submission_sha256": sha256(args.submission.read_bytes()).hexdigest(),
                   "source_sha256": {str(p): sha256(p.read_bytes()).hexdigest() for p in
                                     [Path(__file__), Path(__file__).with_name("models") / "sequence.py"]},
                   "versions": {p: version(p) for p in ["torch", "lightgbm", "xgboost-cpu", "catboost", "numpy", "pandas", "scikit-learn"]}})
        plot(stage, warm, cold)
        stage.rename(output)
    print(warm["case1_metrics"].loc[lambda t: t.scope.isin(["historical_target", "late"]) & t.model.isin(["baseline", *CANDIDATES]),
                                   ["scope", "model", "pearson_r", "pearson_squared", "mae_mm", "recall_thin_10pct"]].round(5).to_string(index=False), flush=True)
    print(cold["case2_ranking"].loc[lambda t: t.model.isin(["baseline", *CANDIDATES])].round(5).to_string(index=False), flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
