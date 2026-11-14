"""A bounded E13 experiment on the saved case 1 folds and previous inspections."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .case_data import model_views
from .case_experiments import CaseModel, predict_models, validate_package
from .models.tabular import TabularModel
from .pipeline import write_json
from .quality import load_case_sources
from .sources import read_config
from .validation import regression_metrics


BASELINE = "blend_gam_catboost"
BLEND = {BASELINE: {"gam": .5, "catboost": .5}}
EDGE_MARGIN_M = 2.0
NEIGHBOR_RADIUS_M = 50.0
EDGE_WEIGHT = .25
MISSING_WEIGHT = .5
MIN_R2_GAIN = .01
MAX_MAE_INCREASE_MM = .01
FEATURE_SETS = {
    BASELINE: [],
    "edge_context": ["edge_min_mm", "edge_min_distance_m"],
    "local_contrast": ["local_contrast_mm"],
    "combined_context": ["edge_min_mm", "edge_min_distance_m", "local_contrast_mm"],
}


def history_features(frame, inspection, target_policy):
    """Build spatial inputs from the anchor campaign as known at each example date."""
    frame = frame.copy()
    features = ["edge_min_mm", "edge_min_distance_m", "neighbor_median_50m_mm", "local_contrast_mm"]
    frame[features] = np.nan
    if not frame.index.is_unique or not frame.anchor_date.lt(frame.target_date).all():
        raise ValueError("Expected unique rows and an earlier anchor date for case 1")
    if target_policy not in {"reported_residual", "corrosion_only"}:
        raise ValueError("Unsupported target policy")
    for (section_id, anchor, target), section in frame.groupby(["id", "anchor_date", "target_date"]):
        points = inspection.loc[
            inspection.id.eq(section_id) & inspection.date.eq(anchor) & inspection.available_at.lt(target)
            & inspection.thickness_mm.notna()
        ]
        if target_policy == "corrosion_only":
            points = points.loc[points.is_corrosion]
        x, thickness = points.distance_m.to_numpy(), points.thickness_mm.to_numpy()
        centers = section.center_m.to_numpy()
        measured = section.anchor_mm.where(section.anchor_observed).to_numpy()
        for row in section.itertuples():
            inside = (x >= row.start_m) & (x < row.end_m)
            exact = thickness[inside].min() if inside.any() else np.nan
            if not np.isclose(exact, row.anchor_mm if row.anchor_observed else np.nan, equal_nan=True):
                raise ValueError(f"Historical points disagree with the prepared anchor for {row.sample_id}")
            near = (x >= row.start_m - EDGE_MARGIN_M) & (x < row.end_m + EDGE_MARGIN_M)
            if near.any():
                level = thickness[near].min()
                distance = np.abs(x[near & (thickness == level)] - row.center_m).min()
                frame.loc[row.Index, ["edge_min_mm", "edge_min_distance_m"]] = [level, distance]
            distance = np.abs(centers - row.center_m)
            neighbors = measured[(distance > 0) & (distance <= NEIGHBOR_RADIUS_M) & np.isfinite(measured)]
            if len(neighbors):
                median = np.median(neighbors)
                frame.at[row.Index, "neighbor_median_50m_mm"] = median
                if row.anchor_observed:
                    frame.at[row.Index, "local_contrast_mm"] = row.anchor_mm - median
    return frame


def heuristic_predictions(train, query, baseline):
    """Test two fixed corrections to the frozen blend on disjoint groups."""
    edge_drop = (query.anchor_mm - query.edge_min_mm).clip(lower=0).fillna(0)
    edge = baseline - EDGE_WEIGHT * edge_drop.where(query.anchor_observed, 0).to_numpy()

    # Add a local contrast to the shared fallback when a cell has no anchor.
    # The reference uses only comparable, labelled training rows with no anchor.
    reference = train.loc[train.observed & ~train.anchor_observed & train.neighbor_median_50m_mm.notna()]
    missing = baseline.copy()
    if len(reference):
        fraction = (reference.neighbor_median_50m_mm / reference.nominal_mm).median()
        eligible = ~query.anchor_observed & query.neighbor_median_50m_mm.notna()
        adjustment = query.neighbor_median_50m_mm - fraction * query.nominal_mm
        missing[eligible.to_numpy()] += MISSING_WEIGHT * adjustment.loc[eligible].to_numpy()
    nominal = query.nominal_mm.to_numpy()
    return {"edge_rule": np.clip(edge, 0, nominal), "missing_anchor_rule": np.clip(missing, 0, nominal)}


def predict_candidates(package, train, query, fallback, seed):
    eligible = train.loc[train.observed & train.anchor_observed]
    results = {}
    for name, added in FEATURE_SETS.items():
        models = {}
        for kind in BLEND[BASELINE]:
            model = TabularModel("between", kind, seed, columns=package["feature_columns"] + added)
            model.fit(eligible, eligible.y_mm.to_numpy())
            models[kind] = CaseModel(model, kind, "case1", "between", pd.Timestamp(package["cutoff_exclusive"]), fallback["fraction"])
        _, bounded = predict_models(models, query, BLEND)
        results[name] = bounded[BASELINE]
    return {**results, **heuristic_predictions(train, query, results[BASELINE])}


def measure(oof, frame, target_id):
    observed = frame.observed
    masks = [("all", observed), ("target_section", observed & frame.id.eq(target_id)),
             ("anchored", observed & frame.anchor_observed), ("no_anchor", observed & ~frame.anchor_observed)]
    masks += [(f"fold_{fold + 1}", observed & frame.fold_id.eq(fold)) for fold in sorted(frame.fold_id.unique())]
    return pd.DataFrame([
        {"scope": scope, "model": name, **regression_metrics(frame.loc[mask, "y_mm"], values[mask.to_numpy()])}
        for scope, mask in masks for name, values in oof.items()
    ])


def screen(metrics):
    baseline = metrics.loc[metrics.model.eq(BASELINE)].set_index("scope")
    records = []
    for name, group in metrics.loc[metrics.model.ne(BASELINE)].groupby("model", sort=False):
        result = group.set_index("scope")
        delta = result.pearson_squared - baseline.pearson_squared
        delta_mae = result.mae_mm - baseline.mae_mm
        fold = result.index.str.startswith("fold_")
        # Signed r prevents a strong reverse ranking from counting as an improvement.
        wins = int((result.loc[fold, "pearson_r"] > baseline.loc[fold, "pearson_r"]).sum())
        passes = (result.loc[["all", "target_section"], "pearson_r"].gt(0).all()
                  and delta.loc[["all", "target_section"]].ge(MIN_R2_GAIN).all()
                  and delta_mae.loc[["all", "target_section"]].le(MAX_MAE_INCREASE_MM).all()
                  and wins >= 2)
        records.append({"model": name, "delta_r2_all": delta.loc["all"], "delta_r2_target": delta.loc["target_section"],
                        "delta_mae_mm_all": delta_mae.loc["all"], "r_fold_wins": wins, "passes": bool(passes)})
    return pd.DataFrame(records)


def sensitivity(oof, frame, query):
    """Re-score fixed OOF predictions under two changes of evaluation mix."""
    data = frame.loc[frame.observed & frame.id.eq(query.id.iloc[0])].copy()
    observed_share = data.anchor_observed.mean()
    query_share = query.anchor_observed.mean()
    if not 0 < observed_share < 1:
        raise ValueError("Coverage sensitivity needs both known and missing historical anchors")
    weights = np.where(data.anchor_observed, query_share / observed_share,
                       (1 - query_share) / (1 - observed_share))
    weights /= weights.sum()
    actual = data.y_mm.to_numpy()
    centered_actual = actual - np.dot(weights, actual)
    actual_variance = np.dot(weights, centered_actual ** 2)
    records = []
    for name, values in oof.items():
        for fold in sorted(frame.fold_id.unique()):
            mask = frame.observed & frame.fold_id.ne(fold)
            for scope, subset in [("all", mask), ("target_section", mask & frame.id.eq(query.id.iloc[0]))]:
                result = regression_metrics(frame.loc[subset, "y_mm"], values[subset.to_numpy()])
                records.append({"check": "omit_fold_from_score", "scope": scope, "omitted_fold_id": fold,
                                "model": name, **result})
        predicted = values[data.index.to_numpy()]
        centered = predicted - np.dot(weights, predicted)
        denominator = np.sqrt(actual_variance * np.dot(weights, centered ** 2))
        correlation = np.dot(weights, centered_actual * centered) / denominator if denominator > 1e-12 else np.nan
        records.append({"check": "reweight_to_query_anchor_coverage", "scope": "target_section",
                        "omitted_fold_id": None, "model": name, "n": len(data), "pearson_r": correlation,
                        "pearson_squared": correlation ** 2, "mae_mm": np.dot(weights, np.abs(predicted - actual))})
    return pd.DataFrame(records)


def replay_reference(path, frame, predictions, *, labelled):
    saved = pd.read_csv(path, dtype={"id": str}).loc[lambda data: data.model.eq(BASELINE)].set_index("sample_id")
    if not saved.index.is_unique or set(saved.index) != set(frame.sample_id):
        raise ValueError("Baseline reference keys differ from the experiment")
    saved = saved.loc[frame.sample_id]
    np.testing.assert_array_equal(saved.id, frame.id)
    np.testing.assert_array_equal(saved.segment_id, frame.segment_id)
    if labelled:
        np.testing.assert_array_equal(saved.fold_id, frame.fold_id)
        np.testing.assert_array_equal(saved.observed, frame.observed)
        np.testing.assert_allclose(saved.y_mm, frame.y_mm, rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(saved.prediction_mm, predictions, rtol=0, atol=1e-11)


def lag_diagnostic(frame, baseline):
    """Same-row comparison of past errors shifted one cell within section and fold."""
    data = frame.assign(prediction_mm=baseline)
    keyed = data.set_index(["id", "fold_id", "segment_id"]).prediction_mm
    shifts = [-1, 0, 1]
    for step in shifts:
        keys = pd.MultiIndex.from_arrays([data.id, data.fold_id, data.segment_id + step], names=keyed.index.names)
        data[step] = keyed.reindex(keys).to_numpy()
    data = data.loc[data.observed].dropna(subset=shifts)
    return pd.DataFrame([{"fold_id": fold, "old_prediction_offset_cells": step,
                          **regression_metrics(block.y_mm, block[step])}
                         for fold, block in data.groupby("fold_id") for step in shifts])


def plot_comparison(out, metrics, candidates):
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MultipleLocator

    names = ["edge_context", "local_contrast", "combined_context", "edge_rule", "missing_anchor_rule"]
    labels = ["Boundary features", "Local contrast", "Both feature sets", "Boundary rule", "Missing-anchor rule"]
    ordered = candidates.set_index("model").loc[names]
    positions = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.barh(positions - .17, ordered.delta_r2_all, .32, label="All historical cells", color="#176285")
    ax.barh(positions + .17, ordered.delta_r2_target, .32, label="Target section in 2022", color="#56a294")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#555555", linewidth=.8)
    ax.axvline(MIN_R2_GAIN, color="#888888", linestyle=":", linewidth=1)
    ax.set_xlabel("Change in Pearson r² vs. frozen 50/50 blend")
    ax.set_title("Effect of each candidate")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="x", alpha=.2)

    ax = axes[1]
    table = metrics.loc[metrics.scope.str.startswith("fold_")].pivot(index="scope", columns="model", values="pearson_r")
    ax.plot(np.arange(1, 5), table[BASELINE], marker="o", label="Frozen blend", color="#176285")
    best = ordered.delta_r2_all.idxmax()
    ax.plot(np.arange(1, 5), table[best], marker="s", label=labels[names.index(best)], color="#ba703d")
    ax.set_xticks(np.arange(1, 5))
    ax.set_xlabel("Historical spatial fold")
    ax.set_ylabel("Signed Pearson r")
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.grid(alpha=.2)
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Best exploratory candidate by pooled r²")
    fig.suptitle("Case 1 | same folds and labels | target-date accuracy unknown")
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)


def run(prepared, reference, output, data_dir, contracts_dir):
    package = joblib.load(prepared)
    validate_package(package)
    if package["case_id"] != "case1" or package["scenario"] != "between":
        raise ValueError("This experiment applies to case 1 only")
    # A neighboring 10 m cell can extend another 5 m from its center.
    if package["config"]["spatial_buffer_m"] < NEIGHBOR_RADIUS_M + 5:
        raise ValueError("Saved spatial buffer is too small for these history inputs")
    cleaned = load_case_sources(data_dir, "case1", contracts_dir, package["config"]["contract"])
    contract = read_config(package["config"]["contract"])
    if cleaned.sources.manifest != package["input_manifest"] or contract["target_policy"] != package["label_policy"]:
        raise ValueError("Recreate inputs from these raw files and target policy before comparing candidates")
    seed = read_config("configs/case_benchmark.yaml")["seed"]
    for key in ["train_frame", "predict_frame"]:
        package[key] = history_features(package[key], cleaned.sources.inspection, package["label_policy"])
    frame = package["train_frame"].reset_index(drop=True)
    fold_ids = np.full(len(frame), -1)
    oof = {}
    with threadpool_limits(limits=1):
        for fold in package["folds"]:
            number = fold["number"]
            print(f"Case 1: experimental fold {number + 1}/{len(package['folds'])}", flush=True)
            train, valid, fallback = model_views(package, number)
            for name, values in predict_candidates(package, train, valid, fallback, seed).items():
                oof.setdefault(name, np.full(len(frame), np.nan))[fold["valid"]] = values
            fold_ids[fold["valid"]] = number
        frame["fold_id"] = fold_ids
        if (fold_ids < 0).any() or any(not np.isfinite(values).all() for values in oof.values()):
            raise ValueError("Incomplete OOF predictions")
        replay_reference(reference / "oof_predictions.csv.gz", frame, oof[BASELINE], labelled=True)
        metrics = measure(oof, frame, package["spec"]["id"])
        candidates = screen(metrics)
        train, query, fallback = model_views(package)
        forecast = predict_candidates(package, train, query, fallback, seed)
        replay_reference(reference / "forecast_predictions.csv.gz", query, forecast[BASELINE], labelled=False)
        if any(not np.isfinite(values).all() for values in forecast.values()):
            raise ValueError("Incomplete forecast predictions")

    with tempfile.TemporaryDirectory(prefix=".case1-heuristics-", dir=output.parent) as temp:
        staged = Path(temp) / "run"
        staged.mkdir()
        metrics.to_csv(staged / "metrics.csv", index=False)
        candidates.to_csv(staged / "comparison.csv", index=False)
        sensitivity(oof, frame, query).to_csv(staged / "sensitivity.csv", index=False)
        lag_diagnostic(frame, oof[BASELINE]).to_csv(staged / "lag_diagnostic.csv", index=False)
        keys = ["sample_id", "id", "segment_id", "target_date", "center_m", "anchor_observed", "anchor_mm",
                "edge_min_mm", "edge_min_distance_m", "local_contrast_mm", "neighbor_median_50m_mm"]
        frame[keys + ["observed", "y_mm", "fold_id"]].assign(**oof).to_csv(staged / "oof.csv", index=False)
        query[keys].assign(**forecast).to_csv(staged / "forecast_provisional.csv", index=False)
        plot_comparison(staged, metrics, candidates)
        write_json(staged / "provenance.json", {
            "prepared_sha256": sha256(prepared.read_bytes()).hexdigest(),
            "reference_sha256": {name: sha256((reference / name).read_bytes()).hexdigest()
                                 for name in ["oof_predictions.csv.gz", "forecast_predictions.csv.gz"]},
            "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "seed": seed, "features": FEATURE_SETS, "edge_margin_m": EDGE_MARGIN_M,
            "neighbor_radius_m": NEIGHBOR_RADIUS_M, "edge_weight": EDGE_WEIGHT, "missing_weight": MISSING_WEIGHT,
            "min_r2_gain_all_and_target": MIN_R2_GAIN, "max_mae_increase_mm": MAX_MAE_INCREASE_MM,
            "min_fold_wins": 2, "historical_labels": int(frame.observed.sum()), "forecast_cells": len(query),
            "forecast_observed_anchors": int(query.anchor_observed.sum()),
            "evaluation": package["evaluation"], "case_cutoff": package["cutoff_exclusive"],
            "baseline_replayed": True, "passed_screen": candidates.loc[candidates.passes, "model"].tolist(),
        })
        staged.rename(output)
    print(candidates.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"Results: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True, help="Case 1 prepared.joblib from prepare-cases")
    parser.add_argument("--reference", type=Path, required=True, help="Case 1 folder from case-benchmark")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--contracts-dir", type=Path, default=Path("contracts"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    run(args.prepared, args.reference, output, args.data_dir, args.contracts_dir)


if __name__ == "__main__":
    main()
