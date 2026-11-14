"""Train and compare cold models under the retrospective case 2 contract."""

from hashlib import sha256
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .agent2_physics import make_model as make_physics
from .case2_data import OPERATION, TERRAIN, WET_TERRAIN, prepare_transfer
from .features import feature_columns
from .models.tabular import make_estimator
from .pipeline import write_json
from .quality import read_raw_sources
from .sources import read_config
from .validation import regression_metrics


SEED = 42
BASE_FEATURES = ["nominal_mm", "diameter_mm", "age_years"]
TERRAIN_FEATURES = [c for c in TERRAIN if c != "slope_50m"]
TABULAR = {
    "catboost_reference_rate": ("catboost", "rate", feature_columns("cold")),
    "catboost_without_position": ("catboost", "rate", [c for c in feature_columns("cold") if c != "position_fraction"]),
    "catboost_terrain_retention": ("catboost", "retention", TERRAIN_FEATURES),
    "catboost_local_retention": ("catboost", "retention", BASE_FEATURES + TERRAIN_FEATURES + OPERATION + WET_TERRAIN),
    "catboost_local_rate": ("catboost", "rate", BASE_FEATURES + TERRAIN_FEATURES + OPERATION + WET_TERRAIN),
    "gam_local_retention": ("gam", "retention", BASE_FEATURES + TERRAIN_FEATURES + OPERATION + WET_TERRAIN),
    "ridge_local_retention": ("bounded_ridge", "retention", BASE_FEATURES + TERRAIN_FEATURES + OPERATION + WET_TERRAIN),
    "relative_terrain_ridge": ("bounded_ridge", "relative", TERRAIN_FEATURES),
    "relative_terrain_catboost": ("catboost", "relative", TERRAIN_FEATURES),
}
PHYSICS = {
    "physics_observed": ("positive_rate", "observed-only"),
    "physics_imputed": ("positive_rate", "train-median"),
    "physics_scale_observed": ("hybrid_scale", "observed-only"),
    "physics_scale_imputed": ("hybrid_scale", "train-median"),
}
BLENDS = {
    "blend_local_retention": (.5, "catboost_local_retention", "gam_local_retention"),
    "blend_terrain": (.5, "relative_terrain_ridge", "relative_terrain_catboost"),
    "blend_local_physics": (.75, "catboost_local_retention", "physics_scale_imputed"),
}


def weights_by_campaign(train):
    """Equal weight per main, then per campaign in that main."""
    group_size = train.groupby(["main_id", "target_date"]).sample_id.transform("size")
    campaigns = train.groupby("main_id").target_date.transform("nunique")
    weights = 1. / (group_size * campaigns)
    return (weights / weights.mean()).to_numpy()


class ColdTabular:
    def __init__(self, kind, response, columns):
        self.kind, self.response, self.columns = kind, response, columns

    def fit(self, frame, weights):
        ratio = frame.y_mm.to_numpy() / frame.nominal_mm.to_numpy()
        self.reference_ = float(np.average(ratio, weights=weights))
        y = ratio
        if self.response == "rate":
            y = (1 - ratio) / frame.horizon_years.to_numpy()
        elif self.response == "relative":
            # Learn deviations within a section/campaign so pipe-level means
            # cannot masquerade as a spatial correlation on a new pipe.
            levels = pd.Series(ratio, index=frame.index).groupby([frame.id, frame.target_date]).transform("mean")
            y = ratio - levels.to_numpy()
        x = frame[self.columns]
        if self.kind == "bounded_ridge":
            self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True).fit(x)
            filled = self.imputer_.transform(x)
            self.scaler_ = StandardScaler().fit(filled)
            self.estimator_ = Ridge(alpha=100.).fit(self._scaled(filled), y, sample_weight=weights)
        else:
            self.estimator_ = make_estimator(self.kind, SEED)
            step = self.estimator_.steps[-1][0]
            self.estimator_.fit(x, y, **{f"{step}__sample_weight": weights})
        return self

    def _scaled(self, filled):
        return np.clip(self.scaler_.transform(filled), -3, 3)

    def predict(self, frame):
        x = frame[self.columns]
        if self.kind == "bounded_ridge":
            x = self._scaled(self.imputer_.transform(x))
        response = self.estimator_.predict(x)
        if self.response == "rate":
            response = 1 - response * frame.horizon_years.to_numpy()
        elif self.response == "relative":
            response = self.reference_ + response
        return response * frame.nominal_mm.to_numpy()


def fit_candidates(train, exposure, cutoff):
    weights = weights_by_campaign(train)
    fitted = {name: ColdTabular(*settings).fit(train, weights) for name, settings in TABULAR.items()}
    legacy = TABULAR["catboost_reference_rate"]
    fitted["catboost_reference_unweighted"] = ColdTabular(*legacy).fit(train, np.ones(len(train)))
    for name, (kind, missing) in PHYSICS.items():
        model = make_physics(kind, scenario="cold", exposure=exposure, cutoff_exclusive=cutoff,
                             feature_columns=TERRAIN_FEATURES, seed=SEED, options={"missing_policy": missing})
        fitted[name] = model.fit(train, train.y_mm.to_numpy())
    return fitted


def predictions(models, query):
    nominal = query.nominal_mm.to_numpy()
    raw = {name: np.asarray(model.predict(query), dtype=float) for name, model in models.items()}
    for name, values in raw.items():
        if values.shape != nominal.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid predictions from {name}")
    bounded = {name: np.clip(y, 0, nominal) for name, y in raw.items()}
    for name, (weight, first, second) in BLENDS.items():
        raw[name] = weight * raw[first] + (1 - weight) * raw[second]
        bounded[name] = weight * bounded[first] + (1 - weight) * bounded[second]
    return raw, bounded


def transfer_scores(frame, bounded):
    rows = []
    for (main, date), group in frame.groupby(["main_id", "target_date"], sort=True):
        idx = group.index.to_numpy()
        for name, values in bounded.items():
            rows.append({"model": name, "held_main": main, "campaign": date,
                         **regression_metrics(group.y_mm, values[idx])})
    return rows


def ranking(scores):
    # Constant predictions get no ranking signal. Averaging signed r before
    # squaring prevents a reversed profile from winning on squared r.
    per_main = scores.assign(r_for_rank=scores.pearson_r.fillna(0)).groupby(["model", "held_main"], as_index=False).agg(
        mean_r=("r_for_rank", "mean"), mae_mm=("mae_mm", "mean"), recall=("recall_thin_10pct", "mean"))
    table = per_main.groupby("model", as_index=False).agg(
        macro_r=("mean_r", "mean"), worst_main_r=("mean_r", "min"), macro_mae_mm=("mae_mm", "mean"),
        macro_thin_recall=("recall", "mean"))
    table["both_mains_positive"] = table.worst_main_r.gt(0)
    return table.sort_values(["both_mains_positive", "macro_r", "macro_mae_mm", "model"], ascending=[False, False, True, True])


def expansion_scores(train, held, cutoff):
    """Replay the agent's donor expansion on exactly the same MAIN-3 labels."""
    train = train.loc[train.main_id.eq("MAIN-2")]
    held = held.loc[held.main_id.eq("MAIN-3")]
    rows = []
    for label, pool in [("2018_only", train.loc[train.target_date.lt("2019-01-01")]),
                        ("2018_plus_2022", train.loc[train.target_date.lt("2023-01-01")]), ("all_main2", train)]:
        if not pool.label_available_at.lt(cutoff).all():
            raise ValueError("Labels exceed the frozen donor pool")
        model = ColdTabular(*TABULAR["catboost_reference_rate"]).fit(pool, np.ones(len(pool)))
        y = np.clip(model.predict(held), 0, held.nominal_mm.to_numpy())
        rows.append({"donor_pool": label, "train_labels": len(pool), **regression_metrics(held.y_mm, y)})
    return pd.DataFrame(rows)


def forecast_diagnostics(query, raw, bounded):
    nominal = query.nominal_mm.to_numpy()
    rows = []
    for name, values in raw.items():
        y = bounded[name]
        # Full saturation or a constant profile cannot rank target cells.
        rows.append({"model": name, "raw_min_mm": values.min(), "raw_max_mm": values.max(),
                     "min_mm": y.min(), "max_mm": y.max(), "std_mm": y.std(),
                     "fraction_outside_bounds": ((values < 0) | (values > nominal)).mean(),
                     "fraction_at_bounds": ((y == 0) | (y == nominal)).mean(),
                     "usable_profile": np.ptp(y) > 1e-6 and ((y == 0) | (y == nominal)).mean() < .5})
    return pd.DataFrame(rows)


def create_plots(output, scores, board, selected, forecast, comparisons):
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    table = scores.groupby(["model", "held_main"]).pearson_r.mean().unstack().reindex(board.model)
    order = table.iloc[::-1]
    order.plot.barh(ax=axes[0], width=.78, color=["#2b7186", "#bd7844"])
    axes[0].axvline(0, color="#555555", linewidth=.7)
    axes[0].set_xlabel("Mean signed Pearson r within campaign")
    axes[0].set_ylabel("")
    axes[0].set_title("Entire main withheld from fitting")
    axes[1].bar(comparisons.donor_pool, comparisons.mae_mm, color="#418376")
    axes[1].set_ylabel("MAE, mm")
    axes[1].set_title("Frozen transfer check on MAIN-3 (94 labels)")
    axes[1].tick_params(axis="x", labelrotation=12)
    fig.savefig(output / "validation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.plot(forecast.center_m, forecast.thickness_pred, color="#2b7186", lw=1.1, label=selected)
    ax.axhline(forecast.nominal_mm.iloc[0], color="#777777", ls="--", label="Passport nominal")
    ax.set(title="Case 2 | provisional profile | target inspection hidden", xlabel="Distance along section, m", ylabel="Remaining wall, mm")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(loc="best")
    fig.savefig(output / "forecast.png", dpi=150)
    plt.close(fig)


def run(data_dir, output, contracts_dir, config_path):
    data_dir, output, contracts_dir, config_path = map(Path, (data_dir, output, contracts_dir, config_path))
    output = output.resolve()
    if output.exists():
        raise ValueError("Use a new output folder")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    cases = read_config(contracts_dir / "cases.yaml")
    policy_path = contracts_dir / "case2/retrospective.yaml"
    policy = read_config(policy_path)
    config = read_config(config_path)
    raw, manifest = read_raw_sources(data_dir)
    train, query, exposure, inference_audit, donor_audit = prepare_transfer(
        raw, manifest, cases, read_config(config["contract"]), policy, config)
    cutoff = pd.Timestamp(policy["pool_cutoff_exclusive"])
    train = train.loc[train.observed].reset_index(drop=True)
    mains = sorted(train.main_id.unique())
    if len(mains) < 2:
        raise ValueError("At least two donor mains are required for transfer validation")
    rows, held_predictions = [], []
    print(train.groupby(["main_id", "target_date"]).size().to_string(), flush=True)
    with threadpool_limits(limits=1):
        for main in mains:
            fit = train.loc[train.main_id.ne(main)]
            held = train.loc[train.main_id.eq(main)].reset_index(drop=True)
            print(f"Case 2 transfer: holding out {main} ({len(held)} labels)", flush=True)
            fitted = fit_candidates(fit, exposure, cutoff)
            _, bounded = predictions(fitted, held.drop(columns=["y_mm", "observed", "label_available_at", "coverage_status"]))
            rows.extend(transfer_scores(held, bounded))
            held_predictions.append(held[["sample_id", "main_id", "id", "segment_id", "target_date", "y_mm"]].assign(**bounded))
        scores = pd.DataFrame(rows)
        board = ranking(scores)
        comparisons = expansion_scores(train, train, cutoff)
        fitted = fit_candidates(train, exposure, cutoff)
        raw_pred, bounded = predictions(fitted, query)
        diagnostics = forecast_diagnostics(query, raw_pred, bounded)
        usable = board.merge(diagnostics[["model", "usable_profile"]], on="model", sort=False)
        valid = usable.loc[usable.usable_profile]
        if valid.empty:
            raise ValueError("All candidate target forecasts are degenerate")
        selected = valid.model.iloc[0]
        forecast = query[["id", "segment_id", "center_m", "target_date", "nominal_mm"]].assign(thickness_pred=bounded[selected])
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".case2-transfer-", dir=output.parent) as path:
            staged = Path(path) / "run"
            staged.mkdir()
            bundle = {"models": fitted, "selected": selected, "policy": policy, "manifest": manifest}
            joblib.dump(bundle, staged / "models.joblib", compress=3)
            _, replay = predictions(joblib.load(staged / "models.joblib")["models"], query)
            for name in replay:
                np.testing.assert_allclose(replay[name], bounded[name], rtol=0, atol=1e-11)
            tables = {"transfer_metrics": scores, "ranking": board, "expansion": comparisons,
                      "forecast": forecast, "forecast_diagnostics": diagnostics,
                      "all_forecasts": query[["id", "segment_id", "target_date"]].assign(**bounded),
                      "raw_forecasts": query[["id", "segment_id", "target_date"]].assign(**raw_pred),
                      "train_campaigns": train.groupby(["main_id", "target_date"], as_index=False).agg(
                          labels=("sample_id", "size"), sections=("id", "nunique")),
                      "held_predictions": pd.concat(held_predictions, ignore_index=True)}
            for name, table in tables.items():
                table.to_csv(staged / f"{name}.csv", index=False)
            for name, audit in [("query", inference_audit), ("donors", donor_audit)]:
                audit.groupby(["source_file", "rule_id", "action"]).size().rename("events").to_csv(staged / f"{name}_cleaning.csv")
            write_json(staged / "summary.json", {"selected": selected, "validation_leader": board.model.iloc[0],
                       "positive_on_both_mains": bool(valid.both_mains_positive.iloc[0]), "train_labels": len(train),
                       "independent_donor_mains": mains, "query_segments": len(query),
                       "query_observed_days": int(exposure.loc[exposure.id.eq(query.id.iloc[0]), "observed"].sum()),
                       "query_coverage": float(query.regime_coverage.iloc[0]), "policy": policy,
                       "manifest": manifest, "seed": SEED, "tabular": TABULAR, "physics": PHYSICS, "blends": BLENDS,
                       "source_sha256": {file.name: sha256(file.read_bytes()).hexdigest() for file in [
                           Path(__file__), Path(__file__).with_name("case2_data.py"), contracts_dir / "cases.yaml",
                           policy_path, config_path, Path(config["contract"])]},
                       "status": "provisional_hidden_target"})
            create_plots(staged, scores, board, selected, forecast, comparisons)
            staged.rename(output)
    print(board.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Case 2 provisional choice: {selected}\nReport: {output}")
