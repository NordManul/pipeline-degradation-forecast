"""Case 2: retrospective cold-start transfer between complete mains."""

import numpy as np
import pandas as pd

from .case_data import canonical_snapshot
from .features import feature_columns
from .geometry import build_grid
from .models.tabular import make_estimator
from .quality import clean_case2_donors, clean_sources
from .targets import build_targets
from .validation import regression_metrics


FINAL_MODEL = "catboost_weighted"


def prepare_transfer(raw, manifest, cases, target_contract, policy, config):
    case = cases["cases"]["case2"]
    if (case["main_id"] != policy["target_main"]
            or pd.Timestamp(case["target_date"]) != pd.Timestamp(policy["inference_cutoff_exclusive"])):
        raise ValueError("Case 2 transfer and inference contracts disagree")
    strict = clean_sources(raw, "case2", cases, target_contract, manifest)
    donors = clean_case2_donors(raw, cases, target_contract, policy["pool_cutoff_exclusive"], manifest)
    donor_grid = build_grid(donors.sources.passport, target_contract["segment_length_m"])
    targets, _ = build_targets(donors.sources, donor_grid, target_contract)
    usable = targets.observed & ~targets.main_id.isin(policy["unresolved_metal_epochs"])
    snapshots = []
    for date, campaign in targets.loc[usable].groupby("date", sort=True):
        spec = {"ids": sorted(campaign.id.unique()), "target_date": date}
        frame = canonical_snapshot(donors, donor_grid, targets, spec, config, target_contract, labelled=True)
        snapshots.append(frame.loc[frame.observed])
    if not snapshots:
        raise ValueError("No eligible donor campaigns")
    train = pd.concat(snapshots, ignore_index=True)
    grid = build_grid(strict.sources.passport, target_contract["segment_length_m"])
    spec = {"ids": [case["id"]], "target_date": case["target_date"]}
    query = canonical_snapshot(strict, grid, None, spec, config, target_contract, labelled=False)
    if (train.main_id.eq(case["main_id"]).any() or train.sample_id.duplicated().any()
            or len(query) != case["expected_provisional_segments"]):
        raise ValueError("Invalid target isolation or query grid")
    return train, query, strict.audit, donors.audit


def weights_by_campaign(train):
    """Equal total weight per main, then per campaign within it."""
    group_size = train.groupby(["main_id", "target_date"]).sample_id.transform("size")
    campaigns = train.groupby("main_id").target_date.transform("nunique")
    weights = 1. / (group_size * campaigns)
    return (weights / weights.mean()).to_numpy()


class ColdStart:
    """CatBoost on yearly loss as a fraction of the nominal wall thickness."""

    def __init__(self, seed, *, balanced=True):
        self.seed = seed
        self.balanced = balanced
        self.columns = feature_columns("cold")

    def fit(self, train):
        ratio = train.y_mm.to_numpy() / train.nominal_mm.to_numpy()
        response = (1 - ratio) / train.horizon_years.to_numpy()
        weights = weights_by_campaign(train) if self.balanced else np.ones(len(train))
        self.estimator_ = make_estimator("catboost", self.seed)
        step = self.estimator_.steps[-1][0]
        self.estimator_.fit(train[self.columns], response, **{f"{step}__sample_weight": weights})
        return self

    def predict(self, query):
        rate = self.estimator_.predict(query[self.columns])
        nominal = query.nominal_mm.to_numpy()
        thickness = (1 - rate * query.horizon_years.to_numpy()) * nominal
        if not np.isfinite(thickness).all():
            raise ValueError("Nonfinite cold-start thickness")
        return np.clip(thickness, 0, nominal)


def fit_case2(train, query, seed):
    """Hold out all campaigns on one main, then train the submission model."""
    mains = sorted(train.main_id.unique())
    if len(mains) < 2:
        raise ValueError("At least two donor mains are required for transfer validation")
    rows = []
    for main in mains:
        donor, held = train.loc[train.main_id.ne(main)], train.loc[train.main_id.eq(main)]
        estimator = ColdStart(seed).fit(donor)
        for date, group in held.groupby("target_date", sort=True):
            rows.append({"split": "holdout_main", "model": FINAL_MODEL, "held_main": main, "campaign": date,
                         **regression_metrics(group.y_mm, estimator.predict(group))})

    # Same MAIN-3 labels in all comparisons of the allowed training pool.
    main2, main3 = train.loc[train.main_id.eq("MAIN-2")], train.loc[train.main_id.eq("MAIN-3")]
    subsets = [("donors_2018", main2.loc[main2.target_date.lt("2019-01-01")]),
               ("donors_2018_2022", main2.loc[main2.target_date.lt("2023-01-01")]),
               ("all_donors_unweighted", main2)]
    for name, donor in subsets:
        estimator = ColdStart(seed, balanced=False).fit(donor)
        rows.append({"split": "holdout_main", "model": name, "held_main": "MAIN-3", "campaign": main3.target_date.iloc[0],
                     **regression_metrics(main3.y_mm, estimator.predict(main3))})
    return ColdStart(seed).fit(train).predict(query), pd.DataFrame(rows)
