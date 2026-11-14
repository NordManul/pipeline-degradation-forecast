"""Input support shifts and prediction saturation, without consulting test labels."""

import numpy as np
import pandas as pd

from .features import feature_columns
from .physical_features import ablation_columns


def input_shift(train, test, scenario, split):
    columns = list(dict.fromkeys(feature_columns(scenario) + ablation_columns("terrain", scenario)))
    records = []
    for name in columns:
        reference = train.loc[train.observed, name].dropna()
        values = test[name].dropna()
        low, high = reference.min(), reference.max()
        tolerance = 1e-9 * max(1, abs(low), abs(high)) if len(reference) else 0
        outside = ((values < low - tolerance) | (values > high + tolerance)).mean() if len(reference) and len(values) else np.nan
        records.append({"split": split, "feature": name, "train_min": low, "train_max": high,
                        "train_std": reference.std(ddof=0), "test_min": values.min(), "test_max": values.max(),
                        "fraction_outside_training_range": outside,
                        "test_missing_fraction": float(test[name].isna().mean())})
    return pd.DataFrame(records)


def prediction_diagnostics(models, frame, split):
    records = []
    nominal = frame.nominal_mm.to_numpy()
    for name, model in models.items():
        raw = np.asarray(model.predict(frame), dtype=float)
        bounded = np.clip(raw, 0, nominal)
        records.append({"split": split, "model": name, "cells": len(frame),
                        "raw_min_mm": float(raw.min()), "raw_max_mm": float(raw.max()),
                        "fraction_above_nominal": float((raw > nominal).mean()),
                        "fraction_at_zero": float((bounded == 0).mean()),
                        "constant_after_bounds": bool(np.std(bounded) <= 1e-12)})
    return pd.DataFrame(records)
