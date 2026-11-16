"""Build a supplemental uncertainty report from published out-of-fold residuals."""

from pathlib import Path

import numpy as np
import pandas as pd


def residual_quantiles(rows, *, balance_mains=False):
    """Return signed error quantiles, optionally weighting mains and campaigns equally."""
    error = (rows.y_mm - rows.thickness_pred).to_numpy()
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("Calibration requires nonempty, finite holdout errors")
    if not balance_mains:
        return np.quantile(error, [.1, .9])
    counts = rows.groupby(["main_id", "target_date"]).y_mm.transform("size")
    campaigns = rows.groupby("main_id").target_date.transform("nunique")
    weights = 1 / (counts * campaigns).to_numpy()
    order = np.argsort(error)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return error[order[np.searchsorted(cumulative, [.1, .9])]]


def thickness_bounds(rows, delta):
    prediction = rows.thickness_pred.to_numpy()
    nominal = rows.nominal_mm.to_numpy()
    return np.clip(prediction + delta[0], 0, nominal), np.clip(prediction + delta[1], 0, nominal)


def evaluate_coverage(label, calibration, control, *, balance_mains=False):
    delta = residual_quantiles(calibration, balance_mains=balance_mains)
    low, high = thickness_bounds(control, delta)
    covered = (low - 1e-9 <= control.y_mm) & (control.y_mm <= high + 1e-9)
    return dict(control=label, calibration_n=len(calibration), control_n=len(control),
                coverage=covered.mean(), mean_width_mm=(high - low).mean(),
                low_offset_mm=delta[0], high_offset_mm=delta[1])


def main():
    root = Path(__file__).resolve().parents[1]
    report = root / "reports"
    observed = pd.read_csv(report / "validation_predictions.csv", dtype={"id": str})
    profiles = pd.read_csv(report / "profile_data.csv", dtype={"id": str})
    submission = pd.read_csv(root / "result.csv", dtype={"id": str})
    historical = observed.loc[observed.split.eq("spatial_2022")]
    late = observed.loc[observed.split.eq("late_2023")]
    transfer = observed.loc[observed.split.eq("holdout_main")]
    checks = [evaluate_coverage("late_2023", historical, late)]
    for held, control in transfer.groupby("main_id"):
        calibration = transfer.loc[transfer.main_id.ne(held)]
        checks.append(evaluate_coverage(held, calibration, control, balance_mains=True))
    cases = submission.merge(profiles[["id", "segment_id", "case", "nominal_mm"]],
                             on=["id", "segment_id"], validate="one_to_one")
    if not len(cases) == len(submission) == len(profiles):
        raise ValueError("The submission and profile keys must match exactly")
    parts = []
    for case, rows in cases.groupby("case", sort=False):
        calibration = historical if case == "case1" else transfer
        delta = residual_quantiles(calibration, balance_mains=case == "case2")
        lower, upper = thickness_bounds(rows, delta)
        if not (np.all(lower <= upper) and np.isfinite(lower).all() and np.isfinite(upper).all()):
            raise ValueError(f"Invalid interval bounds for {case}")
        parts.append(rows.assign(lower_mm=lower, upper_mm=upper)[
            ["id", "segment_id", "case", "thickness_pred", "lower_mm", "upper_mm"]])
    pd.concat(parts).to_csv(report / "uncertainty_intervals.csv", index=False,
                            float_format="%.8f", lineterminator="\n")
    pd.DataFrame(checks).to_csv(report / "uncertainty_validation.csv", index=False,
                                float_format="%.8f", lineterminator="\n")


if __name__ == "__main__":
    main()
