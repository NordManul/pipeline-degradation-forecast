"""Public model factory for the canonical contract 1.0.0 inputs."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd


REQUIRED_MODEL_NAMES = ("constant_rate", "positive_rate", "hybrid_residual")
OPTIONAL_MODEL_NAMES = ("hybrid_scale",)


class ThicknessEstimator(Protocol):
    """Shared estimator interface; labels are separate from prediction inputs."""

    diagnostics_: dict[str, Any]

    def fit(self, frame: pd.DataFrame, y_mm: Sequence[float]) -> "ThicknessEstimator":
        ...

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        ...


def make_model(
    name: str,
    *,
    scenario: Literal["between", "cold"],
    exposure: pd.DataFrame,
    cutoff_exclusive: str,
    feature_columns: Sequence[str],
    seed: int = 42,
    options: Mapping[str, Any] | None = None,
) -> ThicknessEstimator:
    """Create an isolated case/scenario model without raw I/O or target building.

    Fit preprocessing only on the training frame's exposure_id values.
    The caller supplies a separate factory/context for each official case.
    """
    from .models import PhysicsEstimator

    return PhysicsEstimator(name, scenario=scenario, exposure=exposure,
                            cutoff_exclusive=cutoff_exclusive,
                            feature_columns=feature_columns, seed=seed, options=options)
