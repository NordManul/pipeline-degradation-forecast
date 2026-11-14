"""Public factory reserved for the second agent; see contracts/model-api.md.

This scaffold is intentionally unimplemented and is not registered as a model.
"""

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
    """Implement within this package; no raw I/O, target building or CV here.

    Fit preprocessing only on the training frame's exposure_id values.
    The caller supplies a separate factory/context for each official case.
    """
    raise NotImplementedError("Agent 2 implementation pending; contract version 1.0.0")
