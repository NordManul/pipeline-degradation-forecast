"""Model registry. Every estimator returns wall thickness in millimetres."""

from .baselines import GlobalRate, LocalRate, Persistence
from .physics import ExposureModel, PhysicsResidual
from .tabular import TabularModel
from ..physical_features import ablation_columns


MODEL_NAMES = ["persistence", "smooth_persistence", "global_rate", "local_rate", "ridge",
               "ridge_direct", "huber", "gam", "random_forest", "catboost", "physics", "physics_residual",
               "e02_mode", "e02_water", "e02_thermal", "e02_terrain"]


def make_model(name, scenario, seed=42):
    if scenario not in {"between", "cold"}:
        raise ValueError(f"Unknown scenario: {scenario}")
    if name in {"e02_mode", "e02_water", "e02_thermal", "e02_terrain"}:
        return TabularModel(scenario, "ridge_direct", seed, columns=ablation_columns(name.removeprefix("e02_"), scenario))
    if name in {"persistence", "smooth_persistence"}:
        return Persistence(scenario, smooth=name == "smooth_persistence")
    if name == "global_rate":
        return GlobalRate(scenario)
    if name == "local_rate":
        return LocalRate(scenario)
    if name == "physics":
        return ExposureModel(scenario)
    if name == "physics_residual":
        return PhysicsResidual(scenario)
    if name in {"ridge", "ridge_direct", "huber", "gam", "random_forest", "catboost"}:
        return TabularModel(scenario, name, seed)
    raise ValueError(f"Unknown model: {name}")
