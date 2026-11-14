"""Read source tables and apply explicit, audited unit conversions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


FILES = {
    "passport": "pot_south.csv",
    "profile": "altitudes_south.csv",
    "regime": "tech_mode_for_embedings_with_length_with_temperatures_ids16.csv",
    "inspection": "vtd_south_train.csv",
    "ultrasound": "uzt_south.csv",
    "coupons": "repcorr_south.csv",
}
PASSPORT_COLUMNS = {
    "Данные по протяжённости в структуре сети OISPIPE Паспортизация_ID простого участка трубопровода": "id",
    "Данные по протяжённости в структуре сети OISPIPE Паспортизация_ID основного участка трубопровода": "main_id",
    "Данные по протяжённости в структуре сети OISPIPE Паспортизация_ID направления трубопровода": "direction_id",
    "Данные по протяжённости в структуре сети OISPIPE Паспортизация_L простого участка, км": "length_km",
    "Общие данные_D, мм": "diameter_mm",
    "Общие данные_Толщина  стенки по исполнительной доку-ии, мм": "nominal_mm",
    "Общие данные_Дата ввода": "commission_date",
}


@dataclass
class Sources:
    passport: pd.DataFrame
    profile: pd.DataFrame
    regime: pd.DataFrame
    inspection: pd.DataFrame
    ultrasound: pd.DataFrame
    coupons: pd.DataFrame
    manifest: dict


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a configuration mapping in {path}")
    return config


def canonical_ids(values):
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError("Missing or non-finite pipeline ID")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("Pipeline IDs must be integers")
    return numeric.astype("int64").astype(str)
