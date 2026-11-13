"""Read source tables and apply explicit, audited unit conversions."""

from dataclasses import dataclass
from hashlib import sha256
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


def normalize_inspections(raw, contract):
    result = raw.copy()
    result["id"] = canonical_ids(result["id"])
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    date_strings = result["date"].dt.strftime("%Y-%m-%d")
    fractions = date_strings.isin(contract["depth_fraction_dates"])
    percentages = date_strings.isin(contract["depth_percent_dates"])
    if not (fractions | percentages).all():
        raise ValueError("Unknown inspection campaign: confirm its depth units first")
    result["depth_fraction"] = result["Defect Depth in %"] / np.where(fractions, 1, 100)
    result["nominal_observed_mm"] = result["Wall Thickness"]
    result["thickness_mm"] = result["Residual Wall Thickness"]
    result["distance_m"] = result["Distance_adjusted"]
    result["depth_mm"] = result["nominal_observed_mm"] * result["depth_fraction"]
    result["reconstruction_error_mm"] = (
        result["nominal_observed_mm"] - result["depth_mm"] - result["thickness_mm"]
    ).abs()
    if (result["reconstruction_error_mm"] > contract["thickness_tolerance_mm"]).any():
        raise ValueError("Inspection depth units disagree with residual thickness")
    description = result["Feature Description"].fillna("").str.lower()
    result["is_corrosion"] = description.str.contains("корроз|потеря металла", regex=True)
    return result


def load_sources(data_dir, contract):
    if contract.get("grid_scope") != "simple_section" or contract.get("endpoint_policy") != "exclude":
        raise ValueError("Only the audited local grid with excluded out-of-range points is implemented")
    data_dir = Path(data_dir)
    raw, manifest = {}, {}
    for name, filename in FILES.items():
        path = data_dir / filename
        raw[name] = pd.read_csv(path, low_memory=False)
        manifest[filename] = {
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "rows": len(raw[name]),
            "columns": len(raw[name].columns),
        }
    passport = raw["passport"][list(PASSPORT_COLUMNS)].rename(columns=PASSPORT_COLUMNS)
    passport["id"] = canonical_ids(passport["id"])
    passport["length_m"] = passport.pop("length_km") * 1000
    passport["commission_date"] = pd.to_datetime(
        passport["commission_date"], dayfirst=True, errors="raise"
    )
    if passport["id"].duplicated().any():
        raise ValueError("Duplicate passport IDs")
    geometry = passport[["length_m", "diameter_mm", "nominal_mm"]]
    if not np.isfinite(geometry).all().all() or (geometry <= 0).any().any():
        raise ValueError("Missing or non-positive passport geometry")
    if passport["commission_date"].isna().any():
        raise ValueError("Missing commissioning date")
    profile = raw["profile"][["id", "total_distance", "z"]].drop_duplicates().copy()
    profile["id"] = canonical_ids(profile["id"])
    profile = profile.rename(columns={"total_distance": "distance_m", "z": "altitude_m"})
    if profile.duplicated(["id", "distance_m"]).any():
        raise ValueError("Conflicting profile heights at the same distance")
    regime = raw["regime"].rename(columns={"simple_section_id": "id"}).copy()
    regime["id"] = canonical_ids(regime["id"])
    regime["date"] = pd.to_datetime(regime["calc_date"], errors="raise")
    if regime.duplicated(["id", "date"]).any():
        raise ValueError("Duplicate regime timestamps require an explicit resolution")
    regime["water_fraction"] = regime["watercut"] / 100
    regime["oil_volume_m3_day"] = regime["q_oil"] * 1000 / regime["density_oil"].where(
        regime["density_oil"] > 0
    )
    # Zero mixture temperature with no endpoint calculation is an audited placeholder.
    missing_temperature = regime["t_mix"].eq(0) & regime[["t_start", "t_end"]].isna().all(axis=1)
    regime.loc[missing_temperature, "t_mix"] = np.nan
    for column in ["density_liquid_work", "viscosity_liquid_work"]:
        regime.loc[regime[column].eq(0) & regime["q"].eq(0), column] = np.nan
    inspection = normalize_inspections(raw["inspection"], contract)
    known = set(passport["id"])
    for name, table in [("profile", profile), ("regime", regime), ("inspection", inspection)]:
        if not set(table["id"]).issubset(known):
            raise ValueError(f"Unknown passport ID in {name}")
    check = regime.merge(passport[["id", "diameter_mm", "length_m"]], on="id", validate="many_to_one")
    if not np.allclose(check["d_mm"] * 1000, check["diameter_mm"], atol=1e-6):
        raise ValueError("Regime diameter disagrees with passport")
    if not np.allclose(check["length"], check["length_m"], atol=0.1):
        raise ValueError("Regime length disagrees with passport")
    return Sources(passport, profile, regime, inspection, raw["ultrasound"], raw["coupons"], manifest)
