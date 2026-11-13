"""Read only the source columns allowed by the historical E02 contract."""

from pathlib import Path

import numpy as np
import pandas as pd


POT_PREFIX = "Данные по протяжённости в структуре сети OISPIPE Паспортизация_"
POT_COLUMNS = {
    POT_PREFIX + "ID простого участка трубопровода": "id",
    POT_PREFIX + "ID основного участка трубопровода": "main_id",
    POT_PREFIX + "L простого участка, км": "length_km",
}
TECH_COLUMNS = [
    "simple_section_id", "calc_date", "q", "watercut", "fill_coefficient",
    "v_liquid_true", "t_start", "t_end",
]
VTD_COLUMNS = [
    "id", "date", "Distance", "Residual Wall Thickness", "Feature Description",
    "Defect Type", "trait",
]


def canonical_ids(values: pd.Series) -> pd.Series:
    ids = values.astype("string").str.strip()
    if not ids.str.fullmatch(r"\d+(?:\.0+)?", na=False).all():
        raise ValueError("Section IDs must be non-null integer identifiers")
    return ids.str.replace(r"\.0+$", "", regex=True)


def read_sources(raw_dir: Path, terrain: bool = False) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    sections = pd.read_csv(raw_dir / "pot_south.csv", usecols=list(POT_COLUMNS))
    sections = sections.rename(columns=POT_COLUMNS)
    sections["id"] = canonical_ids(sections["id"])
    sections["length_m"] = pd.to_numeric(sections.pop("length_km")) * 1000
    if sections.id.duplicated().any() or sections.main_id.isna().any():
        raise ValueError("Passport must have one row and a MAIN group per section")
    if not (np.isfinite(sections.length_m) & (sections.length_m > 0)).all():
        raise ValueError("Passport lengths must be finite and positive")

    tech = pd.read_csv(
        raw_dir / "tech_mode_for_embedings_with_length_with_temperatures_ids16.csv",
        usecols=TECH_COLUMNS,
    ).rename(columns={"simple_section_id": "id"})
    tech["id"] = canonical_ids(tech["id"])
    tech["calc_date"] = pd.to_datetime(tech.calc_date, errors="raise")
    vtd = pd.read_csv(raw_dir / "vtd_south_train.csv", usecols=VTD_COLUMNS)
    vtd["id"] = canonical_ids(vtd["id"])
    vtd["date"] = pd.to_datetime(vtd.date, errors="raise")
    if tech.calc_date.isna().any() or vtd.date.isna().any():
        raise ValueError("Missing source timestamps")
    sources = {"sections": sections, "tech": tech, "vtd": vtd}
    if terrain:
        profile = pd.read_csv(
            raw_dir / "altitudes_south.csv", usecols=["id", "total_distance", "z"]
        )
        profile["id"] = canonical_ids(profile["id"])
        sources["profile"] = profile
    return sources
