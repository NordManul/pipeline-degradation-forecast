"""Case-specific source cleaning with provenance and exclusion before statistics.

Raw input frames are never changed. Unverified auxiliary measurements remain in
the audit rather than silently becoming wall-thickness labels or zero exposure.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from .sources import FILES, PASSPORT_COLUMNS, Sources, read_config


AUDIT_COLUMNS = ["case_id", "source_file", "source_row_id", "source_sha256",
                 "rule_id", "field", "raw_value", "action", "reason"]
REGIME_NUMERIC = ["q", "q_oil", "q_gas", "watercut", "fill_coefficient", "gas_content_rate",
                  "v_liquid_true", "v_mix", "density_oil", "density_liquid_work",
                  "viscosity_liquid_work", "t_mix", "t_start", "t_end", "d_mm", "length",
                  "p_start", "p_end", "p_start_fact", "p_end_fact"]
INSPECTION_NUMERIC = ["row_id", "Distance_adjusted", "Distance", "Distance_raw", "accumulated_value",
                      "Wall Thickness", "Residual Wall Thickness", "Defect Depth in %", "Defect Depth",
                      "Angle", "Feature Length", "Feature Width", "Relative Distance", "Pipe Length",
                      "Section Number", "Distance to Reference Point 1, m", "Distance to Reference Point 2, m"]


def local_times(values, *, dayfirst=False):
    """Naive timestamps are Moscow civil time; explicit offsets are converted."""
    def parse(value):
        if pd.isna(value) or not isinstance(value, (str, pd.Timestamp)):
            return pd.NaT
        try:
            parsed = pd.to_datetime(value, dayfirst=dayfirst, errors="raise")
            return parsed.tz_convert("Europe/Moscow").tz_localize(None) if parsed.tzinfo else parsed
        except (ValueError, TypeError, OverflowError):
            return pd.NaT
    # Timestamps recur across sections; parse distinct values only.
    unique = values.drop_duplicates()
    mapping = {value: parse(value) for value in unique if not pd.isna(value)}
    return pd.to_datetime(values.map(mapping), errors="coerce")


@dataclass
class CleanedSources:
    sources: Sources
    audit: pd.DataFrame
    case_id: str
    cutoff: pd.Timestamp
    contract: dict


class Audit:
    def __init__(self, case_id, manifest):
        self.case_id, self.manifest, self.parts = case_id, manifest, []

    def add(self, source, frame, mask, rule, field, action, reason):
        selected = frame.loc[mask]
        if selected.empty:
            return
        values = selected[field].astype(str) if field in selected else pd.Series("", index=selected.index)
        self.parts.append(pd.DataFrame({
            "case_id": self.case_id, "source_file": FILES[source],
            "source_row_id": selected.source_row_id.to_numpy(),
            "source_sha256": self.manifest.get(FILES[source], {}).get("sha256", "in_memory"),
            "rule_id": rule, "field": field, "raw_value": values.to_numpy(), "action": action, "reason": reason,
        }))

    def exclude(self, source, frame, mask, rule, field, reason, action="quarantine_row"):
        mask = pd.Series(mask, index=frame.index).fillna(True).astype(bool)
        self.add(source, frame, mask, rule, field, action, reason)
        return frame.loc[~mask].copy()

    def mask(self, source, frame, mask, rule, field, reason):
        mask = pd.Series(mask, index=frame.index).fillna(False).astype(bool)
        self.add(source, frame, mask, rule, field, "mask_channel", reason)
        frame.loc[mask, field] = np.nan

    def table(self):
        return pd.concat(self.parts, ignore_index=True)[AUDIT_COLUMNS] if self.parts else pd.DataFrame(columns=AUDIT_COLUMNS)


def read_raw_sources(data_dir):
    raw, manifest = {}, {}
    for name, filename in FILES.items():
        path = Path(data_dir) / filename
        raw[name] = pd.read_csv(path, low_memory=False)
        manifest[filename] = {"sha256": sha256(path.read_bytes()).hexdigest(),
                              "rows": len(raw[name]), "columns": len(raw[name].columns)}
    return raw, manifest


def _indexed(frame):
    result = frame.copy(deep=True).reset_index(drop=True)
    result["source_row_id"] = np.arange(len(result), dtype=int) + 2  # CSV line including header
    return result


def _dated(source, frame, field, cutoff, audit):
    event = local_times(frame[field])
    frame = audit.exclude(source, frame, event.isna(), "Q_KEY", field, "invalid_event_date")
    frame["event_at"] = event.loc[frame.index]
    # Exclude future events before parsing units, IDs, or validating their payload.
    frame = audit.exclude(source, frame, frame.event_at.ge(cutoff), "TIME", field,
                          "event_on_or_after_case_cutoff", "exclude_temporally")
    supplied = frame["available_at"] if "available_at" in frame else pd.Series(pd.NaT, index=frame.index)
    available = local_times(supplied)
    invalid = supplied.notna() & available.isna()
    frame = audit.exclude(source, frame, invalid, "Q_KEY", "available_at", "invalid_availability_date")
    frame["available_at"] = available.loc[frame.index].fillna(frame.event_at)
    frame = audit.exclude(source, frame, frame.available_at.lt(frame.event_at), "Q_KEY", "available_at",
                          "availability_precedes_observation")
    return audit.exclude(source, frame, frame.available_at.ge(cutoff), "TIME", "available_at",
                         "available_on_or_after_case_cutoff", "exclude_temporally")


def _ids(source, frame, field, audit, allowed=None):
    numeric = pd.to_numeric(frame[field], errors="coerce")
    bad = ~np.isfinite(numeric) | numeric.le(0) | numeric.ne(np.floor(numeric))
    frame = audit.exclude(source, frame, bad, "Q_KEY", field, "invalid_integer_id")
    frame["id"] = numeric.loc[frame.index].astype("int64").astype(str)
    if allowed is not None:
        frame = audit.exclude(source, frame, ~frame.id.isin(allowed), "Q_KEY", field, "unknown_passport_id")
    return frame


def _numbers(source, frame, columns, audit):
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
        raw = frame[column]
        numeric = pd.to_numeric(raw, errors="coerce").astype(float)
        bad = raw.notna() & ~np.isfinite(numeric)
        audit.add(source, frame, bad, "Q_NUMBER", column, "mask_channel", "nonfinite_or_non_numeric")
        frame[column] = numeric.where(np.isfinite(numeric))
    return frame


def _deduplicate(source, frame, keys, audit):
    # Source line metadata must not make two identical measurements different.
    columns = [c for c in frame if c != "source_row_id"]
    frame = audit.exclude(source, frame, frame.duplicated(columns), "Q_DUPLICATE", keys[0],
                          "exact_duplicate", "deduplicate")
    return audit.exclude(source, frame, frame.duplicated(keys, keep=False), "Q_DUPLICATE", keys[0],
                         "conflicting_duplicate_key")


def _passport(raw, cutoff, audit):
    frame = _ids("passport", _indexed(raw), next(iter(PASSPORT_COLUMNS)), audit)
    # Only allowlisted static fields enter any model-facing table.
    frame = frame[[*PASSPORT_COLUMNS, "source_row_id"]].rename(columns=PASSPORT_COLUMNS)
    frame["id"] = pd.to_numeric(frame.id).astype("int64").astype(str)
    frame = _numbers("passport", frame, ["length_km", "diameter_mm", "nominal_mm"], audit)
    frame["commission_date"] = local_times(frame.commission_date, dayfirst=True)
    bad = frame[["length_km", "diameter_mm", "nominal_mm"]].isna().any(axis=1)
    bad |= frame[["length_km", "diameter_mm", "nominal_mm"]].le(0).any(axis=1)
    bad |= frame.diameter_mm.le(2 * frame.nominal_mm) | frame.commission_date.isna()
    frame = audit.exclude("passport", frame, bad, "Q_GEOMETRY", "nominal_mm", "invalid_static_passport_geometry_or_date")
    frame = audit.exclude("passport", frame, frame.commission_date.ge(cutoff), "TIME", "commission_date",
                          "commission_not_before_case_target", "exclude_temporally")
    frame["length_m"] = frame.pop("length_km") * 1000
    return _deduplicate("passport", frame, ["id"], audit).reset_index(drop=True)


def _profile(raw, passport, audit):
    frame = _ids("profile", _indexed(raw), "id", audit, set(passport.id))
    frame = frame[["id", "total_distance", "z", "source_row_id"]].rename(columns={"total_distance": "distance_m", "z": "altitude_m"})
    frame = _numbers("profile", frame, ["distance_m", "altitude_m"], audit)
    lengths = frame.id.map(passport.set_index("id").length_m)
    bad = frame[["distance_m", "altitude_m"]].isna().any(axis=1) | frame.distance_m.lt(0) | frame.distance_m.gt(lengths + 0.1)
    frame = audit.exclude("profile", frame, bad, "Q_GEOMETRY", "distance_m", "invalid_profile_point_rounding_tolerance_0_1_m")
    return _deduplicate("profile", frame, ["id", "distance_m"], audit).reset_index(drop=True)


def _inspection(raw, passport, cutoff, contract, audit, *, campaign=None):
    frame = _dated("inspection", _indexed(raw), "date", cutoff, audit)
    frame = _ids("inspection", frame, "id", audit, set(passport.id))
    frame["date"] = frame.event_at.dt.normalize()
    if campaign is not None:
        frame = audit.exclude("inspection", frame, frame.date.ne(campaign), "Q_CAMPAIGN", "date",
                              "outside_allowed_peer_campaign", "exclude_temporally")
    commissioned = frame.id.map(passport.set_index("id").commission_date)
    frame = audit.exclude("inspection", frame, frame.date.lt(commissioned), "Q_EPOCH", "date", "inspection_before_resolved_commission")
    dates = frame.date.dt.strftime("%Y-%m-%d")
    known = dates.isin(contract["depth_fraction_dates"] + contract["depth_percent_dates"])
    frame = audit.exclude("inspection", frame, ~known, "Q_DEPTH", "date", "unknown_campaign_depth_unit")
    frame = _numbers("inspection", frame, INSPECTION_NUMERIC, audit)
    bad_id = frame.row_id.isna() | frame.row_id.ne(np.floor(frame.row_id))
    frame = audit.exclude("inspection", frame, bad_id, "Q_KEY", "row_id", "invalid_inspection_row_id")
    frame["row_id"] = frame.row_id.astype("int64")
    frame = _deduplicate("inspection", frame, ["row_id"], audit)
    lengths = frame.id.map(passport.set_index("id").length_m)
    bad = frame.Distance_adjusted.isna() | frame.Distance_adjusted.lt(0) | frame.Distance_adjusted.ge(lengths)
    frame = audit.exclude("inspection", frame, bad, "Q_GEOMETRY", "Distance_adjusted", "outside_local_half_open_section")
    frame["distance_m"] = frame.Distance_adjusted
    frame["nominal_observed_mm"] = frame["Wall Thickness"]
    frame["thickness_mm"] = frame["Residual Wall Thickness"]
    fractions = frame.date.dt.strftime("%Y-%m-%d").isin(contract["depth_fraction_dates"])
    frame["depth_fraction"] = frame["Defect Depth in %"] / np.where(fractions, 1, 100)
    frame["depth_mm"] = frame.nominal_observed_mm * frame.depth_fraction
    frame["reconstruction_error_mm"] = (frame.nominal_observed_mm - frame.depth_mm - frame.thickness_mm).abs()
    tolerance = contract["thickness_tolerance_mm"]
    invalid = frame.thickness_mm.notna() & (
        frame.thickness_mm.le(0) | frame.nominal_observed_mm.isna() | frame.nominal_observed_mm.le(0)
        | frame.thickness_mm.gt(frame.nominal_observed_mm + tolerance))
    invalid_depth = frame.depth_fraction.notna() & ~frame.depth_fraction.between(0, 1)
    invalid |= invalid_depth | frame.reconstruction_error_mm.gt(tolerance)
    # Retain geometry for weld/defect matching but forbid an invalid measurement as a label/rate.
    audit.mask("inspection", frame, invalid, "Q_THICKNESS", "thickness_mm", "invalid_residual_or_depth_reconstruction")
    frame["is_corrosion"] = frame["Feature Description"].fillna("").str.lower().str.contains("корроз|потеря металла", regex=True)
    return frame.sort_values(["date", "id", "row_id"]).reset_index(drop=True)


def _regime(raw, passport, cutoff, audit):
    frame = _dated("regime", _indexed(raw), "calc_date", cutoff, audit)
    frame = _ids("regime", frame, "simple_section_id", audit, set(passport.id))
    frame["simple_section_id"] = frame.id
    frame["date"] = frame.event_at
    frame["calc_date"] = frame.event_at
    commissioned = frame.id.map(passport.set_index("id").commission_date)
    frame = audit.exclude("regime", frame, frame.date.lt(commissioned), "Q_EPOCH", "date", "regime_before_resolved_commission")
    known_event = frame.id.eq("1250009583") & frame.date.dt.normalize().eq(pd.Timestamp("2018-10-07"))
    frame = audit.exclude("regime", frame, known_event, "Q_KNOWN_EVENT", "q", "unresolved_joint_flow_pressure_anomaly_U09")
    frame = _numbers("regime", frame, REGIME_NUMERIC, audit)
    frame = _deduplicate("regime", frame, ["id", "date"], audit)
    lookup = passport.set_index("id")
    mismatch = ~np.isclose(frame.d_mm * 1000, frame.id.map(lookup.diameter_mm), atol=1e-6, rtol=0)
    mismatch |= ~np.isclose(frame.length, frame.id.map(lookup.length_m), atol=0.1, rtol=0)
    frame = audit.exclude("regime", frame, mismatch, "Q_GEOMETRY", "d_mm", "regime_geometry_disagrees_with_passport")
    for column in ["q", "q_oil", "q_gas", "v_liquid_true", "v_mix"]:
        audit.mask("regime", frame, frame[column].lt(0), "Q_FLOW", column, "negative_unsigned_flow_or_speed")
    for column, maximum in [("watercut", 100), ("fill_coefficient", 1), ("gas_content_rate", 1)]:
        audit.mask("regime", frame, frame[column].notna() & ~frame[column].between(0, maximum), "Q_FRACTION", column, "outside_declared_fraction_range")
    for column in ["density_oil", "density_liquid_work", "viscosity_liquid_work"]:
        audit.mask("regime", frame, frame[column].le(0), "Q_DENSITY_VISCOSITY", column, "nonpositive_value_unavailable_for_physical_division")
    placeholder = frame.t_mix.eq(0) & frame[["t_start", "t_end"]].isna().all(axis=1)
    audit.mask("regime", frame, placeholder, "Q_TEMPERATURE", "t_mix", "zero_mixture_temperature_without_endpoint_calculation")
    for column in ["p_start_fact", "p_end_fact"]:
        audit.mask("regime", frame, frame[column].eq(0), "Q_PRESSURE", column, "unconfirmed_zero_fact_pressure_sentinel")
    comparable = frame.q.gt(0) & frame.density_oil.gt(0) & frame.q_oil.notna() & frame.watercut.notna()
    reconstructed = 100 * (1 - 1000 * frame.q_oil / (frame.density_oil * frame.q))
    inconsistent = comparable & (reconstructed - frame.watercut).abs().gt(0.02)
    for column in ["q", "q_oil", "density_oil", "watercut"]:
        audit.mask("regime", frame, inconsistent, "Q_BALANCE", column, "inconsistent_oil_water_volume_balance_unresolved_input")
    frame["water_fraction"] = frame.watercut / 100
    frame["oil_volume_m3_day"] = 1000 * frame.q_oil / frame.density_oil
    frame["gas_flow_m3_day_source_conditions"] = 1000 * frame.q_gas
    frame["liquid_flow_m3_day"] = frame.q
    frame["oil_mass_tonne_day"] = frame.q_oil
    # Unknown origin of the precomputed corrosion index; it cannot reach an allowlist by accident.
    frame = frame.drop(columns=["pit_corrosion_intensity"], errors="ignore")
    return frame.sort_values(["id", "date"]).reset_index(drop=True)


def _auxiliary(raw, passport, cutoff, audit):
    ultrasound = _dated("ultrasound", _indexed(raw["ultrasound"]), "date", cutoff, audit)
    ultrasound = _ids("ultrasound", ultrasound, "id", audit, set(passport.id))
    audit.add("ultrasound", ultrasound, np.ones(len(ultrasound), dtype=bool), "Q_ULTRASOUND",
              "residual_wall_thickness", "exclude_unverified_source", "object_nominal_or_measurement_method_unresolved")
    coupons = _dated("coupons", _indexed(raw["coupons"]), "Sample Removal Date", cutoff, audit)
    coupons = _ids("coupons", coupons, "Simple Section ID", audit, set(passport.id))
    installed = local_times(coupons["Sample Installation Date"])
    coupons = audit.exclude("coupons", coupons, installed.isna() | installed.gt(coupons.event_at),
                            "Q_COUPONS", "Sample Installation Date", "invalid_or_reversed_exposure")
    duration = (coupons.event_at - installed.loc[coupons.index]).dt.total_seconds() / 86400
    reported = pd.to_numeric(coupons["Exposure Time, days"], errors="coerce")
    coupons = audit.exclude("coupons", coupons, reported.isna() | reported.le(0) | (duration - reported).abs().gt(1e-6),
                            "Q_COUPONS", "Exposure Time, days", "unresolved_exposure_duration")
    audit.add("coupons", coupons, np.ones(len(coupons), dtype=bool), "Q_COUPONS", "Measurement Method",
              "exclude_unverified_source", "laboratory_availability_and_units_unconfirmed")
    return ultrasound.iloc[:0].copy(), coupons.iloc[:0].copy()


def clean_sources(raw, case_id, contract, target_contract, manifest=None):
    case = contract["cases"][case_id]
    cutoff = pd.Timestamp(case["cutoff_exclusive"])
    if cutoff != pd.Timestamp(case["target_date"]) or contract["time_policy"]["cutoff_operator"] != "strictly_before":
        raise ValueError("Only the declared strict pre-target cutoff is implemented")
    return _clean_as_of(raw, case_id, contract, target_contract, cutoff, manifest)


def clean_case2_donors(raw, contract, target_contract, pool_cutoff, manifest=None):
    """Separate retrospective pool. Remove the complete target main before cleaning dated inputs."""
    return _clean_as_of(raw, "case2", contract, target_contract, pd.Timestamp(pool_cutoff), manifest, donors_only=True)


def clean_case1_peer_inspections(raw_inspection, strict, peer_ids, target_contract, freeze):
    """Read the eligible peer inspection only, leaving the strict feature sources intact."""
    case = strict.contract["cases"]["case1"]
    passport = strict.sources.passport
    peers = passport.loc[passport.id.isin(peer_ids)]
    if (strict.case_id != "case1" or set(peers.id) != set(peer_ids) or peers.empty
            or peers.id.eq(case["id"]).any() or not peers.main_id.eq(case["main_id"]).all()
            or not strict.cutoff < freeze):
        raise ValueError("Peer labels require explicit non-target sections of the same main and a later dataset freeze")
    audit = Audit("case1_peers", strict.sources.manifest)
    inspection = _inspection(raw_inspection, peers, freeze, target_contract, audit, campaign=strict.cutoff)
    return inspection, audit.table()


def _clean_as_of(raw, case_id, contract, target_contract, cutoff, manifest, *, donors_only=False):
    case = contract["cases"][case_id]
    audit = Audit(case_id, manifest or {})
    passport = _passport(raw["passport"], cutoff, audit)
    target = passport.loc[passport.id.eq(case["id"])]
    if len(target) != 1 or not target.main_id.eq(case["main_id"]).all():
        raise ValueError("Official target passport failed validation")
    if donors_only:
        passport = passport.loc[passport.main_id.ne(case["main_id"])].copy()
    profile = _profile(raw["profile"], passport, audit)
    inspection = _inspection(raw["inspection"], passport, cutoff, target_contract, audit)
    regime = _regime(raw["regime"], passport, cutoff, audit)
    ultrasound, coupons = _auxiliary(raw, passport, cutoff, audit)
    sources = Sources(passport, profile, regime, inspection, ultrasound, coupons, manifest or {})
    return CleanedSources(sources, audit.table(), case_id, cutoff, contract)


def load_case_sources(data_dir, case_id, contracts_dir="contracts", target_contract_path="configs/data_contract.yaml"):
    contract = read_config(Path(contracts_dir) / "cases.yaml")
    raw, manifest = read_raw_sources(data_dir)
    return clean_sources(raw, case_id, contract, read_config(target_contract_path), manifest)
