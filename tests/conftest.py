from pathlib import Path

import pytest

from pipetwin.quality import clean_sources, read_raw_sources
from pipetwin.sources import read_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def contract():
    return read_config(ROOT / "configs/data_contract.yaml")


@pytest.fixture(scope="session")
def data_inputs(contract):
    raw, manifest = read_raw_sources(ROOT / "data/raw")
    return raw, manifest, read_config(ROOT / "contracts/cases.yaml"), contract, read_config(ROOT / "configs/forecast.yaml")


@pytest.fixture(scope="session")
def sources(data_inputs):
    raw, manifest, cases, contract, _ = data_inputs
    return clean_sources(raw, "case1", cases, contract, manifest).sources
