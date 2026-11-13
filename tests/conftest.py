from pathlib import Path

import pytest

from pipetwin.sources import load_sources, read_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def contract():
    return read_config(ROOT / "configs/data_contract.yaml")


@pytest.fixture(scope="session")
def sources(contract):
    return load_sources(ROOT / "data/raw", contract)
