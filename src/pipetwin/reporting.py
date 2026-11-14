"""Atomic publication of reproducible experiment artifacts."""

import json
from pathlib import Path
import tempfile

import joblib


def write_experiment(output_dir: Path, report: dict, bundle: dict, tables: dict):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("Output directory already exists; select a new run directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".experiment-", dir=output_dir.parent) as staging:
        staged_run = Path(staging) / "run"
        staged_run.mkdir()
        for name, table in tables.items():
            table.to_csv(staged_run / name, index=False)
        (staged_run / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        joblib.dump(bundle, staged_run / "models.joblib")
        staged_run.rename(output_dir)
