from pathlib import Path

import pytest

from credit_monitoring.config import load_demo_config


def write_config(path: Path, *, extra: str = "", omit: str = "") -> None:
    values = {
        "seed": "42",
        "training_rows": "500",
        "scoring_rows": "200",
        "review_threshold": "0.45",
        "limit_balance_multiplier": "0.65",
        "age_shift": "8",
        "experiment_name": "credit-monitoring-d4",
        "runtime_dir": "artifacts/runtime",
    }
    values.pop(omit, None)
    body = "\n".join(f"{key}: {value}" for key, value in values.items())
    path.write_text(f"{body}\n{extra}", encoding="utf-8")


def test_load_demo_config() -> None:
    config = load_demo_config("configs/d4-demo.yaml")

    assert config.seed == 42
    assert config.runtime_dir == Path("artifacts/runtime")


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "unknown.yaml"
    write_config(path, extra="unexpected: true\n")

    with pytest.raises(ValueError, match="unknown configuration fields: unexpected"):
        load_demo_config(path)


def test_config_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"
    write_config(path, omit="seed")

    with pytest.raises(ValueError, match="missing configuration fields: seed"):
        load_demo_config(path)
