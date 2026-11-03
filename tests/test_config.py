from pathlib import Path

import pytest

from credit_monitoring.config import load_demo_config, load_training_config


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


def test_load_training_config() -> None:
    config = load_training_config("configs/training.yaml")

    assert config.train_fraction == 0.6
    assert config.model_version == "real-logistic-v1"


def test_training_config_rejects_fractions_that_do_not_sum_to_one(tmp_path: Path) -> None:
    path = tmp_path / "training.yaml"
    path.write_text(
        """seed: 42
train_fraction: 0.60
validation_fraction: 0.20
test_fraction: 0.15
simulation_fraction: 0.10
experiment_name: credit-training-e4
model_version: real-logistic-v1
data_dir: data
runtime_dir: artifacts/runtime
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="partition fractions must sum to 1"):
        load_training_config(path)
