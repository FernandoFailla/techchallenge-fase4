"""Strict YAML configuration for deterministic workflows."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEMO_EXPECTED_FIELDS = {
    "seed",
    "training_rows",
    "scoring_rows",
    "review_threshold",
    "limit_balance_multiplier",
    "age_shift",
    "experiment_name",
    "runtime_dir",
}

TRAINING_EXPECTED_FIELDS = {
    "seed",
    "train_fraction",
    "validation_fraction",
    "test_fraction",
    "simulation_fraction",
    "experiment_name",
    "model_version",
    "data_dir",
    "runtime_dir",
}


@dataclass(frozen=True)
class DemoConfig:
    seed: int
    training_rows: int
    scoring_rows: int
    review_threshold: float
    limit_balance_multiplier: float
    age_shift: int
    experiment_name: str
    runtime_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    simulation_fraction: float
    experiment_name: str
    model_version: str
    data_dir: Path
    runtime_dir: Path


def _require_integer(payload: dict[str, Any], field: str, *, minimum: int) -> int:
    value = payload[field]
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_float(payload: dict[str, Any], field: str) -> float:
    value = payload[field]
    if type(value) not in {int, float}:
        raise ValueError(f"{field} must be numeric")
    return float(value)


def load_demo_config(path: str | Path) -> DemoConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration must be a YAML mapping")

    fields = set(payload)
    missing = sorted(DEMO_EXPECTED_FIELDS - fields)
    unknown = sorted(fields - DEMO_EXPECTED_FIELDS)
    if missing:
        raise ValueError(f"missing configuration fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(unknown)}")

    seed = _require_integer(payload, "seed", minimum=0)
    training_rows = _require_integer(payload, "training_rows", minimum=100)
    scoring_rows = _require_integer(payload, "scoring_rows", minimum=100)
    age_shift = _require_integer(payload, "age_shift", minimum=-30)
    if age_shift == 0 or age_shift > 30:
        raise ValueError("age_shift must be between -30 and 30 and cannot be zero")

    review_threshold = _require_float(payload, "review_threshold")
    if not 0 <= review_threshold <= 1:
        raise ValueError("review_threshold must be between 0 and 1")

    limit_balance_multiplier = _require_float(payload, "limit_balance_multiplier")
    if not 0 < limit_balance_multiplier < 1:
        raise ValueError("limit_balance_multiplier must be between 0 and 1")

    experiment_name = payload["experiment_name"]
    runtime_dir = payload["runtime_dir"]
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("experiment_name must be a non-empty string")
    if not isinstance(runtime_dir, str) or not runtime_dir.strip():
        raise ValueError("runtime_dir must be a non-empty path")

    return DemoConfig(
        seed=seed,
        training_rows=training_rows,
        scoring_rows=scoring_rows,
        review_threshold=review_threshold,
        limit_balance_multiplier=limit_balance_multiplier,
        age_shift=age_shift,
        experiment_name=experiment_name,
        runtime_dir=Path(runtime_dir),
    )


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration must be a YAML mapping")

    fields = set(payload)
    missing = sorted(TRAINING_EXPECTED_FIELDS - fields)
    unknown = sorted(fields - TRAINING_EXPECTED_FIELDS)
    if missing:
        raise ValueError(f"missing configuration fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(unknown)}")

    fractions = {
        name: _require_float(payload, name)
        for name in (
            "train_fraction",
            "validation_fraction",
            "test_fraction",
            "simulation_fraction",
        )
    }
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("partition fractions must be greater than zero")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("partition fractions must sum to 1")

    string_fields = ("experiment_name", "model_version", "data_dir", "runtime_dir")
    for field in string_fields:
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"{field} must be a non-empty string")

    return TrainingConfig(
        seed=_require_integer(payload, "seed", minimum=0),
        **fractions,
        experiment_name=payload["experiment_name"],
        model_version=payload["model_version"],
        data_dir=Path(payload["data_dir"]),
        runtime_dir=Path(payload["runtime_dir"]),
    )
