"""Strict YAML configuration for deterministic workflows."""

import math
from dataclasses import dataclass
from datetime import datetime
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

DRIFT_EXPECTED_FIELDS = {
    "seed",
    "cohort_rows",
    "stress_strength",
    "concept_delta",
    "psi_bins",
    "psi_smoothing",
    "psi_minimum_rows",
    "bootstrap_samples",
    "experiment_name",
    "expected_model_version",
    "data_path",
    "model_path",
    "runtime_dir",
    "scoring_at",
    "outcome_available_at",
}

FAIRNESS_EXPECTED_FIELDS = {
    "seed",
    "experiment_name",
    "e5_config_path",
    "minimum_group_rows",
    "minimum_group_events",
    "minimum_group_non_events",
    "bootstrap_samples",
    "disparity_review_threshold",
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


@dataclass(frozen=True)
class DriftConfig:
    seed: int
    cohort_rows: int
    stress_strength: float
    concept_delta: float
    psi_bins: int
    psi_smoothing: float
    psi_minimum_rows: int
    bootstrap_samples: int
    experiment_name: str
    expected_model_version: str
    data_path: Path
    model_path: Path
    runtime_dir: Path
    scoring_at: datetime
    outcome_available_at: datetime


@dataclass(frozen=True)
class FairnessConfig:
    seed: int
    experiment_name: str
    e5_config_path: Path
    minimum_group_rows: int
    minimum_group_events: int
    minimum_group_non_events: int
    bootstrap_samples: int
    disparity_review_threshold: float
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
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


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


def load_drift_config(path: str | Path) -> DriftConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration must be a YAML mapping")

    fields = set(payload)
    missing = sorted(DRIFT_EXPECTED_FIELDS - fields)
    unknown = sorted(fields - DRIFT_EXPECTED_FIELDS)
    if missing:
        raise ValueError(f"missing configuration fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(unknown)}")

    cohort_rows = _require_integer(payload, "cohort_rows", minimum=100)
    psi_bins = _require_integer(payload, "psi_bins", minimum=2)
    psi_minimum_rows = _require_integer(payload, "psi_minimum_rows", minimum=50)
    bootstrap_samples = _require_integer(payload, "bootstrap_samples", minimum=50)
    stress_strength = _require_float(payload, "stress_strength")
    concept_delta = _require_float(payload, "concept_delta")
    psi_smoothing = _require_float(payload, "psi_smoothing")
    if stress_strength <= 0 or concept_delta <= 0 or psi_smoothing <= 0:
        raise ValueError("drift strengths and PSI smoothing must be greater than zero")

    string_fields = (
        "experiment_name",
        "expected_model_version",
        "data_path",
        "model_path",
        "runtime_dir",
        "scoring_at",
        "outcome_available_at",
    )
    for field in string_fields:
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    try:
        scoring_at = datetime.fromisoformat(payload["scoring_at"])
        outcome_available_at = datetime.fromisoformat(payload["outcome_available_at"])
    except ValueError as error:
        raise ValueError("drift timestamps must use ISO-8601") from error
    if scoring_at.tzinfo is not None or outcome_available_at.tzinfo is not None:
        raise ValueError("drift timestamps must use naive UTC")
    if scoring_at >= outcome_available_at:
        raise ValueError("scoring_at must precede outcome_available_at")

    return DriftConfig(
        seed=_require_integer(payload, "seed", minimum=0),
        cohort_rows=cohort_rows,
        stress_strength=stress_strength,
        concept_delta=concept_delta,
        psi_bins=psi_bins,
        psi_smoothing=psi_smoothing,
        psi_minimum_rows=psi_minimum_rows,
        bootstrap_samples=bootstrap_samples,
        experiment_name=payload["experiment_name"],
        expected_model_version=payload["expected_model_version"],
        data_path=Path(payload["data_path"]),
        model_path=Path(payload["model_path"]),
        runtime_dir=Path(payload["runtime_dir"]),
        scoring_at=scoring_at,
        outcome_available_at=outcome_available_at,
    )


def load_fairness_config(path: str | Path) -> FairnessConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration must be a YAML mapping")

    fields = set(payload)
    missing = sorted(FAIRNESS_EXPECTED_FIELDS - fields)
    unknown = sorted(fields - FAIRNESS_EXPECTED_FIELDS)
    if missing:
        raise ValueError(f"missing configuration fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(unknown)}")

    integer_fields = (
        "minimum_group_rows",
        "minimum_group_events",
        "minimum_group_non_events",
        "bootstrap_samples",
    )
    integers = {field: _require_integer(payload, field, minimum=1) for field in integer_fields}
    if integers["bootstrap_samples"] < 50:
        raise ValueError("bootstrap_samples must be at least 50")
    disparity_review_threshold = _require_float(payload, "disparity_review_threshold")
    if not 0 <= disparity_review_threshold <= 1:
        raise ValueError("disparity_review_threshold must be between 0 and 1")

    string_fields = ("experiment_name", "e5_config_path", "runtime_dir")
    for field in string_fields:
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"{field} must be a non-empty string")

    return FairnessConfig(
        seed=_require_integer(payload, "seed", minimum=0),
        experiment_name=payload["experiment_name"],
        e5_config_path=Path(payload["e5_config_path"]),
        **integers,
        disparity_review_threshold=disparity_review_threshold,
        runtime_dir=Path(payload["runtime_dir"]),
    )
