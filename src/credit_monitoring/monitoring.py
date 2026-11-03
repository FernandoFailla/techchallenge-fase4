"""Data-drift evaluation at the Evidently boundary."""

from dataclasses import dataclass
from typing import Any

import polars as pl
from evidently import Report
from evidently.presets import DataDriftPreset

from credit_monitoring.contracts import FEATURE_COLUMNS, validate_scoring_batch


@dataclass(frozen=True)
class DriftSummary:
    drifted_features: tuple[str, ...]
    drifted_count: int
    drifted_share: float
    alert: bool


@dataclass(frozen=True)
class DriftAnalysis:
    snapshot: Any
    summary: DriftSummary


def _metric_detected_drift(metric: dict[str, Any]) -> bool:
    config = metric["config"]
    score = float(metric["value"])
    threshold = float(config["threshold"])
    method = str(config["method"]).lower()
    return score < threshold if "p_value" in method else score >= threshold


def analyze_data_drift(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    alert_feature_count: int = 2,
) -> DriftAnalysis:
    if alert_feature_count < 1:
        raise ValueError("alert_feature_count must be at least 1")

    validated_reference = validate_scoring_batch(reference)
    validated_current = validate_scoring_batch(current)
    snapshot = Report([DataDriftPreset()]).run(
        current_data=validated_current.select(FEATURE_COLUMNS).to_pandas(),
        reference_data=validated_reference.select(FEATURE_COLUMNS).to_pandas(),
    )

    drifted_features = tuple(
        metric["config"]["column"]
        for metric in snapshot.dict()["metrics"]
        if metric["config"]["type"].endswith(":ValueDrift") and _metric_detected_drift(metric)
    )
    drifted_count = len(drifted_features)
    return DriftAnalysis(
        snapshot=snapshot,
        summary=DriftSummary(
            drifted_features=drifted_features,
            drifted_count=drifted_count,
            drifted_share=drifted_count / len(FEATURE_COLUMNS),
            alert=drifted_count >= alert_feature_count,
        ),
    )
