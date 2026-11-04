"""Data-drift evaluation at the Evidently boundary."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import polars as pl
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from credit_monitoring.contracts import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_model_scoring_batch,
    validate_scoring_batch,
)
from credit_monitoring.training import ModelMetrics, evaluate_scored_predictions


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


class PsiLevel(StrEnum):
    INFORMATIONAL = "informational"
    WARNING = "warning"
    RELEVANT = "relevant"


@dataclass(frozen=True)
class PsiResult:
    column: str
    value: float
    level: PsiLevel
    cut_points: tuple[float, ...]
    reference_proportions: tuple[float, ...]
    current_proportions: tuple[float, ...]


@dataclass(frozen=True)
class PopulationStability:
    results: tuple[PsiResult, ...]
    feature_warning_count: int
    score_relevant: bool
    alert: bool


@dataclass(frozen=True)
class MetricShift:
    metric: str
    baseline: float
    current: float
    difference: float
    confidence_lower: float | None
    confidence_upper: float | None
    bootstrap_samples: int
    alert: bool


@dataclass(frozen=True)
class PerformanceComparison:
    baseline: ModelMetrics
    current: ModelMetrics
    shifts: tuple[MetricShift, ...]
    alert: bool


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


def analyze_model_data_drift(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    alert_feature_count: int = 2,
) -> DriftAnalysis:
    if alert_feature_count < 1:
        raise ValueError("alert_feature_count must be at least 1")
    validated_reference = validate_model_scoring_batch(
        reference.select(ID_COLUMN, *MODEL_FEATURE_COLUMNS)
    )
    validated_current = validate_model_scoring_batch(
        current.select(ID_COLUMN, *MODEL_FEATURE_COLUMNS)
    )
    snapshot = Report([DataDriftPreset()]).run(
        current_data=validated_current.select(MODEL_FEATURE_COLUMNS).to_pandas(),
        reference_data=validated_reference.select(MODEL_FEATURE_COLUMNS).to_pandas(),
    )
    drifted_features = tuple(
        metric["config"]["column"]
        for metric in snapshot.dict()["metrics"]
        if metric["config"]["type"].endswith(":ValueDrift") and _metric_detected_drift(metric)
    )
    return DriftAnalysis(
        snapshot=snapshot,
        summary=DriftSummary(
            drifted_features=drifted_features,
            drifted_count=len(drifted_features),
            drifted_share=len(drifted_features) / len(MODEL_FEATURE_COLUMNS),
            alert=len(drifted_features) >= alert_feature_count,
        ),
    )


def calculate_psi(
    reference: pl.Series,
    current: pl.Series,
    *,
    bins: int = 10,
    smoothing: float = 0.5,
    minimum_rows: int = 100,
) -> PsiResult:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("smoothing must be greater than zero")
    if reference.len() < minimum_rows or current.len() < minimum_rows:
        raise ValueError(f"PSI requires at least {minimum_rows} rows per dataset")
    if reference.null_count() or current.null_count():
        raise ValueError("PSI inputs must not contain null values")

    reference_values = reference.to_numpy()
    current_values = current.to_numpy()
    if not np.isfinite(reference_values).all() or not np.isfinite(current_values).all():
        raise ValueError("PSI inputs must contain only finite values")
    if np.ptp(reference_values) == 0:
        constant = float(reference_values[0])
        cut_points = np.array([np.nextafter(constant, -np.inf), np.nextafter(constant, np.inf)])
    else:
        cut_points = np.unique(
            np.quantile(reference_values, np.arange(1, bins, dtype=float) / bins)
        )
    edges = np.concatenate(([-np.inf], cut_points, [np.inf]))
    reference_counts = np.histogram(reference_values, bins=edges)[0]
    current_counts = np.histogram(current_values, bins=edges)[0]
    bin_count = len(reference_counts)
    reference_proportions = (reference_counts + smoothing) / (
        reference_counts.sum() + smoothing * bin_count
    )
    current_proportions = (current_counts + smoothing) / (
        current_counts.sum() + smoothing * bin_count
    )
    value = float(
        np.sum(
            (current_proportions - reference_proportions)
            * np.log(current_proportions / reference_proportions)
        )
    )
    level = (
        PsiLevel.RELEVANT
        if value >= 0.25
        else PsiLevel.WARNING
        if value >= 0.10
        else PsiLevel.INFORMATIONAL
    )
    return PsiResult(
        column=reference.name,
        value=value,
        level=level,
        cut_points=tuple(float(point) for point in cut_points),
        reference_proportions=tuple(float(value) for value in reference_proportions),
        current_proportions=tuple(float(value) for value in current_proportions),
    )


def analyze_population_stability(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
    score_column: str = "default_probability",
    bins: int = 10,
    smoothing: float = 0.5,
    minimum_rows: int = 100,
) -> PopulationStability:
    columns = (*feature_columns, score_column)
    missing = sorted(set(columns) - set(reference.columns) | (set(columns) - set(current.columns)))
    if missing:
        raise ValueError(f"missing PSI columns: {', '.join(missing)}")
    results = tuple(
        calculate_psi(
            reference.get_column(column),
            current.get_column(column),
            bins=bins,
            smoothing=smoothing,
            minimum_rows=minimum_rows,
        )
        for column in columns
    )
    feature_warning_count = sum(
        result.level is not PsiLevel.INFORMATIONAL
        for result in results
        if result.column in feature_columns
    )
    score_result = next(result for result in results if result.column == score_column)
    score_relevant = score_result.level is PsiLevel.RELEVANT
    return PopulationStability(
        results=results,
        feature_warning_count=feature_warning_count,
        score_relevant=score_relevant,
        alert=feature_warning_count >= 2 or score_relevant,
    )


def _performance_values(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    false_positive_rate, true_positive_rate, _ = roc_curve(target, probability)
    return {
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "ks_discrimination": float(max(true_positive_rate - false_positive_rate)),
        "brier_score": float(brier_score_loss(target, probability)),
    }


def compare_paired_performance(
    baseline: pl.DataFrame,
    current: pl.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> PerformanceComparison:
    if bootstrap_samples < 50:
        raise ValueError("bootstrap_samples must be at least 50")
    baseline = baseline.sort(ID_COLUMN)
    current = current.sort(ID_COLUMN)
    if not baseline.get_column(ID_COLUMN).equals(current.get_column(ID_COLUMN)):
        raise ValueError("paired performance datasets must contain the same identifiers")

    baseline_metrics = evaluate_scored_predictions(baseline)
    current_metrics = evaluate_scored_predictions(current)
    baseline_target = baseline.get_column(TARGET_COLUMN).to_numpy()
    current_target = current.get_column(TARGET_COLUMN).to_numpy()
    probability = baseline.get_column("default_probability").to_numpy()
    if not np.array_equal(probability, current.get_column("default_probability").to_numpy()):
        raise ValueError("paired performance datasets must contain the same scores")

    random = np.random.default_rng(seed)
    bootstrap_differences: dict[str, list[float]] = {
        "roc_auc": [],
        "average_precision": [],
        "ks_discrimination": [],
        "brier_score": [],
    }
    for _ in range(bootstrap_samples):
        indices = random.integers(0, len(probability), len(probability))
        baseline_sample = baseline_target[indices]
        current_sample = current_target[indices]
        sampled_probability = probability[indices]
        bootstrap_differences["brier_score"].append(
            float(
                brier_score_loss(current_sample, sampled_probability)
                - brier_score_loss(baseline_sample, sampled_probability)
            )
        )
        if len(np.unique(baseline_sample)) < 2 or len(np.unique(current_sample)) < 2:
            continue
        baseline_values = _performance_values(baseline_sample, sampled_probability)
        current_values = _performance_values(current_sample, sampled_probability)
        for metric in ("roc_auc", "average_precision", "ks_discrimination"):
            bootstrap_differences[metric].append(current_values[metric] - baseline_values[metric])

    adverse_increase = {"brier_score"}
    shifts = []
    for metric, differences in bootstrap_differences.items():
        baseline_value = float(getattr(baseline_metrics, metric))
        current_value = float(getattr(current_metrics, metric))
        enough_samples = metric == "brier_score" or len(differences) >= max(
            20, bootstrap_samples * 0.5
        )
        if enough_samples:
            lower, upper = (float(value) for value in np.quantile(differences, [0.025, 0.975]))
            alert = lower > 0 if metric in adverse_increase else upper < 0
        else:
            lower, upper, alert = None, None, False
        shifts.append(
            MetricShift(
                metric=metric,
                baseline=baseline_value,
                current=current_value,
                difference=current_value - baseline_value,
                confidence_lower=lower,
                confidence_upper=upper,
                bootstrap_samples=len(differences),
                alert=alert,
            )
        )
    return PerformanceComparison(
        baseline=baseline_metrics,
        current=current_metrics,
        shifts=tuple(shifts),
        alert=any(shift.alert for shift in shifts),
    )


def default_rate_by_score_band(data: pl.DataFrame) -> pl.DataFrame:
    required = {TARGET_COLUMN, "default_probability"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing score-band columns: {', '.join(missing)}")
    return (
        data.lazy()
        .with_columns(
            pl.when(pl.col("default_probability") < 0.1)
            .then(pl.lit("[0.0, 0.1)"))
            .when(pl.col("default_probability") < 0.2)
            .then(pl.lit("[0.1, 0.2)"))
            .when(pl.col("default_probability") < 0.4)
            .then(pl.lit("[0.2, 0.4)"))
            .when(pl.col("default_probability") < 0.6)
            .then(pl.lit("[0.4, 0.6)"))
            .otherwise(pl.lit("[0.6, 1.0]"))
            .alias("score_band")
        )
        .group_by("score_band")
        .agg(
            pl.len().alias("rows"),
            pl.col(TARGET_COLUMN).sum().alias("events"),
            pl.col(TARGET_COLUMN).mean().alias("event_rate"),
            pl.col("default_probability").mean().alias("mean_probability"),
        )
        .with_columns(pl.col("event_rate", "mean_probability").round(12))
        .sort("score_band")
        .collect()
    )
