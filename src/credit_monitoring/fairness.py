"""Restricted fairness views and aggregate Fairlearn assessment."""

from dataclasses import dataclass
from functools import partial

import numpy as np
import polars as pl
from fairlearn.metrics import (
    MetricFrame,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score

from credit_monitoring.contracts import ID_COLUMN, TARGET_COLUMN

AGE_GROUP_COLUMN = "AGE_AUDIT_GROUP"
SEX_GROUP_COLUMN = "SEX_AUDIT_GROUP"
AGE_BANDS = ("21-29", "30-39", "40-49", "50-59", "60+")
ERROR_METRICS = (
    "recall",
    "false_negative_rate",
    "false_positive_rate",
    "precision",
    "accuracy",
)


@dataclass(frozen=True)
class FairnessSummary:
    sensitive_feature: str
    evidence_sufficient: bool
    published_group_count: int
    suppressed_group_count: int
    equalized_odds_difference: float | None
    equalized_odds_ci_lower: float | None
    equalized_odds_ci_upper: float | None
    disparity_review_threshold: float
    investigation_required: bool
    recommendation: str


@dataclass(frozen=True)
class FairnessAnalysis:
    group_metrics: pl.DataFrame
    metric_differences: pl.DataFrame
    summary: FairnessSummary


def resolve_fairness_state(*, investigation_required: bool, evidence_sufficient: bool) -> str:
    if investigation_required and not evidence_sufficient:
        return "INVESTIGATION_REQUIRED_WITH_INSUFFICIENT_DATA"
    if investigation_required:
        return "INVESTIGATION_REQUIRED"
    if not evidence_sufficient:
        return "EVALUATION_BLOCKED_INSUFFICIENT_DATA"
    return "MONITORING_CONTINUES"


def _redact_suppressed_groups(aggregate: pl.DataFrame) -> pl.DataFrame:
    return aggregate.with_columns(
        pl.when("publish_error_metrics")
        .then(pl.col(column))
        .otherwise(pl.lit(None, dtype=dtype))
        .alias(column)
        for column, dtype in (
            ("events", pl.Int64),
            ("non_events", pl.Int64),
            ("mean_probability", pl.Float64),
            ("review_rate", pl.Float64),
        )
    )


def _equalized_odds_bootstrap_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    strata = [
        np.flatnonzero((sensitive_features == group) & (y_true == target))
        for group in np.unique(sensitive_features)
        for target in (0, 1)
    ]
    estimates = []
    for _ in range(bootstrap_samples):
        sample = np.concatenate(
            [rng.choice(indices, size=indices.size, replace=True) for indices in strata]
        )
        estimates.append(
            equalized_odds_difference(
                y_true[sample],
                y_pred[sample],
                sensitive_features=sensitive_features[sample],
            )
        )
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def assign_age_audit_groups(data: pl.DataFrame, *, minimum_rows: int) -> pl.DataFrame:
    required = {ID_COLUMN, "AGE_GROUP"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing age-audit columns: {', '.join(missing)}")
    if minimum_rows < 1:
        raise ValueError("minimum_rows must be at least 1")
    if data.height == 0 or data.get_column(ID_COLUMN).n_unique() != data.height:
        raise ValueError("age-audit data must contain unique rows")
    if data.select(ID_COLUMN, "AGE_GROUP").null_count().sum_horizontal().item() != 0:
        raise ValueError("age-audit data must not contain null values")

    observed = set(data.get_column("AGE_GROUP").unique())
    unknown = sorted(observed - set(AGE_BANDS))
    if unknown:
        raise ValueError(f"unknown age bands: {', '.join(unknown)}")
    mapping = pl.DataFrame(
        {
            "AGE_GROUP": list(AGE_BANDS),
            AGE_GROUP_COLUMN: ["21-29", "30-39", "40+", "40+", "40+"],
        }
    )
    return data.join(mapping, on="AGE_GROUP", how="left", validate="m:1")


def build_restricted_audit_view(
    cohort: pl.DataFrame,
    matured_scores: pl.DataFrame,
    *,
    minimum_age_rows: int,
) -> pl.DataFrame:
    cohort_columns = {ID_COLUMN, "SEX", "AGE_GROUP"}
    score_columns = {ID_COLUMN, TARGET_COLUMN, "default_probability", "review_required"}
    missing = sorted(
        (cohort_columns - set(cohort.columns)) | (score_columns - set(matured_scores.columns))
    )
    if missing:
        raise ValueError(f"missing restricted-audit columns: {', '.join(missing)}")
    if cohort.get_column(ID_COLUMN).n_unique() != cohort.height:
        raise ValueError("audit cohort identifiers must be unique")
    if matured_scores.get_column(ID_COLUMN).n_unique() != matured_scores.height:
        raise ValueError("matured score identifiers must be unique")
    if set(cohort.get_column(ID_COLUMN)) != set(matured_scores.get_column(ID_COLUMN)):
        raise ValueError("audit cohort and matured scores must contain the same identifiers")
    if not set(cohort.get_column("SEX").unique()) <= {1, 2}:
        raise ValueError("SEX must use the documented UCI codes 1 or 2")

    age_groups = assign_age_audit_groups(
        cohort.select(ID_COLUMN, "AGE_GROUP"),
        minimum_rows=minimum_age_rows,
    )
    restricted_groups = (
        cohort.lazy()
        .select(
            pl.col(ID_COLUMN),
            pl.when(pl.col("SEX") == 1)
            .then(pl.lit("male"))
            .otherwise(pl.lit("female"))
            .alias(SEX_GROUP_COLUMN),
        )
        .collect()
        .join(
            age_groups.select(ID_COLUMN, AGE_GROUP_COLUMN),
            on=ID_COLUMN,
            how="inner",
            validate="1:1",
        )
    )
    return restricted_groups.join(
        matured_scores.select(
            ID_COLUMN,
            TARGET_COLUMN,
            "default_probability",
            "review_required",
        ),
        on=ID_COLUMN,
        how="inner",
        validate="1:1",
    ).sort(ID_COLUMN)


def audit_group_fairness(
    data: pl.DataFrame,
    *,
    sensitive_feature: str,
    minimum_rows: int,
    minimum_events: int,
    minimum_non_events: int,
    bootstrap_samples: int,
    seed: int,
    disparity_review_threshold: float,
) -> FairnessAnalysis:
    required = {
        ID_COLUMN,
        TARGET_COLUMN,
        "default_probability",
        "review_required",
        sensitive_feature,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing fairness columns: {', '.join(missing)}")
    if min(minimum_rows, minimum_events, minimum_non_events) < 1:
        raise ValueError("fairness publication minimums must be positive")
    if bootstrap_samples < 50:
        raise ValueError("bootstrap_samples must be at least 50")
    if not 0 <= disparity_review_threshold <= 1:
        raise ValueError("disparity_review_threshold must be between 0 and 1")
    if data.height == 0 or data.get_column(ID_COLUMN).n_unique() != data.height:
        raise ValueError("fairness data must contain unique rows")
    if data.select(*required).null_count().sum_horizontal().item() != 0:
        raise ValueError("fairness data must not contain null values")
    if not set(data.get_column(TARGET_COLUMN).unique().to_list()) <= {0, 1}:
        raise ValueError(f"{TARGET_COLUMN} must be binary")
    if not set(data.get_column("review_required").unique().to_list()) <= {0, 1}:
        raise ValueError("review_required must be binary")
    probabilities = data.get_column("default_probability")
    if not probabilities.is_finite().all() or probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError("default_probability must contain finite values between 0 and 1")

    aggregate = (
        data.lazy()
        .group_by(sensitive_feature)
        .agg(
            pl.len().alias("rows"),
            pl.col(TARGET_COLUMN).sum().alias("events"),
            (pl.len() - pl.col(TARGET_COLUMN).sum()).alias("non_events"),
            pl.col("default_probability").mean().alias("mean_probability"),
            pl.col("review_required").mean().alias("review_rate"),
        )
        .with_columns(
            (
                (pl.col("rows") >= minimum_rows)
                & (pl.col("events") >= minimum_events)
                & (pl.col("non_events") >= minimum_non_events)
            ).alias("publish_error_metrics")
        )
        .with_columns(
            pl.when(pl.col("publish_error_metrics"))
            .then(pl.lit("published"))
            .when(pl.col("rows") < minimum_rows)
            .then(pl.lit("suppressed: insufficient rows"))
            .otherwise(pl.lit("suppressed: insufficient class support"))
            .alias("publication_status")
        )
        .sort(sensitive_feature)
        .collect()
    )
    published_groups = aggregate.filter(pl.col("publish_error_metrics")).get_column(
        sensitive_feature
    )
    suppressed_count = aggregate.height - published_groups.len()
    public_aggregate = _redact_suppressed_groups(aggregate)
    if published_groups.len() < 2:
        empty_metrics = public_aggregate
        for metric in ERROR_METRICS:
            empty_metrics = empty_metrics.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(metric),
                pl.lit(None, dtype=pl.Float64).alias(f"{metric}_ci_lower"),
                pl.lit(None, dtype=pl.Float64).alias(f"{metric}_ci_upper"),
            )
        return FairnessAnalysis(
            group_metrics=empty_metrics,
            metric_differences=pl.DataFrame(),
            summary=FairnessSummary(
                sensitive_feature=sensitive_feature,
                evidence_sufficient=False,
                published_group_count=published_groups.len(),
                suppressed_group_count=suppressed_count,
                equalized_odds_difference=None,
                equalized_odds_ci_lower=None,
                equalized_odds_ci_upper=None,
                disparity_review_threshold=disparity_review_threshold,
                investigation_required=False,
                recommendation="collect additional labeled observations before comparison",
            ),
        )

    published = data.filter(pl.col(sensitive_feature).is_in(published_groups.to_list()))
    metric_frame = MetricFrame(
        metrics={
            "recall": partial(recall_score, zero_division=0),
            "false_negative_rate": false_negative_rate,
            "false_positive_rate": false_positive_rate,
            "precision": partial(precision_score, zero_division=0),
            "accuracy": accuracy_score,
        },
        y_true=published.get_column(TARGET_COLUMN).to_numpy(),
        y_pred=published.get_column("review_required").to_numpy(),
        sensitive_features=published.get_column(sensitive_feature).to_numpy(),
        n_boot=bootstrap_samples,
        ci_quantiles=[0.025, 0.975],
        random_state=seed,
    )
    lower_by_group, upper_by_group = metric_frame.by_group_ci
    fairlearn_rows = []
    for group, values in metric_frame.by_group.iterrows():
        row: dict[str, object] = {sensitive_feature: group}
        for metric in ERROR_METRICS:
            row[metric] = float(values[metric])
            row[f"{metric}_ci_lower"] = float(lower_by_group.loc[group, metric])
            row[f"{metric}_ci_upper"] = float(upper_by_group.loc[group, metric])
        fairlearn_rows.append(row)
    group_metrics = public_aggregate.join(
        pl.DataFrame(fairlearn_rows),
        on=sensitive_feature,
        how="left",
        validate="1:1",
    )

    differences = metric_frame.difference()
    lower_difference, upper_difference = metric_frame.difference_ci()
    metric_differences = pl.DataFrame(
        {
            "metric": list(ERROR_METRICS),
            "difference": [float(differences[metric]) for metric in ERROR_METRICS],
            "ci_lower": [float(lower_difference[metric]) for metric in ERROR_METRICS],
            "ci_upper": [float(upper_difference[metric]) for metric in ERROR_METRICS],
        }
    )
    equalized_odds = float(
        equalized_odds_difference(
            published.get_column(TARGET_COLUMN).to_numpy(),
            published.get_column("review_required").to_numpy(),
            sensitive_features=published.get_column(sensitive_feature).to_numpy(),
        )
    )
    equalized_odds_lower, equalized_odds_upper = _equalized_odds_bootstrap_interval(
        published.get_column(TARGET_COLUMN).to_numpy(),
        published.get_column("review_required").to_numpy(),
        published.get_column(sensitive_feature).to_numpy(),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    investigation_required = equalized_odds_lower > disparity_review_threshold
    evidence_sufficient = suppressed_count == 0
    recommendation = (
        "investigate supported disparities and collect labels for suppressed groups"
        if investigation_required and not evidence_sufficient
        else (
            "investigate proxies, labels, threshold policy, and group data quality"
            if investigation_required
            else (
                "collect additional labeled observations for suppressed groups"
                if not evidence_sufficient
                else "retain monitoring; no automatic mitigation or threshold change"
            )
        )
    )
    return FairnessAnalysis(
        group_metrics=group_metrics,
        metric_differences=metric_differences,
        summary=FairnessSummary(
            sensitive_feature=sensitive_feature,
            evidence_sufficient=evidence_sufficient,
            published_group_count=published_groups.len(),
            suppressed_group_count=suppressed_count,
            equalized_odds_difference=equalized_odds,
            equalized_odds_ci_lower=equalized_odds_lower,
            equalized_odds_ci_upper=equalized_odds_upper,
            disparity_review_threshold=disparity_review_threshold,
            investigation_required=investigation_required,
            recommendation=recommendation,
        ),
    )
