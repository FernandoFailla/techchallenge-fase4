"""Minimal logistic model and blocking scoring orchestration."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_monitoring.contracts import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    BatchContractError,
    ContractViolation,
    validate_historical_data,
    validate_scoring_batch,
)


class ProbabilityModel(Protocol):
    def predict_proba(self, data: object) -> object: ...


class BatchState(StrEnum):
    REJECTED = "REJECTED"
    SCORED = "SCORED"


@dataclass(frozen=True)
class BatchResult:
    state: BatchState
    scores: pl.DataFrame
    violations: tuple[ContractViolation, ...] = ()
    failure_count: int = 0


def fit_baseline_model(data: pl.DataFrame, *, random_state: int = 42) -> Pipeline:
    validated = validate_historical_data(data)
    features = validated.select(FEATURE_COLUMNS).to_numpy()
    target = validated.get_column(TARGET_COLUMN).to_numpy()
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(random_state=random_state, max_iter=1_000),
            ),
        ]
    )
    return model.fit(features, target)


def _empty_scores() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            ID_COLUMN: pl.Int64,
            "default_probability": pl.Float64,
            "review_required": pl.Boolean,
        }
    )


def score_batch(
    model: ProbabilityModel,
    data: pl.DataFrame,
    *,
    review_threshold: float = 0.5,
) -> BatchResult:
    if not 0 <= review_threshold <= 1:
        raise ValueError("review_threshold must be between 0 and 1")

    try:
        validated = validate_scoring_batch(data)
    except BatchContractError as error:
        return BatchResult(
            state=BatchState.REJECTED,
            scores=_empty_scores(),
            violations=error.violations,
            failure_count=error.failure_count,
        )

    probabilities = model.predict_proba(validated.select(FEATURE_COLUMNS).to_numpy())[:, 1]
    scores = pl.DataFrame(
        {
            ID_COLUMN: validated.get_column(ID_COLUMN),
            "default_probability": probabilities,
        }
    ).with_columns((pl.col("default_probability") >= review_threshold).alias("review_required"))
    return BatchResult(state=BatchState.SCORED, scores=scores)
