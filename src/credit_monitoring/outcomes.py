"""Matured-outcome validation and one-to-one association."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import polars as pl

from credit_monitoring.contracts import (
    ID_COLUMN,
    BatchContractError,
    ContractViolation,
    validate_outcome_feed,
)


class OutcomeState(StrEnum):
    MATURED = "MATURED"
    OUTCOME_REJECTED = "OUTCOME_REJECTED"
    EVALUATION_BLOCKED = "EVALUATION_BLOCKED"


@dataclass(frozen=True)
class OutcomeAssociation:
    state: OutcomeState
    data: pl.DataFrame
    reason: str | None = None
    violations: tuple[ContractViolation, ...] = ()


def associate_mature_outcomes(
    scores: pl.DataFrame,
    outcomes: pl.DataFrame,
    *,
    as_of: datetime,
) -> OutcomeAssociation:
    score_columns = {ID_COLUMN, "default_probability", "review_required"}
    missing_score_columns = sorted(score_columns - set(scores.columns))
    if missing_score_columns:
        raise ValueError(f"missing score columns: {', '.join(missing_score_columns)}")
    if scores.height == 0 or scores.get_column(ID_COLUMN).n_unique() != scores.height:
        raise ValueError("scores must contain unique rows")
    if scores.select(*score_columns).null_count().sum_horizontal().item() != 0:
        raise ValueError("scores must not contain null values")
    if scores.schema[ID_COLUMN] != pl.Int64:
        raise ValueError("score identifiers must use Int64")
    if scores.schema["default_probability"] not in {pl.Float32, pl.Float64}:
        raise ValueError("default_probability must be floating point")
    if not scores.get_column("default_probability").is_between(0, 1).all():
        raise ValueError("default_probability must be between 0 and 1")
    if scores.schema["review_required"] != pl.Boolean:
        raise ValueError("review_required must be boolean")

    try:
        validated = validate_outcome_feed(outcomes)
    except BatchContractError as error:
        return OutcomeAssociation(
            state=OutcomeState.OUTCOME_REJECTED,
            data=pl.DataFrame(),
            reason=str(error),
            violations=error.violations,
        )

    score_ids = set(scores.get_column(ID_COLUMN))
    outcome_ids = set(validated.get_column(ID_COLUMN))
    if score_ids != outcome_ids:
        missing = len(score_ids - outcome_ids)
        unexpected = len(outcome_ids - score_ids)
        return OutcomeAssociation(
            state=OutcomeState.EVALUATION_BLOCKED,
            data=pl.DataFrame(),
            reason=f"outcome keys do not match scores: missing={missing}, unexpected={unexpected}",
        )
    if validated.get_column("available_at").max() > as_of:
        return OutcomeAssociation(
            state=OutcomeState.EVALUATION_BLOCKED,
            data=pl.DataFrame(),
            reason="outcomes are not mature at the evaluation time",
        )

    return OutcomeAssociation(
        state=OutcomeState.MATURED,
        data=scores.join(validated, on=ID_COLUMN, how="inner", validate="1:1").sort(ID_COLUMN),
    )
