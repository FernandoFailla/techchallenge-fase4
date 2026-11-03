from datetime import datetime, timedelta

import polars as pl
import pytest

from credit_monitoring.contracts import ID_COLUMN, TARGET_COLUMN
from credit_monitoring.outcomes import OutcomeState, associate_mature_outcomes


def scores_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            ID_COLUMN: [1, 2, 3, 4],
            "default_probability": [0.1, 0.2, 0.7, 0.8],
            "review_required": [False, False, True, True],
        }
    )


def outcomes_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            ID_COLUMN: [1, 2, 3, 4],
            TARGET_COLUMN: [0, 0, 1, 1],
            "available_at": [datetime(2026, 4, 1)] * 4,
        }
    )


def test_mature_outcomes_are_associated_one_to_one() -> None:
    result = associate_mature_outcomes(
        scores_fixture(),
        outcomes_fixture(),
        as_of=datetime(2026, 4, 1),
    )

    assert result.state is OutcomeState.MATURED
    assert result.data.height == 4
    assert TARGET_COLUMN in result.data.columns


def test_duplicate_outcomes_are_rejected_by_contract() -> None:
    duplicated = outcomes_fixture().with_columns(
        pl.when(pl.col(ID_COLUMN) == 2)
        .then(pl.lit(1))
        .otherwise(pl.col(ID_COLUMN))
        .alias(ID_COLUMN)
    )

    result = associate_mature_outcomes(
        scores_fixture(),
        duplicated,
        as_of=datetime(2026, 4, 1),
    )

    assert result.state is OutcomeState.OUTCOME_REJECTED
    assert result.data.is_empty()


def test_missing_or_premature_outcomes_block_evaluation() -> None:
    missing = associate_mature_outcomes(
        scores_fixture(),
        outcomes_fixture().head(3),
        as_of=datetime(2026, 4, 1),
    )
    premature = associate_mature_outcomes(
        scores_fixture(),
        outcomes_fixture().with_columns(pl.col("available_at") + timedelta(days=1)),
        as_of=datetime(2026, 4, 1),
    )

    assert missing.state is OutcomeState.EVALUATION_BLOCKED
    assert premature.state is OutcomeState.EVALUATION_BLOCKED


def test_invalid_scores_do_not_reach_maturation() -> None:
    invalid_scores = scores_fixture().with_columns(pl.lit(1.2).alias("default_probability"))

    with pytest.raises(ValueError, match="default_probability must be between 0 and 1"):
        associate_mature_outcomes(
            invalid_scores,
            outcomes_fixture(),
            as_of=datetime(2026, 4, 1),
        )
