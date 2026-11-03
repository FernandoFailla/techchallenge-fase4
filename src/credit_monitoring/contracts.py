"""Pandera contracts for historical, scoring, and outcome data."""

from dataclasses import dataclass

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaErrors

ID_COLUMN = "ID"
TARGET_COLUMN = "default_payment_next_month"
FEATURE_COLUMNS = ("LIMIT_BAL", "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1")
MODEL_FEATURE_COLUMNS = (
    "LIMIT_BAL",
    "PAY_0",
    "MAX_DELAY_6M",
    "DELAY_MONTHS_6M",
    "UTILIZATION_CURRENT",
    "UTILIZATION_MEAN_6M",
    "PAYMENT_TO_BILL_RATIO_6M",
)


def _base_columns() -> dict[str, pa.Column]:
    return {
        ID_COLUMN: pa.Column(pl.Int64, nullable=False, unique=True),
        "LIMIT_BAL": pa.Column(pl.Float64, checks=pa.Check.gt(0), nullable=False),
        "AGE": pa.Column(pl.Int64, checks=pa.Check.in_range(18, 100), nullable=False),
        "PAY_0": pa.Column(pl.Int64, checks=pa.Check.in_range(-2, 8), nullable=False),
        "BILL_AMT1": pa.Column(pl.Float64, nullable=False),
        "PAY_AMT1": pa.Column(pl.Float64, checks=pa.Check.ge(0), nullable=False),
    }


SCORING_SCHEMA = pa.DataFrameSchema(
    _base_columns(),
    strict=True,
    ordered=True,
    coerce=False,
    name="scoring_v1",
)

HISTORICAL_SCHEMA = pa.DataFrameSchema(
    {
        **_base_columns(),
        TARGET_COLUMN: pa.Column(
            pl.Int64,
            checks=pa.Check.isin([0, 1]),
            nullable=False,
        ),
    },
    strict=True,
    ordered=True,
    coerce=False,
    name="historical_v1",
)

MODEL_SCORING_SCHEMA = pa.DataFrameSchema(
    {
        ID_COLUMN: pa.Column(pl.Int64, nullable=False, unique=True),
        "LIMIT_BAL": pa.Column(pl.Float64, checks=pa.Check.gt(0), nullable=False),
        "PAY_0": pa.Column(pl.Int64, checks=pa.Check.in_range(-2, 8), nullable=False),
        "MAX_DELAY_6M": pa.Column(
            pl.Int64,
            checks=pa.Check.in_range(-2, 8),
            nullable=False,
        ),
        "DELAY_MONTHS_6M": pa.Column(
            pl.Int64,
            checks=pa.Check.in_range(0, 6),
            nullable=False,
        ),
        "UTILIZATION_CURRENT": pa.Column(
            pl.Float64,
            checks=pa.Check.in_range(-1, 2),
            nullable=False,
        ),
        "UTILIZATION_MEAN_6M": pa.Column(
            pl.Float64,
            checks=pa.Check.in_range(-1, 2),
            nullable=False,
        ),
        "PAYMENT_TO_BILL_RATIO_6M": pa.Column(
            pl.Float64,
            checks=pa.Check.in_range(0, 5),
            nullable=False,
        ),
    },
    strict=True,
    ordered=True,
    coerce=False,
    name="model_scoring_v1",
)

OUTCOME_SCHEMA = pa.DataFrameSchema(
    {
        ID_COLUMN: pa.Column(pl.Int64, nullable=False, unique=True),
        TARGET_COLUMN: pa.Column(
            pl.Int64,
            checks=pa.Check.isin([0, 1]),
            nullable=False,
        ),
        "available_at": pa.Column(pl.Datetime, nullable=False),
    },
    strict=True,
    ordered=True,
    coerce=False,
    name="outcome_v1",
)


@dataclass(frozen=True, order=True)
class ContractViolation:
    column: str
    check: str


class BatchContractError(ValueError):
    def __init__(self, violations: tuple[ContractViolation, ...], failure_count: int) -> None:
        super().__init__(f"batch violates {len(violations)} contract rules")
        self.violations = violations
        self.failure_count = failure_count


def _summarize_schema_errors(error: SchemaErrors) -> BatchContractError:
    failure_cases = error.failure_cases
    summary = (
        failure_cases.lazy()
        .select(
            pl.col("column").cast(pl.String, strict=False).fill_null("<schema>").alias("column"),
            pl.col("check").cast(pl.String, strict=False).fill_null("<unknown>").alias("check"),
        )
        .unique()
        .sort("column", "check")
        .collect()
    )
    violations = tuple(ContractViolation(**row) for row in summary.iter_rows(named=True))
    return BatchContractError(violations=violations, failure_count=failure_cases.height)


def _validate(data: pl.DataFrame, schema: pa.DataFrameSchema) -> pl.DataFrame:
    try:
        return schema.validate(data, lazy=True)
    except SchemaErrors as error:
        raise _summarize_schema_errors(error) from error


def validate_scoring_batch(data: pl.DataFrame) -> pl.DataFrame:
    return _validate(data, SCORING_SCHEMA)


def validate_historical_data(data: pl.DataFrame) -> pl.DataFrame:
    return _validate(data, HISTORICAL_SCHEMA)


def validate_model_scoring_batch(data: pl.DataFrame) -> pl.DataFrame:
    return _validate(data, MODEL_SCORING_SCHEMA)


def validate_outcome_feed(data: pl.DataFrame) -> pl.DataFrame:
    return _validate(data, OUTCOME_SCHEMA)
