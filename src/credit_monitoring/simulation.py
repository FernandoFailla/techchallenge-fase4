"""Deterministic synthetic data for the vertical slice."""

from random import Random

import polars as pl

from credit_monitoring.contracts import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    validate_historical_data,
    validate_scoring_batch,
)


def generate_historical_data(rows: int, *, seed: int, id_start: int = 1) -> pl.DataFrame:
    if rows < 2:
        raise ValueError("rows must be at least 2")

    random = Random(seed)
    limits = [
        float(random.choice([20_000, 30_000, 50_000, 80_000, 120_000, 200_000, 350_000]))
        for _ in range(rows)
    ]
    ages = [random.randint(21, 70) for _ in range(rows)]
    payment_status = random.choices([-2, -1, 0, 1, 2, 3], weights=[8, 12, 45, 18, 12, 5], k=rows)
    bills = [round(limit * random.uniform(0.05, 1.10), 2) for limit in limits]
    payments = [round(limit * random.uniform(0.00, 0.25), 2) for limit in limits]
    draws = [random.random() for _ in range(rows)]

    base = pl.DataFrame(
        {
            ID_COLUMN: range(id_start, id_start + rows),
            "LIMIT_BAL": limits,
            "AGE": ages,
            "PAY_0": payment_status,
            "BILL_AMT1": bills,
            "PAY_AMT1": payments,
            "_draw": draws,
        },
        schema={
            ID_COLUMN: pl.Int64,
            "LIMIT_BAL": pl.Float64,
            "AGE": pl.Int64,
            "PAY_0": pl.Int64,
            "BILL_AMT1": pl.Float64,
            "PAY_AMT1": pl.Float64,
            "_draw": pl.Float64,
        },
    )
    generated = (
        base.lazy()
        .with_columns(
            (pl.col("BILL_AMT1") / pl.col("LIMIT_BAL")).alias("_utilization"),
            (pl.col("PAY_AMT1") / pl.col("LIMIT_BAL")).alias("_payment_ratio"),
        )
        .with_columns(
            (
                -2.2
                + pl.when(pl.col("PAY_0") > 0).then(pl.col("PAY_0") * 0.75).otherwise(0)
                + pl.col("_utilization") * 1.8
                - pl.col("_payment_ratio") * 2.0
                - (pl.col("AGE") - 35) * 0.01
            ).alias("_logit")
        )
        .with_columns((1 / (1 + (-pl.col("_logit")).exp())).alias("_probability"))
        .with_columns(
            (pl.col("_draw") < pl.col("_probability")).cast(pl.Int64).alias(TARGET_COLUMN)
        )
        .select(ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN)
        .collect()
    )
    return validate_historical_data(generated)


def scoring_view(data: pl.DataFrame) -> pl.DataFrame:
    validated = validate_historical_data(data)
    return validated.select(ID_COLUMN, *FEATURE_COLUMNS)


def simulate_data_drift(
    data: pl.DataFrame,
    *,
    limit_balance_multiplier: float,
    age_shift: int,
) -> pl.DataFrame:
    if not 0 < limit_balance_multiplier < 1:
        raise ValueError("limit_balance_multiplier must be between 0 and 1")
    if age_shift == 0:
        raise ValueError("age_shift cannot be zero")

    validated = validate_scoring_batch(data)
    shifted = (
        validated.lazy()
        .with_columns(
            (pl.col("LIMIT_BAL") * limit_balance_multiplier).alias("LIMIT_BAL"),
            (pl.col("AGE") + age_shift).clip(18, 100).cast(pl.Int64).alias("AGE"),
        )
        .select(ID_COLUMN, *FEATURE_COLUMNS)
        .collect()
    )
    return validate_scoring_batch(shifted)


def corrupt_scoring_batch(data: pl.DataFrame) -> pl.DataFrame:
    validated = validate_scoring_batch(data)
    if validated.height < 3:
        raise ValueError("at least 3 rows are required to corrupt a batch")

    first_id = validated.item(0, ID_COLUMN)
    return (
        validated.head(10)
        .with_row_index("_row")
        .with_columns(
            pl.when(pl.col("_row") <= 1)
            .then(pl.lit(first_id))
            .otherwise(pl.col(ID_COLUMN))
            .alias(ID_COLUMN),
            pl.when(pl.col("_row") == 0)
            .then(pl.lit(-1.0))
            .otherwise(pl.col("LIMIT_BAL"))
            .alias("LIMIT_BAL"),
            pl.when(pl.col("_row") == 2).then(pl.lit(10)).otherwise(pl.col("AGE")).alias("AGE"),
        )
        .drop("_row")
    )
