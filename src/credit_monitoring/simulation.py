"""Deterministic synthetic data and post-freeze drift scenarios."""

from dataclasses import dataclass
from datetime import datetime
from random import Random

import numpy as np
import polars as pl

from credit_monitoring.contracts import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_historical_data,
    validate_model_scoring_batch,
    validate_outcome_feed,
    validate_scoring_batch,
)
from credit_monitoring.data import AUDIT_COLUMNS
from credit_monitoring.training import CreditModel, predict_credit_risk

MONTH_COLUMN = "month"
SCENARIO_COLUMN = "scenario"
SCENARIOS = {"M1": "stable", "M2": "data_drift", "M3": "concept_drift"}


@dataclass(frozen=True)
class ConceptDriftSimulation:
    evidence: pl.DataFrame
    outcomes: pl.DataFrame


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


def build_drift_cohorts(
    pool: pl.DataFrame,
    *,
    seed: int,
    rows_per_month: int,
    stress_strength: float,
) -> pl.DataFrame:
    required = {ID_COLUMN, *MODEL_FEATURE_COLUMNS}
    missing = sorted(required - set(pool.columns))
    if missing:
        raise ValueError(f"missing simulation columns: {', '.join(missing)}")
    if pool.get_column(ID_COLUMN).n_unique() != pool.height:
        raise ValueError("simulation pool identifiers must be unique")
    if rows_per_month < 100:
        raise ValueError("rows_per_month must be at least 100")
    selection_candidates = pool.height - 2 * rows_per_month
    if selection_candidates < rows_per_month + max(1, rows_per_month // 2):
        raise ValueError("simulation pool needs a selection reserve for the drift cohort")
    if not np.isfinite(stress_strength) or stress_strength <= 0:
        raise ValueError("stress_strength must be greater than zero")

    ordered = pool.sort(ID_COLUMN)
    validate_model_scoring_batch(ordered.select(ID_COLUMN, *MODEL_FEATURE_COLUMNS))
    random = np.random.default_rng(seed)
    ranked = (
        ordered.lazy()
        .with_columns(
            (1 - (pl.col("LIMIT_BAL").rank("average") - 1) / (pl.len() - 1)).alias(
                "_LOW_LIMIT_RANK"
            ),
            ((pl.col("MAX_DELAY_6M").rank("average") - 1) / (pl.len() - 1)).alias("_DELAY_RANK"),
            ((pl.col("UTILIZATION_CURRENT").rank("average") - 1) / (pl.len() - 1)).alias(
                "_UTILIZATION_RANK"
            ),
            pl.Series("_SELECTION_DRAW", random.random(ordered.height)),
            pl.Series("_SPLIT_DRAW", random.random(ordered.height)),
        )
        .with_columns(
            (
                0.4 * pl.col("_LOW_LIMIT_RANK")
                + 0.4 * pl.col("_DELAY_RANK")
                + 0.2 * pl.col("_UTILIZATION_RANK")
            ).alias("_STRESS_SCORE")
        )
        .with_columns(
            (
                -pl.col("_SELECTION_DRAW").log() / (stress_strength * pl.col("_STRESS_SCORE")).exp()
            ).alias("_SELECTION_PRIORITY")
        )
        .collect()
    )
    stable_candidates = ranked.sort("_SPLIT_DRAW")
    m1_ids = stable_candidates.head(rows_per_month).select(ID_COLUMN)
    m3_ids = stable_candidates.slice(rows_per_month, rows_per_month).select(ID_COLUMN)
    stable_ids = pl.concat((m1_ids, m3_ids))
    m2_ids = (
        ranked.join(stable_ids, on=ID_COLUMN, how="anti")
        .sort("_SELECTION_PRIORITY")
        .head(rows_per_month)
        .select(ID_COLUMN)
    )
    month_ids = {
        "M1": m1_ids,
        "M2": m2_ids,
        "M3": m3_ids,
    }
    audit_columns = tuple(column for column in AUDIT_COLUMNS if column in ordered.columns)
    cohorts = []
    for month, identifiers in month_ids.items():
        cohort = (
            ordered.join(identifiers, on=ID_COLUMN, how="semi")
            .select(ID_COLUMN, *MODEL_FEATURE_COLUMNS, *audit_columns)
            .with_columns(
                pl.lit(month).alias(MONTH_COLUMN),
                pl.lit(SCENARIOS[month]).alias(SCENARIO_COLUMN),
            )
        )
        cohorts.append(cohort)

    result = pl.concat(cohorts).sort(MONTH_COLUMN, ID_COLUMN)
    if result.get_column(ID_COLUMN).n_unique() != result.height:
        raise AssertionError("generated cohorts are not disjoint")
    return result


def simulate_concept_drift(
    model: CreditModel,
    data: pl.DataFrame,
    *,
    delta: float,
    seed: int,
    available_at: datetime,
) -> ConceptDriftSimulation:
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be greater than zero")

    scoring_data = validate_model_scoring_batch(
        data.select(ID_COLUMN, *MODEL_FEATURE_COLUMNS).sort(ID_COLUMN)
    )
    scores = predict_credit_risk(model, scoring_data)
    random = np.random.default_rng(seed)
    evidence = (
        scoring_data.join(scores, on=ID_COLUMN, how="inner", validate="1:1")
        .lazy()
        .with_columns(
            (
                0.6 * pl.col("UTILIZATION_CURRENT").clip(0, 2) / 2
                + 0.4 * pl.col("MAX_DELAY_6M").clip(0, 8) / 8
            ).alias("vulnerability"),
            pl.Series("shared_uniform", random.random(scoring_data.height)),
        )
        .with_columns(
            pl.col("default_probability").alias("p0"),
            pl.col("default_probability").clip(1e-9, 1 - 1e-9).alias("_P0_CLIPPED"),
        )
        .with_columns(
            (
                1
                / (
                    1
                    + (
                        -(
                            (pl.col("_P0_CLIPPED") / (1 - pl.col("_P0_CLIPPED"))).log()
                            + delta * pl.col("vulnerability")
                        )
                    ).exp()
                )
            ).alias("p1")
        )
        .with_columns(
            (pl.col("shared_uniform") < pl.col("p0")).cast(pl.Int64).alias("y0"),
            (pl.col("shared_uniform") < pl.col("p1")).cast(pl.Int64).alias("y1"),
        )
        .select(ID_COLUMN, "p0", "p1", "vulnerability", "shared_uniform", "y0", "y1")
        .collect()
    )
    outcomes = validate_outcome_feed(
        evidence.select(
            pl.col(ID_COLUMN),
            pl.col("y1").alias(TARGET_COLUMN),
            pl.lit(available_at).alias("available_at"),
        )
    )
    return ConceptDriftSimulation(evidence=evidence, outcomes=outcomes)
