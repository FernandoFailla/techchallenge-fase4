from datetime import datetime

import polars as pl
import pytest

from credit_monitoring.contracts import ID_COLUMN, MODEL_FEATURE_COLUMNS, TARGET_COLUMN
from credit_monitoring.simulation import (
    MONTH_COLUMN,
    build_drift_cohorts,
    simulate_concept_drift,
)
from credit_monitoring.training import fit_credit_model


def modeling_fixture(rows: int = 600) -> pl.DataFrame:
    return (
        pl.DataFrame({ID_COLUMN: range(1, rows + 1)})
        .lazy()
        .with_columns(
            (20_000 + pl.col(ID_COLUMN) % 20 * 10_000).cast(pl.Float64).alias("LIMIT_BAL"),
            ((pl.col(ID_COLUMN) % 6) - 2).alias("PAY_0"),
            ((pl.col(ID_COLUMN) % 8) - 2).alias("MAX_DELAY_6M"),
            (pl.col(ID_COLUMN) % 7).alias("DELAY_MONTHS_6M"),
            (pl.col(ID_COLUMN) % 100 / 80).alias("UTILIZATION_CURRENT"),
            (pl.col(ID_COLUMN) % 90 / 80).alias("UTILIZATION_MEAN_6M"),
            (pl.col(ID_COLUMN) % 20 / 20).alias("PAYMENT_TO_BILL_RATIO_6M"),
        )
        .with_columns(
            (
                (pl.col("PAY_0") > 0)
                | (pl.col("UTILIZATION_CURRENT") > 0.9)
                | (pl.col(ID_COLUMN) % 11 == 0)
            )
            .cast(pl.Int64)
            .alias(TARGET_COLUMN)
        )
        .collect()
    )


def test_drift_cohorts_are_deterministic_disjoint_and_do_not_edit_features() -> None:
    pool = modeling_fixture(900)
    first = build_drift_cohorts(
        pool,
        seed=43,
        rows_per_month=200,
        stress_strength=2.0,
    )
    second = build_drift_cohorts(
        pool,
        seed=43,
        rows_per_month=200,
        stress_strength=2.0,
    )
    stronger = build_drift_cohorts(
        pool,
        seed=43,
        rows_per_month=200,
        stress_strength=4.0,
    )

    selected_source = pool.join(first.select(ID_COLUMN), on=ID_COLUMN, how="semi").select(
        ID_COLUMN, *MODEL_FEATURE_COLUMNS
    )
    selected_cohorts = first.select(ID_COLUMN, *MODEL_FEATURE_COLUMNS)

    assert first.equals(second)
    assert first.height == 600
    assert first.get_column(ID_COLUMN).n_unique() == first.height
    assert set(first.get_column(MONTH_COLUMN).unique()) == {"M1", "M2", "M3"}
    assert selected_source.sort(ID_COLUMN).equals(selected_cohorts.sort(ID_COLUMN))
    assert (
        first.filter(pl.col(MONTH_COLUMN) != "M2")
        .get_column(ID_COLUMN)
        .equals(stronger.filter(pl.col(MONTH_COLUMN) != "M2").get_column(ID_COLUMN))
    )
    assert (
        first.filter(pl.col(MONTH_COLUMN) == "M2").get_column("LIMIT_BAL").mean()
        < first.filter(pl.col(MONTH_COLUMN) == "M1").get_column("LIMIT_BAL").mean()
    )


def test_concept_drift_uses_paired_potential_outcomes() -> None:
    data = modeling_fixture()
    model = fit_credit_model(
        data.head(350),
        data.slice(350, 125),
        seed=42,
        model_version="test-v1",
    )
    current = data.tail(125).drop(TARGET_COLUMN)

    result = simulate_concept_drift(
        model,
        current,
        delta=2.0,
        seed=144,
        available_at=datetime(2026, 4, 1),
    )

    assert (result.evidence.get_column("p1") >= result.evidence.get_column("p0")).all()
    assert (result.evidence.get_column("p1") > result.evidence.get_column("p0")).any()
    assert (result.evidence.get_column("y1") >= result.evidence.get_column("y0")).all()
    assert result.outcomes.get_column(TARGET_COLUMN).equals(result.evidence.get_column("y1"))


def test_drift_cohort_requires_candidates_beyond_the_three_months() -> None:
    with pytest.raises(ValueError, match="selection reserve"):
        build_drift_cohorts(
            modeling_fixture(600),
            seed=43,
            rows_per_month=200,
            stress_strength=2.0,
        )


def test_drift_simulation_rejects_non_finite_strengths() -> None:
    pool = modeling_fixture(900)
    with pytest.raises(ValueError, match="stress_strength must be greater than zero"):
        build_drift_cohorts(
            pool,
            seed=43,
            rows_per_month=200,
            stress_strength=float("nan"),
        )

    model = fit_credit_model(
        pool.head(500),
        pool.slice(500, 200),
        seed=42,
        model_version="test-v1",
    )
    with pytest.raises(ValueError, match="delta must be greater than zero"):
        simulate_concept_drift(
            model,
            pool.tail(200).drop(TARGET_COLUMN),
            delta=float("inf"),
            seed=144,
            available_at=datetime(2026, 4, 1),
        )
