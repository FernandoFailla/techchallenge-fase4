import polars as pl

from credit_monitoring.contracts import FEATURE_COLUMNS, TARGET_COLUMN
from credit_monitoring.simulation import (
    corrupt_scoring_batch,
    generate_historical_data,
    scoring_view,
    simulate_data_drift,
)


def test_historical_generation_is_reproducible() -> None:
    first = generate_historical_data(200, seed=42)
    second = generate_historical_data(200, seed=42)

    assert first.equals(second)
    assert first.get_column(TARGET_COLUMN).n_unique() == 2


def test_data_drift_changes_only_two_features() -> None:
    reference = scoring_view(generate_historical_data(200, seed=43, id_start=10_000))

    current = simulate_data_drift(
        reference,
        limit_balance_multiplier=0.65,
        age_shift=8,
    )

    changed = {
        column
        for column in FEATURE_COLUMNS
        if not reference.get_column(column).equals(current.get_column(column))
    }
    assert changed == {"LIMIT_BAL", "AGE"}
    assert reference.get_column("ID").equals(current.get_column("ID"))


def test_corrupt_batch_has_deterministic_anomalies() -> None:
    reference = scoring_view(generate_historical_data(100, seed=44))

    invalid = corrupt_scoring_batch(reference)

    assert invalid.height == 10
    assert invalid.get_column("ID").n_unique() == 9
    assert invalid.filter(pl.col("LIMIT_BAL") <= 0).height == 1
    assert invalid.filter(pl.col("AGE") < 18).height == 1
