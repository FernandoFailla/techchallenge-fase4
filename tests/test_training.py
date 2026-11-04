import polars as pl
import pytest

from credit_monitoring.contracts import ID_COLUMN, TARGET_COLUMN, BatchContractError
from credit_monitoring.data import MODEL_FEATURE_COLUMNS
from credit_monitoring.training import (
    evaluate_credit_model,
    fit_credit_model,
    predict_credit_risk,
    validate_partition_isolation,
)


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


def test_fit_selects_threshold_on_validation_and_evaluates() -> None:
    data = modeling_fixture()
    train = data.head(350)
    validation = data.slice(350, 125)
    test = data.tail(125)

    model = fit_credit_model(train, validation, seed=42, model_version="test-v1")
    metrics = evaluate_credit_model(model, test)
    scoring_data = test.drop(TARGET_COLUMN)
    scores = predict_credit_risk(model, scoring_data)

    assert 0 <= model.threshold <= 1
    assert model.feature_columns == MODEL_FEATURE_COLUMNS
    assert metrics.roc_auc > 0.8
    assert scores.height == test.height
    assert scores.get_column("review_required").dtype == pl.Boolean
    assert TARGET_COLUMN not in scoring_data.columns


def test_validation_labels_do_not_change_fitted_coefficients() -> None:
    data = modeling_fixture()
    train = data.head(350)
    validation = data.slice(350, 125)
    inverted_validation = validation.with_columns((1 - pl.col(TARGET_COLUMN)).alias(TARGET_COLUMN))

    first = fit_credit_model(train, validation, seed=42, model_version="test-v1")
    second = fit_credit_model(train, inverted_validation, seed=42, model_version="test-v1")

    first_coefficients = first.pipeline.named_steps["classifier"].coef_.tolist()
    second_coefficients = second.pipeline.named_steps["classifier"].coef_.tolist()
    assert first_coefficients == second_coefficients


def test_real_scoring_contract_blocks_out_of_range_features() -> None:
    data = modeling_fixture()
    model = fit_credit_model(
        data.head(350),
        data.slice(350, 125),
        seed=42,
        model_version="test-v1",
    )
    invalid = (
        data.tail(125).drop(TARGET_COLUMN).with_columns(pl.lit(3.0).alias("UTILIZATION_CURRENT"))
    )

    with pytest.raises(BatchContractError):
        predict_credit_risk(model, invalid)


def test_partition_isolation_rejects_duplicate_or_overlapping_identifiers() -> None:
    data = modeling_fixture()
    duplicate = pl.concat((data.head(100), data.head(1)))

    with pytest.raises(ValueError, match="identifiers must be unique"):
        validate_partition_isolation({"train": duplicate, "validation": data.slice(100, 100)})
    with pytest.raises(ValueError, match="overlaps earlier partitions"):
        validate_partition_isolation({"train": data.head(100), "validation": data.slice(99, 100)})
