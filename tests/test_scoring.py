import polars as pl

from credit_monitoring.contracts import FEATURE_COLUMNS, TARGET_COLUMN
from credit_monitoring.scoring import BatchState, fit_baseline_model, score_batch

SCHEMA = {
    "ID": pl.Int64,
    "LIMIT_BAL": pl.Float64,
    "AGE": pl.Int64,
    "PAY_0": pl.Int64,
    "BILL_AMT1": pl.Float64,
    "PAY_AMT1": pl.Float64,
}


def historical_fixture() -> pl.DataFrame:
    rows = {
        "ID": list(range(1, 13)),
        "LIMIT_BAL": [
            20_000.0,
            30_000.0,
            40_000.0,
            50_000.0,
            60_000.0,
            80_000.0,
            100_000.0,
            120_000.0,
            150_000.0,
            180_000.0,
            200_000.0,
            250_000.0,
        ],
        "AGE": [24, 27, 31, 35, 42, 29, 38, 45, 51, 33, 47, 56],
        "PAY_0": [3, 2, 1, 0, 2, -1, 0, -1, 1, 0, -2, -1],
        "BILL_AMT1": [
            19_000.0,
            28_000.0,
            32_000.0,
            30_000.0,
            55_000.0,
            20_000.0,
            50_000.0,
            45_000.0,
            130_000.0,
            40_000.0,
            30_000.0,
            20_000.0,
        ],
        "PAY_AMT1": [
            0.0,
            500.0,
            1_000.0,
            4_000.0,
            500.0,
            8_000.0,
            6_000.0,
            9_000.0,
            1_000.0,
            7_000.0,
            12_000.0,
            15_000.0,
        ],
        TARGET_COLUMN: [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    }
    return pl.DataFrame(rows, schema={**SCHEMA, TARGET_COLUMN: pl.Int64})


def scoring_fixture() -> pl.DataFrame:
    return historical_fixture().filter(pl.col("ID") <= 3).select("ID", *FEATURE_COLUMNS)


def invalid_scoring_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ID": [1, 1],
            "LIMIT_BAL": [-10.0, 20_000.0],
            "AGE": [12, 35],
            "PAY_0": [0, 9],
            "BILL_AMT1": [1_000.0, 2_000.0],
            "PAY_AMT1": [-1.0, 100.0],
            TARGET_COLUMN: [0, 1],
        },
        schema={**SCHEMA, TARGET_COLUMN: pl.Int64},
    )


def test_valid_batch_is_scored() -> None:
    model = fit_baseline_model(historical_fixture())

    result = score_batch(model, scoring_fixture(), review_threshold=0.4)

    assert result.state is BatchState.SCORED
    assert result.scores.columns == ["ID", "default_probability", "review_required"]
    assert result.scores.height == 3
    assert result.scores.get_column("default_probability").is_between(0, 1).all()
    assert result.violations == ()


def test_invalid_batch_is_rejected_before_model_call() -> None:
    class ModelMustNotBeCalled:
        def predict_proba(self, data: object) -> object:
            raise AssertionError("invalid batch reached the model")

    result = score_batch(ModelMustNotBeCalled(), invalid_scoring_fixture())

    assert result.state is BatchState.REJECTED
    assert result.scores.height == 0
    assert result.failure_count >= 6
    assert ("ID", "field_uniqueness") in {
        (violation.column, violation.check) for violation in result.violations
    }
    assert "column_in_schema" in {violation.check for violation in result.violations}


def test_review_threshold_must_be_a_probability() -> None:
    model = fit_baseline_model(historical_fixture())

    try:
        score_batch(model, scoring_fixture(), review_threshold=1.1)
    except ValueError as error:
        assert str(error) == "review_threshold must be between 0 and 1"
    else:
        raise AssertionError("invalid threshold was accepted")
