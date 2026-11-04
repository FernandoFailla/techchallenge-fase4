import polars as pl
import pytest

from credit_monitoring.contracts import ID_COLUMN, TARGET_COLUMN
from credit_monitoring.fairness import (
    AGE_GROUP_COLUMN,
    SEX_GROUP_COLUMN,
    assign_age_audit_groups,
    audit_group_fairness,
    build_restricted_audit_view,
)


def age_fixture() -> pl.DataFrame:
    bands = ["21-29"] * 210 + ["30-39"] * 310 + ["40-49"] * 160
    bands += ["50-59"] * 50 + ["60+"] * 20
    return pl.DataFrame({ID_COLUMN: range(1, 751), "AGE_GROUP": bands})


def fairness_fixture() -> pl.DataFrame:
    return (
        pl.DataFrame(
            {
                ID_COLUMN: range(1, 601),
                "group": ["a"] * 300 + ["b"] * 300,
            }
        )
        .lazy()
        .with_columns(((pl.col(ID_COLUMN) % 2) == 0).cast(pl.Int64).alias(TARGET_COLUMN))
        .with_columns(
            pl.when(pl.col("group") == "a")
            .then(pl.col(TARGET_COLUMN) == 1)
            .otherwise(
                pl.when(pl.col(TARGET_COLUMN) == 1)
                .then(pl.col(ID_COLUMN) % 4 == 0)
                .otherwise(pl.col(ID_COLUMN) % 5 == 0)
            )
            .alias("review_required")
        )
        .with_columns(
            pl.when(pl.col("review_required"))
            .then(pl.lit(0.8))
            .otherwise(pl.lit(0.2))
            .alias("default_probability")
        )
        .collect()
    )


def test_age_groups_merge_adjacent_small_bands_without_labels() -> None:
    grouped = assign_age_audit_groups(age_fixture(), minimum_rows=200)

    counts = grouped.group_by(AGE_GROUP_COLUMN).len().sort(AGE_GROUP_COLUMN)
    assert counts.to_dict(as_series=False) == {
        AGE_GROUP_COLUMN: ["21-29", "30-39", "40+"],
        "len": [210, 310, 230],
    }


def test_age_group_boundaries_do_not_adapt_to_current_cohort_counts() -> None:
    data = pl.DataFrame(
        {
            ID_COLUMN: [1, 2, 3, 4, 5],
            "AGE_GROUP": ["21-29", "30-39", "40-49", "50-59", "60+"],
        }
    )

    grouped = assign_age_audit_groups(data, minimum_rows=200)

    assert grouped.get_column(AGE_GROUP_COLUMN).to_list() == [
        "21-29",
        "30-39",
        "40+",
        "40+",
        "40+",
    ]


def test_restricted_view_excludes_raw_demographics_and_model_features() -> None:
    ages = age_fixture()
    cohort = ages.with_columns(
        pl.when(pl.col(ID_COLUMN) <= 338).then(pl.lit(1)).otherwise(pl.lit(2)).alias("SEX")
    )
    matured = (
        ages.lazy()
        .select(ID_COLUMN)
        .with_columns(((pl.col(ID_COLUMN) % 2) == 0).cast(pl.Int64).alias(TARGET_COLUMN))
        .with_columns((pl.col(TARGET_COLUMN) == 1).alias("review_required"))
        .with_columns(
            pl.when(pl.col("review_required"))
            .then(pl.lit(0.8))
            .otherwise(pl.lit(0.2))
            .alias("default_probability")
        )
        .collect()
    )

    restricted = build_restricted_audit_view(cohort, matured, minimum_age_rows=200)

    assert restricted.columns == [
        ID_COLUMN,
        SEX_GROUP_COLUMN,
        AGE_GROUP_COLUMN,
        TARGET_COLUMN,
        "default_probability",
        "review_required",
    ]
    assert "SEX" not in restricted.columns
    assert "AGE_GROUP" not in restricted.columns


def test_fairlearn_audit_publishes_supported_groups_and_opens_investigation() -> None:
    result = audit_group_fairness(
        fairness_fixture(),
        sensitive_feature="group",
        minimum_rows=200,
        minimum_events=30,
        minimum_non_events=30,
        bootstrap_samples=50,
        seed=42,
        disparity_review_threshold=0.10,
    )

    assert result.summary.published_group_count == 2
    assert result.summary.evidence_sufficient is True
    assert result.summary.suppressed_group_count == 0
    assert result.summary.equalized_odds_difference > 0.1
    assert result.summary.investigation_required is True
    assert result.group_metrics.get_column("recall").null_count() == 0


def test_fairlearn_audit_suppresses_small_groups() -> None:
    data = fairness_fixture().with_columns(
        pl.when(pl.col(ID_COLUMN) <= 100)
        .then(pl.lit("small"))
        .otherwise(pl.lit("large"))
        .alias("group")
    )

    result = audit_group_fairness(
        data,
        sensitive_feature="group",
        minimum_rows=200,
        minimum_events=30,
        minimum_non_events=30,
        bootstrap_samples=50,
        seed=42,
        disparity_review_threshold=0.10,
    )

    assert result.summary.published_group_count == 1
    assert result.summary.evidence_sufficient is False
    assert result.summary.suppressed_group_count == 1
    assert result.summary.equalized_odds_difference is None
    suppressed = result.group_metrics.filter(pl.col("group") == "small")
    for column in (
        "events",
        "non_events",
        "mean_probability",
        "review_rate",
        "recall",
    ):
        assert suppressed.item(0, column) is None
    assert suppressed.item(0, "publication_status").startswith("suppressed")


def test_suppressed_group_is_redacted_when_two_other_groups_are_published() -> None:
    data = fairness_fixture().with_columns(
        pl.when(pl.col(ID_COLUMN) <= 20)
        .then(pl.lit("small"))
        .when(pl.col(ID_COLUMN) <= 310)
        .then(pl.lit("a"))
        .otherwise(pl.lit("b"))
        .alias("group")
    )

    result = audit_group_fairness(
        data,
        sensitive_feature="group",
        minimum_rows=200,
        minimum_events=30,
        minimum_non_events=30,
        bootstrap_samples=50,
        seed=42,
        disparity_review_threshold=0.10,
    )

    assert result.summary.evidence_sufficient is False
    assert result.summary.published_group_count == 2
    assert result.summary.suppressed_group_count == 1
    suppressed = result.group_metrics.filter(pl.col("group") == "small")
    assert suppressed.item(0, "rows") == 20
    assert suppressed.item(0, "events") is None
    assert suppressed.item(0, "mean_probability") is None
    assert suppressed.item(0, "recall") is None


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (TARGET_COLUMN, 2, f"{TARGET_COLUMN} must be binary"),
        ("review_required", 2, "review_required must be binary"),
        ("default_probability", float("inf"), "finite values between 0 and 1"),
    ],
)
def test_fairness_rejects_invalid_statistical_inputs(
    column: str,
    value: float,
    message: str,
) -> None:
    data = fairness_fixture().with_columns(
        pl.when(pl.col(ID_COLUMN) == 1).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
    )

    with pytest.raises(ValueError, match=message):
        audit_group_fairness(
            data,
            sensitive_feature="group",
            minimum_rows=200,
            minimum_events=30,
            minimum_non_events=30,
            bootstrap_samples=50,
            seed=42,
            disparity_review_threshold=0.10,
        )


def test_fairness_threshold_must_be_a_probability() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        audit_group_fairness(
            fairness_fixture(),
            sensitive_feature="group",
            minimum_rows=200,
            minimum_events=30,
            minimum_non_events=30,
            bootstrap_samples=50,
            seed=42,
            disparity_review_threshold=1.1,
        )
