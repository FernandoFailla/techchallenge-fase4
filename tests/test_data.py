import hashlib
from zipfile import ZipFile

import polars as pl
import pytest

from credit_monitoring.contracts import ID_COLUMN, TARGET_COLUMN
from credit_monitoring.data import (
    AUDIT_COLUMNS,
    BILL_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    PAY_STATUS_COLUMNS,
    PAYMENT_COLUMNS,
    UCI_WORKBOOK_MEMBER,
    assign_stratified_partitions,
    build_modeling_dataset,
    build_partition_manifest,
    download_uci_credit_data,
)


def source_fixture(rows: int = 100) -> pl.DataFrame:
    identifiers = list(range(1, rows + 1))
    return pl.DataFrame(
        {
            ID_COLUMN: identifiers,
            "LIMIT_BAL": [10_000 + index * 1_000 for index in identifiers],
            "SEX": [1 + index % 2 for index in identifiers],
            "EDUCATION": [index % 7 for index in identifiers],
            "MARRIAGE": [index % 4 for index in identifiers],
            "AGE": [21 + index % 50 for index in identifiers],
            **{column: [(index % 6) - 2 for index in identifiers] for column in PAY_STATUS_COLUMNS},
            **{column: [index * 500 for index in identifiers] for column in BILL_COLUMNS},
            **{column: [index * 100 for index in identifiers] for column in PAYMENT_COLUMNS},
            TARGET_COLUMN: [index % 2 for index in identifiers],
        }
    )


def test_download_validates_checksum_and_extracts_workbook(tmp_path) -> None:
    archive_path = tmp_path / "fixture.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(UCI_WORKBOOK_MEMBER, b"workbook")
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    result = download_uci_credit_data(
        tmp_path / "raw",
        source_url=archive_path.as_uri(),
        expected_sha256=checksum,
    )

    assert result.workbook_path.read_bytes() == b"workbook"
    assert result.archive_sha256 == checksum


def test_download_rejects_checksum_mismatch(tmp_path) -> None:
    archive_path = tmp_path / "fixture.zip"
    archive_path.write_bytes(b"not-the-expected-archive")

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_uci_credit_data(
            tmp_path / "raw",
            source_url=archive_path.as_uri(),
            expected_sha256="0" * 64,
        )


def test_curated_features_exclude_demographics_and_bound_ratios() -> None:
    source = source_fixture(100).with_columns(
        pl.when(pl.col(ID_COLUMN) == 1)
        .then(pl.lit(1))
        .otherwise(pl.col("BILL_AMT1"))
        .alias("BILL_AMT1"),
        pl.when(pl.col(ID_COLUMN) == 1)
        .then(pl.lit(100_000))
        .otherwise(pl.col("PAY_AMT1"))
        .alias("PAY_AMT1"),
    )

    curated = build_modeling_dataset(source)

    assert set(MODEL_FEATURE_COLUMNS).isdisjoint(AUDIT_COLUMNS)
    assert curated.get_column("PAYMENT_TO_BILL_RATIO_6M").max() <= 5.0
    assert set(curated.get_column("EDUCATION").unique()) <= {1, 2, 3, 4}
    assert set(curated.get_column("MARRIAGE").unique()) <= {1, 2, 3}


def test_stratified_partitions_are_deterministic_and_disjoint() -> None:
    curated = build_modeling_dataset(source_fixture(1_000))
    kwargs = {
        "seed": 42,
        "train_fraction": 0.6,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "simulation_fraction": 0.1,
    }

    first = assign_stratified_partitions(curated, **kwargs)
    second = assign_stratified_partitions(curated, **kwargs)
    manifest = build_partition_manifest(first)

    assert first.equals(second)
    assert first.get_column(ID_COLUMN).n_unique() == first.height
    assert set(first.get_column("partition").unique()) == {
        "train",
        "validation",
        "test",
        "simulation",
    }
    assert manifest.get_column("rows").sum() == 1_000
    assert manifest.get_column("event_rate").max() - manifest.get_column("event_rate").min() < 0.01
