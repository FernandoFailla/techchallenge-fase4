"""UCI ingestion, behavioral features, and leakage-safe partitions."""

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import polars as pl

from credit_monitoring.contracts import ID_COLUMN, MODEL_FEATURE_COLUMNS, TARGET_COLUMN

UCI_SOURCE_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
UCI_ARCHIVE_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
UCI_WORKBOOK_MEMBER = "default of credit card clients.xls"
UCI_DOI = "10.24432/C55S3H"
UCI_LICENSE = "CC BY 4.0"

PAY_STATUS_COLUMNS = ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
BILL_COLUMNS = tuple(f"BILL_AMT{month}" for month in range(1, 7))
PAYMENT_COLUMNS = tuple(f"PAY_AMT{month}" for month in range(1, 7))
RAW_COLUMNS = (
    ID_COLUMN,
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_STATUS_COLUMNS,
    *BILL_COLUMNS,
    *PAYMENT_COLUMNS,
    TARGET_COLUMN,
)
AUDIT_COLUMNS = ("SEX", "EDUCATION", "MARRIAGE", "AGE", "AGE_GROUP")
PARTITIONS = ("train", "validation", "test", "simulation")


@dataclass(frozen=True)
class DownloadedDataset:
    archive_path: Path
    workbook_path: Path
    archive_sha256: str


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def download_uci_credit_data(
    raw_dir: str | Path,
    *,
    source_url: str = UCI_SOURCE_URL,
    expected_sha256: str = UCI_ARCHIVE_SHA256,
) -> DownloadedDataset:
    destination = Path(raw_dir)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "default-credit-card-clients.zip"
    workbook_path = destination / UCI_WORKBOOK_MEMBER

    if archive_path.exists():
        archive_content = archive_path.read_bytes()
    else:
        with urlopen(source_url, timeout=60) as response:
            archive_content = response.read()

    archive_sha256 = hashlib.sha256(archive_content).hexdigest()
    if archive_sha256 != expected_sha256:
        raise ValueError(
            f"UCI archive checksum mismatch: expected {expected_sha256}, got {archive_sha256}"
        )

    if not archive_path.exists():
        _atomic_write(archive_path, archive_content)
    with ZipFile(BytesIO(archive_content)) as archive:
        if UCI_WORKBOOK_MEMBER not in archive.namelist():
            raise ValueError(f"UCI archive does not contain {UCI_WORKBOOK_MEMBER}")
        workbook_content = archive.read(UCI_WORKBOOK_MEMBER)
    _atomic_write(workbook_path, workbook_content)

    return DownloadedDataset(
        archive_path=archive_path,
        workbook_path=workbook_path,
        archive_sha256=archive_sha256,
    )


def normalize_uci_frame(raw: pl.DataFrame) -> pl.DataFrame:
    if raw.width != len(RAW_COLUMNS) or raw.height < 3:
        raise ValueError("unexpected UCI workbook dimensions")

    headers = tuple(str(value).strip().replace(" ", "_") for value in raw.row(1))
    if headers != RAW_COLUMNS:
        raise ValueError("unexpected UCI workbook headers")

    return (
        raw.slice(2)
        .lazy()
        .select(
            pl.col(source).cast(pl.Int64).alias(target)
            for source, target in zip(raw.columns, RAW_COLUMNS, strict=True)
        )
        .collect()
    )


def load_uci_credit_data(workbook_path: str | Path) -> pl.DataFrame:
    raw = pl.read_excel(workbook_path, has_header=False, engine="calamine")
    data = normalize_uci_frame(raw)
    if data.height != 30_000:
        raise ValueError(f"expected 30000 UCI rows, got {data.height}")
    if data.get_column(ID_COLUMN).n_unique() != data.height:
        raise ValueError("UCI identifiers must be unique")
    if data.null_count().select(pl.sum_horizontal(pl.all())).item() != 0:
        raise ValueError("UCI data must not contain null values")
    if set(data.get_column(TARGET_COLUMN).unique()) != {0, 1}:
        raise ValueError("UCI target must be binary")
    return data


def build_modeling_dataset(data: pl.DataFrame) -> pl.DataFrame:
    missing = sorted(set(RAW_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"missing UCI columns: {', '.join(missing)}")

    curated = (
        data.lazy()
        .with_columns(
            pl.max_horizontal(*PAY_STATUS_COLUMNS).alias("MAX_DELAY_6M"),
            pl.sum_horizontal(
                *((pl.col(column) > 0).cast(pl.Int64) for column in PAY_STATUS_COLUMNS)
            ).alias("DELAY_MONTHS_6M"),
            (pl.col("BILL_AMT1") / pl.col("LIMIT_BAL"))
            .clip(-1.0, 2.0)
            .alias("UTILIZATION_CURRENT"),
            (pl.mean_horizontal(*BILL_COLUMNS) / pl.col("LIMIT_BAL"))
            .clip(-1.0, 2.0)
            .alias("UTILIZATION_MEAN_6M"),
            pl.sum_horizontal(*PAYMENT_COLUMNS).alias("_TOTAL_PAYMENT_6M"),
            pl.sum_horizontal(*(pl.col(column).clip(lower_bound=0) for column in BILL_COLUMNS))
            .cast(pl.Float64)
            .alias("_POSITIVE_BILL_6M"),
            pl.when(pl.col("EDUCATION").is_in([1, 2, 3]))
            .then(pl.col("EDUCATION"))
            .otherwise(pl.lit(4))
            .alias("EDUCATION"),
            pl.when(pl.col("MARRIAGE").is_in([1, 2]))
            .then(pl.col("MARRIAGE"))
            .otherwise(pl.lit(3))
            .alias("MARRIAGE"),
            pl.when(pl.col("AGE") < 30)
            .then(pl.lit("21-29"))
            .when(pl.col("AGE") < 40)
            .then(pl.lit("30-39"))
            .when(pl.col("AGE") < 50)
            .then(pl.lit("40-49"))
            .when(pl.col("AGE") < 60)
            .then(pl.lit("50-59"))
            .otherwise(pl.lit("60+"))
            .alias("AGE_GROUP"),
        )
        .with_columns(
            pl.when(pl.col("_POSITIVE_BILL_6M") > 0)
            .then(pl.col("_TOTAL_PAYMENT_6M") / pl.col("_POSITIVE_BILL_6M"))
            .otherwise(0.0)
            .clip(0.0, 5.0)
            .alias("PAYMENT_TO_BILL_RATIO_6M")
        )
        .select(
            pl.col(ID_COLUMN),
            pl.col("LIMIT_BAL").cast(pl.Float64),
            *MODEL_FEATURE_COLUMNS[1:],
            pl.col(TARGET_COLUMN),
            *AUDIT_COLUMNS,
        )
        .collect()
    )
    if curated.get_column(ID_COLUMN).n_unique() != curated.height:
        raise ValueError("curated identifiers must be unique")
    if curated.null_count().select(pl.sum_horizontal(pl.all())).item() != 0:
        raise ValueError("curated data must not contain null values")
    return curated


def assign_stratified_partitions(
    data: pl.DataFrame,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    simulation_fraction: float,
) -> pl.DataFrame:
    fractions = (train_fraction, validation_fraction, test_fraction, simulation_fraction)
    if any(fraction <= 0 for fraction in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("partition fractions must be positive and sum to 1")
    if data.get_column(ID_COLUMN).n_unique() != data.height:
        raise ValueError("identifiers must be unique before partitioning")

    validation_end = train_fraction + validation_fraction
    test_end = validation_end + test_fraction
    partitioned = (
        data.lazy()
        .with_columns(
            (((pl.col(ID_COLUMN) * 48_271 + seed) % 2_147_483_647).rank("ordinal"))
            .over(TARGET_COLUMN)
            .alias("_STRATIFIED_RANK"),
            pl.len().over(TARGET_COLUMN).alias("_CLASS_ROWS"),
        )
        .with_columns(
            ((pl.col("_STRATIFIED_RANK") - 1) / pl.col("_CLASS_ROWS")).alias("_PARTITION_POSITION")
        )
        .with_columns(
            pl.when(pl.col("_PARTITION_POSITION") < train_fraction)
            .then(pl.lit("train"))
            .when(pl.col("_PARTITION_POSITION") < validation_end)
            .then(pl.lit("validation"))
            .when(pl.col("_PARTITION_POSITION") < test_end)
            .then(pl.lit("test"))
            .otherwise(pl.lit("simulation"))
            .alias("partition")
        )
        .drop("_STRATIFIED_RANK", "_CLASS_ROWS", "_PARTITION_POSITION")
        .sort(ID_COLUMN)
        .collect()
    )
    if set(partitioned.get_column("partition").unique()) != set(PARTITIONS):
        raise ValueError("all configured partitions must contain rows")
    return partitioned


def build_partition_manifest(data: pl.DataFrame) -> pl.DataFrame:
    return (
        data.lazy()
        .group_by("partition")
        .agg(
            pl.len().alias("rows"),
            pl.col(TARGET_COLUMN).sum().alias("events"),
            pl.col(TARGET_COLUMN).mean().alias("event_rate"),
        )
        .with_columns((pl.col("rows") / pl.col("rows").sum()).alias("fraction"))
        .sort("partition")
        .collect()
    )
