"""Download, curate, partition, and persist the UCI credit dataset."""

import argparse
import json

from credit_monitoring.config import load_training_config
from credit_monitoring.contracts import TARGET_COLUMN
from credit_monitoring.data import (
    UCI_DOI,
    UCI_LICENSE,
    UCI_SOURCE_URL,
    assign_stratified_partitions,
    build_modeling_dataset,
    build_partition_manifest,
    download_uci_credit_data,
    load_uci_credit_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()

    config = load_training_config(args.config)
    raw_dir = config.data_dir / "raw"
    processed_dir = config.data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    download = download_uci_credit_data(raw_dir)
    source = load_uci_credit_data(download.workbook_path)
    modeling = build_modeling_dataset(source)
    partitioned = assign_stratified_partitions(
        modeling,
        seed=config.seed,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        simulation_fraction=config.simulation_fraction,
    )
    manifest = build_partition_manifest(partitioned)

    dataset_path = processed_dir / "credit-modeling.parquet"
    temporary_path = dataset_path.with_suffix(".parquet.tmp")
    partitioned.write_parquet(temporary_path)
    temporary_path.replace(dataset_path)

    source_metadata = {
        "name": "Default of Credit Card Clients",
        "creator": "I-Cheng Yeh",
        "doi": UCI_DOI,
        "license": UCI_LICENSE,
        "source_url": UCI_SOURCE_URL,
        "archive_sha256": download.archive_sha256,
        "rows": source.height,
        "raw_feature_count": source.width - 2,
        "target": TARGET_COLUMN,
        "event_rate": source.get_column(TARGET_COLUMN).mean(),
    }
    metadata_path = processed_dir / "source-metadata.json"
    metadata_path.write_text(
        json.dumps(source_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = processed_dir / "split-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dicts(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "rows": partitioned.height,
                "event_rate": source_metadata["event_rate"],
                "archive_sha256": download.archive_sha256,
                "partitions": manifest.to_dicts(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
