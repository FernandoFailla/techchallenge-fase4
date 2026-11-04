"""Download, curate, partition, and persist the UCI credit dataset."""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from time import perf_counter

import mlflow

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
from credit_monitoring.health import collect_process_health
from credit_monitoring.provenance import collect_source_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()

    config = load_training_config(args.config)
    raw_dir = config.data_dir / "raw"
    processed_dir = config.data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_root = config.runtime_dir / "data-preparation-e4"
    artifact_store = config.runtime_dir / "mlflow-artifacts"
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_store.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{(config.runtime_dir / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "credit-data-preparation-e4"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_store.resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    events: list[dict[str, object]] = []
    started = perf_counter()
    with mlflow.start_run(experiment_id=experiment_id, run_name="uci-credit-preparation") as run:
        run_id = run.info.run_id
        output_dir = output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        events_path = output_dir / "events.jsonl"

        def record_event(
            stage: str,
            status: str,
            message: str,
            duration_seconds: float,
            **details: object,
        ) -> None:
            events.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "level": "INFO" if status != "FAILED" else "ERROR",
                    "run_id": run_id,
                    "batch_id": "uci-credit-preparation",
                    "stage": stage,
                    "status": status,
                    "duration_seconds": duration_seconds,
                    "message": message,
                    **details,
                }
            )

        stage = "download"
        stage_started = perf_counter()
        record_event(stage, "STARTED", "Official dataset download started", 0.0)
        try:
            download = download_uci_credit_data(raw_dir)
            record_event(
                stage,
                "COMPLETED",
                "Official archive checksum verified",
                perf_counter() - stage_started,
            )

            stage = "ingestion"
            stage_started = perf_counter()
            source = load_uci_credit_data(download.workbook_path)
            record_event(
                stage,
                "COMPLETED",
                "Workbook schema loaded",
                perf_counter() - stage_started,
                rows=source.height,
            )

            stage = "transformation"
            stage_started = perf_counter()
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
            record_event(
                stage,
                "COMPLETED",
                "Features and disjoint partitions built",
                perf_counter() - stage_started,
                rows=partitioned.height,
            )

            stage = "persistence"
            stage_started = perf_counter()
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
            record_event(
                stage,
                "COMPLETED",
                "Prepared dataset persisted atomically",
                perf_counter() - stage_started,
            )
        except Exception as error:
            record_event(
                stage,
                "FAILED",
                "Data preparation stopped before completion",
                perf_counter() - stage_started,
                error_type=type(error).__name__,
            )
            events_path.write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
                encoding="utf-8",
            )
            mlflow.set_tags({"pipeline_state": "FAILED", "failed_stage": stage})
            mlflow.log_artifact(str(events_path), artifact_path="data-preparation-e4")
            raise

        process_health = collect_process_health()
        source_provenance = collect_source_provenance()
        summary = {
            "run_id": run_id,
            "pipeline_state": "PREPARED",
            "stages": {
                "download": "COMPLETED",
                "ingestion": "COMPLETED",
                "transformation": "COMPLETED",
                "persistence": "COMPLETED",
            },
            "dataset_path": str(dataset_path),
            "rows": partitioned.height,
            "event_rate": source_metadata["event_rate"],
            "archive_sha256": download.archive_sha256,
            "partitions": manifest.to_dicts(),
            "duration_seconds": perf_counter() - started,
            "process_health": asdict(process_health),
            "source_provenance": asdict(source_provenance),
        }
        (output_dir / "run-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        events_path.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
            encoding="utf-8",
        )
        mlflow.log_params(
            {
                "seed": config.seed,
                "source_url": UCI_SOURCE_URL,
                "archive_sha256": download.archive_sha256,
                "source_revision": source_provenance.revision,
                "source_tree_sha256": source_provenance.tree_sha256,
                "source_worktree_dirty": source_provenance.worktree_dirty,
                "python_version": process_health.python_version,
                "platform": process_health.platform,
            }
        )
        mlflow.log_metrics(
            {
                "duration_seconds": summary["duration_seconds"],
                "rows": partitioned.height,
                "event_rate": source_metadata["event_rate"],
                "partition_count": manifest.height,
                "process_cpu_count": process_health.cpu_count,
                "process_max_rss_mb": process_health.max_rss_mb,
            }
        )
        mlflow.set_tags(
            {
                "pipeline_state": "PREPARED",
                "ingestion_state": "COMPLETED",
                "transformation_state": "COMPLETED",
                "source_worktree_dirty": source_provenance.worktree_dirty,
            }
        )
        mlflow.log_artifacts(output_dir, artifact_path="data-preparation-e4")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
