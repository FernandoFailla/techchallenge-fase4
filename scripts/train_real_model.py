"""Train and register the real-data logistic baseline."""

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict

import mlflow
import polars as pl

from credit_monitoring.config import load_training_config
from credit_monitoring.contracts import ID_COLUMN, BatchContractError
from credit_monitoring.data import MODEL_FEATURE_COLUMNS, PARTITIONS
from credit_monitoring.health import collect_process_health
from credit_monitoring.provenance import collect_source_provenance, file_sha256
from credit_monitoring.training import (
    evaluate_credit_model,
    fit_credit_model,
    predict_credit_risk,
    validate_partition_isolation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()

    config = load_training_config(args.config)
    dataset_path = config.data_dir / "processed" / "credit-modeling.parquet"
    source_metadata_path = config.data_dir / "processed" / "source-metadata.json"
    split_manifest_path = config.data_dir / "processed" / "split-manifest.json"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"prepared dataset not found: run prepare_data.py for {dataset_path}"
        )

    data = pl.scan_parquet(dataset_path).collect()
    if set(data.get_column("partition").unique()) != set(PARTITIONS):
        raise ValueError("prepared dataset must contain the four documented partitions")
    partitions = {
        partition: data.filter(pl.col("partition") == partition) for partition in PARTITIONS
    }
    validate_partition_isolation(partitions)
    train = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]
    model = fit_credit_model(
        train,
        validation,
        seed=config.seed,
        model_version=config.model_version,
    )
    validation_metrics = evaluate_credit_model(model, validation)
    test_metrics = evaluate_credit_model(model, test)
    invalid_source = test.select(ID_COLUMN, *MODEL_FEATURE_COLUMNS).head(4)
    first_identifier = invalid_source.item(0, ID_COLUMN)
    invalid_batch = invalid_source.with_columns(
        pl.when(pl.col(ID_COLUMN) == first_identifier)
        .then(pl.lit(-1.0))
        .otherwise(pl.col("LIMIT_BAL"))
        .alias("LIMIT_BAL"),
        pl.when(pl.col(ID_COLUMN) == first_identifier)
        .then(pl.lit(9))
        .otherwise(pl.col("PAY_0"))
        .alias("PAY_0"),
        pl.when(pl.col(ID_COLUMN) == first_identifier)
        .then(pl.lit(3.0))
        .otherwise(pl.col("UTILIZATION_CURRENT"))
        .alias("UTILIZATION_CURRENT"),
    )
    try:
        invalid_scores = predict_credit_risk(model, invalid_batch)
    except BatchContractError as error:
        contract_evidence = {
            "state": "REJECTED",
            "scored_rows": 0,
            "failure_count": error.failure_count,
            "violations": [asdict(violation) for violation in error.violations],
        }
    else:
        raise RuntimeError(
            f"invalid real-data batch bypassed the contract and scored {invalid_scores.height} rows"
        )

    output_dir = config.runtime_dir / "training-e4"
    artifact_store = config.runtime_dir / "mlflow-artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_store.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "credit-model.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)
    dataset_sha256 = file_sha256(dataset_path)
    model_sha256 = file_sha256(model_path)
    source = collect_source_provenance()
    process_health = collect_process_health()

    resolved_config = {
        **asdict(config),
        "data_dir": str(config.data_dir),
        "runtime_dir": str(config.runtime_dir),
    }
    metrics = {
        "validation": asdict(validation_metrics),
        "test": asdict(test_metrics),
    }
    model_metadata = {
        "model_version": model.model_version,
        "model_type": "standardized_logistic_regression",
        "scorecard_active": False,
        "feature_columns": model.feature_columns,
        "review_threshold": model.threshold,
        "threshold_objective": "F2 on validation",
        "reference_partition": "test",
        "model_sha256": model_sha256,
        "source_provenance": asdict(source),
        "process_health": asdict(process_health),
    }
    (output_dir / "resolved-config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "model-metadata.json").write_text(
        json.dumps(model_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "contract-evidence.json").write_text(
        json.dumps(contract_evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "source-metadata.json").write_text(
        source_metadata_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (output_dir / "split-manifest.json").write_text(
        split_manifest_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    config_payload = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"))
    config_sha256 = hashlib.sha256(config_payload.encode()).hexdigest()
    tracking_uri = f"sqlite:///{(config.runtime_dir / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            config.experiment_name,
            artifact_location=artifact_store.resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=config.model_version) as run:
        mlflow.log_params(
            {
                "model_version": config.model_version,
                "model_type": model_metadata["model_type"],
                "scorecard_active": False,
                "seed": config.seed,
                "review_threshold": model.threshold,
                "threshold_objective": model_metadata["threshold_objective"],
                "feature_columns": ",".join(MODEL_FEATURE_COLUMNS),
                "dataset_sha256": dataset_sha256,
                "model_sha256": model_sha256,
                "config_sha256": config_sha256,
                "source_revision": source.revision,
                "source_tree_sha256": source.tree_sha256,
                "source_worktree_dirty": source.worktree_dirty,
                "python_version": process_health.python_version,
                "platform": process_health.platform,
            }
        )
        mlflow.log_metrics(
            {
                **{
                    f"validation_{name}": value
                    for name, value in asdict(validation_metrics).items()
                },
                **{f"test_{name}": value for name, value in asdict(test_metrics).items()},
                "invalid_batch_failure_count": contract_evidence["failure_count"],
                "invalid_batch_scored_rows": contract_evidence["scored_rows"],
                "process_cpu_count": process_health.cpu_count,
                "process_max_rss_mb": process_health.max_rss_mb,
            }
        )
        mlflow.set_tags(
            {
                "model_version": config.model_version,
                "pipeline_state": "TRAINED",
                "reference_partition": "test",
                "invalid_batch_state": contract_evidence["state"],
                "source_worktree_dirty": source.worktree_dirty,
            }
        )
        mlflow.log_artifacts(output_dir, artifact_path="training-e4")
        run_id = run.info.run_id

    print(
        json.dumps(
            {
                "run_id": run_id,
                "model_version": config.model_version,
                "review_threshold": model.threshold,
                "validation": asdict(validation_metrics),
                "test": asdict(test_metrics),
                "invalid_batch": contract_evidence,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
