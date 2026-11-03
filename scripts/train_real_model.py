"""Train and register the real-data logistic baseline."""

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict

import mlflow
import polars as pl

from credit_monitoring.config import load_training_config
from credit_monitoring.data import MODEL_FEATURE_COLUMNS
from credit_monitoring.training import evaluate_credit_model, fit_credit_model


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
    train = data.filter(pl.col("partition") == "train")
    validation = data.filter(pl.col("partition") == "validation")
    test = data.filter(pl.col("partition") == "test")
    model = fit_credit_model(
        train,
        validation,
        seed=config.seed,
        model_version=config.model_version,
    )
    validation_metrics = evaluate_credit_model(model, validation)
    test_metrics = evaluate_credit_model(model, test)

    output_dir = config.runtime_dir / "training-e4"
    artifact_store = config.runtime_dir / "mlflow-artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_store.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "credit-model.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)

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
    (output_dir / "source-metadata.json").write_text(
        source_metadata_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (output_dir / "split-manifest.json").write_text(
        split_manifest_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
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
                "config_sha256": config_sha256,
            }
        )
        mlflow.log_metrics(
            {
                **{
                    f"validation_{name}": value
                    for name, value in asdict(validation_metrics).items()
                },
                **{f"test_{name}": value for name, value in asdict(test_metrics).items()},
            }
        )
        mlflow.set_tags(
            {
                "model_version": config.model_version,
                "pipeline_state": "TRAINED",
                "reference_partition": "test",
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
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
