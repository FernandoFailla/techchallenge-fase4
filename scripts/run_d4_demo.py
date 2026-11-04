"""Run the deterministic D4 vertical slice and persist local evidence."""

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict
from datetime import UTC, datetime
from time import perf_counter

import mlflow

from credit_monitoring.config import load_demo_config
from credit_monitoring.health import collect_process_health
from credit_monitoring.monitoring import analyze_data_drift
from credit_monitoring.scoring import fit_baseline_model, score_batch
from credit_monitoring.simulation import (
    corrupt_scoring_batch,
    generate_historical_data,
    scoring_view,
    simulate_data_drift,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/d4-demo.yaml")
    args = parser.parse_args()

    config = load_demo_config(args.config)
    output_dir = config.runtime_dir / "d4-demo"
    artifact_store = config.runtime_dir / "mlflow-artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_store.mkdir(parents=True, exist_ok=True)

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

    resolved_config = {**asdict(config), "runtime_dir": str(config.runtime_dir)}
    config_payload = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(config_payload.encode()).hexdigest()
    events: list[dict[str, object]] = []

    with mlflow.start_run(experiment_id=experiment_id, run_name="d4-demo") as run:
        run_id = run.info.run_id

        def record_event(stage: str, status: str, message: str) -> None:
            events.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "level": "INFO",
                    "run_id": run_id,
                    "batch_id": "d4-current",
                    "stage": stage,
                    "status": status,
                    "message": message,
                }
            )

        total_started = perf_counter()
        record_event("demo", "STARTED", "D4 demonstration started")

        training_started = perf_counter()
        historical = generate_historical_data(config.training_rows, seed=config.seed)
        model = fit_baseline_model(historical, random_state=config.seed)
        training_seconds = perf_counter() - training_started
        record_event("training", "COMPLETED", "Baseline model trained")

        scoring_started = perf_counter()
        reference = scoring_view(
            generate_historical_data(
                config.scoring_rows,
                seed=config.seed + 1,
                id_start=10_000,
            )
        )
        current = simulate_data_drift(
            reference,
            limit_balance_multiplier=config.limit_balance_multiplier,
            age_shift=config.age_shift,
        )
        scored = score_batch(model, current, review_threshold=config.review_threshold)
        rejected = score_batch(
            model,
            corrupt_scoring_batch(reference),
            review_threshold=config.review_threshold,
        )
        scoring_seconds = perf_counter() - scoring_started
        record_event("scoring", scored.state.value, "Valid drifted batch processed")
        record_event("contract", rejected.state.value, "Invalid batch blocked before scoring")

        monitoring_started = perf_counter()
        drift = analyze_data_drift(reference, current)
        monitoring_seconds = perf_counter() - monitoring_started
        processing_state = "PROCESSED_WITH_ALERTS" if drift.summary.alert else "PROCESSED"
        record_event("monitoring", processing_state, "Data drift evaluation completed")

        model_path = output_dir / "baseline-model.pkl"
        report_html_path = output_dir / "data-drift-report.html"
        report_json_path = output_dir / "data-drift-report.json"
        config_path = output_dir / "resolved-config.json"
        summary_path = output_dir / "run-summary.json"
        log_path = output_dir / "events.jsonl"

        with model_path.open("wb") as model_file:
            pickle.dump(model, model_file)
        drift.snapshot.save_html(str(report_html_path))
        drift.snapshot.save_json(str(report_json_path))
        config_path.write_text(
            json.dumps(resolved_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        duration_seconds = perf_counter() - total_started
        process_health = collect_process_health()
        summary = {
            "run_id": run_id,
            "processing_state": processing_state,
            "valid_batch_state": scored.state.value,
            "invalid_batch_state": rejected.state.value,
            "scored_rows": scored.scores.height,
            "invalid_batch_scored_rows": rejected.scores.height,
            "invalid_batch_failure_count": rejected.failure_count,
            "invalid_batch_violations": [asdict(violation) for violation in rejected.violations],
            "drifted_features": drift.summary.drifted_features,
            "drifted_feature_count": drift.summary.drifted_count,
            "drifted_feature_share": drift.summary.drifted_share,
            "review_required_count": scored.scores.get_column("review_required").sum(),
            "duration_seconds": duration_seconds,
            "process_health": asdict(process_health),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        record_event("demo", "COMPLETED", "Artifacts persisted without row-level data")
        log_path.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
            encoding="utf-8",
        )

        mlflow.log_params(
            {
                "seed": config.seed,
                "training_rows": config.training_rows,
                "scoring_rows": config.scoring_rows,
                "review_threshold": config.review_threshold,
                "limit_balance_multiplier": config.limit_balance_multiplier,
                "age_shift": config.age_shift,
                "config_sha256": config_hash,
                "model_version": "baseline-logistic-v1",
                "contract_version": "scoring_v1",
            }
        )
        mlflow.log_metrics(
            {
                "duration_seconds": duration_seconds,
                "training_seconds": training_seconds,
                "scoring_seconds": scoring_seconds,
                "monitoring_seconds": monitoring_seconds,
                "received_rows": current.height,
                "scored_rows": scored.scores.height,
                "invalid_batch_failure_count": rejected.failure_count,
                "invalid_batch_scored_rows": rejected.scores.height,
                "drifted_feature_count": drift.summary.drifted_count,
                "drifted_feature_share": drift.summary.drifted_share,
                "alert_count": int(drift.summary.alert),
                "review_required_count": summary["review_required_count"],
                "process_cpu_count": process_health.cpu_count,
                "process_max_rss_mb": process_health.max_rss_mb,
            }
        )
        mlflow.set_tags(
            {
                "batch_id": "d4-current",
                "batch_state": scored.state.value,
                "invalid_batch_state": rejected.state.value,
                "processing_state": processing_state,
                "drifted_features": ",".join(drift.summary.drifted_features),
                "python_version": process_health.python_version,
                "platform": process_health.platform,
            }
        )
        mlflow.log_artifacts(output_dir, artifact_path="d4-demo")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
