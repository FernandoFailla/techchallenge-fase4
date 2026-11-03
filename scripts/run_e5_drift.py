"""Run the three post-freeze drift cohorts and register aggregate E5 evidence."""

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import mlflow
import polars as pl

from credit_monitoring.config import load_drift_config
from credit_monitoring.contracts import ID_COLUMN, MODEL_FEATURE_COLUMNS, TARGET_COLUMN
from credit_monitoring.monitoring import (
    analyze_model_data_drift,
    analyze_population_stability,
    compare_paired_performance,
    default_rate_by_score_band,
)
from credit_monitoring.outcomes import OutcomeState, associate_mature_outcomes
from credit_monitoring.simulation import (
    MONTH_COLUMN,
    build_drift_cohorts,
    simulate_concept_drift,
)
from credit_monitoring.training import CreditModel, predict_credit_risk


def _json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e5-drift.yaml")
    args = parser.parse_args()

    config = load_drift_config(args.config)
    if not config.data_path.exists():
        raise FileNotFoundError(f"prepared dataset not found: {config.data_path}")
    if not config.model_path.exists():
        raise FileNotFoundError(f"trained model not found: {config.model_path}")

    with config.model_path.open("rb") as model_file:
        model = pickle.load(model_file)
    if not isinstance(model, CreditModel):
        raise TypeError("model artifact does not contain a CreditModel")
    if model.model_version != config.expected_model_version:
        raise ValueError(
            f"expected model {config.expected_model_version}, got {model.model_version}"
        )

    data = pl.scan_parquet(config.data_path).collect()
    reference = data.filter(pl.col("partition") == "test").select(ID_COLUMN, *MODEL_FEATURE_COLUMNS)
    pool = data.filter(pl.col("partition") == "simulation")
    reference_scores = predict_credit_risk(model, reference)
    reference_monitoring = reference.join(
        reference_scores,
        on=ID_COLUMN,
        how="inner",
        validate="1:1",
    )

    output_dir = config.runtime_dir / "drift-e5"
    artifact_store = config.runtime_dir / "mlflow-artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_store.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        **asdict(config),
        "data_path": str(config.data_path),
        "model_path": str(config.model_path),
        "runtime_dir": str(config.runtime_dir),
        "outcome_available_at": config.outcome_available_at.isoformat(),
    }
    config_payload = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"))
    config_sha256 = hashlib.sha256(config_payload.encode()).hexdigest()
    dataset_sha256 = hashlib.sha256(config.data_path.read_bytes()).hexdigest()

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

    started = perf_counter()
    events: list[dict[str, object]] = []
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"{config.expected_model_version}-three-months",
    ) as run:
        run_id = run.info.run_id
        run_output_dir = output_dir / run_id
        run_output_dir.mkdir(parents=True, exist_ok=False)

        def record_event(stage: str, status: str, message: str) -> None:
            events.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "run_id": run_id,
                    "stage": stage,
                    "status": status,
                    "message": message,
                }
            )

        record_event("simulation", "STARTED", "E5 post-freeze scenario started")
        cohorts = build_drift_cohorts(
            pool,
            seed=config.seed,
            rows_per_month=config.cohort_rows,
            stress_strength=config.stress_strength,
        )
        month_summaries: dict[str, dict[str, object]] = {}
        month_scores: dict[str, pl.DataFrame] = {}
        for month in ("M1", "M2", "M3"):
            current = cohorts.filter(pl.col(MONTH_COLUMN) == month).select(
                ID_COLUMN, *MODEL_FEATURE_COLUMNS
            )
            scores = predict_credit_risk(model, current)
            month_scores[month] = scores
            current_monitoring = current.join(scores, on=ID_COLUMN, how="inner", validate="1:1")
            drift = analyze_model_data_drift(reference, current)
            stability = analyze_population_stability(
                reference_monitoring,
                current_monitoring,
                bins=config.psi_bins,
                smoothing=config.psi_smoothing,
                minimum_rows=config.psi_minimum_rows,
            )
            drift.snapshot.save_html(str(run_output_dir / f"{month.lower()}-data-drift.html"))
            drift.snapshot.save_json(str(run_output_dir / f"{month.lower()}-data-drift.json"))
            alerted = drift.summary.alert or stability.alert
            processing_state = "PROCESSED_WITH_ALERTS" if alerted else "PROCESSED"
            month_summaries[month] = {
                "rows": current.height,
                "processing_state": processing_state,
                "evidently": asdict(drift.summary),
                "psi": {
                    "feature_warning_count": stability.feature_warning_count,
                    "score_relevant": stability.score_relevant,
                    "alert": stability.alert,
                    "results": [asdict(result) for result in stability.results],
                },
            }
            record_event("scoring", processing_state, f"{month} scoring and drift completed")

        if month_summaries["M1"]["processing_state"] != "PROCESSED":
            raise RuntimeError("M1 must remain stable")
        if month_summaries["M2"]["processing_state"] != "PROCESSED_WITH_ALERTS":
            raise RuntimeError("M2 must produce a data-drift alert")
        if month_summaries["M3"]["processing_state"] != "PROCESSED":
            raise RuntimeError("M3 inputs must remain stable before outcomes mature")

        m3 = cohorts.filter(pl.col(MONTH_COLUMN) == "M3")
        concept = simulate_concept_drift(
            model,
            m3,
            delta=config.concept_delta,
            seed=config.seed + 101,
            available_at=config.outcome_available_at,
        )
        outcome_association = associate_mature_outcomes(
            month_scores["M3"],
            concept.outcomes,
            as_of=config.outcome_available_at,
        )
        if outcome_association.state is not OutcomeState.MATURED:
            raise RuntimeError(f"M3 outcome evaluation failed: {outcome_association.reason}")

        baseline_performance_data = month_scores["M3"].join(
            concept.evidence.select(ID_COLUMN, pl.col("y0").alias(TARGET_COLUMN)),
            on=ID_COLUMN,
            how="inner",
            validate="1:1",
        )
        performance = compare_paired_performance(
            baseline_performance_data,
            outcome_association.data,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed + 202,
        )
        brier_shift = next(shift for shift in performance.shifts if shift.metric == "brier_score")
        if not brier_shift.alert:
            raise RuntimeError("M3 must produce a significant adverse Brier shift")

        concept_summary = concept.evidence.select(
            pl.col("p0").mean().alias("mean_p0"),
            pl.col("p1").mean().alias("mean_p1"),
            (pl.col("p1") - pl.col("p0")).mean().alias("mean_probability_lift"),
            pl.col("vulnerability").mean().alias("mean_vulnerability"),
            (pl.col("y1") != pl.col("y0")).sum().alias("changed_outcomes"),
            pl.col("y0").mean().alias("y0_event_rate"),
            pl.col("y1").mean().alias("y1_event_rate"),
        ).to_dicts()[0]
        score_bands = default_rate_by_score_band(outcome_association.data).to_dicts()
        duration_seconds = perf_counter() - started
        summary = {
            "run_id": run_id,
            "model_version": model.model_version,
            "cohort_ids_disjoint": cohorts.get_column(ID_COLUMN).n_unique() == cohorts.height,
            "months": month_summaries,
            "outcome_state": outcome_association.state,
            "concept_drift": concept_summary,
            "performance": {
                "baseline": asdict(performance.baseline),
                "current": asdict(performance.current),
                "shifts": [asdict(shift) for shift in performance.shifts],
                "alert": performance.alert,
            },
            "default_rate_by_score_band": score_bands,
            "duration_seconds": duration_seconds,
        }
        record_event("outcomes", outcome_association.state, "M3 outcomes matured one-to-one")
        record_event("evaluation", "EVALUATED_WITH_ALERTS", "M3 concept drift evaluated")
        record_event("simulation", "COMPLETED", "Aggregate E5 evidence persisted")

        _json_write(run_output_dir / "resolved-config.json", resolved_config)
        _json_write(run_output_dir / "run-summary.json", summary)
        (run_output_dir / "events.jsonl").write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
            encoding="utf-8",
        )

        mlflow.log_params(
            {
                "seed": config.seed,
                "cohort_rows": config.cohort_rows,
                "stress_strength": config.stress_strength,
                "concept_delta": config.concept_delta,
                "psi_bins": config.psi_bins,
                "psi_smoothing": config.psi_smoothing,
                "bootstrap_samples": config.bootstrap_samples,
                "model_version": model.model_version,
                "dataset_sha256": dataset_sha256,
                "config_sha256": config_sha256,
                "cohort_selection": "weighted_exponential_race",
                "concept_mechanism": "sigmoid(logit(p0)+delta*g(x))",
            }
        )
        metrics: dict[str, float] = {
            "duration_seconds": duration_seconds,
            "concept_mean_p0": concept_summary["mean_p0"],
            "concept_mean_p1": concept_summary["mean_p1"],
            "concept_mean_probability_lift": concept_summary["mean_probability_lift"],
            "concept_changed_outcomes": concept_summary["changed_outcomes"],
            "baseline_brier_score": performance.baseline.brier_score,
            "current_brier_score": performance.current.brier_score,
            "brier_score_difference": brier_shift.difference,
            "performance_alert_count": sum(shift.alert for shift in performance.shifts),
        }
        for month, month_summary in month_summaries.items():
            metrics[f"{month.lower()}_rows"] = month_summary["rows"]
            metrics[f"{month.lower()}_drifted_feature_count"] = month_summary["evidently"][
                "drifted_count"
            ]
            metrics[f"{month.lower()}_psi_feature_warning_count"] = month_summary["psi"][
                "feature_warning_count"
            ]
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "pipeline_state": "EVALUATED_WITH_ALERTS",
                "m1_state": month_summaries["M1"]["processing_state"],
                "m2_state": month_summaries["M2"]["processing_state"],
                "m3_scoring_state": month_summaries["M3"]["processing_state"],
                "m3_outcome_state": outcome_association.state,
                "model_version": model.model_version,
            }
        )
        mlflow.log_artifacts(run_output_dir, artifact_path="drift-e5")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
