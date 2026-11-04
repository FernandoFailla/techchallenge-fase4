"""Run the restricted E6 fairness audit and persist aggregate evidence."""

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import mlflow
import polars as pl

from credit_monitoring.config import load_drift_config, load_fairness_config
from credit_monitoring.contracts import ID_COLUMN, MODEL_FEATURE_COLUMNS
from credit_monitoring.fairness import (
    AGE_GROUP_COLUMN,
    SEX_GROUP_COLUMN,
    audit_group_fairness,
    build_restricted_audit_view,
    resolve_fairness_state,
)
from credit_monitoring.health import collect_process_health
from credit_monitoring.outcomes import OutcomeState, associate_mature_outcomes
from credit_monitoring.provenance import collect_source_provenance, file_sha256
from credit_monitoring.simulation import (
    MONTH_COLUMN,
    build_drift_cohorts,
    simulate_concept_drift,
)
from credit_monitoring.training import CreditModel, predict_credit_risk


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _serialize_config(config: object) -> dict[str, object]:
    return {
        key: value.isoformat()
        if isinstance(value, datetime)
        else str(value)
        if isinstance(value, Path)
        else value
        for key, value in asdict(config).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e6-fairness.yaml")
    args = parser.parse_args()

    config = load_fairness_config(args.config)
    drift_config = load_drift_config(config.e5_config_path)
    if not drift_config.data_path.exists():
        raise FileNotFoundError(f"prepared dataset not found: {drift_config.data_path}")
    if not drift_config.model_path.exists():
        raise FileNotFoundError(f"trained model not found: {drift_config.model_path}")
    dataset_sha256 = file_sha256(drift_config.data_path)
    model_sha256 = file_sha256(drift_config.model_path)
    with drift_config.model_path.open("rb") as model_file:
        model = pickle.load(model_file)
    if not isinstance(model, CreditModel):
        raise TypeError("model artifact does not contain a CreditModel")
    if model.model_version != drift_config.expected_model_version:
        raise ValueError(
            f"expected model {drift_config.expected_model_version}, got {model.model_version}"
        )

    started = perf_counter()
    data = pl.scan_parquet(drift_config.data_path).collect()
    pool = data.filter(pl.col("partition") == "simulation")
    cohorts = build_drift_cohorts(
        pool,
        seed=drift_config.seed,
        rows_per_month=drift_config.cohort_rows,
        stress_strength=drift_config.stress_strength,
    )
    m3 = cohorts.filter(pl.col(MONTH_COLUMN) == "M3")
    scores = predict_credit_risk(model, m3.select(ID_COLUMN, *MODEL_FEATURE_COLUMNS))
    concept = simulate_concept_drift(
        model,
        m3,
        delta=drift_config.concept_delta,
        seed=drift_config.seed + 101,
        available_at=drift_config.outcome_available_at,
    )
    premature_association = associate_mature_outcomes(
        scores,
        concept.outcomes,
        as_of=drift_config.scoring_at,
    )
    if premature_association.state is not OutcomeState.EVALUATION_BLOCKED:
        raise RuntimeError("fairness outcomes must remain blocked at the scoring time")
    outcome_association = associate_mature_outcomes(
        scores,
        concept.outcomes,
        as_of=drift_config.outcome_available_at,
    )
    if outcome_association.state is not OutcomeState.MATURED:
        raise RuntimeError(
            f"fairness evaluation requires mature outcomes: {outcome_association.reason}"
        )
    restricted = build_restricted_audit_view(
        m3,
        outcome_association.data,
        minimum_age_rows=config.minimum_group_rows,
    )
    demographic_columns = {"SEX", "AGE", "AGE_GROUP", "EDUCATION", "MARRIAGE"}
    demographics_excluded = not demographic_columns.intersection(model.feature_columns)
    if not demographics_excluded:
        raise RuntimeError("demographic columns must not participate in model scoring")
    common_threshold = restricted.select(
        (pl.col("review_required") == (pl.col("default_probability") >= model.threshold)).all()
    ).item()
    if not common_threshold:
        raise RuntimeError("group decisions must use the frozen common model threshold")
    analyses = {
        feature: audit_group_fairness(
            restricted,
            sensitive_feature=feature,
            minimum_rows=config.minimum_group_rows,
            minimum_events=config.minimum_group_events,
            minimum_non_events=config.minimum_group_non_events,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed,
            disparity_review_threshold=config.disparity_review_threshold,
        )
        for feature in (SEX_GROUP_COLUMN, AGE_GROUP_COLUMN)
    }
    investigation_required = any(
        analysis.summary.investigation_required for analysis in analyses.values()
    )
    evidence_sufficient = all(
        analysis.summary.evidence_sufficient for analysis in analyses.values()
    )
    fairness_state = resolve_fairness_state(
        investigation_required=investigation_required,
        evidence_sufficient=evidence_sufficient,
    )

    resolved_config = {
        "fairness": _serialize_config(config),
        "drift": _serialize_config(drift_config),
        "model_version": model.model_version,
        "model_sha256": model_sha256,
        "dataset_sha256": dataset_sha256,
        "outcome_mechanism": "synthetic_p1",
    }
    config_payload = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"))
    config_sha256 = hashlib.sha256(config_payload.encode()).hexdigest()
    source = collect_source_provenance()
    artifact_store = config.runtime_dir / "mlflow-artifacts"
    output_root = config.runtime_dir / "fairness-e6"
    artifact_store.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

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

    with mlflow.start_run(experiment_id=experiment_id, run_name="m3-restricted-audit") as run:
        run_id = run.info.run_id
        output_dir = output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        analysis_summary = {
            feature: asdict(analysis.summary) for feature, analysis in analyses.items()
        }
        playbook = {
            "state": fairness_state,
            "assessment_scope": "group error-rate comparison on synthetic matured M3 outcomes",
            "checks": {
                "demographics_excluded_from_model": demographics_excluded,
                "common_threshold_across_groups": common_threshold,
                "no_groups_suppressed": all(
                    analysis.summary.suppressed_group_count == 0 for analysis in analyses.values()
                ),
            },
            "label_source": "simulate_concept_drift synthetic_p1",
            "recommendation": (
                "investigate supported disparities and collect labels for blocked comparisons"
                if investigation_required and not evidence_sufficient
                else (
                    "risk and governance review before any model, feature, or threshold change"
                    if investigation_required
                    else (
                        "collect additional labeled observations before group comparison"
                        if not evidence_sufficient
                        else (
                            "retain monitoring; collect real temporal outcomes "
                            "before policy changes"
                        )
                    )
                )
            ),
            "automatic_mitigation": False,
            "automatic_retraining": False,
        }
        process_health = collect_process_health()
        summary = {
            "run_id": run_id,
            "model_version": model.model_version,
            "rows": restricted.height,
            "outcome_state": outcome_association.state,
            "outcome_timeline": {
                "scoring_at": drift_config.scoring_at.isoformat(),
                "available_at": drift_config.outcome_available_at.isoformat(),
                "state_at_scoring": premature_association.state,
                "state_at_availability": outcome_association.state,
            },
            "fairness_state": fairness_state,
            "source_provenance": {
                "revision": source.revision,
                "tree_sha256": source.tree_sha256,
                "worktree_dirty": source.worktree_dirty,
            },
            "process_health": asdict(process_health),
            "analyses": analysis_summary,
            "playbook": playbook,
            "duration_seconds": perf_counter() - started,
        }
        _write_json(output_dir / "resolved-config.json", resolved_config)
        _write_json(output_dir / "fairness-summary.json", summary)
        _write_json(output_dir / "investigation-playbook.json", playbook)
        for feature, analysis in analyses.items():
            prefix = feature.lower()
            analysis.group_metrics.write_csv(output_dir / f"{prefix}-group-metrics.csv")
            analysis.metric_differences.write_csv(output_dir / f"{prefix}-differences.csv")

        mlflow.log_params(
            {
                "model_version": model.model_version,
                "seed": config.seed,
                "minimum_group_rows": config.minimum_group_rows,
                "minimum_group_events": config.minimum_group_events,
                "minimum_group_non_events": config.minimum_group_non_events,
                "bootstrap_samples": config.bootstrap_samples,
                "disparity_review_threshold": config.disparity_review_threshold,
                "dataset_sha256": dataset_sha256,
                "model_sha256": model_sha256,
                "config_sha256": config_sha256,
                "source_revision": source.revision,
                "source_tree_sha256": source.tree_sha256,
                "source_worktree_dirty": source.worktree_dirty,
                "python_version": process_health.python_version,
                "platform": process_health.platform,
                "audit_features": f"{SEX_GROUP_COLUMN},{AGE_GROUP_COLUMN}",
            }
        )
        metrics = {
            "rows": restricted.height,
            "investigation_required": int(investigation_required),
            "process_cpu_count": process_health.cpu_count,
            "process_max_rss_mb": process_health.max_rss_mb,
        }
        for feature, analysis in analyses.items():
            prefix = feature.lower()
            metrics[f"{prefix}_published_groups"] = analysis.summary.published_group_count
            metrics[f"{prefix}_suppressed_groups"] = analysis.summary.suppressed_group_count
            for metric_name, value in (
                ("equalized_odds_difference", analysis.summary.equalized_odds_difference),
                ("equalized_odds_ci_lower", analysis.summary.equalized_odds_ci_lower),
                ("equalized_odds_ci_upper", analysis.summary.equalized_odds_ci_upper),
            ):
                if value is not None:
                    metrics[f"{prefix}_{metric_name}"] = value
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "pipeline_state": (
                    "FAIRNESS_EVALUATED"
                    if evidence_sufficient
                    else "EVALUATION_BLOCKED_INSUFFICIENT_DATA"
                ),
                "fairness_state": fairness_state,
                "outcome_state": outcome_association.state,
                "model_version": model.model_version,
                "row_level_data_persisted": False,
                "source_worktree_dirty": source.worktree_dirty,
            }
        )
        mlflow.log_artifacts(output_dir, artifact_path="fairness-e6")

    (output_root / "latest-run-id.txt").write_text(f"{run_id}\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
