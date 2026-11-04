"""Export compact, aggregate delivery evidence from completed local runs."""

import html
import json
from pathlib import Path

RUNTIME = Path("artifacts/runtime")
EVIDENCE = Path("artifacts/evidence")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_run(root: Path, filename: str) -> tuple[str, dict[str, object]]:
    pointer = root / "latest-run-id.txt"
    if pointer.exists():
        run_id = pointer.read_text(encoding="utf-8").strip()
        path = root / run_id / filename
    else:
        candidates = [path for path in root.glob(f"*/{filename}") if path.is_file()]
        if not candidates:
            raise FileNotFoundError(f"no completed {filename} under {root}")
        path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
        run_id = path.parent.name
    return run_id, _read_json(path)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _score_psi(month: dict[str, object]) -> float:
    results = month["psi"]["results"]
    return next(result["value"] for result in results if result["column"] == "default_probability")


def _format_metric(value: float | None) -> str:
    return "indisponível" if value is None else f"{value:.4f}"


def main() -> None:
    d4 = _read_json(RUNTIME / "d4-demo" / "run-summary.json")
    preparation_run_id, preparation = _latest_run(
        RUNTIME / "data-preparation-e4", "run-summary.json"
    )
    training_metrics = _read_json(RUNTIME / "training-e4" / "metrics.json")
    training_contract = _read_json(RUNTIME / "training-e4" / "contract-evidence.json")
    training_metadata = _read_json(RUNTIME / "training-e4" / "model-metadata.json")
    e5_run_id, e5 = _latest_run(RUNTIME / "drift-e5", "run-summary.json")
    e6_run_id, e6 = _latest_run(RUNTIME / "fairness-e6", "fairness-summary.json")

    preparation_evidence = {
        "run_id": preparation_run_id,
        "pipeline_state": preparation["pipeline_state"],
        "stages": preparation["stages"],
        "rows": preparation["rows"],
        "event_rate": preparation["event_rate"],
        "archive_sha256": preparation["archive_sha256"],
        "duration_seconds": preparation["duration_seconds"],
        "process_health": preparation["process_health"],
        "source_provenance": preparation["source_provenance"],
    }

    months = {
        month: {
            "rows": details["rows"],
            "processing_state": details["processing_state"],
            "evidently_drifted_features": details["evidently"]["drifted_features"],
            "evidently_drifted_count": details["evidently"]["drifted_count"],
            "psi_feature_warning_count": details["psi"]["feature_warning_count"],
            "score_psi": _score_psi(details),
        }
        for month, details in e5["months"].items()
    }
    brier_shift = next(
        shift for shift in e5["performance"]["shifts"] if shift["metric"] == "brier_score"
    )
    e5_evidence = {
        "run_id": e5_run_id,
        "model_version": e5["model_version"],
        "cohort_ids_disjoint": e5["cohort_ids_disjoint"],
        "months": months,
        "outcome_timeline": e5["outcome_timeline"],
        "process_health": e5["process_health"],
        "concept_drift": e5["concept_drift"],
        "brier_shift": brier_shift,
        "source_provenance": e5["source_provenance"],
    }
    e6_evidence = {
        "run_id": e6_run_id,
        "model_version": e6["model_version"],
        "rows": e6["rows"],
        "outcome_state": e6["outcome_state"],
        "fairness_state": e6["fairness_state"],
        "analyses": e6["analyses"],
        "playbook": e6["playbook"],
        "source_provenance": e6["source_provenance"],
        "process_health": e6["process_health"],
        "row_level_data_persisted": False,
    }
    monitoring_evidence = {
        "d4": d4,
        "data_preparation": preparation_evidence,
        "training": {
            "model": training_metadata,
            "metrics": training_metrics,
            "invalid_batch": training_contract,
        },
        "drift": e5_evidence,
        "fairness": e6_evidence,
    }

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    _write_json(EVIDENCE / "monitoring-summary.json", monitoring_evidence)
    _write_json(EVIDENCE / "data-preparation-summary.json", preparation_evidence)
    _write_json(EVIDENCE / "e5-drift-summary.json", e5_evidence)
    _write_json(EVIDENCE / "e6-fairness-summary.json", e6_evidence)

    month_rows = "".join(
        "<tr>"
        f"<td>{month}</td>"
        f"<td>{details['rows']}</td>"
        f"<td><code>{html.escape(details['processing_state'])}</code></td>"
        f"<td>{details['evidently_drifted_count']}</td>"
        f"<td>{details['psi_feature_warning_count']}</td>"
        f"<td>{details['score_psi']:.4f}</td>"
        "</tr>"
        for month, details in months.items()
    )
    violation_rows = "".join(
        "<li>"
        f"<code>{html.escape(violation['column'])}</code>: "
        f"{html.escape(violation['check'])}"
        "</li>"
        for violation in training_contract["violations"]
    )
    fairness_rows = "".join(
        "<tr>"
        f"<td>{html.escape(feature)}</td>"
        f"<td>{summary['published_group_count']}</td>"
        f"<td>{_format_metric(summary['equalized_odds_difference'])}</td>"
        f"<td>{_format_metric(summary['equalized_odds_ci_lower'])}; "
        f"{_format_metric(summary['equalized_odds_ci_upper'])}</td>"
        f"<td>{summary['investigation_required']}</td>"
        "</tr>"
        for feature, summary in e6["analyses"].items()
    )
    report = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Credit Monitoring - Evidência de Drift e Observabilidade</title>
  <style>
    body {{ font: 16px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 1100px;
      padding: 0 1rem; color: #17202a; }}
    h1, h2 {{ color: #173f5f; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #ccd6dd; padding: .55rem; text-align: left; }}
    th {{ background: #eaf2f8; }}
    code {{ background: #f2f4f4; padding: .1rem .25rem; }}
    .alert {{ border-left: 5px solid #c0392b; padding: .75rem; background: #fdf2f2; }}
    .ok {{ border-left: 5px solid #238636; padding: .75rem; background: #f0fff4; }}
  </style>
</head>
<body>
  <h1>Evidência agregada de drift e observabilidade</h1>
  <p>Relatório customizado gerado de runs locais concluídas.
    Não contém IDs ou linhas de clientes.</p>
  <h2>Ingestão e transformação</h2>
  <p class="ok">Run MLflow <code>{html.escape(preparation_run_id)}</code>;
    estado <code>{preparation["pipeline_state"]}</code>; {preparation["rows"]} linhas.
    Download, ingestão, transformação e persistência terminaram em
    <code>COMPLETED</code>.</p>
  <h2>Contrato de dados no pipeline real</h2>
  <p class="ok">Estado <code>{training_contract["state"]}</code>; linhas pontuadas:
    <strong>{training_contract["scored_rows"]}</strong>;
    falhas: {training_contract["failure_count"]}.</p>
  <ul>{violation_rows}</ul>
  <h2>Data Drift e saúde mensal</h2>
  <p>Run MLflow E5: <code>{html.escape(e5_run_id)}</code>. Evidently compara cada safra com
    a referência; PSI usa bins fixados na referência.</p>
  <table><thead><tr><th>Safra</th><th>Volume</th><th>Estado</th><th>Features Evidently</th>
    <th>Alertas PSI</th><th>PSI do score</th></tr></thead><tbody>{month_rows}</tbody></table>
  <p>Saúde do processo E5: {e5["process_health"]["cpu_count"]} CPUs disponíveis;
    pico RSS {e5["process_health"]["max_rss_mb"]:.1f} MB;
    Python {html.escape(e5["process_health"]["python_version"])}.</p>
  <h2>Desfecho atrasado e Concept Drift</h2>
  <p>Na emissão do score: <code>{e5["outcome_timeline"]["state_at_scoring"]}</code>.
    Na disponibilidade: <code>{e5["outcome_timeline"]["state_at_availability"]}</code>, após
    {e5["outcome_timeline"]["delay_days"]} dias.</p>
  <p class="alert">Brier: {e5["performance"]["baseline"]["brier_score"]:.4f} →
    {e5["performance"]["current"]["brier_score"]:.4f}; diferença
    {brier_shift["difference"]:.4f}, IC [{brier_shift["confidence_lower"]:.4f};
    {brier_shift["confidence_upper"]:.4f}].</p>
  <h2>Fairness agregada</h2>
  <p>Run MLflow E6: <code>{html.escape(e6_run_id)}</code>; estado
    <code>{e6["fairness_state"]}</code>.</p>
  <table><thead><tr><th>Atributo</th><th>Grupos</th><th>Equalized odds</th>
    <th>IC 95%</th><th>Investigação</th></tr></thead><tbody>{fairness_rows}</tbody></table>
  <h2>Limites</h2>
  <p>As safras e os rótulos são sintéticos. O relatório demonstra mecanismos e controles
    programados; não estima degradação econômica real, discriminação causal
    ou conformidade legal.</p>
</body></html>
"""
    (EVIDENCE / "drift-observability-report.html").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
