import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # E5: safras OOT, Data Drift e Concept Drift

    Investigação reproduzível sobre três lotes disjuntos gerados somente após o
    congelamento do modelo `real-logistic-v1`. O notebook explora hipóteses e
    importa a lógica durável de `src/`; a execução oficial e os artefatos ficam
    sob responsabilidade de `scripts/run_e5_drift.py`.
    """)
    return


@app.cell(hide_code=True)
def _():
    import pickle

    import polars as pl

    from credit_monitoring.config import load_drift_config
    from credit_monitoring.contracts import ID_COLUMN, MODEL_FEATURE_COLUMNS, TARGET_COLUMN
    from credit_monitoring.monitoring import (
        analyze_model_data_drift,
        analyze_population_stability,
        compare_paired_performance,
    )
    from credit_monitoring.outcomes import OutcomeState, associate_mature_outcomes
    from credit_monitoring.simulation import (
        MONTH_COLUMN,
        build_drift_cohorts,
        simulate_concept_drift,
    )
    from credit_monitoring.training import predict_credit_risk

    return (
        ID_COLUMN,
        MODEL_FEATURE_COLUMNS,
        MONTH_COLUMN,
        OutcomeState,
        TARGET_COLUMN,
        analyze_model_data_drift,
        analyze_population_stability,
        associate_mature_outcomes,
        build_drift_cohorts,
        compare_paired_performance,
        load_drift_config,
        pickle,
        pl,
        predict_credit_risk,
        simulate_concept_drift,
    )


@app.cell(hide_code=True)
def _(ID_COLUMN, MODEL_FEATURE_COLUMNS, load_drift_config, pickle, pl):
    config = load_drift_config("configs/e5-drift.yaml")
    dataset = pl.scan_parquet(config.data_path).collect()
    reference = dataset.filter(pl.col("partition") == "test").select(
        ID_COLUMN, *MODEL_FEATURE_COLUMNS
    )
    simulation_pool = dataset.filter(pl.col("partition") == "simulation")
    with config.model_path.open("rb") as _model_file:
        model = pickle.load(_model_file)

    {
        "model_version": model.model_version,
        "reference_rows": reference.height,
        "simulation_pool_rows": simulation_pool.height,
        "cohort_rows": config.cohort_rows,
    }
    return config, model, reference, simulation_pool


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hipóteses

    - **M1 saudável:** seleção aleatória do complemento deve permanecer próxima da referência.
    - **M2 Data Drift:** uma corrida exponencial ponderada deve aumentar a participação de
      baixo limite, atraso e utilização sem editar nenhum valor individual.
    - **M3 Concept Drift:** as entradas devem permanecer estáveis; após a maturação,
      `p1(X) = sigmoid(logit(p0(X)) + delta * g(X))` deve piorar a calibração do score
      congelado. `Y0` e `Y1` usam o mesmo sorteio por linha.
    """)
    return


@app.cell(hide_code=True)
def _(MONTH_COLUMN, build_drift_cohorts, config, mo, pl, simulation_pool):
    cohorts = build_drift_cohorts(
        simulation_pool,
        seed=config.seed,
        rows_per_month=config.cohort_rows,
        stress_strength=config.stress_strength,
    )
    cohort_profile = (
        cohorts.group_by(MONTH_COLUMN)
        .agg(
            pl.len().alias("rows"),
            pl.col("LIMIT_BAL").mean().round(1).alias("mean_limit_balance"),
            pl.col("MAX_DELAY_6M").mean().round(3).alias("mean_max_delay"),
            pl.col("UTILIZATION_CURRENT").mean().round(3).alias("mean_utilization"),
        )
        .sort(MONTH_COLUMN)
    )
    mo.vstack([mo.md("## Evidência de composição"), cohort_profile])
    return (cohorts,)


@app.cell(hide_code=True)
def _(
    ID_COLUMN,
    MODEL_FEATURE_COLUMNS,
    MONTH_COLUMN,
    analyze_model_data_drift,
    analyze_population_stability,
    cohorts,
    config,
    model,
    pl,
    predict_credit_risk,
    reference,
):
    reference_scores = predict_credit_risk(model, reference)
    reference_monitoring = reference.join(reference_scores, on=ID_COLUMN, validate="1:1")
    _monitoring_rows = []
    for _month in ("M1", "M2", "M3"):
        _current = cohorts.filter(pl.col(MONTH_COLUMN) == _month).select(
            ID_COLUMN, *MODEL_FEATURE_COLUMNS
        )
        _current_scores = predict_credit_risk(model, _current)
        _current_monitoring = _current.join(_current_scores, on=ID_COLUMN, validate="1:1")
        _drift = analyze_model_data_drift(reference, _current)
        _stability = analyze_population_stability(
            reference_monitoring,
            _current_monitoring,
            bins=config.psi_bins,
            smoothing=config.psi_smoothing,
            minimum_rows=config.psi_minimum_rows,
        )
        _monitoring_rows.append(
            {
                "month": _month,
                "evidently_drifted_features": _drift.summary.drifted_count,
                "features": ", ".join(_drift.summary.drifted_features) or "none",
                "psi_feature_warnings": _stability.feature_warning_count,
                "score_psi": next(
                    _result.value
                    for _result in _stability.results
                    if _result.column == "default_probability"
                ),
                "dataset_alert": _drift.summary.alert or _stability.alert,
            }
        )
    monitoring_summary = pl.DataFrame(_monitoring_rows)
    monitoring_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resposta sobre Data Drift

    M1 e M3 não acionam alerta de distribuição. M2 aciona o estado
    `PROCESSED_WITH_ALERTS`: a seleção altera `P(X)` em múltiplas features e quatro
    delas têm PSI acima de `0,10` (uma relevante e três em nível de warning).
    Como as linhas são apenas selecionadas do pool OOT, essa evidência não depende
    de valores artificiais editados.
    """)
    return


@app.cell(hide_code=True)
def _(
    ID_COLUMN,
    MODEL_FEATURE_COLUMNS,
    MONTH_COLUMN,
    OutcomeState,
    TARGET_COLUMN,
    associate_mature_outcomes,
    cohorts,
    compare_paired_performance,
    config,
    mo,
    model,
    pl,
    predict_credit_risk,
    simulate_concept_drift,
):
    m3 = cohorts.filter(pl.col(MONTH_COLUMN) == "M3")
    m3_scores = predict_credit_risk(model, m3.select(ID_COLUMN, *MODEL_FEATURE_COLUMNS))
    concept = simulate_concept_drift(
        model,
        m3,
        delta=config.concept_delta,
        seed=config.seed + 101,
        available_at=config.outcome_available_at,
    )
    outcome_association = associate_mature_outcomes(
        m3_scores,
        concept.outcomes,
        as_of=config.outcome_available_at,
    )
    assert outcome_association.state is OutcomeState.MATURED
    baseline_outcomes = m3_scores.join(
        concept.evidence.select(ID_COLUMN, pl.col("y0").alias(TARGET_COLUMN)),
        on=ID_COLUMN,
        validate="1:1",
    )
    performance_comparison = compare_paired_performance(
        baseline_outcomes,
        outcome_association.data,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed + 202,
    )
    concept_summary = concept.evidence.select(
        pl.col("p0").mean().alias("mean_p0"),
        pl.col("p1").mean().alias("mean_p1"),
        (pl.col("p1") - pl.col("p0")).mean().alias("mean_probability_lift"),
        (pl.col("y1") != pl.col("y0")).sum().alias("changed_outcomes"),
        pl.col("y0").mean().alias("y0_event_rate"),
        pl.col("y1").mean().alias("y1_event_rate"),
    )
    performance_shifts = pl.DataFrame(
        {
            "metric": [_shift.metric for _shift in performance_comparison.shifts],
            "baseline": [_shift.baseline for _shift in performance_comparison.shifts],
            "current": [_shift.current for _shift in performance_comparison.shifts],
            "difference": [_shift.difference for _shift in performance_comparison.shifts],
            "ci_lower": [_shift.confidence_lower for _shift in performance_comparison.shifts],
            "ci_upper": [_shift.confidence_upper for _shift in performance_comparison.shifts],
            "alert": [_shift.alert for _shift in performance_comparison.shifts],
        }
    )
    mo.vstack([mo.md("## Evidência de Concept Drift"), concept_summary, performance_shifts])
    return (outcome_association,)


@app.cell(hide_code=True)
def _(config, mo, outcome_association):
    mo.md(
        f"""
        ## Decisão

        A configuração versionada é adequada para a demonstração E5:

        - as três safras têm `{config.cohort_rows}` linhas e IDs disjuntos;
        - `stress_strength={config.stress_strength}` isola Data Drift em M2;
        - `concept_delta={config.concept_delta}` aumenta a probabilidade média em M3;
        - a associação de desfechos termina em `{outcome_association.state}`;
        - o bootstrap pareado sinaliza degradação de Brier, enquanto mudanças de AUC/KS
          são tratadas como consequências e não como definição de Concept Drift.

        Limite causal: o contraste recupera somente o efeito programado no simulador;
        não identifica efeito econômico real.
        """
    )
    return


if __name__ == "__main__":
    app.run()
