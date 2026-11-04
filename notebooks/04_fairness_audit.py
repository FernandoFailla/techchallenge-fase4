import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def e6_0():
    import pickle

    import marimo as mo
    import polars as pl
    from plotnine import aes, coord_flip, geom_col, ggplot, labs, theme_minimal

    from credit_monitoring.config import load_drift_config, load_fairness_config
    from credit_monitoring.contracts import ID_COLUMN, MODEL_FEATURE_COLUMNS
    from credit_monitoring.fairness import (
        AGE_GROUP_COLUMN,
        SEX_GROUP_COLUMN,
        audit_group_fairness,
        build_restricted_audit_view,
    )
    from credit_monitoring.outcomes import associate_mature_outcomes
    from credit_monitoring.simulation import (
        MONTH_COLUMN,
        build_drift_cohorts,
        simulate_concept_drift,
    )
    from credit_monitoring.training import predict_credit_risk

    return (
        AGE_GROUP_COLUMN,
        ID_COLUMN,
        MODEL_FEATURE_COLUMNS,
        MONTH_COLUMN,
        SEX_GROUP_COLUMN,
        aes,
        associate_mature_outcomes,
        audit_group_fairness,
        build_drift_cohorts,
        build_restricted_audit_view,
        coord_flip,
        geom_col,
        ggplot,
        labs,
        load_drift_config,
        load_fairness_config,
        mo,
        pickle,
        pl,
        predict_credit_risk,
        simulate_concept_drift,
        theme_minimal,
    )


@app.cell
def e6_1(mo):
    mo.md("""
    # E6 - Auditoria de equidade com Fairlearn

    Auditoria agregada do modelo congelado `real-logistic-v1` sobre desfechos M3
    maturados. Sexo e idade aparecem somente nesta visão restrita de auditoria:
    não entram nas features do modelo nem são persistidos em nível individual.
    """)
    return


@app.cell
def e6_2(load_drift_config, load_fairness_config, pickle, pl):
    fairness_config = load_fairness_config("configs/e6-fairness.yaml")
    drift_config = load_drift_config(fairness_config.e5_config_path)
    with drift_config.model_path.open("rb") as _model_file:
        credit_model = pickle.load(_model_file)
    source_data = pl.scan_parquet(drift_config.data_path).collect()
    simulation_pool = source_data.filter(pl.col("partition") == "simulation")
    return credit_model, drift_config, fairness_config, simulation_pool


@app.cell
def e6_3(
    ID_COLUMN,
    MODEL_FEATURE_COLUMNS,
    MONTH_COLUMN,
    associate_mature_outcomes,
    build_drift_cohorts,
    build_restricted_audit_view,
    credit_model,
    drift_config,
    fairness_config,
    pl,
    predict_credit_risk,
    simulate_concept_drift,
    simulation_pool,
):
    drift_cohorts = build_drift_cohorts(
        simulation_pool,
        seed=drift_config.seed,
        rows_per_month=drift_config.cohort_rows,
        stress_strength=drift_config.stress_strength,
    )
    m3_cohort = drift_cohorts.filter(pl.col(MONTH_COLUMN) == "M3")
    m3_scores = predict_credit_risk(
        credit_model,
        m3_cohort.select(ID_COLUMN, *MODEL_FEATURE_COLUMNS),
    )
    concept_result = simulate_concept_drift(
        credit_model,
        m3_cohort,
        delta=drift_config.concept_delta,
        seed=drift_config.seed + 101,
        available_at=drift_config.outcome_available_at,
    )
    matured_result = associate_mature_outcomes(
        m3_scores,
        concept_result.outcomes,
        as_of=drift_config.outcome_available_at,
    )
    restricted_audit = build_restricted_audit_view(
        m3_cohort,
        matured_result.data,
        minimum_age_rows=fairness_config.minimum_group_rows,
    )
    return (restricted_audit,)


@app.cell
def e6_4(
    AGE_GROUP_COLUMN,
    SEX_GROUP_COLUMN,
    audit_group_fairness,
    fairness_config,
    restricted_audit,
):
    sex_analysis = audit_group_fairness(
        restricted_audit,
        sensitive_feature=SEX_GROUP_COLUMN,
        minimum_rows=fairness_config.minimum_group_rows,
        minimum_events=fairness_config.minimum_group_events,
        minimum_non_events=fairness_config.minimum_group_non_events,
        bootstrap_samples=fairness_config.bootstrap_samples,
        seed=fairness_config.seed,
        disparity_review_threshold=fairness_config.disparity_review_threshold,
    )
    age_analysis = audit_group_fairness(
        restricted_audit,
        sensitive_feature=AGE_GROUP_COLUMN,
        minimum_rows=fairness_config.minimum_group_rows,
        minimum_events=fairness_config.minimum_group_events,
        minimum_non_events=fairness_config.minimum_group_non_events,
        bootstrap_samples=fairness_config.bootstrap_samples,
        seed=fairness_config.seed,
        disparity_review_threshold=fairness_config.disparity_review_threshold,
    )
    return age_analysis, sex_analysis


@app.cell
def e6_5(age_analysis, mo, pl, sex_analysis):
    audit_summary = pl.DataFrame(
        [
            {
                "attribute": "sexo",
                "evidence_sufficient": sex_analysis.summary.evidence_sufficient,
                "published_groups": sex_analysis.summary.published_group_count,
                "suppressed_groups": sex_analysis.summary.suppressed_group_count,
                "equalized_odds_difference": sex_analysis.summary.equalized_odds_difference,
                "ci_lower": sex_analysis.summary.equalized_odds_ci_lower,
                "ci_upper": sex_analysis.summary.equalized_odds_ci_upper,
                "investigation_required": sex_analysis.summary.investigation_required,
            },
            {
                "attribute": "idade",
                "evidence_sufficient": age_analysis.summary.evidence_sufficient,
                "published_groups": age_analysis.summary.published_group_count,
                "suppressed_groups": age_analysis.summary.suppressed_group_count,
                "equalized_odds_difference": age_analysis.summary.equalized_odds_difference,
                "ci_lower": age_analysis.summary.equalized_odds_ci_lower,
                "ci_upper": age_analysis.summary.equalized_odds_ci_upper,
                "investigation_required": age_analysis.summary.investigation_required,
            },
        ]
    )
    mo.vstack([mo.md("## Resumo da auditoria"), mo.ui.table(audit_summary)])
    return (audit_summary,)


@app.cell
def e6_6(age_analysis, mo, pl, sex_analysis):
    sex_group_metrics = sex_analysis.group_metrics.with_columns(pl.lit("sexo").alias("attribute"))
    age_group_metrics = age_analysis.group_metrics.with_columns(pl.lit("idade").alias("attribute"))
    published_group_metrics = pl.concat(
        (sex_group_metrics, age_group_metrics), how="diagonal_relaxed"
    ).filter(pl.col("publish_error_metrics"))
    mo.vstack(
        [
            mo.md("## Métricas publicáveis por grupo"),
            mo.md(
                "Cada linha cumpre o suporte mínimo de 200 observações, "
                "30 eventos e 30 não eventos."
            ),
            mo.ui.table(published_group_metrics),
        ]
    )
    return


@app.cell
def e6_7(age_analysis, mo, pl, sex_analysis):
    difference_metrics = pl.concat(
        (
            sex_analysis.metric_differences.with_columns(pl.lit("sexo").alias("attribute")),
            age_analysis.metric_differences.with_columns(pl.lit("idade").alias("attribute")),
        )
    )
    mo.vstack([mo.md("## Diferenças entre grupos e incerteza"), mo.ui.table(difference_metrics)])
    return


@app.cell
def e6_8(
    aes,
    audit_summary,
    coord_flip,
    geom_col,
    ggplot,
    labs,
    theme_minimal,
):
    eo_chart_data = audit_summary.select("attribute", "equalized_odds_difference")
    disparity_chart = (
        ggplot(
            eo_chart_data.to_pandas(),
            aes(x="attribute", y="equalized_odds_difference", fill="attribute"),
        )
        + geom_col(show_legend=False)
        + coord_flip()
        + labs(
            title="Diferença de equalized odds por atributo",
            x="Atributo auditado",
            y="Diferença absoluta",
        )
        + theme_minimal()
    )
    disparity_chart
    return


@app.cell
def e6_9(age_analysis, fairness_config, mo, sex_analysis):
    if not (sex_analysis.summary.evidence_sufficient and age_analysis.summary.evidence_sufficient):
        _interpretation = mo.md(
            "## Interpretação\n\n"
            "Avaliação bloqueada: faltam ao menos dois grupos publicáveis por atributo."
        )
    else:
        _sex_difference = sex_analysis.summary.equalized_odds_difference
        _sex_lower = sex_analysis.summary.equalized_odds_ci_lower
        _sex_upper = sex_analysis.summary.equalized_odds_ci_upper
        _age_difference = age_analysis.summary.equalized_odds_difference
        _age_lower = age_analysis.summary.equalized_odds_ci_lower
        _age_upper = age_analysis.summary.equalized_odds_ci_upper
        _review_threshold = fairness_config.disparity_review_threshold
        _interpretation = mo.md(
            f"""
            ## Interpretação

            - Sexo: diferença de equalized odds de **{_sex_difference:.3f}**,
              IC bootstrap **[{_sex_lower:.3f}, {_sex_upper:.3f}]**.
            - Idade: diferença de equalized odds de **{_age_difference:.3f}**,
              IC bootstrap **[{_age_lower:.3f}, {_age_upper:.3f}]**.
            - O limite inferior dos dois intervalos permanece abaixo do gatilho
              **{_review_threshold:.2f}**.

            Resultado: manter monitoramento. A evidência não autoriza mitigação, retreino ou
            limiar específico por grupo. Os rótulos são sintéticos; uma decisão real exige
            validação com desfechos temporais observados e revisão conjunta de Risco e Governança.
            """
        )
    _interpretation
    return


@app.cell
def e6_10(mo):
    mo.md("""
    ## Controles de governança

    - Atributos demográficos não integram `MODEL_FEATURE_COLUMNS`.
    - A visão restrita contém apenas chave, grupos derivados, score, decisão e alvo maturado.
    - O MLflow recebe somente métricas agregadas e artefatos sem linhas individuais.
    - Grupos sem suporte são suprimidos, nunca combinados com base no alvo.
    - O playbook permite investigação humana, mas proíbe mitigação e retreino automáticos.
    """)
    return


if __name__ == "__main__":
    app.run()
