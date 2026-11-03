import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def title():
    import marimo as mo

    mo.md("""# EDA D4: drift sintetico

    Analise exploratoria executavel da fatia vertical de monitoramento.""")
    return (mo,)


@app.cell
def imports():
    import polars as pl
    from plotnine import aes, facet_wrap, geom_density, ggplot, labs, theme_minimal

    from credit_monitoring.config import load_demo_config
    from credit_monitoring.contracts import FEATURE_COLUMNS
    from credit_monitoring.simulation import (
        generate_historical_data,
        scoring_view,
        simulate_data_drift,
    )

    return (
        FEATURE_COLUMNS,
        aes,
        facet_wrap,
        generate_historical_data,
        geom_density,
        ggplot,
        labs,
        load_demo_config,
        pl,
        scoring_view,
        simulate_data_drift,
        theme_minimal,
    )


@app.cell(hide_code=True)
def hypothesis(mo):
    mo.md("""
    ## Hipotese

    A simulacao deve introduzir drift somente em `LIMIT_BAL` e `AGE`.
    As demais features e os identificadores devem permanecer identicos, isolando
    o sinal que a etapa de observabilidade precisa detectar.
    """)
    return


@app.cell(hide_code=True)
def load_data(
    generate_historical_data,
    load_demo_config,
    scoring_view,
    simulate_data_drift,
):
    config = load_demo_config("configs/d4-demo.yaml")
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

    {
        "seed": config.seed,
        "rows_per_dataset": reference.height,
        "limit_balance_multiplier": config.limit_balance_multiplier,
        "age_shift": config.age_shift,
    }
    return current, reference


@app.cell(hide_code=True)
def evidence(FEATURE_COLUMNS, current, mo, pl, reference):
    combined = pl.concat(
        [
            reference.with_columns(pl.lit("reference").alias("dataset")),
            current.with_columns(pl.lit("current").alias("dataset")),
        ]
    )
    changed_features = [
        feature
        for feature in FEATURE_COLUMNS
        if not reference.get_column(feature).equals(current.get_column(feature))
    ]
    feature_evidence = pl.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "changed": [feature in changed_features for feature in FEATURE_COLUMNS],
        }
    )
    mean_summary = (
        combined.group_by("dataset")
        .agg(*(pl.col(feature).mean().round(3).alias(feature) for feature in FEATURE_COLUMNS))
        .sort("dataset")
    )

    mo.vstack(
        [
            mo.md("## Evidencia tabular"),
            mean_summary,
            feature_evidence,
        ]
    )
    return changed_features, combined, mean_summary


@app.cell(hide_code=True)
def plot_distributions(
    aes,
    combined,
    facet_wrap,
    geom_density,
    ggplot,
    labs,
    theme_minimal,
):
    distribution_data = combined.unpivot(
        index=["ID", "dataset"],
        on=["LIMIT_BAL", "AGE"],
        variable_name="feature",
        value_name="value",
    )
    drift_plot = (
        ggplot(
            distribution_data.to_pandas(),
            aes("value", color="dataset", fill="dataset"),
        )
        + geom_density(alpha=0.2)
        + facet_wrap("~feature", scales="free", ncol=2)
        + labs(
            title="Distribuicoes antes e depois do drift",
            x="Valor",
            y="Densidade",
        )
        + theme_minimal()
    )
    drift_plot
    return


@app.cell(hide_code=True)
def conclusion(changed_features, mean_summary, mo, pl):
    reference_means = mean_summary.filter(pl.col("dataset") == "reference")
    current_means = mean_summary.filter(pl.col("dataset") == "current")

    mo.md(
        f"""
        ## Resposta e decisao

        A hipotese foi confirmada: **{", ".join(changed_features)}** sao as unicas
        features alteradas. A media de `LIMIT_BAL` caiu de
        **{reference_means.item(0, "LIMIT_BAL"):,.1f}** para
        **{current_means.item(0, "LIMIT_BAL"):,.1f}**, enquanto a media de `AGE`
        subiu de **{reference_means.item(0, "AGE"):.1f}** para
        **{current_means.item(0, "AGE"):.1f}**.

        A configuracao sintetica e adequada para a entrega D4: ela fornece um caso
        deterministico e interpretavel para validar o relatorio Evidently, sem
        contaminar as demais features. A persistencia dos artefatos permanece fora
        deste notebook e sera feita pelo executor centralizado em `scripts/`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
