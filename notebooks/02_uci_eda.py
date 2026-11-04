import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def title():
    import marimo as mo

    mo.md("""# EDA E4: dados reais e baseline sem leakage

    Investigacao reproduzivel com analises supervisionadas restritas ao treino.""")
    return (mo,)


@app.cell
def imports():
    import polars as pl
    from plotnine import aes, geom_line, geom_point, ggplot, labs, theme_minimal

    from credit_monitoring.config import load_training_config
    from credit_monitoring.contracts import ID_COLUMN, TARGET_COLUMN
    from credit_monitoring.data import (
        BILL_COLUMNS,
        MODEL_FEATURE_COLUMNS,
        PAYMENT_COLUMNS,
        build_modeling_dataset,
        build_partition_manifest,
        load_uci_credit_data,
    )

    return (
        BILL_COLUMNS,
        ID_COLUMN,
        MODEL_FEATURE_COLUMNS,
        PAYMENT_COLUMNS,
        TARGET_COLUMN,
        aes,
        build_modeling_dataset,
        build_partition_manifest,
        geom_line,
        geom_point,
        ggplot,
        labs,
        load_training_config,
        load_uci_credit_data,
        pl,
        theme_minimal,
    )


@app.cell(hide_code=True)
def questions(mo):
    mo.md("""
    ## Perguntas

    1. O arquivo oficial tem o volume, alvo e qualidade declarados pela UCI?
    2. Quais anomalias de codificacao e extremos exigem tratamento deterministico?
    3. As particoes 60/15/15/10 preservam a prevalencia sem reutilizar IDs?
    4. Um conjunto pequeno de features comportamentais sustenta o baseline sem usar
       sexo, idade, educacao ou estado civil como entradas do modelo?
    """)
    return


@app.cell(hide_code=True)
def load_data(
    ID_COLUMN,
    MODEL_FEATURE_COLUMNS,
    build_modeling_dataset,
    load_training_config,
    load_uci_credit_data,
    pl,
):
    config = load_training_config("configs/training.yaml")
    workbook_path = config.data_dir / "raw" / "default of credit card clients.xls"
    processed_path = config.data_dir / "processed" / "credit-modeling.parquet"
    raw_data = load_uci_credit_data(workbook_path)
    partitioned_data = pl.scan_parquet(processed_path).collect()
    train_ids = partitioned_data.filter(pl.col("partition") == "train").select(ID_COLUMN)
    train_raw_data = raw_data.join(train_ids, on=ID_COLUMN, how="inner", validate="1:1")
    train_modeling_data = build_modeling_dataset(train_raw_data)

    {
        "raw_shape": raw_data.shape,
        "train_modeling_shape": train_modeling_data.shape,
        "model_features": MODEL_FEATURE_COLUMNS,
    }
    return partitioned_data, raw_data, train_modeling_data, train_raw_data


@app.cell(hide_code=True)
def quality_evidence(ID_COLUMN, TARGET_COLUMN, mo, pl, raw_data, train_raw_data):
    quality_summary = raw_data.select(
        pl.len().alias("rows"),
        pl.col(ID_COLUMN).n_unique().alias("unique_ids"),
        pl.sum_horizontal(pl.all().null_count()).alias("null_values"),
        pl.col("LIMIT_BAL").min().alias("limit_min"),
        pl.col("LIMIT_BAL").max().alias("limit_max"),
        pl.col("AGE").min().alias("age_min"),
        pl.col("AGE").max().alias("age_max"),
    ).with_columns(
        pl.lit(train_raw_data.get_column(TARGET_COLUMN).mean()).alias("train_event_rate")
    )

    mo.vstack([mo.md("## Evidencia de qualidade"), quality_summary])
    return (quality_summary,)


@app.cell(hide_code=True)
def anomaly_evidence(BILL_COLUMNS, PAYMENT_COLUMNS, mo, pl, train_raw_data):
    category_anomalies = pl.concat(
        [
            train_raw_data.group_by("EDUCATION")
            .len()
            .rename({"EDUCATION": "code"})
            .with_columns(pl.lit("EDUCATION").alias("field")),
            train_raw_data.group_by("MARRIAGE")
            .len()
            .rename({"MARRIAGE": "code"})
            .with_columns(pl.lit("MARRIAGE").alias("field")),
        ]
    ).sort("field", "code")
    ratio_evidence = (
        train_raw_data.lazy()
        .with_columns(
            pl.sum_horizontal(*PAYMENT_COLUMNS).alias("total_payment"),
            pl.sum_horizontal(
                *(pl.col(column).clip(lower_bound=0) for column in BILL_COLUMNS)
            ).alias("positive_bill"),
        )
        .with_columns(
            pl.when(pl.col("positive_bill") > 0)
            .then(pl.col("total_payment") / pl.col("positive_bill"))
            .otherwise(0.0)
            .alias("payment_ratio")
        )
        .select(
            pl.any_horizontal(*(pl.col(column) < 0 for column in BILL_COLUMNS))
            .sum()
            .alias("rows_with_negative_bill"),
            (pl.col("positive_bill") <= 0).sum().alias("rows_without_positive_bill"),
            (pl.col("payment_ratio") > 5).sum().alias("payment_ratio_above_5"),
            pl.col("payment_ratio").quantile(0.995).alias("payment_ratio_q995"),
            pl.col("payment_ratio").max().alias("payment_ratio_max"),
        )
        .collect()
    )

    mo.vstack(
        [
            mo.md("## Evidencia para tratamento"),
            category_anomalies,
            ratio_evidence,
        ]
    )
    return (ratio_evidence,)


@app.cell(hide_code=True)
def curation_decisions(mo, quality_summary, ratio_evidence):
    mo.md(
        f"""
        ## Resposta e decisoes de curadoria

        - O alvo no treino tem prevalencia de
          **{quality_summary.item(0, "train_event_rate"):.2%}** e nao ha nulos.
        - Codigos residuais de `EDUCATION` (0, 5, 6) sao consolidados em 4, e o
          codigo 0 de `MARRIAGE` e consolidado em 3.
        - Saldos negativos sao preservados como credito/estorno; utilizacoes sao
          limitadas ao intervalo [-1, 2].
        - A razao pagamento/fatura e limitada em 5: apenas
          **{ratio_evidence.item(0, "payment_ratio_above_5")}** linhas excedem o
          corte, enquanto o maximo bruto chega a
          **{ratio_evidence.item(0, "payment_ratio_max"):.0f}**.
        - Decisoes supervisionadas e de tratamento usam somente o treino; teste e
          simulacao permanecem fechados. Sexo, idade, educacao e estado civil ficam na visao de
          auditoria e nao entram em `MODEL_FEATURE_COLUMNS`.
        """
    )
    return


@app.cell(hide_code=True)
def split_evidence(ID_COLUMN, build_partition_manifest, mo, partitioned_data):
    split_manifest = build_partition_manifest(partitioned_data)
    max_event_rate_gap = (
        split_manifest.get_column("event_rate").max()
        - split_manifest.get_column("event_rate").min()
    )

    mo.vstack(
        [
            mo.md(
                f"""## Evidencia das particoes

    Maior diferenca de prevalencia: **{max_event_rate_gap:.4%}**.
    IDs unicos: **{partitioned_data.get_column(ID_COLUMN).n_unique() == partitioned_data.height}**.
    """
            ),
            split_manifest,
        ]
    )
    return


@app.cell(hide_code=True)
def delay_plot(
    TARGET_COLUMN,
    aes,
    geom_line,
    geom_point,
    ggplot,
    labs,
    train_modeling_data,
    pl,
    theme_minimal,
):
    delay_risk = (
        train_modeling_data.lazy()
        .group_by("MAX_DELAY_6M")
        .agg(
            pl.len().alias("rows"),
            pl.col(TARGET_COLUMN).mean().alias("event_rate"),
        )
        .sort("MAX_DELAY_6M")
        .collect()
    )
    risk_plot = (
        ggplot(delay_risk.to_pandas(), aes("MAX_DELAY_6M", "event_rate"))
        + geom_line(color="#264653")
        + geom_point(color="#e76f51", size=2.5)
        + labs(
            title="Inadimplencia cresce com o maior atraso em seis meses",
            x="Maior atraso em seis meses",
            y="Taxa de inadimplencia",
        )
        + theme_minimal()
    )
    risk_plot
    return


@app.cell(hide_code=True)
def conclusion(mo):
    mo.md("""
    ## Conclusao

    A fonte oficial e adequada ao baseline: 30 mil clientes, alvo binario e
    prevalencia suficiente no treino. O conjunto comportamental curado
    e as particoes estratificadas podem seguir para treino. A entrega usa a Pipeline
    logistica oficial de fallback; WOE e tabela de pontos continuam como melhoria
    posterior, sem afirmar que um scorecard esta ativo.
    """)
    return


if __name__ == "__main__":
    app.run()
