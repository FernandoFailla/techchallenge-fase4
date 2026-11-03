# ADR 0005: Usar scorecard WOE com regressão logística

- Status: aceito
- Data: 2026-10-23

## Contexto

Um classificador genérico atenderia ao enunciado, mas um scorecard aproxima o exercício das práticas tradicionais de risco de crédito. Uma implementação artesanal de discretização, WOE e conversão em pontos aumentaria o volume de código estatístico e o risco de vazamento ou inconsistência.

## Decisão

O modelo baseline será um scorecard implementado com:

- `optbinning.BinningProcess` para discretização supervisionada e transformação WOE;
- `sklearn.linear_model.LogisticRegression` como estimador;
- `optbinning.scorecard.Scorecard` para probabilidade, score e tabela auditável de pontos.

O binning, a seleção de variáveis e a regressão serão ajustados exclusivamente no conjunto de treino. Validação, teste/referência de monitoramento e safras sintéticas somente aplicarão os artefatos já ajustados.

O evento será `default=1`. A implementação terá teste explícito de direção: maior probabilidade de inadimplência deve produzir menor score. O limiar de `review_required` será escolhido na validação maximizando F2 e então congelado para o teste e para as safras sintéticas.

O Evidently continuará sendo a fonte principal dos relatórios de drift exigidos pelo trabalho. Métricas de estabilidade oferecidas pelo OptBinning não substituirão esse relatório.

## Consequências

- A tabela detalhada do scorecard passa a ser um artefato versionado do modelo.
- O monitoramento deve incluir atributos de entrada, probabilidade e score.
- As faixas aprendidas não podem ser recalculadas silenciosamente em lotes de produção.
- O processo de treino precisará verificar monotonicidade, tamanho mínimo dos bins e estabilidade das variáveis selecionadas.
- Os parâmetros de escala e o valor numérico do limiar serão fechados após a EDA; o objetivo do limiar já está fixado em maximizar F2 na validação.
- Se o fluxo OptBinning não estiver integrado e testado até o fim do terceiro dia de implementação, o fallback será uma regressão logística simples em `sklearn.pipeline.Pipeline`; WOE e pontos passam a melhoria posterior, preservando os entregáveis obrigatórios.

WOE é o caminho preferencial de aprendizagem, não um requisito da entrega. Se o fallback for acionado, documentação, arquitetura e vídeo deverão apresentar a Pipeline logística realmente entregue e remover afirmações de que existe tabela de pontos.

## Experimento de escala planejado

Após o ajuste do modelo, diferentes combinações de ponto-base, odds-base e PDO serão aplicadas às mesmas probabilidades no conjunto de validação. A escolha considerará faixa observada, percentis e legibilidade. Ela não alterará o limiar probabilístico escolhido por F2 nem será comparada por AUC ou KS, pois uma transformação linear do log-odds não altera o poder preditivo.
