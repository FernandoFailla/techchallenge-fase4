# Credit Monitoring Tech Challenge

Implementação local e reproduzível de um pipeline de risco de crédito com contratos de dados, simulação de drift, observabilidade e auditoria.

O planejamento detalhado está em [`docs/README.md`](docs/README.md). O estado das entregas está em [`docs/implementation-plan.md`](docs/implementation-plan.md).

## Requisitos

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Make

## Ambiente

```bash
make setup
make check
```

`make check` executa lint, verificação de formatação e testes rápidos. Dados e artefatos gerados permanecem fora do Git.

## Comandos

| Comando | Finalidade |
|---|---|
| `make setup` | Sincronizar o ambiente a partir do lockfile |
| `make lint` | Verificar lint e formatação |
| `make test` | Executar a suíte rápida |
| `make notebook-check` | Validar estaticamente o notebook Marimo |
| `make check` | Executar lint, testes e validação do notebook |
| `make demo` | Executar a fatia vertical D4 e registrar um run MLflow local |
| `make prepare-data` | Baixar, verificar e particionar o dataset UCI |
| `make train-real` | Treinar e registrar o baseline com dados reais preparados |
| `make e4` | Executar preparação e treino da entrega E4 |
| `make e5` | Executar as três safras, maturar M3 e registrar drift no MLflow |
| `make notebook` | Abrir a EDA de drift sintético no Marimo |
| `make notebook-real` | Abrir a EDA do dataset UCI no Marimo |
| `make notebook-drift` | Abrir a investigação das safras E5 no Marimo |
| `make mlflow-ui` | Abrir o MLflow local em `http://127.0.0.1:5000` |

## Demonstração D4

```bash
make demo
make mlflow-ui
```

A demonstração treina a regressão logística, processa um lote válido com drift,
bloqueia um lote inválido antes da inferência e gera um relatório Evidently. Os
artefatos agregados ficam em `artifacts/runtime/d4-demo/`; metadados e artefatos
do run ficam no store local sob `artifacts/runtime/`. Nenhum dado em nível de
cliente é persistido nos logs ou no MLflow.

## Dados reais e baseline

```bash
make e4
make notebook-real
make mlflow-ui
```

O projeto usa o dataset [Default of Credit Card Clients](https://doi.org/10.24432/C55S3H),
criado por I-Cheng Yeh e distribuído pela UCI sob licença CC BY 4.0. O download
é aceito somente com o SHA-256
`56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`.

A EDA fechou partições estratificadas de 60% para treino, 15% para validação,
15% para teste/referência e 10% para simulação. O modelo real atual é a
`Pipeline` logística de fallback, versão `real-logistic-v1`; não há WOE, escala
de pontos nem tabela de scorecard ativa. O limiar `review_required` é escolhido
por F2 somente na validação e congelado antes da avaliação no teste.

Resultados verificados no teste/referência:

| Métrica | Valor |
|---|---:|
| ROC-AUC | 0,7235 |
| Average Precision | 0,5142 |
| KS de discriminação | 0,3912 |
| Brier score | 0,1428 |
| Recall no limiar F2 | 0,5974 |

Fonte, perfil, tratamentos e manifesto das partições estão documentados em
[`docs/data-source.md`](docs/data-source.md).

## Safras e drift

```bash
make e5
make notebook-drift
make mlflow-ui
```

A E5 usa 2.250 registros do pool pós-congelamento para criar M1, M2 e M3 com
750 IDs disjuntos por mês. M1 e M3 são reservados aleatoriamente primeiro. M2
altera somente a seleção do restante: uma corrida exponencial ponderada aumenta a
participação de baixo limite, atraso e utilização, sem editar valores individuais.

No scoring, o Evidently e o PSI usam o teste como referência. M2 termina em
`PROCESSED_WITH_ALERTS`, com as sete features sinalizadas pelo Evidently, quatro
PSI acima de 0,10 e PSI do score igual a 0,2573. M1 e M3 têm somente PSI
informativo e terminam em `PROCESSED`.

Após a maturação de M3, o feed separado é associado um-para-um aos scores. O
simulador compara `p0(X)` com
`p1(X) = sigmoid(logit(p0(X)) + 2 * g(X))` usando o mesmo sorteio para `Y0` e
`Y1`. A probabilidade média sobe de 0,2163 para 0,2727, 55 desfechos pareados
mudam e o Brier do modelo congelado piora de 0,1594 para 0,1904. O intervalo
bootstrap da diferença, `[0,0212; 0,0415]`, aciona o alerta de desempenho.

O contraste demonstra apenas o mecanismo programado no simulador; não estima um
efeito causal econômico real. Relatórios agregados e logs ficam em
`artifacts/runtime/drift-e5/`; o run verificado é
`6de5c6ace45a4c8fadef55f173f97a78`.
