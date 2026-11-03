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
| `make notebook` | Abrir a EDA de drift sintético no Marimo |
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
