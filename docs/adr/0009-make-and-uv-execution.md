# ADR 0009: Usar Makefile como interface e uv para o ambiente

- Status: aceito
- Data: 2026-10-23

## Contexto

O projeto precisa ser simples de executar e reproduzir. Notebooks como mecanismo principal introduziriam estado implícito, enquanto um orquestrador de workflows seria desproporcional ao MVP.

## Decisão

O repositório usará:

- `pyproject.toml` e lockfile do uv para dependências e ambiente;
- notebooks Marimo orientados por hipótese para EDA e decisões, sem persistência de dados ou artefatos;
- Polars para manipulação e Plotnine com tema próprio para visualizações;
- um pacote Python em `src/` para toda lógica testável;
- `scripts/` para execuções determinísticas e persistência;
- YAML para treino, simulação e outros fluxos com iteração esperada;
- `Makefile` como interface humana para instalação, qualidade, treino, simulação, monitoramento, demonstração e interface do MLflow;
- notebooks que importam funções promovidas de `src/`, mas não são necessários para reproduzir entregáveis.

Targets candidatos incluem `setup`, `test`, `lint`, `notebook`, `train`, `simulate`, `validate-bad-batch`, `monitor`, `demo` e `mlflow-ui`. A lista final deverá permanecer pequena e cada target apenas delegará a scripts Python.

## Consequências

- O avaliador terá um comando principal para gerar a demonstração completa.
- Regras de negócio não serão implementadas em receitas do Makefile.
- Somente scripts persistirão datasets derivados, modelos, relatórios e gráficos oficiais.
- A versão de Python será fixada após validar a compatibilidade conjunta das bibliotecas.
- Dados baixados, ambiente virtual, banco do MLflow e artefatos de runtime serão ignorados pelo Git; somente evidências agregadas allowlisted em `artifacts/evidence/` poderão ser versionadas.
