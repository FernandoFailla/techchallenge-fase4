# ADR 0007: Centralizar rastreamento local no MLflow

- Status: aceito
- Data: 2026-10-23

## Contexto

O enunciado aceita MLflow como ambiente de rastreamento e pede visibilidade sobre execução do pipeline, latência, volume, falhas e drift. Prometheus e Grafana aproximariam o projeto de uma infraestrutura de produção, mas aumentariam o custo operacional do MVP. Um dashboard customizado duplicaria visualizações já fornecidas por MLflow e Evidently.

## Decisão

Usaremos MLflow local como interface de rastreamento e logs JSON estruturados como registro de diagnóstico.

- O treino do scorecard será um run com parâmetros, métricas e artefatos do modelo.
- Cada processamento de lote será um run associado a `batch_id`, mês simulado, versão do contrato e versão do modelo.
- O run registrará hash da configuração YAML completamente resolvida, manifesto das partições e revisão do código quando disponível.
- Relatórios HTML e JSON do Evidently serão artefatos do run correspondente.
- Os logs terão, no mínimo, timestamp, nível, `run_id`, `batch_id`, etapa, status, duração e mensagem, e serão anexados ao run como artefatos.
- Estado do lote e severidade dos alertas serão tags ou parâmetros; o status nativo do MLflow representará somente sucesso ou falha técnica da execução.
- A execução local da interface será documentada por um único comando.

As métricas operacionais mínimas serão duração total e por etapa, volume recebido/processado, quantidade de violações de qualidade, quantidade de features com drift, quantidade de alertas e uso básico de CPU/memória do processo quando suportado pela versão fixada. Após a maturação, também serão registradas métricas de desempenho e equidade.

## Consequências

- Não haverá Prometheus, Grafana, Datadog, cloud ou dashboard Streamlit no MVP.
- Dados e artefatos gerados localmente não serão versionados indiscriminadamente; o repositório conterá instruções e evidências selecionadas.
- `artifacts/evidence/` poderá versionar somente HTML, JSON e imagens agregadas selecionadas; dados brutos, artefatos por linha, `artifacts/runtime/` e o store local do MLflow serão ignorados.
- Logs e métricas não poderão conter identificadores de clientes nem atributos demográficos em nível de linha.
- O status do run no MLflow e o código de saída do processo deverão refletir falhas técnicas; rejeições esperadas serão runs concluídos com `batch_state=REJECTED` e evidências da quarentena.
