# ADR 0008: Quarentenar dados inválidos e não bloquear por drift

- Status: aceito
- Data: 2026-10-23

## Contexto

O contrato precisa bloquear a ingestão de dados ruins, mas nem toda anomalia de monitoramento torna uma predição inválida. Data drift pode representar uma mudança legítima da população, e degradação de desempenho precisa de investigação antes de qualquer retreino.

## Decisão

- Violação do contrato: o lote recebe status `REJECTED`, é colocado em quarentena, não gera scores e registra as violações.
- Feed de desfechos inválido: o feed recebe status `OUTCOME_REJECTED`, os scores existentes são preservados e somente a avaliação rotulada fica `EVALUATION_BLOCKED`.
- Data drift ou prediction drift: o lote continua, um alerta é registrado e o relatório é anexado ao run.
- Degradação de desempenho ou disparidade entre grupos: o lote já processado permanece auditável e é criada uma recomendação de investigação e possível retreino.
- Erro técnico inesperado: a execução recebe status `FAILED` e termina sem produzir saídas parciais como se fossem válidas.

A demonstração de um lote inválido poderá capturar a rejeição de forma controlada e ainda gerar evidências, sem permitir sua entrada no scorecard.

## Consequências

- Status de lote e status da execução são conceitos diferentes.
- Rejeição de scoring e rejeição de desfechos são estados diferentes e não apagam resultados anteriores válidos.
- Não haverá retreino, aprovação ou implantação automática do modelo.
- Todo alerta deverá indicar severidade, causa observada e ação recomendada.
- A presença de drift, isoladamente, não será interpretada como erro do modelo.
