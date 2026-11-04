# Plano de auditoria de equidade

## Finalidade

Avaliar se o scorecard e o sinalizador de revisão apresentam comportamentos ou erros materialmente diferentes entre grupos. A auditoria não usa atributos demográficos para gerar scores e não declara conformidade legal a partir de uma única métrica.

## Dados

- A visão de scoring exclui `ID`, `SEX`, `AGE`, `EDUCATION` e `MARRIAGE`.
- A visão restrita de auditoria relaciona a predição a `SEX` e a faixas de `AGE`.
- As faixas etárias são fixas entre runs: `21-29`, `30-39` e `40+`. A consolidação de `40+` foi decidida na EDA de treino, sem alvo da safra auditada. Após a maturação, métricas de erro só são publicadas quando houver pelo menos 200 registros, 30 eventos e 30 não eventos no grupo.
- `EDUCATION` e `MARRIAGE` ficam fora da auditoria inicial para evitar multiplicação de grupos pequenos sem uma hipótese de dano definida.

## Métricas sem desfecho

- Quantidade de registros por grupo.
- Score médio e distribuição do score por grupo.
- Taxa de `review_required` por grupo.

Essas diferenças descrevem alocação de revisão, mas não medem erros do modelo.

## Métricas após maturação

- Recall da classe de inadimplência e taxa de falsos negativos.
- Taxa de falsos positivos.
- Precisão e acurácia por grupo como contexto.
- `equalized_odds_difference` calculada sobre `review_required`.

O Fairlearn `MetricFrame` produzirá valores globais e desagregados. Intervalos bootstrap representarão incerteza amostral da linha de base quando o tamanho dos grupos permitir. Para equalized odds, cada réplica reamostra dentro dos estratos grupo-alvo e recalcula a métrica composta antes dos quantis.

Grupos que não cumprirem a regra de publicação aparecerão apenas com contagem e motivo da supressão, sem estimativa pontual de erro.

## Interpretação e resposta

- Falso positivo: cliente sem inadimplência encaminhado desnecessariamente para revisão reforçada.
- Falso negativo: risco de inadimplência não sinalizado, aumentando a exposição da fintech.
- Disparidade além da incerteza da linha de base abre investigação; não prova discriminação por si só.
- O MVP não aplica mitigação automática nem limiares diferentes por grupo.
- A investigação revisará proxies, bins, qualidade dos rótulos e a política do limiar; comparará uma política candidata com o baseline e documentará impacto em desempenho e grupos.
- O responsável por risco/governança aprovará remediação, coleta adicional, retirada de feature ou suspensão do modelo. Retreino e publicação nunca ocorrerão automaticamente.

## Implementação e resultado E6

- `src/credit_monitoring/fairness.py` concentra agrupamento por idade, construção da visão restrita, regra de publicação e métricas Fairlearn.
- `configs/e6-fairness.yaml` fixa semente 48, 200 reamostragens bootstrap e gatilho operacional de 0,10.
- `scripts/run_e6_fairness.py` reconstrói M3, associa os desfechos maturados e registra somente resultados agregados.
- `notebooks/04_fairness_audit.py` apresenta a investigação sem persistir artefatos.
- A configuração resolvida incorpora os parâmetros E5 efetivos e os hashes do dataset e do modelo; a run também informa revisão Git, estado do worktree e hash determinístico dos arquivos de execução, incluindo arquivos ainda não commitados.

Na execução canônica, sexo publicou dois grupos e idade publicou `21-29`, `30-39` e `40+`; nenhum grupo foi suprimido. A diferença de equalized odds foi 0,0458 por sexo, IC `[0,0092; 0,1556]`, e 0,1344 por idade, IC `[0,0662; 0,2766]`. Os limites inferiores não ultrapassaram o gatilho de 0,10, portanto o estado foi `MONITORING_CONTINUES`.

O playbook calcula em runtime a exclusão de demográficos, confere o limiar comum e verifica suporte dos grupos; os rótulos são identificados como sintéticos. A resposta é manter monitoramento e validar com desfechos temporais reais antes de qualquer mudança de política. O resumo da run mais recente está em `artifacts/evidence/e6-fairness-summary.json`.
