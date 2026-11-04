# Rastreabilidade dos requisitos

| Requisito do enunciado | Decisão do projeto | Evidência |
|---|---|---|
| Classificador base de crédito | Pipeline logística simples `real-logistic-v1`; WOE permanece inativo | `artifacts/evidence/monitoring-summary.json` e artefatos locais MLflow |
| Dataset binário com pelo menos 5.000 amostras | UCI Default of Credit Card Clients, 30.000 registros | Fonte, licença, checksum e perfil em `docs/data-source.md` |
| Separar referência e produção | Treino, validação, teste/referência e pool sintético disjuntos | Manifesto 60/15/15/10 com hashes de IDs e testes contra sobreposição |
| Pelo menos três regras rígidas | Pandera com esquema, nulos, unicidade, ranges e alvo | Testes do contrato e relatório de validação |
| Lote inválido bloqueado | Status `REJECTED`, violações sanitizadas e zero scores | Seção de contrato em `artifacts/evidence/drift-observability-report.html` |
| Alterar ao menos duas variáveis | Mudança de composição sobre entradas retidas e influentes no modelo | `configs/e5-drift.yaml`, `build_drift_cohorts` e notebook `03_drift_scenarios.py` |
| Gerar predições no lote novo | Probabilidade e `review_required` para cada safra | Volumes e estados em `artifacts/evidence/e5-drift-summary.json` |
| Detectar Data Drift | Evidently e PSI explícito | `artifacts/evidence/drift-observability-report.html` e relatórios completos locais |
| Demonstrar Concept Drift | Contraste pareado `p0(X)`/`p1(X)` nas mesmas entradas e maturação sob choque | `simulate_concept_drift`, associação um-para-um e alerta adverso de Brier |
| Teste estatístico PSI ou KS | PSI operacional e KS de discriminação nomeado separadamente | Bins PSI fixados na referência e resumo agregado do run E5 |
| Centralizar logs e métricas | MLflow local com runs de preparação, treino e monitoramento; logs JSON incluem sucesso/falha sanitizada | Tags, métricas, eventos e relatórios resumidos em `artifacts/evidence/monitoring-summary.json` |
| Métricas de saúde | Status, duração, volume, falhas, drift, alertas, CPUs disponíveis e pico RSS | `artifacts/evidence/drift-observability-report.html` e MLflow local |
| Dashboard ou ambiente de rastreamento | MLflow Tracking UI e relatório HTML estático | `make mlflow-ui` e `artifacts/evidence/drift-observability-report.html` |
| Governança e LGPD | Plano em duas camadas, base legal e retenção | Seção consolidada no README e `docs/governance-plan.md` |
| Mitigação de vieses | Auditoria Fairlearn e playbook sem mitigação automática | `notebooks/04_fairness_audit.py` e `artifacts/evidence/e6-fairness-summary.json` |
| Causalidade | DAG e intervenção/contrafactual sintéticos | Seção no README e `docs/causal-model.md` |
| Histórico de commits semântico | Commits pequenos por etapa | Histórico Git local por entrega; publicação GitHub ainda pendente |
| Vídeo STAR de até cinco minutos | Roteiro sincronizado com a demo | `docs/video-star-plan.md`; link final ainda depende da gravação |

## Critérios de aceite da entrega

- Uma instalação limpa consegue executar a demo pelos targets documentados.
- O lote inválido não produz scores.
- O mês 2 mostra drift relevante em pelo menos duas entradas.
- O mês 3 mostra diferença explícita entre `p0(X)` e `p1(X)` nas mesmas entradas e degradação compatível após maturação.
- Relatórios não contêm identificadores individuais.
- README distingue claramente implementação real, simulação e prescrição de produção.
- Todos os links e artefatos usados no vídeo existem na versão entregue.
