# Rastreabilidade dos requisitos

| Requisito do enunciado | Decisão do projeto | Evidência planejada |
|---|---|---|
| Classificador base de crédito | Pipeline logística simples `real-logistic-v1`; WOE permanece inativo | Métricas e artefatos do run MLflow `8f0ae13d86ae43bd9615c563b87d48db` |
| Dataset binário com pelo menos 5.000 amostras | UCI Default of Credit Card Clients, 30.000 registros | Fonte, licença, checksum e perfil em `docs/data-source.md` |
| Separar referência e produção | Treino, validação, teste/referência e pool sintético disjuntos | Manifesto 60/15/15/10, semente 42 e testes contra leakage |
| Pelo menos três regras rígidas | Pandera com esquema, nulos, unicidade, ranges e alvo | Testes do contrato e relatório de validação |
| Lote inválido bloqueado | Status `REJECTED`, quarentena e zero scores | Run do MLflow e log da demonstração do lote corrompido |
| Alterar ao menos duas variáveis | Mudança de composição sobre entradas retidas e influentes no modelo | `configs/e5-drift.yaml`, `build_drift_cohorts` e notebook `03_drift_scenarios.py` |
| Gerar predições no lote novo | Probabilidade e `review_required` para cada safra | Métricas agregadas do run E5 `6de5c6ace45a4c8fadef55f173f97a78` |
| Detectar Data Drift | Evidently e PSI explícito | Relatórios HTML/JSON de M1/M2/M3 em `artifacts/runtime/drift-e5/` |
| Demonstrar Concept Drift | Contraste pareado `p0(X)`/`p1(X)` nas mesmas entradas e maturação sob choque | `simulate_concept_drift`, associação um-para-um e alerta adverso de Brier |
| Teste estatístico PSI ou KS | PSI operacional e KS de discriminação nomeado separadamente | Bins PSI fixados na referência e resumo agregado do run E5 |
| Centralizar logs e métricas | MLflow local com logs JSON anexados aos runs | Run E5 com tags de estado, métricas, eventos e relatórios Evidently |
| Métricas de saúde | Status, duração, volume, falhas, drift e alertas | `run-summary.json`, `events.jsonl` e métricas mensais no MLflow |
| Dashboard ou ambiente de rastreamento | MLflow Tracking UI | Instrução `make mlflow-ui` e evidência visual selecionada |
| Governança e LGPD | Plano em duas camadas, base legal e retenção | Seção consolidada no README e plano detalhado em docs |
| Mitigação de vieses | Auditoria Fairlearn e playbook de investigação/remediação | Relatório agregado, regra de grupos mínimos e resposta documentada |
| Causalidade | DAG e intervenção/contrafactual sintéticos | Diagrama, contraste e limites de interpretação |
| Histórico de commits semântico | Commits pequenos por etapa | Histórico Git com mensagens convencionais e PR/checks |
| Vídeo STAR de até cinco minutos | Roteiro sincronizado com a demo | Link no README e roteiro cronometrado |

## Critérios de aceite da entrega

- Uma instalação limpa consegue executar a demo pelos targets documentados.
- O lote inválido não produz scores.
- O mês 2 mostra drift relevante em pelo menos duas entradas.
- O mês 3 mostra diferença explícita entre `p0(X)` e `p1(X)` nas mesmas entradas e degradação compatível após maturação.
- Relatórios não contêm identificadores individuais.
- README distingue claramente implementação real, simulação e prescrição de produção.
- Todos os links e artefatos usados no vídeo existem na versão entregue.
