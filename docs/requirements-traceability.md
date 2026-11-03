# Rastreabilidade dos requisitos

| Requisito do enunciado | Decisão do projeto | Evidência planejada |
|---|---|---|
| Classificador base de crédito | Preferência por scorecard WOE; fallback oficial para Pipeline logística simples | Métricas de teste e artefato efetivamente entregue no MLflow; tabela de pontos somente se WOE estiver ativo |
| Dataset binário com pelo menos 5.000 amostras | UCI Default of Credit Card Clients, 30.000 registros | Fonte, licença, checksum e perfil do dataset |
| Separar referência e produção | Treino, validação, teste/referência e pool sintético disjuntos | Manifesto das partições, sementes e testes contra leakage |
| Pelo menos três regras rígidas | Pandera com esquema, nulos, unicidade, ranges e alvo | Testes do contrato e relatório de validação |
| Lote inválido bloqueado | Status `REJECTED`, quarentena e zero scores | Run do MLflow e log da demonstração do lote corrompido |
| Alterar ao menos duas variáveis | Mudança de composição sobre duas entradas retidas e influentes no modelo | Configuração da simulação, importância e comparações de distribuição |
| Gerar predições no lote novo | Score, probabilidade, faixa e `review_required` | Saída agregada e métricas do run |
| Detectar Data Drift | Evidently e PSI explícito | Relatório HTML/JSON por mês e alertas registrados |
| Demonstrar Concept Drift | Contraste pareado `p0(X)`/`p1(X)` nas mesmas entradas e maturação sob choque | Mecanismos versionados, contraste e degradação consequente |
| Teste estatístico PSI ou KS | PSI operacional e KS de duas amostras quando adequado; KS de discriminação separado | Tabelas e documentação metodológica |
| Centralizar logs e métricas | MLflow local com logs JSON anexados aos runs | Interface do MLflow, tags, métricas, alertas e artefatos |
| Métricas de saúde | Status, duração, volume, falhas, drift e alertas | Comparação de runs mensais no MLflow |
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
