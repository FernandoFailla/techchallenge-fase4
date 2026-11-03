# Plano de implementação

## Contexto de execução

- Equipe: uma pessoa.
- Prazo: duas a três semanas.
- Estratégia: MVP didático e extensível, executável localmente.

## Backlog executável

| Entrega | Resultado verificável | Critério de aceite | Estado |
|---|---|---|---|
| E0: baseline de planejamento | Repositório local, ADRs, planos e mapa visual em commits semânticos | Histórico Git revisável e tutorial cronológico renderizável | Concluída |
| E1: bootstrap técnico | Projeto uv, pacote `src/`, Makefile, README, lint, testes e CI mínima | `uv sync` e `make check` passam em ambiente limpo | Concluída |
| E2: contrato e scoring mínimo | Pipeline logística simples, schema de scoring e rejeição controlada | Lote válido gera scores; lote inválido termina `REJECTED` com zero scores | Concluída |
| E3: observabilidade D4 | Simulação determinística em duas entradas, relatório Evidently e run MLflow | `make demo` produz artefato de modelo, relatório e run local sem dados individuais em logs | Em andamento |
| E4: dados e scorecard | Download reproduzível, EDA, partições sem leakage e scorecard opcional | Decisões da EDA registradas; modelo e limites versionados | Pendente |
| E5: safras e drift | M1 estável, M2 com data drift e M3 com concept drift maturado | Alertas e métricas correspondem aos mecanismos configurados | Pendente |
| E6: fairness, governança e entrega | Auditoria agregada, documentação final, demo limpa e roteiro STAR | Rastreabilidade aponta para evidências existentes | Pendente |

### Cadência de execução

- Manter somente uma entrega em andamento por vez.
- Encerrar cada entrega com targets Make reproduzíveis, testes verdes e commits semânticos.
- Atualizar o estado deste backlog somente depois de verificar o critério de aceite.
- Criar ao final um tutorial cronológico em `tutorial/NNN-*.qmd`; a pasta permanece ignorada pelo Git.
- Não antecipar módulos vazios: criar somente a estrutura necessária para a entrega corrente.

## Fluxo de trabalho

- Marimo documenta cada investigação como hipótese, evidência, resposta e decisão.
- Notebooks usam Polars e Plotnine, mas não persistem datasets, modelos, gráficos ou relatórios.
- Lógica que permanecerá em preparação, modelagem, avaliação ou monitoramento é promovida para `src/`, com testes e programação defensiva.
- Scripts determinísticos importam `src/`, carregam YAML validado por código testado de `src/`, persistem artefatos e registram runs oficiais no MLflow.
- Conversão para pandas ou NumPy ocorre uma única vez na fronteira de OptBinning, Scikit-Learn, Evidently, Fairlearn ou Plotnine quando necessária.

## Gate de proteção do prazo

Até o fim do dia 4 deverá existir uma fatia vertical executável: regressão logística simples, rejeição Pandera de lote inválido, simulação mínima de duas entradas, um relatório Evidently, um run MLflow com logs e um README inicial. Se OptBinning não estiver estável no dia 3, essa fatia usa a regressão logística simples e o scorecard WOE vira melhoria time-boxed.

## Semana 1: fundação, dados e scorecard

### Dias 1 e 2: bootstrap e EDA

- Criar `pyproject.toml`, ambiente uv, Makefile, `notebooks/`, `src/`, `scripts/`, `configs/`, testes e CI.
- Fazer um teste de compatibilidade entre Pandera, OptBinning, Evidently, Fairlearn e MLflow.
- Baixar o dataset por script reproduzível e registrar licença, checksum e metadados.
- Criar notebook Marimo de EDA, com questões explícitas sobre tipos, ranges, categorias, duplicatas, prevalência e grupos de auditoria.
- Criar as primeiras visualizações Plotnine e iniciar o tema do projeto sem salvar artefatos pelo notebook.
- Fechar os percentuais de partição, ranges do contrato e faixas etárias.

Aceite: `make check` passa; EDA gera um resumo versionável; decisões condicionadas à EDA são registradas.

### Dias 3 a 5: fatia vertical e scorecard

- Separar treino, validação, teste/referência e pool de simulação sem leakage.
- Construir o conjunto curado de atributos comportamentais e documentar suas fórmulas.
- Entregar primeiro uma `Pipeline` de regressão logística simples e o smoke ponta a ponta.
- Se o spike passar no dia 3, ajustar `BinningProcess`, WOE e regressão logística somente no treino.
- Escolher parâmetros e `review_required` por F2 na validação; medir AUC, PR-AUC, KS, Brier e calibração final no teste.
- Se WOE estiver ativo, comparar escalas por offset/PDO sem alterar o limiar probabilístico.
- Executar treino por script com YAML resolvido e registrar pipeline, config, split manifest e métricas no MLflow.

Aceite: o modelo e os limites têm versão; com WOE ativo, score e probabilidade têm direção consistente e a tabela de pontos é auditável.

## Semana 2: contrato, simulação e monitoramento

### Dias 6 e 7: contrato e quarentena

- Implementar schemas Pandera para treino, scoring sem alvo e feed de desfechos.
- Criar fixtures válidas e inválidas.
- Implementar estados do lote, quarentena e logs JSON.
- Demonstrar que o lote corrompido produz zero scores.
- Testar associação um-para-um e rejeição de desfechos ausentes, duplicados ou prematuros.

Aceite: o run inválido aparece como `REJECTED`, lista anomalias e não chama o modelo.

### Dias 8 a 10: três meses e Data Drift

- Gerar três safras OOT sintéticas, disjuntas, somente depois de congelar o modelo.
- Confirmar que o mês 1 é estável e que o mês 2 altera ao menos duas entradas retidas e influentes.
- Gerar predições, score quando disponível, PSI, relatório Evidently HTML/JSON e alertas.
- Registrar duração, volume, tags de estado, features com drift, logs e artefatos no MLflow.

Aceite: mês 2 mostra drift intencional e o pipeline continua com `PROCESSED_WITH_ALERTS`.

### Dias 11 e 12: maturação, Concept Drift e fairness

- Anexar desfechos um mês depois e aplicar `p0(X)` do modelo congelado e `p1(X)` com sorteio compartilhado no mês 3.
- Construir intervalos de incerteza da linha de base e diferenças por bootstrap de duas amostras.
- Demonstrar Concept Drift pelo contraste pareado e comparar AUC, KS, Brier e evento por faixa de score como consequências.
- Executar auditoria Fairlearn por sexo e faixa etária.
- Executar o playbook de investigação sobre qualquer disparidade material e conferir as limitações declaradas.

Aceite: `p0(X)` e `p1(X)` diferem conforme a regra versionada, o lote mostra degradação compatível e métricas de grupo não expõem linhas.

## Semana 3: integração, documentação e apresentação

### Dias 13 e 14: demo e documentação

- Integrar `make demo` em uma execução limpa.
- Consolidar arquitetura, comandos, resultados, LGPD, fairness e causalidade no README.
- Selecionar relatórios e imagens agregados pequenos para `artifacts/evidence/` no GitHub.
- Refatorar notebooks para importar toda lógica promovida de `src/` e remover duplicações.
- Executar checklist completa a partir de um clone ou ambiente limpo.

Aceite: cada linha de `docs/requirements-traceability.md` aponta para uma evidência existente.

### Dia 15: vídeo e margem

- Ensaiar e gravar o vídeo STAR com duração alvo de 4min30s.
- Demonstrar rejeição do lote inválido, drift, MLflow e resultado maturado.
- Verificar link do vídeo, histórico de commits, licença e instruções finais.
- Reservar o restante do prazo para correções, não para novas funcionalidades.

## Pontos de decisão após a EDA

- Percentuais e tamanhos finais de treino, validação, teste/referência e pool de simulação.
- Ranges e códigos aceitos pelo contrato.
- Se WOE estiver ativo, variáveis mantidas pelo Information Value e estabilidade dos bins.
- Faixas etárias com tamanho amostral suficiente.
- Escala de pontos e PDO; o objetivo do limiar já está fixado em F2.
- Intensidade da amostragem sob estresse e da intervenção de concept drift.
- Quantidade de reamostragens bootstrap compatível com tempo e estabilidade.

## Riscos principais

| Risco | Resposta |
|---|---|
| Incompatibilidade entre bibliotecas | Smoke test e lockfile no primeiro dia |
| OptBinning comprometer o prazo | Fallback obrigatório para `Pipeline` logística simples no dia 3 |
| Mudanças de API no Evidently | Fixar versão e encapsular geração de relatórios em um módulo |
| Drift simulado fraco ou instável | Sementes fixas e teste quantitativo da intensidade da simulação |
| Bins instáveis ou grupos pequenos | Restringir quantidade/tamanho de bins e consolidar grupos após EDA |
| Demo longa ou frágil | Artefatos cacheáveis, target de smoke e ensaio em ambiente limpo |
| README afirmar controles inexistentes | Revisão explícita entre implementado e prescrito |

## Fora do escopo

- API web e interface para analistas.
- Decisão automática de crédito.
- Prometheus, Grafana, cloud e Kubernetes.
- Banco operacional de clientes.
- Model Registry e implantação online.
- Retreino ou promoção automática.
- Inferência causal sobre uma economia real.
- Mitigação automática de fairness.

## Ordem de cortes se o prazo apertar

1. Reduzir variações de escala/PDO e manter uma única escala legível.
2. Reduzir quantidade de reamostragens bootstrap, preservando o método.
3. Produzir menos gráficos customizados, mantendo Evidently e MLflow.
4. Usar regressão logística simples no lugar de OptBinning.

Nunca cortar contrato bloqueante, duas features com drift, relatório Evidently, observabilidade, plano LGPD ou vídeo STAR.
