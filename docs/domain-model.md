# Modelo de domínio

Este documento é atualizado durante a sessão de planejamento. Decisões consolidadas ficam registradas em `docs/adr/`.

## Narrativa atual

Uma fintech recebe lotes de dados de clientes existentes para monitoramento de risco da carteira. Antes de qualquer predição, o lote precisa cumprir um contrato de scoring sem alvo. Para cada registro válido, o modelo estima a probabilidade de inadimplência no mês seguinte e produz uma faixa de risco para apoiar revisão ou renovação de limite.

Uma avaliação de alto risco gera apenas o sinalizador `review_required`. A fila, a interface e a decisão do analista ficam fora do MVP; não haverá rejeição automática de crédito.

No momento do scoring, o desfecho ainda não existe. O sistema compara o lote pontuado com uma referência e monitora qualidade, dados de entrada e predições. Um mês depois, o lote matura e recebe os desfechos, permitindo medir desempenho, concept drift e disparidades de erro entre grupos. O processo deve produzir evidências auditáveis e respeitar princípios de privacidade e revisão humana.

## Atores identificados

- Cliente de crédito: titular ao qual a avaliação se refere.
- Analista de crédito: interpreta a avaliação e toma a decisão de negócio.
- Equipe de dados/modelo: treina, versiona e monitora o modelo.
- Responsável por governança: define controles de privacidade, retenção e auditoria.

## Conceitos candidatos

- Lote de dados
- Contrato de dados
- Versão do contrato
- Resultado de validação
- Modelo de risco
- Scorecard
- Versão do modelo
- Avaliação de risco
- Faixa de risco
- Execução do pipeline
- Run de monitoramento
- Análise de drift
- Alerta
- Evidência de auditoria
- Desfecho
- Visão de scoring
- Visão de auditoria

Os conceitos são candidatos, não uma prescrição de classes ou tabelas.

## Invariantes já aceitas

- Um lote inválido não pode chegar à etapa de predição.
- Violações de qualidade devem rejeitar o lote inteiro e permanecer auditáveis.
- O identificador do dataset não pode ser usado como atributo preditivo.
- Atributos demográficos não podem entrar no modelo; seu uso fica restrito à auditoria de equidade.
- Uma estimativa do modelo não equivale à decisão final de crédito.
- Uma avaliação de alto risco deve solicitar revisão humana, não rejeitar crédito automaticamente.
- Dados de referência e dados de análise/produção devem permanecer separados.
- A coluna-alvo não pode existir no lote de scoring e só pode chegar por feed maturado validado.
- Resultados de validação, drift e execução devem ser rastreáveis até o lote processado.
- Logs e métricas de observabilidade não podem expor dados pessoais em nível de cliente.
- Controles implementados e controles prescritos para produção não podem ser apresentados como equivalentes.
- As faixas e os valores WOE aprendidos no treino não podem ser reajustados com dados de produção.
- Limites de desempenho devem ser derivados do teste/referência congelado e versionados com o modelo.
- Features derivadas devem ter a mesma fórmula versionada no treino e na produção.

## Eventos de domínio candidatos

- Lote recebido
- Validação aprovada
- Lote rejeitado
- Visões de scoring e auditoria separadas
- Avaliações de risco geradas
- Revisão humana solicitada
- Drift de dados detectado
- Lote maturado
- Desempenho degradado
- Disparidade entre grupos detectada
- Alerta aberto

## Política operacional

- `REJECTED`: lote em quarentena por violação do contrato; nenhuma avaliação é gerada.
- `PROCESSED_WITH_ALERTS`: lote válido e pontuado, mas com drift ou outro sinal para investigação.
- `PROCESSED`: lote válido e sem alertas.
- `FAILED`: falha técnica impediu a conclusão confiável da execução.
- `EVALUATION_BLOCKED`: scores válidos existem, mas um feed de desfechos rejeitado impede métricas rotuladas.

Drift não interrompe scoring e nenhum alerta dispara retreino automático.

## Linha do tempo da simulação

- Mês 1: lote saudável, usado para demonstrar a operação normal.
- Mês 2: mudança na composição da população, com maior presença de limite baixo e atraso de pagamento.
- Mês 3: choque econômico sintético altera a chance de inadimplência para perfis de alta utilização e atraso.

Cada mês passa primeiro pelo scoring sem desfecho e, depois, pela maturação com desfecho.

## Decisões condicionadas à EDA

- Percentuais exatos de treino, validação, teste/referência e pool de simulação.
- Quantidade de registros em cada lote mensal.
- Limites de domínio do contrato derivados do perfil real dos dados.
- Parâmetros da escala de pontos; o limiar probabilístico seguirá F2 na validação.
- Quantidade de reamostragens bootstrap; fairness já exige 200 registros e 30 casos por classe para publicação.

Independentemente dos percentuais, as partições serão disjuntas e estratificadas: treino ajusta o modelo; validação escolhe parâmetros, escala e limiar; teste mede a linha de base final e serve de referência de monitoramento; o pool de simulação gera as safras OOT sintéticas somente após o congelamento. As safras não reutilizam registros entre meses.
