# Plano do vídeo STAR

Duração alvo: 4min30s. Limite absoluto: 5 minutos.

## 0:00-0:30 - Situation

- Explicar degradação silenciosa em crédito: a entrada muda antes de o desfecho chegar.
- Mostrar a linha do tempo scoring agora, inadimplência observada um mês depois.
- Enunciar o risco para clientes, fintech e governança.

## 0:30-0:50 - Task

- Interceptar dados ruins.
- Monitorar Data Drift sem rótulos.
- Detectar Concept Drift após maturação.
- Produzir trilha de auditoria, fairness e plano LGPD.

## 0:50-4:05 - Action

### 0:50-1:10 - Modelo

- Mostrar rapidamente a Pipeline logística efetivamente entregue; se WOE estiver ativo, incluir a tabela de pontos.
- Explicar que a saída pede revisão, não rejeição automática; mencionar direção do score apenas se houver scorecard.

### 1:10-1:55 - Qualidade

- Executar ou mostrar o run do lote corrompido.
- Apontar as violações Pandera, status `REJECTED` e ausência de scores.

### 1:55-2:45 - Data Drift

- Comparar mês saudável e mês de mudança populacional.
- Abrir o relatório Evidently e destacar pelo menos duas features com PSI relevante.
- Mostrar que o lote foi processado com alerta, não bloqueado.

### 2:45-3:30 - Concept Drift e fairness

- Mostrar `p0(X)` e `p1(X)` sobre as mesmas entradas e o lote maturado sob choque.
- Explicar que o contraste demonstra a mudança programada em `P(Y|X)` e que a piora de desempenho é consequência.
- Exibir uma tabela agregada do Fairlearn sem dados individuais.

### 3:30-4:05 - Observabilidade

- Mostrar no MLflow os runs, estado do lote, duração, volume, saúde do processo, alertas, logs e relatórios anexados.

## 4:05-4:30 - Result

- Recapitular que qualidade bloqueia, drift alerta e degradação recomenda investigação.
- Mostrar o DAG e esclarecer que a causalidade é sintética, não inferida do mundo real.
- Citar base legal de proteção do crédito, minimização, retenção e revisão humana.
- Encerrar com as limitações e a principal lição aprendida.

## Regras de gravação

- Usar artefatos previamente gerados; não depender de esperar treinamento durante o vídeo.
- Ensaiar transições de tela e manter uma versão de backup dos relatórios.
- Não mostrar caminhos locais, IDs de linha ou dados individuais.
- Cronometrar duas vezes e cortar detalhes de implementação antes de acelerar a fala.
