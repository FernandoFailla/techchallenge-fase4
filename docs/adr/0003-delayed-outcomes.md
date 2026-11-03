# ADR 0003: Modelar desfechos com atraso de um mês

- Status: aceito
- Data: 2026-10-23

## Contexto

O modelo estima inadimplência no mês seguinte. No momento da avaliação de risco, o desfecho verdadeiro ainda não existe. Data drift pode ser observado sem rótulos, mas desempenho, concept drift e métricas de erro por grupo exigem os desfechos.

## Decisão

Cada lote simulará dois momentos:

1. No scoring, o sistema valida dados sem alvo, gera predições e monitora qualidade, distribuições de entrada, distribuição das predições e métricas operacionais.
2. Após um mês simulado, um feed separado associa os desfechos ao lote e permite calcular desempenho, indicadores de concept drift e métricas de equidade dependentes do rótulo.

Haverá contratos distintos para dados históricos rotulados, scoring e desfechos maturados. O contrato de scoring proibirá a coluna-alvo. O feed de desfechos exigirá chave, alvo e `available_at`, e a associação será validada como um-para-um antes de qualquer métrica rotulada.

## Consequências

- Uma queda de desempenho não será inferida apenas de data drift.
- Alertas sem rótulo e alertas com rótulo terão significados e respostas operacionais diferentes.
- A chave técnica precisará associar cada avaliação ao seu desfecho sem entrar no treinamento.
- A demonstração poderá explicar por que a degradação do modelo é detectada depois da mudança nos dados.
- Desfechos ausentes, duplicados, prematuros ou sem predição correspondente impedirão a avaliação do lote, sem alterar os scores já emitidos.
