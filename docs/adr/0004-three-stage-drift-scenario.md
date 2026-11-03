# ADR 0004: Simular drift em três estágios distintos

- Status: aceito
- Data: 2026-10-23

## Contexto

O enunciado pede Data Drift e Concept Drift, mas alterar distribuições de entrada não demonstra, por si só, mudança na relação entre atributos e desfecho. Misturar os dois fenômenos em um único lote também dificultaria explicar qual alerta aparece primeiro e por quê.

O dataset é uma fotografia histórica sem uma sequência de meses de produção. Qualquer evolução temporal deverá ser explicitamente simulada e reproduzível.

## Decisão

A demonstração terá três lotes mensais:

1. Mês saudável: população semelhante à referência e mecanismo de desfecho sem choque.
2. Mês de data drift: maior participação de perfis definidos por pelo menos duas entradas retidas e influentes no scorecard, alterando `P(X)` sem editar seus valores individuais.
3. Mês de concept drift: sobre as mesmas linhas de entrada do terceiro lote, compara o mecanismo-base `p0(X)` com o mecanismo sob choque `p1(X)`, no qual utilização e atraso modificam explicitamente a chance de inadimplência.

Os dados históricos serão separados em treino, validação e teste. Um pool adicional e disjunto só será usado depois do congelamento do modelo para gerar as três safras OOT sintéticas, sem reutilizar registros entre meses. A semente, as regras de amostragem e os mecanismos `p0`/`p1` serão versionados.

## Consequências

- O segundo mês deverá acionar alertas de distribuição antes da disponibilidade dos desfechos.
- A mudança em `P(Y|X)` será demonstrada diretamente pelo contraste `p0(X)` versus `p1(X)` nas mesmas entradas; degradação de desempenho será apresentada como consequência, não como prova isolada.
- A simulação será descrita como cenário sintético, não como observação causal real da economia.
- A análise causal deverá distinguir hipótese de mecanismo, intervenção simulada e evidência observada.
