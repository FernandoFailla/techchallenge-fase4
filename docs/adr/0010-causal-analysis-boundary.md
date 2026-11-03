# ADR 0010: Limitar causalidade ao DAG e à intervenção sintética

- Status: aceito
- Data: 2026-10-23

## Contexto

O enunciado pede análise causal, mas o dataset não contém uma variável de tratamento econômico, indicadores macroeconômicos, sequência temporal de produção ou desenho experimental. Aplicar uma biblioteca de inferência causal não resolveria a falta de identificação.

## Decisão

A análise causal terá:

- um DAG limitado ao mecanismo realmente simulado: perfil comportamental, choque, interação e inadimplência;
- uma descrição explícita de confundimento e viés de seleção na carteira observada;
- uma intervenção sintética versionada no terceiro mês;
- probabilidades `p0(X)` e `p1(X)` e um sorteio uniforme compartilhado para construir desfechos potenciais pareados dentro do simulador.

`p0(X)` será o `predict_proba` do modelo monitorado depois de congelado. Os rótulos originais do pool não gerarão desfechos das safras sintéticas. Isso torna o mecanismo sem choque correto por construção e limita seu uso à demonstração do monitoramento, não à avaliação independente da qualidade do modelo.

Não usaremos DoWhy nem apresentaremos a diferença simulada como estimativa de um efeito causal no mundo real.

## Consequências

- Será possível explicar a causa da degradação dentro do gerador controlado.
- O contraste pareado recuperará somente o efeito programado pelo gerador, não um efeito unitário observado no mundo real.
- A documentação deverá separar observação estatística, hipótese causal e regra de simulação.
- A análise reconhecerá que somente clientes já presentes na carteira são observados, criando potencial viés de seleção.
- Para estimar efeitos reais seriam necessários dados temporais, variáveis econômicas, política de concessão e hipóteses de identificação adicionais.
