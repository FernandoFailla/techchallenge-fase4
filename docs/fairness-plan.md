# Plano de auditoria de equidade

## Finalidade

Avaliar se o scorecard e o sinalizador de revisão apresentam comportamentos ou erros materialmente diferentes entre grupos. A auditoria não usa atributos demográficos para gerar scores e não declara conformidade legal a partir de uma única métrica.

## Dados

- A visão de scoring exclui `ID`, `SEX`, `AGE`, `EDUCATION` e `MARRIAGE`.
- A visão restrita de auditoria relaciona a predição a `SEX` e a faixas de `AGE`.
- As faixas etárias serão definidas usando somente contagens sem alvo. Faixas adjacentes serão agregadas até atingir pelo menos 200 registros; após a maturação, métricas de erro só serão publicadas quando houver também pelo menos 30 eventos e 30 não eventos no grupo.
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

O Fairlearn `MetricFrame` produzirá valores globais e desagregados. Intervalos bootstrap representarão incerteza amostral da linha de base quando o tamanho dos grupos permitir.

Grupos que não cumprirem a regra de publicação aparecerão apenas com contagem e motivo da supressão, sem estimativa pontual de erro.

## Interpretação e resposta

- Falso positivo: cliente sem inadimplência encaminhado desnecessariamente para revisão reforçada.
- Falso negativo: risco de inadimplência não sinalizado, aumentando a exposição da fintech.
- Disparidade além da incerteza da linha de base abre investigação; não prova discriminação por si só.
- O MVP não aplica mitigação automática nem limiares diferentes por grupo.
- A investigação revisará proxies, bins, qualidade dos rótulos e a política do limiar; comparará uma política candidata com o baseline e documentará impacto em desempenho e grupos.
- O responsável por risco/governança aprovará remediação, coleta adicional, retirada de feature ou suspensão do modelo. Retreino e publicação nunca ocorrerão automaticamente.
