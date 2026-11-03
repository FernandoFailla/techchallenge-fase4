# ADR 0011: Separar estabilidade populacional de degradação de desempenho

- Status: aceito
- Data: 2026-10-23

## Contexto

Data drift pode ser medido sem desfechos, enquanto concept drift exige evidência de mudança na relação entre entradas e alvo. Em scorecards, PSI e KS aparecem com significados diferentes e precisam ser nomeados de forma inequívoca.

## Decisão

No scoring, sem rótulos:

- o Evidently gerará o relatório geral de Data Drift sobre os atributos do modelo;
- PSI será explicitado para as variáveis manipuladas e para o score;
- PSI menor que 0,10 será informativo, entre 0,10 e 0,25 gerará aviso e a partir de 0,25 indicará mudança relevante;
- bins de PSI serão fixados na referência, com suavização de contagens zero e tamanho mínimo de amostra documentados;
- um alerta de dataset será aberto quando pelo menos duas entradas relevantes atingirem mudança operacionalmente relevante ou quando a distribuição do score atingir esse nível.

Após a maturação:

- serão calculados ROC-AUC, PR-AUC, KS de discriminação, Brier score e taxa observada de inadimplência por faixa de score;
- o teste histórico será reamostrado por bootstrap, com semente fixa, para estimar incerteza amostral da linha de base, não variabilidade temporal histórica;
- diferenças entre baseline e lote serão avaliadas por bootstrap de duas amostras; AUC/KS menores ou Brier maior além do intervalo gerarão alerta de degradação;
- concept drift será demonstrado diretamente pela diferença entre `p0(X)` e `p1(X)` nas mesmas entradas do mês 3; degradação de discriminação ou calibração será uma consequência monitorada.

## Consequências

- Os limites de desempenho serão artefatos versionados junto ao modelo.
- Lotes de produção não recalibrarão automaticamente os limites.
- O relatório deverá distinguir o KS de distribuição entre datasets do KS de discriminação do scorecard.
- Os cortes convencionais de PSI serão documentados como heurísticas operacionais, não como leis estatísticas universais.
- PSI não será descrito como teste de significância estatística.
