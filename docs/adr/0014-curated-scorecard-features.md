# ADR 0014: Usar features comportamentais curadas no scorecard

- Status: aceito
- Data: 2026-10-23

## Contexto

As 19 variáveis financeiras incluem seis meses de status, faturas e pagamentos, com forte correlação temporal. Incluir tudo produziria um scorecard extenso e difícil de explicar. Restringir o modelo a poucas colunas brutas também desperdiçaria o histórico disponível.

## Decisão

A EDA partirá de um conjunto pequeno de features financeiras e comportamentais interpretáveis:

- limite de crédito;
- atraso mais recente;
- maior atraso nos seis meses;
- quantidade de meses com atraso;
- utilização de crédito atual e média;
- razão entre pagamentos e faturas;
- tendência de saldo, somente se permanecer estável e fácil de explicar.

O tratamento de saldos negativos, faturas zeradas e razões extremas será definido a partir dos dados. OptBinning e critérios de Information Value/estabilidade poderão remover candidatas, mas não adicionar atributos demográficos.

Depois do ajuste, a simulação escolherá pelo menos duas entradas que tenham sido efetivamente retidas e demonstrem influência no score. Limite e atraso são candidatos de negócio, não nomes garantidos antes da EDA.

## Consequências

- A geração de features será determinística, versionada e compartilhada por treino e produção.
- Cada feature selecionada deverá ter definição de negócio no README ou no dicionário de dados.
- Redução de dimensionalidade não poderá usar produção nem o alvo de validação.
- A tabela final deverá permanecer pequena o suficiente para ser apresentada no vídeo.
