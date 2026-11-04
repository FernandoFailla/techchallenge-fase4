# Fonte de dados, EDA e baseline real

## Fonte e licença

- Dataset: Default of Credit Card Clients.
- Criador: I-Cheng Yeh.
- Repositório: UCI Machine Learning Repository.
- DOI: [10.24432/C55S3H](https://doi.org/10.24432/C55S3H).
- Licença: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Arquivo oficial: `default of credit card clients.xls`.
- Registros: 30.000.
- Atributos explicativos brutos: 23.
- Alvo: `default_payment_next_month`, com evento igual a 1.
- SHA-256 do ZIP oficial: `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`.

O script `scripts/prepare_data.py` recusa o download quando o checksum diverge,
normaliza os dois cabeçalhos do workbook e persiste os dados somente em `data/`,
que permanece fora do Git.

## Evidências da EDA

A análise executada em `notebooks/02_uci_eda.py` usa o conjunto completo somente
para verificações de schema, volume e qualidade sem alvo. Relações com o alvo e
decisões de tratamento são calculadas exclusivamente na partição de treino. Ela confirmou:

- 30.000 identificadores únicos e nenhum valor nulo;
- prevalência do evento de 22,12% no treino;
- `LIMIT_BAL` entre 10.000 e 1.000.000;
- idade entre 21 e 79 anos;
- status de pagamento entre -2 e 8;
- códigos residuais 0, 5 e 6 em educação e código 0 em estado civil;
- saldos de fatura negativos, compatíveis com crédito ou estorno;
- 582 registros de treino sem fatura positiva acumulada em seis meses;
- 110 registros de treino com razão pagamento/fatura acima de 5 e máximo bruto de 797.

## Decisões de curadoria

As features do modelo são:

| Feature | Definição |
|---|---|
| `LIMIT_BAL` | Limite de crédito atual |
| `PAY_0` | Status de pagamento mais recente |
| `MAX_DELAY_6M` | Maior status de atraso nos seis meses |
| `DELAY_MONTHS_6M` | Quantidade de meses com status positivo de atraso |
| `UTILIZATION_CURRENT` | Fatura atual dividida pelo limite, limitada a [-1, 2] |
| `UTILIZATION_MEAN_6M` | Média das seis faturas dividida pelo limite, limitada a [-1, 2] |
| `PAYMENT_TO_BILL_RATIO_6M` | Pagamentos sobre faturas positivas, zero sem denominador e limitada a [0, 5] |

Educação 0, 5 e 6 é consolidada em 4 (outros). Estado civil 0 é consolidado
em 3 (outros). Sexo, educação, estado civil, idade e faixa etária ficam na
visão de auditoria e não participam do ajuste do modelo.

## Partições

As partições são determinísticas, disjuntas e estratificadas pelo alvo com
semente 42:

| Partição | Registros | Eventos | Prevalência |
|---|---:|---:|---:|
| Treino | 18.001 | 3.982 | 22,1210% |
| Validação | 4.499 | 995 | 22,1160% |
| Teste/referência | 4.501 | 996 | 22,1284% |
| Pool de simulação | 2.999 | 663 | 22,1074% |

O manifesto inclui um SHA-256 da lista ordenada de IDs de cada partição, e o
treinamento aborta se houver IDs duplicados ou sobrepostos. O treino ajusta os coeficientes. A validação escolhe somente o limiar que
maximiza F2. O teste mede a linha de base final e será a referência congelada
de monitoramento. O pool de simulação não participa do ajuste nem da escolha do
limiar.

## Baseline entregue

A versão `real-logistic-v1` é uma `Pipeline` com `StandardScaler` e
`LogisticRegression`. O caminho WOE/OptBinning planejado na ADR 0005 não está
ativo; portanto, o projeto não publica score, PDO nem tabela de pontos.

O limiar F2 escolhido na validação foi `0,1762696902`. No teste/referência:

| Métrica | Valor |
|---|---:|
| ROC-AUC | 0,7235266 |
| Average Precision | 0,5142285 |
| KS de discriminação | 0,3912283 |
| Brier score | 0,1427518 |
| F2 | 0,5595260 |
| Precisão | 0,4463616 |
| Recall | 0,5973896 |
| Taxa de revisão | 0,2961564 |

O modelo, o limiar, as métricas, a configuração resolvida, a identidade da
fonte, os hashes do modelo/código e o manifesto foram registrados no MLflow.
O mesmo run comprova que um lote inválido no pipeline real termina `REJECTED`
com zero scores e lista somente coluna/regra das violações.
