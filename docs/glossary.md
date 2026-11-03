# Glossário

## Termos do domínio

**Analista de crédito**: pessoa que usa a estimativa de risco como um dos insumos para revisar ou renovar o limite de um cliente existente.

**Avaliação de risco**: processo que transforma os dados validados de um cliente em probabilidade e faixa de risco de inadimplência.

**Bin**: intervalo ou grupo de valores de uma variável ao qual o scorecard atribui um WOE e uma quantidade de pontos.

**Desfecho**: rótulo observado após a janela de previsão, indicando se houve inadimplência no mês seguinte.

**Cliente de crédito**: titular representado por um registro do dataset. Não equivale, necessariamente, a um novo solicitante de empréstimo.

**Inadimplência**: não pagamento no mês seguinte, representado pelo alvo binário do dataset.

**Faixa de risco**: categoria derivada da probabilidade estimada de inadimplência. Seus limites ainda serão definidos.

**Falso negativo**: cliente que inadimple, mas não é sinalizado como alto risco pelo limiar do modelo.

**Falso positivo**: cliente que não inadimple, mas é sinalizado como alto risco e encaminhado à revisão reforçada.

**Revisão humana reforçada**: análise conceitualmente solicitada para avaliações de alto risco. No MVP, será representada por `review_required`; a execução da revisão fica fora do escopo.

**Scorecard**: modelo aditivo que transforma os bins das variáveis em pontos. A soma dos pontos representa o risco estimado.

## Termos técnicos

**Concept drift**: mudança na relação entre os atributos de entrada e o desfecho; pode degradar o modelo mesmo sem grande mudança na distribuição isolada dos atributos.

**Choque econômico sintético**: intervenção controlada na simulação que altera a probabilidade de inadimplência de determinados perfis. Não representa evidência causal obtida de dados reais.

**Contrato de dados**: conjunto executável de regras de esquema e qualidade que um lote deve satisfazer antes de seguir no pipeline.

**Contrafactual sintético**: desfecho que o gerador atribuiria ao mesmo registro sem a intervenção simulada. Só tem validade dentro das regras da simulação.

**PDO (Points to Double the Odds)**: quantidade de pontos necessária para dobrar as odds representadas pelo score. Altera a escala, não a qualidade preditiva.

**Population Stability Index (PSI)**: índice de mudança entre distribuições segmentadas. No projeto, será usado para atributos do modelo e score.

**Data drift**: mudança na distribuição dos dados de entrada entre uma população de referência e uma população de análise.

**DAG**: grafo direcionado acíclico usado para declarar uma hipótese sobre relações causais e possíveis fontes de confundimento ou seleção.

**Dataset de análise/produção**: lote posterior usado para simular operação e no qual podem ocorrer problemas de qualidade ou drift.

**Dataset de referência de monitoramento**: partição histórica limpa, mantida fora do ajuste, usada para estabelecer a linha de base final e comparar drift após o congelamento do modelo.

**Lote**: conjunto de registros processado como uma unidade para validação, predição e monitoramento.

**Lote maturado**: lote cujos desfechos já estão disponíveis e podem ser associados às predições anteriores.

**Lote pontuado**: lote validado que já recebeu predições, mas cujo período de observação do desfecho ainda não terminou.

**OOT sintético**: safra futura simulada depois do congelamento do modelo. Não equivale a uma coorte temporal observada no dataset original.

**Observabilidade**: capacidade de entender a saúde do pipeline e do modelo por meio de logs, métricas, relatórios e alertas.

**PII**: informação que identifica ou pode tornar identificável uma pessoa natural. O identificador do dataset não será usado como atributo preditivo.

**Visão de auditoria**: projeção restrita que relaciona grupos demográficos, predições e desfechos para medir disparidades.

**Visão de scoring**: projeção de atributos financeiros e comportamentais autorizados para entrada do modelo.

**Weight of Evidence (WOE)**: transformação que representa, em cada bin, a relação entre eventos e não eventos. Os bins e seus valores devem ser aprendidos somente no treino.

**KS de discriminação**: maior separação entre as distribuições acumuladas de score dos eventos e não eventos em um lote rotulado.

**KS de duas amostras**: teste que compara duas distribuições empíricas, como uma feature na referência e em produção.
