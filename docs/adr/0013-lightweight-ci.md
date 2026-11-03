# ADR 0013: Manter CI pequena e determinística

- Status: aceito
- Data: 2026-10-23

## Contexto

Uma GitHub Action fornece evidência de qualidade no repositório, mas baixar o dataset, resolver o scorecard e gerar todos os relatórios em cada push tornaria a CI lenta e dependente de serviços externos.

## Decisão

A CI executará somente lint e testes rápidos com fixtures locais pequenas. A demonstração completa continuará disponível por `make demo` e será executada localmente.

Os testes mínimos cobrirão:

- aceitação de lote válido e quarentena de lote inválido;
- proibição do alvo no scoring e validação do feed de desfechos;
- associação um-para-um entre predição e desfecho, incluindo duplicatas e ausências;
- ausência de atributos proibidos na visão de scoring;
- ajuste de binning somente no treino;
- reprodutibilidade da simulação com semente fixa;
- direção consistente entre score e probabilidade de inadimplência;
- rejeição de YAML com campos ausentes ou desconhecidos e resolução determinística de defaults;
- repetibilidade de manifestos, features e predições sob a mesma semente;
- preservação de nomes, ordem e dtypes na fronteira Polars para pandas/NumPy;
- política de alertas e estados do lote;
- geração dos artefatos esperados em um smoke test reduzido.

## Consequências

- A CI não dependerá da UCI, de servidor MLflow nem de navegador.
- Artefatos completos serão verificados por uma checklist de entrega e não a cada commit.
- `pytest` será o executor de testes e `ruff` cuidará de lint e formatação.
