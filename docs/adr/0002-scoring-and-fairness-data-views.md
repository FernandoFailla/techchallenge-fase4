# ADR 0002: Separar dados de scoring e de auditoria de equidade

- Status: aceito
- Data: 2026-10-23

## Contexto

O dataset contém `ID`, `SEX`, `EDUCATION`, `MARRIAGE` e `AGE`. Usar atributos demográficos no modelo pode criar ou ampliar tratamento desigual. Excluí-los de todo o processo, por outro lado, impede a medição de disparidades entre grupos.

## Decisão

O pipeline produzirá duas visões com acessos e finalidades diferentes:

- A visão de scoring conterá apenas atributos financeiros e comportamentais.
- A visão de auditoria conterá a predição, o desfecho quando disponível e os grupos derivados de `SEX` e `AGE`.

`ID`, `SEX`, `EDUCATION`, `MARRIAGE` e `AGE` não serão atributos do modelo. `ID` será usado apenas como chave técnica pseudonimizada para relacionar predição e desfecho na simulação. `EDUCATION` e `MARRIAGE` não serão necessários na auditoria inicial.

O Fairlearn será usado para avaliação desagregada. O uso inicial será de `MetricFrame` e métricas por grupo; nenhuma métrica isolada será apresentada como prova de justiça ou conformidade legal.

## Consequências

- O contrato de entrada ainda deve validar os campos antes da separação das visões.
- A documentação deve declarar a finalidade específica dos atributos demográficos mantidos para auditoria.
- A visão de auditoria terá retenção e acesso mais restritos que os artefatos agregados de monitoramento.
- A eventual mitigação de disparidades será uma decisão posterior à avaliação, não uma otimização automática assumida desde o início.
