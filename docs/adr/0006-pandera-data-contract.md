# ADR 0006: Implementar o contrato de dados com Pandera

- Status: aceito
- Data: 2026-10-23

## Contexto

O projeto precisa demonstrar pelo menos três regras rígidas de qualidade e bloquear a ingestão de um lote com erros. Great Expectations forneceria documentação visual própria, mas adicionaria uma segunda estrutura de configuração; o projeto já produzirá relatórios HTML com Evidently.

## Decisão

O contrato será implementado com o backend Polars do Pandera. A validação ocorrerá antes da separação entre scoring e auditoria e acumulará as falhas do lote para gerar um relatório útil. Conversões para pandas ou NumPy ocorrerão somente nas fronteiras de bibliotecas que não aceitem Polars.

Haverá três schemas:

- dados históricos rotulados, com alvo binário obrigatório;
- scoring, com features obrigatórias e alvo proibido;
- desfechos, com chave, alvo e `available_at`.

O contrato inicial cobrirá, no mínimo:

- esquema estrito de colunas e tipos;
- ausência de nulos nos campos obrigatórios;
- unicidade de `ID` dentro de cada lote;
- `AGE` dentro de um intervalo adulto plausível;
- `LIMIT_BAL` estritamente positivo;
- alvo binário nos dados rotulados e ausência do alvo no scoring;
- associação um-para-um entre predição e desfecho maturado.

Os limites e conjuntos de valores serão fechados após o perfilamento do arquivo original. O lote inválido de demonstração terá anomalias intencionais e determinísticas, como nulo, idade impossível, limite inválido e identificador duplicado.

## Consequências

- Qualquer violação rejeitará o lote inteiro; o MVP não corrigirá nem descartará linhas silenciosamente.
- O erro estruturado e a quantidade de violações serão registrados como evidência operacional.
- O contrato será versionado e testado com lotes válidos e inválidos.
- Regras de qualidade não serão usadas para detectar drift; contratos e drift têm finalidades diferentes.
- Testes cobrirão desfechos duplicados, ausentes, prematuros e sem predição correspondente.
