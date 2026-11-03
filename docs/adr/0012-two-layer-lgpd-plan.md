# ADR 0012: Documentar LGPD em duas camadas

- Status: aceito
- Data: 2026-10-23

## Contexto

O repositório é uma demonstração educacional com dataset público, mas o enunciado pede um plano de proteção aplicável a uma fintech. Descrever somente o ambiente local seria superficial; descrever controles hipotéticos como se estivessem implementados seria enganoso.

## Decisão

O plano de governança separará:

1. Controles implementados e verificáveis no projeto.
2. Controles prescritos para uma implantação real e ainda não implementados.

A base legal assumida para o cenário brasileiro será proteção do crédito, conforme o art. 7º, X da LGPD. A decisão final não será exclusivamente automatizada, mas o plano ainda preverá transparência, explicação e contestação.

O uso restrito de sexo e faixa etária para auditoria será tratado como finalidade secundária que exige análise documentada de compatibilidade, necessidade, salvaguardas e base legal pelo encarregado; o projeto não presumirá que a base do scoring resolve automaticamente esse uso.

Prazos de retenção da camada de produção serão apresentados como política proposta e sujeitos à validação do encarregado e da assessoria jurídica.

## Consequências

- O README não poderá afirmar que controles prescritos foram implementados.
- A origem taiwanesa e histórica do dataset será tratada como limitação de representatividade.
- Sexo e idade serão tratados como dados pessoais com risco discriminatório, sem classificá-los automaticamente como dados pessoais sensíveis segundo a LGPD.
- Atribuição à UCI e a licença CC BY 4.0 serão documentadas.
