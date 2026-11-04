# Plano de governança e LGPD

## Escopo e papéis

No cenário hipotético, a fintech é a controladora e define as finalidades do scoring e do monitoramento. Equipes de dados, risco e auditoria atuam com acessos distintos. O projeto acadêmico usa um dataset público como representação e não opera decisões reais de crédito.

## Base legal e finalidade

- Finalidade principal: estimar risco de inadimplência na carteira existente para apoiar monitoramento, revisão ou renovação de limite.
- Base legal assumida: proteção do crédito, art. 7º, X da LGPD.
- Finalidade secundária restrita: usar sexo e faixa etária para auditoria de disparidades, sem incluí-los no scorecard. Compatibilidade, necessidade, base legal específica e balanceamento dessa finalidade deverão ser validados pelo encarregado antes de produção.
- A saída de alto risco solicita revisão reforçada; não existe negativa automática.

## Controles implementados no projeto

- `ID` não participa do treinamento nem do score.
- A visão de scoring exclui atributos demográficos.
- Logs e métricas não armazenam linhas de clientes nem atributos demográficos individuais.
- Dados baixados e artefatos de linha permanecem fora do Git.
- Relatórios selecionados para entrega contêm somente estatísticas agregadas.
- Sementes, contrato e regras da simulação são versionados; runs registram hashes do dataset, modelo e árvore de código executada.
- Um procedimento de limpeza removerá dados e artefatos locais reproduzíveis após a avaliação acadêmica.
- A auditoria E6 mantém a visão individual restrita somente em memória e registra no MLflow contagens, métricas de erro, intervalos e decisão do playbook agregados.
- A regra operacional suprime métricas de erro para grupos com menos de 200 registros, 30 eventos ou 30 não eventos.
- O pickle do modelo é aceito somente como artefato local confiável gerado pela E4; produção exigiria formato e verificação de integridade aprovados antes da carga.

## Controles prescritos para produção

- Criptografia em trânsito e em repouso.
- Controle de acesso por função para separar scoring, análise de crédito e auditoria de equidade.
- Cofre de chaves e pseudonimização do identificador operacional.
- Registro de acesso e trilha imutável de versões do contrato, modelo, score e decisão humana.
- Avaliação de impacto à proteção de dados antes da implantação.
- Canal para correção de dados, explicação, contestação e revisão de decisões.
- Processo para atender direitos aplicáveis do art. 18, com autenticação do titular e rastreio da resposta.
- Processo formal de aprovação, rollback e retreino; alertas não promovem modelos automaticamente.
- Donos de acesso definidos para scoring, auditoria e exclusão, com revisão periódica de permissões.

## Retenção proposta para produção

- Lote em quarentena: até 30 dias para correção e investigação, seguido de exclusão segura.
- Associação individual entre score e desfecho: 13 meses, cobrindo a janela de um mês e um ciclo anual de monitoramento.
- Visão restrita de auditoria: mesma janela necessária à análise, seguida de anonimização ou exclusão.
- Métricas agregadas, contratos, scorecards e relatórios de validação: durante a vida do modelo e por cinco anos adicionais para auditoria interna.
- Exclusões devem gerar evidência verificável; bloqueios legais de descarte precisam de justificativa, responsável e prazo de revisão.

Esses prazos são uma proposta arquitetural. Obrigações regulatórias, defesa de direitos e políticas institucionais podem exigir ajustes aprovados pelo encarregado e pela assessoria jurídica.

## Riscos e limitações

- O dataset representa clientes de cartão de crédito em Taiwan em 2005 e não garante validade para uma fintech brasileira atual.
- A carteira observada exclui pessoas que não receberam crédito, criando viés de seleção.
- Métricas de grupo não provam ausência ou presença de discriminação causal.
- A simulação econômica não estima o efeito real de inflação ou desemprego.
- O projeto não implementa autenticação, autorização, criptografia ou atendimento ao titular; esses itens pertencem à camada prescrita.
- Os desfechos M3 são sintéticos e a auditoria E6 não autoriza conclusão sobre impacto real, discriminação causal ou conformidade legal.
