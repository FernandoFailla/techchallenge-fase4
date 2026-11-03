# ADR 0001: Escopo do projeto e enquadramento do domínio

- Status: aceito
- Data: 2026-10-23

## Contexto

O Tech Challenge pede uma demonstração integrada de validação de dados, detecção de drift, observabilidade e governança para um modelo de risco de crédito. O projeto também precisa ser compreensível, reproduzível e demonstrável em um vídeo de até cinco minutos.

O dataset escolhido, [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), contém 30.000 registros de clientes de cartão de crédito em Taiwan, 23 atributos explicativos e o indicador de inadimplência no mês seguinte. Ele não contém pedidos de empréstimo nem decisões históricas de aprovação de crédito.

## Decisão

Construiremos um MVP didático e extensível que cubra integralmente a rubrica e possa ser executado localmente.

O modelo estimará a probabilidade de inadimplência no mês seguinte e produzirá uma faixa de risco. Como o dataset representa clientes de cartão já existentes, essa saída apoiará monitoramento da carteira e revisão ou renovação de limite; ela não será tratada como decisão final automática nem como modelo de originação de novos clientes.

Usaremos o dataset Default of Credit Card Clients, publicado pela UCI sob licença CC BY 4.0, como fonte dos dados de referência e das simulações de produção.

## Consequências

- O projeto deve separar a estimativa de risco da decisão de crédito.
- A documentação e o vídeo não devem afirmar que o dataset contém pedidos de crédito ou aprovações reais.
- Concessão para novos clientes será citada apenas como extrapolação fora do escopo dos dados.
- A revisão humana e a possibilidade de contestação farão parte do desenho de governança.
- A arquitetura será pequena e local, mas seus componentes terão responsabilidades separadas para permitir evolução.
- A atribuição ao autor e à UCI deverá constar na documentação final.

## Alternativas rejeitadas

- German Credit: possui cerca de 1.000 registros e não atende ao mínimo de 5.000 amostras indicado no enunciado.
- Home Credit e Lending Club: oferecem maior riqueza, mas aumentam substancialmente a preparação dos dados e a dependência de fontes externas.
- Decisão totalmente automática: não é necessária para cumprir a rubrica e ampliaria o risco regulatório sem benefício didático proporcional.
