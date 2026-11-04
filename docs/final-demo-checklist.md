# Checklist da demonstração final

## Antes da gravação

- [ ] Executar `make setup` em Python 3.12.
- [ ] Executar `make check` e confirmar lint, 57 testes e quatro notebooks válidos.
- [ ] Executar `make demo` e confirmar runs D4, E4, E5 e E6 no MLflow.
- [ ] Abrir `make mlflow-ui` e localizar os runs apontados pelos arquivos `latest-run-id.txt`.
- [ ] Separar os relatórios agregados que serão mostrados; não abrir dados individuais.
- [ ] Confirmar que o link do vídeo foi adicionado ao README após a gravação.

## Sequência STAR

- [ ] Situação e atraso de um mês do desfecho em até 30 segundos.
- [ ] Lote corrompido em `REJECTED` e zero scores.
- [ ] M2 com Data Drift e M3 com Concept Drift maturado.
- [ ] Fairlearn agregado e decisão `MONITORING_CONTINUES`.
- [ ] Runs, métricas, logs e artefatos na interface do MLflow.
- [ ] Limites causais, LGPD, revisão humana e ausência de automação de retreino.

## Segurança da apresentação

- [ ] Duração máxima de cinco minutos, alvo de 4min30s.
- [ ] Nenhum ID, linha individual, caminho pessoal ou tela com dado bruto.
- [ ] Artefatos abertos previamente e versão de backup disponível.
- [ ] Afirmações conferidas contra `docs/requirements-traceability.md`.
