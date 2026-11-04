.PHONY: setup lint format test notebook-check check demo d4-demo prepare-data train-real e4 e5 e6 evidence notebook notebook-real notebook-drift notebook-fairness mlflow-ui

setup:
	uv sync --frozen --group dev --group notebook

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

notebook-check:
	uv run --group notebook marimo check notebooks/01_d4_synthetic_eda.py
	uv run --group notebook marimo check notebooks/02_uci_eda.py
	uv run --group notebook marimo check notebooks/03_drift_scenarios.py
	uv run --group notebook marimo check notebooks/04_fairness_audit.py

check: lint test notebook-check

demo: d4-demo
	$(MAKE) e5
	uv run python scripts/run_e6_fairness.py
	uv run python scripts/export_evidence.py

d4-demo:
	uv run python scripts/run_d4_demo.py

prepare-data:
	uv run python scripts/prepare_data.py

train-real: prepare-data
	uv run python scripts/train_real_model.py

e4: train-real

e5: e4
	uv run python scripts/run_e5_drift.py

e6: e4
	uv run python scripts/run_e6_fairness.py

evidence:
	uv run python scripts/export_evidence.py

notebook:
	uv run --group notebook marimo edit notebooks/01_d4_synthetic_eda.py

notebook-real:
	uv run --group notebook marimo edit notebooks/02_uci_eda.py

notebook-drift:
	uv run --group notebook marimo edit notebooks/03_drift_scenarios.py

notebook-fairness:
	uv run --group notebook marimo edit notebooks/04_fairness_audit.py

mlflow-ui:
	uv run mlflow server --backend-store-uri sqlite:///$(abspath artifacts/runtime/mlflow.db) --default-artifact-root file://$(abspath artifacts/runtime/mlflow-artifacts) --host 127.0.0.1 --port 5000
