.PHONY: setup lint format test notebook-check check demo notebook mlflow-ui

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

check: lint test notebook-check

demo:
	uv run python scripts/run_d4_demo.py

notebook:
	uv run --group notebook marimo edit notebooks/01_d4_synthetic_eda.py

mlflow-ui:
	uv run mlflow server --backend-store-uri sqlite:///$(abspath artifacts/runtime/mlflow.db) --default-artifact-root file://$(abspath artifacts/runtime/mlflow-artifacts) --host 127.0.0.1 --port 5000
