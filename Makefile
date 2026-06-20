# wc2026-predictor — common tasks.
# All Python tasks run inside the project-local virtualenv (.venv).

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
WC2026 := $(VENV)/bin/wc2026

.DEFAULT_GOAL := help

.PHONY: help setup lock data train evaluate predict simulate test lint format typecheck clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

lock: ## Compile requirements.lock from pyproject.toml (pinned).
	$(VENV)/bin/pip-compile --extra dev --output-file requirements.lock pyproject.toml

setup: ## Sync the venv to the lock file and install the package (editable).
	$(VENV)/bin/pip-sync requirements.lock
	$(PIP) install -e .

data: ## Download/refresh sources and build processed datasets (prompts before network calls).
	$(WC2026) data

train: ## Train, evaluate, and persist both models (baseline + Bayesian).
	$(WC2026) train --model baseline
	$(WC2026) train --model bayesian

evaluate: ## Run the temporal backtest and print metrics.
	$(WC2026) evaluate

predict: ## Predict a single match, e.g. make predict ARGS="--home Ecuador --away Argentina".
	$(WC2026) predict $(ARGS)

simulate: ## Monte Carlo tournament simulation, e.g. make simulate ARGS="--n 10000".
	$(WC2026) simulate $(ARGS)

test: ## Run the test suite.
	$(VENV)/bin/pytest

lint: ## Lint with ruff.
	$(VENV)/bin/ruff check src tests

format: ## Auto-format with ruff.
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

typecheck: ## Type-check src with mypy.
	$(VENV)/bin/mypy

clean: ## Remove caches and build artifacts (keeps the venv and data).
	rm -rf build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
