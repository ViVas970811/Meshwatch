# ---------------------------------------------------------------------------
# Fraud Detection GNN -- developer Makefile
# ---------------------------------------------------------------------------
# Works with GNU make on Linux/macOS and mingw32-make on Windows.
# All targets assume the active venv already has the package installed.
# ---------------------------------------------------------------------------

PY       ?= python
PIP      ?= $(PY) -m pip
PYTEST   ?= $(PY) -m pytest
RUFF     ?= $(PY) -m ruff
MYPY     ?= $(PY) -m mypy

SRC_DIR  := src/fraud_detection
TEST_DIR := tests

.PHONY: help install install-dev install-all \
        lint format typecheck test test-unit test-integration test-cov \
        download-data preprocess split eda \
        build-graph build-graph-subset feast-apply graph-eda \
        train train-ensemble evaluate-gnn evaluate-ensemble \
        gnn-eda ensemble-eda colab-eda mlflow-ui \
        clean clean-data clean-all \
        tag-phase-1 tag-phase-2 tag-phase-3

help: ## Show this help
	@echo "Fraud Detection GNN -- common targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install: ## Install base (runtime) dependencies
	$(PIP) install -e .

install-dev: ## Install base + dev tooling
	$(PIP) install -e ".[dev]"

install-all: ## Install everything (dev + train + serve + agent + monitor)
	$(PIP) install -e ".[all]"

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
lint: ## Run ruff lint
	$(RUFF) check $(SRC_DIR) $(TEST_DIR) scripts

format: ## Auto-format with ruff
	$(RUFF) format $(SRC_DIR) $(TEST_DIR) scripts
	$(RUFF) check --fix $(SRC_DIR) $(TEST_DIR) scripts

typecheck: ## Run mypy
	$(MYPY) $(SRC_DIR)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test: test-unit ## Alias for unit tests

test-unit: ## Run unit tests
	$(PYTEST) $(TEST_DIR)/unit -v

test-integration: ## Run integration tests (requires docker services)
	$(PYTEST) $(TEST_DIR)/integration -v -m integration

test-cov: ## Run unit tests with coverage
	$(PYTEST) $(TEST_DIR)/unit -v --cov --cov-report=term-missing --cov-report=html

# ---------------------------------------------------------------------------
# Data pipeline (Phase 1)
# ---------------------------------------------------------------------------
download-data: ## Download IEEE-CIS dataset from Kaggle
	$(PY) scripts/download_data.py

preprocess: ## Run preprocessing on raw data
	$(PY) scripts/preprocess.py

split: ## Create temporal train/val/test splits
	$(PY) scripts/split_data.py

eda: ## Launch EDA notebook
	$(PY) -m jupyter notebook notebooks/01_eda.ipynb

# ---------------------------------------------------------------------------
# Graph + features (Phase 2)
# ---------------------------------------------------------------------------
build-graph: ## Build HeteroData + 119 engineered features (writes data/graphs/)
	$(PY) scripts/build_graph.py

build-graph-subset: ## Same as build-graph but on a 100K-row subset (dev mode)
	$(PY) scripts/build_graph.py --nrows 100000

feast-apply: ## Apply Feast feature views from configs/feast/features.py
	cd configs/feast && feast apply

graph-eda: ## Launch Phase 2 graph EDA notebook
	$(PY) -m jupyter notebook notebooks/02_graph.ipynb

# ---------------------------------------------------------------------------
# Model + training (Phase 3)
# ---------------------------------------------------------------------------
train: ## Train the heterogeneous GNN (writes data/models/gnn/)
	$(PY) scripts/train.py

train-ensemble: ## Train the GNN+XGBoost ensemble (needs `make train` first)
	$(PY) scripts/train_ensemble.py

evaluate-gnn: ## Evaluate the GNN-only model on the test split
	$(PY) scripts/evaluate.py --model-dir data/models/gnn --split test

evaluate-ensemble: ## Evaluate the ensemble on the test split
	$(PY) scripts/evaluate.py --model-dir data/models/ensemble --split test

gnn-eda: ## Launch Phase 3 GNN training notebook
	$(PY) -m jupyter notebook notebooks/03_gnn.ipynb

ensemble-eda: ## Launch Phase 3 ensemble notebook
	$(PY) -m jupyter notebook notebooks/04_ensemble.ipynb

colab-eda: ## Launch the Colab GPU training notebook
	$(PY) -m jupyter notebook notebooks/06_colab_training.ipynb

mlflow-ui: ## Launch the MLflow UI (http://localhost:5000) -- reads ./mlflow.db (sqlite default)
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-data: ## Remove processed data (raw/ is preserved)
	rm -rf data/processed/* data/splits/* data/graphs/*

clean-all: clean clean-data ## Remove everything (including raw data)
	rm -rf data/raw/*

# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------
tag-phase-1: ## Tag v0.1.0-data-foundation
	git tag -a v0.1.0-data-foundation -m "Phase 1: Foundation & Data Pipeline"
	@echo "Tag created. Push with: git push origin v0.1.0-data-foundation"

tag-phase-2: ## Tag v0.2.0-graph-engine
	git tag -a v0.2.0-graph-engine -m "Phase 2: Graph & Features"
	@echo "Tag created. Push with: git push origin v0.2.0-graph-engine"

tag-phase-3: ## Tag v0.3.0-gnn-model
	git tag -a v0.3.0-gnn-model -m "Phase 3: GNN Model & Training"
	@echo "Tag created. Push with: git push origin v0.3.0-gnn-model"
