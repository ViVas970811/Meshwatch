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
        clean clean-data clean-all \
        tag-phase-1

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
