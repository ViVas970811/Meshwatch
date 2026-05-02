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
        serve serve-ray serve-dev demo-stream demo-stream-batch \
        compose-up compose-down compose-logs docker-build-serving \
        investigate investigate-critical investigate-ollama \
        dashboard-install dashboard-dev dashboard-build dashboard-test dashboard-lint dashboard-preview \
        docker-build-dashboard \
        clean clean-data clean-all \
        tag-phase-1 tag-phase-2 tag-phase-3 tag-phase-4 tag-phase-5 tag-phase-6

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
# Serving (Phase 4)
# ---------------------------------------------------------------------------
serve: ## Start FastAPI on http://127.0.0.1:8000 (single worker, prod-style)
	$(PY) scripts/serve.py

serve-dev: ## Start FastAPI with auto-reload (dev mode)
	$(PY) scripts/serve.py --reload

serve-ray: ## Start the API behind Ray Serve (multi-replica)
	$(PY) scripts/serve.py --ray --num-replicas 2

demo-stream: ## Replay 200 transactions at 5 RPS to /api/v1/predict
	$(PY) scripts/demo_stream.py --n 200 --rps 5

demo-stream-batch: ## Stress test: 2000 txns @ 50 RPS via /api/v1/predict/batch
	$(PY) scripts/demo_stream.py --n 2000 --rps 50 --batch 50 --summary-only

# ---------------------------------------------------------------------------
# Docker Compose (Phase 4 infra)
# ---------------------------------------------------------------------------
compose-up: ## Bring up Redis + Kafka + MLflow + Prometheus + Grafana + API
	docker compose up -d

compose-down: ## Tear down compose stack (keeps named volumes)
	docker compose down

compose-logs: ## Tail logs from the API container
	docker compose logs -f api

docker-build-serving: ## Build the API image locally
	docker build -t meshwatch/api:dev -f Dockerfile.serving .

# ---------------------------------------------------------------------------
# Agentic Investigator (Phase 5)
# ---------------------------------------------------------------------------
investigate: ## Run the LangGraph investigator on a synthetic HIGH alert (stub LLM)
	$(PY) scripts/investigate.py --score 0.85 --amount 420

investigate-critical: ## Run the investigator on a synthetic CRITICAL alert
	$(PY) scripts/investigate.py --score 0.95 --amount 4210

investigate-ollama: ## Same as `investigate` but routes through a local Ollama daemon
	$(PY) scripts/investigate.py --score 0.95 --amount 4210 --use-ollama

# ---------------------------------------------------------------------------
# React Dashboard (Phase 6)
# ---------------------------------------------------------------------------
dashboard-install: ## Install dashboard npm dependencies
	cd dashboard && npm install

dashboard-dev: ## Run the Vite dev server on http://localhost:5173 (proxies /api -> :8000)
	cd dashboard && npm run dev

dashboard-build: ## Production build (Vite -> dashboard/dist)
	cd dashboard && npm run build

dashboard-preview: ## Serve the built dist/ locally
	cd dashboard && npm run preview

dashboard-test: ## Run the Vitest suite (32 tests)
	cd dashboard && npm test

dashboard-lint: ## Type-check the dashboard
	cd dashboard && npx tsc -b

docker-build-dashboard: ## Build the dashboard nginx image locally
	docker build -t meshwatch/dashboard:dev -f Dockerfile.dashboard .

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

tag-phase-4: ## Tag v0.4.0-serving-pipeline
	git tag -a v0.4.0-serving-pipeline -m "Phase 4: Real-Time Serving Pipeline"
	@echo "Tag created. Push with: git push origin v0.4.0-serving-pipeline"

tag-phase-5: ## Tag v0.5.0-agent-investigator
	git tag -a v0.5.0-agent-investigator -m "Phase 5: Agentic AI Investigator"
	@echo "Tag created. Push with: git push origin v0.5.0-agent-investigator"

tag-phase-6: ## Tag v0.6.0-dashboard
	git tag -a v0.6.0-dashboard -m "Phase 6: React Dashboard"
	@echo "Tag created. Push with: git push origin v0.6.0-dashboard"
