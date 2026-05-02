# Meshwatch -- Real-Time Transaction Fraud Detection with GNNs + Agentic AI

> Graph-native fraud detection in real time. A heterogeneous GNN (PyTorch
> Geometric) + XGBoost ensemble catches collusion rings tabular ML misses,
> streams transactions via Kafka, scores in <50 ms through FastAPI + Ray
> Serve, and auto-investigates alerts with a LangGraph agent. A React
> dashboard visualises the fraud network. Built on IEEE-CIS (590K transactions).

[![Phase](https://img.shields.io/badge/phase-6%2F8-blue)](./Fraud_Detection_GNN_Implementation_Plan.pdf)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()
[![Tests](https://img.shields.io/badge/tests-299%20%2B%2032-success)]()
[![Latency](https://img.shields.io/badge/predict-P95%201.6ms-success)]()
[![Agent](https://img.shields.io/badge/investigate-%3C50ms-success)]()
[![Dashboard](https://img.shields.io/badge/dashboard-Vite%206-blue)]()

Production-grade fraud detection on the **IEEE-CIS** dataset (590,540 transactions, 3.5% fraud rate)
combining a **heterogeneous GNN** (PyTorch Geometric) with an **XGBoost ensemble**, served in
real time (<50ms P95) via FastAPI + Ray Serve + Kafka, and investigated automatically by a
**LangGraph agent** backed by a local Ollama LLM. Results surface in a React.js dashboard.

> See [Fraud_Detection_GNN_Implementation_Plan.pdf](./Fraud_Detection_GNN_Implementation_Plan.pdf)
> for the complete 8-phase roadmap. Current status is in the
> [Roadmap section](#roadmap) at the bottom of this file.

---

## Quick start

```bash
# 1. Clone + enter
git clone https://github.com/ViVas970811/Meshwatch.git
cd Meshwatch

# 2. Set up environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
make install-all                   # dev + train + serve + agent + monitor

# 3. Configure Kaggle credentials
cp .env.example .env
#  pick ONE of:
#    (a) put KAGGLE_API_TOKEN=KGAT_xxxxxxxx in .env (new token format)
#    (b) put KAGGLE_USERNAME + KAGGLE_KEY in .env (classic)
#    (c) drop kaggle.json at ~/.kaggle/kaggle.json
# Then accept the competition rules at
# https://www.kaggle.com/c/ieee-fraud-detection/rules

# 4. Phase 1: download -> preprocess -> split
make download-data
make preprocess
make split
make eda                  # -> opens notebooks/01_eda.ipynb

# 5. Phase 2: heterograph + 119 engineered features
make build-graph-subset   # 100K subset, ~5 min  (recommended for local iteration)
# OR
make build-graph          # full 590K, ~25-30 min  (production run)

make feast-apply          # register 3 feature views with Feast
make graph-eda            # -> opens notebooks/02_graph.ipynb

# 6. Phase 3: train GNN + ensemble + MLflow UI
make train                # GNN training -> data/models/gnn/
make train-ensemble       # GNN+XGBoost ensemble -> data/models/ensemble/
make evaluate-ensemble    # PR/ROC/calibration on the test split
make mlflow-ui            # -> http://localhost:5000

# Notebook walkthroughs
make gnn-eda              # notebooks/03_gnn.ipynb
make ensemble-eda         # notebooks/04_ensemble.ipynb
make colab-eda            # notebooks/06_colab_training.ipynb (Colab T4)

# 7. Phase 4: real-time serving
make serve                # FastAPI on http://127.0.0.1:8000 -- /docs for OpenAPI
make demo-stream          # replay 200 txns at 5 RPS to /api/v1/predict
make compose-up           # full Docker stack: Redis + Kafka + MLflow + Prometheus + Grafana + API + Dashboard
make compose-logs         # tail logs from the API container

# 8. Phase 5: agentic investigator (LangGraph + Ollama)
make investigate          # run the agent on a synthetic HIGH alert (stub LLM)
make investigate-critical # CRITICAL alert -> all 8 tools fire
make investigate-ollama   # route through a local Ollama daemon (llama3.1:8b)
# or hit the live API:
#   curl -s http://localhost:8000/api/v1/investigate \
#        -H 'content-type: application/json' \
#        -d '{"prediction": {...}}' | jq

# 9. Phase 6: React dashboard (Vite + TanStack Query + Recharts + force-graph)
make dashboard-install    # one-time: cd dashboard && npm install
make dashboard-dev        # http://localhost:5173 (proxies /api + /ws to :8000)
make dashboard-test       # vitest run (32 tests)
make dashboard-build      # production bundle -> dashboard/dist/
# OR run dashboard alongside the rest of the stack:
make compose-up           # api + dashboard + redis + kafka + mlflow + prometheus + grafana
#                         # dashboard at http://localhost:5173, api at http://localhost:8000
```

---

## Project layout

```
Meshwatch/
├── .github/workflows/        # CI (lint + tests across Py 3.10 / 3.11 / 3.12)
├── configs/
│   ├── base.yaml             # project-wide settings
│   └── feast/                # feature store config + feature views
├── data/                     # raw / processed / splits / graphs (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb              # Phase 1 -- raw-data EDA
│   ├── 02_graph.ipynb            # Phase 2 -- graph + features EDA
│   ├── 03_gnn.ipynb              # Phase 3 -- GNN training
│   ├── 04_ensemble.ipynb         # Phase 3 -- ensemble & feature importance
│   └── 06_colab_training.ipynb   # Phase 3 -- Colab T4 production run
├── scripts/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── split_data.py
│   ├── build_graph.py            # Phase 2: heterograph + features
│   ├── train.py                  # Phase 3: GNN training + MLflow
│   ├── train_ensemble.py         # Phase 3: GNN+XGBoost ensemble
│   ├── evaluate.py               # Phase 3: PR/ROC/calibration on val or test
│   ├── serve.py                  # Phase 4: FastAPI / Ray Serve launcher
│   ├── demo_stream.py            # Phase 4: replay txns to /api/v1/predict
│   └── investigate.py            # Phase 5: run the LangGraph investigator on a synthetic alert
├── dashboard/                    # Phase 6: React 18 + TS + Vite frontend
│   ├── src/{api,components,pages,store,lib}/
│   ├── tests/                    # 32 Vitest tests
│   ├── tailwind.config.ts
│   └── README.md
├── src/fraud_detection/
│   ├── data/                 # download, preprocessing, splits, graph_builder
│   ├── features/             # temporal, aggregated, graph_features, pipeline
│   ├── models/               # hetero_gnn, gnn_layers, xgboost_model, ensemble, losses
│   ├── training/             # trainer, callbacks, evaluator
│   ├── serving/              # FastAPI app, Ray Serve deployment, schemas   (Phase 4)
│   ├── streaming/            # Kafka producer/consumer                      (Phase 4)
│   ├── agent/                # LangGraph agent, 8 tools, prompts            (Phase 5)
│   ├── monitoring/           # Evidently drift, Prometheus metrics          (Phase 7)
│   └── utils/                # config, logging, timing
├── tests/{unit,integration,e2e}/
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Development

Python (backend, agent, training):

```bash
make lint            # ruff check
make format          # ruff format + autofix
make typecheck       # mypy
make test            # pytest unit tests (299 tests, ~50 s)
make test-cov        # unit tests with coverage report
```

TypeScript (dashboard):

```bash
make dashboard-lint  # tsc -b (composite project type-check)
make dashboard-test  # vitest run (32 tests)
make dashboard-build # production Vite bundle -> dashboard/dist/
```

CI runs the full Python matrix (Python 3.10 / 3.11 / 3.12) on every push + PR
via `.github/workflows/ci.yml`. The dashboard test suite is run locally; a
dedicated CI job lands in Phase 7 alongside the MLOps work.

---

## Roadmap

| Phase | Tag | Status |
| :-- | :-- | :-- |
| 1. Foundation & Data Pipeline | `v0.1.0-data-foundation` | ✅ Complete |
| 2. Graph & Features | `v0.2.0-graph-engine` | ✅ Complete |
| 3. GNN Model & Training | `v0.3.0-gnn-model` | ✅ Complete |
| 4. Real-Time Serving | `v0.4.0-serving-pipeline` | ✅ Complete |
| 5. Agentic Investigator | `v0.5.0-agent-investigator` | ✅ Complete |
| 6. React Dashboard | `v0.6.0-dashboard` | ✅ Complete |
| 7. MLOps & Monitoring | `v0.7.0-mlops` | 🚧 Up next |
| 8. Docs, Demo & Polish | `v1.0.0-release` | ⏳ Planned |

## License

MIT -- see [LICENSE](LICENSE).
