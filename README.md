# Meshwatch -- Real-Time Transaction Fraud Detection with GNNs + Agentic AI

> Graph-native fraud detection in real time. A heterogeneous GNN (PyTorch
> Geometric) + XGBoost ensemble catches collusion rings tabular ML misses,
> streams transactions via Kafka, scores in <50 ms through FastAPI + Ray
> Serve, and auto-investigates alerts with a LangGraph agent. A React
> dashboard visualises the fraud network. Built on IEEE-CIS (590K transactions).

[![Phase](https://img.shields.io/badge/phase-1%2F8-blue)](./Fraud_Detection_GNN_Implementation_Plan.pdf)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

Production-grade fraud detection on the **IEEE-CIS** dataset (590K transactions, 3.5% fraud rate)
combining a **heterogeneous GNN** (PyTorch Geometric) with an **XGBoost ensemble**, served in
real time (<50ms P95) via FastAPI + Ray Serve + Kafka, and investigated automatically by a
**LangGraph agent** backed by a local Ollama LLM. Results surface in a React.js dashboard.

> This repository is currently at **Phase 1: Foundation & Data Pipeline (v0.1.0-data-foundation)**.
> See [Fraud_Detection_GNN_Implementation_Plan.pdf](../Fraud_Detection_GNN_Implementation_Plan.pdf)
> for the complete 8-phase roadmap.

---

## Phase 1 deliverables

| Area | File | Purpose |
| --- | --- | --- |
| Scaffolding | `pyproject.toml`, `Makefile`, `.gitignore`, `.env.example` | Reproducible environment, hatchling build with `dev`/`train`/`serve`/`agent` groups |
| Config | [`src/fraud_detection/utils/config.py`](src/fraud_detection/utils/config.py) | Pydantic `AppConfig` loading YAML + env vars |
| Logging | [`src/fraud_detection/utils/logging.py`](src/fraud_detection/utils/logging.py) | Structured JSON logs via `structlog` |
| Download | [`src/fraud_detection/data/download.py`](src/fraud_detection/data/download.py) | Kaggle CLI wrapper for `ieee-fraud-detection` |
| Preprocess | [`src/fraud_detection/data/preprocessing.py`](src/fraud_detection/data/preprocessing.py) | `IEEECISPreprocessor` -- missing-value strategy per column group |
| Splits | [`src/fraud_detection/data/splits.py`](src/fraud_detection/data/splits.py) | Chronological `TemporalSplitter` (60/20/20, no leakage) |
| EDA | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Class balance, missingness, temporal fraud rate |

---

## Quick start

```bash
# 1. Clone & enter
git clone https://github.com/ViVas970811/fraud-detection-gnn.git
cd fraud-detection-gnn

# 2. Set up environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
make install-dev

# 3. Configure Kaggle credentials
cp .env.example .env
# edit .env -- add KAGGLE_USERNAME and KAGGLE_KEY
#   OR place kaggle.json at ~/.kaggle/kaggle.json (chmod 600)

# 4. Download + preprocess + split
make download-data
make preprocess
make split

# 5. Explore
make eda
```

---

## Project layout

```
fraud-detection-gnn/
├── .github/workflows/        # CI (Phase 7)
├── configs/                  # YAML configs (base/train/serve/feast/kafka)
├── data/                     # Raw, processed, splits, graphs (gitignored)
├── notebooks/                # 01_eda -> 06_colab
├── scripts/                  # download_data, preprocess, split_data, train, ...
├── src/fraud_detection/
│   ├── data/                 # download, preprocessing, splits, graph_builder
│   ├── features/             # temporal, aggregated, graph_features, pipeline
│   ├── models/               # hetero_gnn, xgboost_model, ensemble, losses
│   ├── training/             # trainer, callbacks, evaluator, hyperopt
│   ├── serving/              # FastAPI app, Ray Serve deployment, schemas
│   ├── streaming/            # Kafka producer/consumer
│   ├── agent/                # LangGraph agent, 8 tools, prompts
│   ├── monitoring/           # Evidently drift, Prometheus metrics
│   └── utils/                # config, logging, timing
├── tests/{unit,integration,e2e}/
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Development

```bash
make lint         # ruff
make format       # ruff format + fix
make typecheck    # mypy
make test         # pytest unit tests
make test-cov     # unit tests + coverage report
```

---

## Roadmap

| Phase | Tag | Status |
| :-- | :-- | :-- |
| 1. Foundation & Data Pipeline | `v0.1.0-data-foundation` | In progress |
| 2. Graph & Features | `v0.2.0-graph-engine` | Planned |
| 3. GNN Model & Training | `v0.3.0-gnn-model` | Planned |
| 4. Real-Time Serving | `v0.4.0-serving-pipeline` | Planned |
| 5. Agentic Investigator | `v0.5.0-agent-investigator` | Planned |
| 6. React Dashboard | `v0.6.0-dashboard` | Planned |
| 7. MLOps & Monitoring | `v0.7.0-mlops` | Planned |
| 8. Docs, Demo & Polish | `v1.0.0-release` | Planned |

## License

MIT -- see [LICENSE](LICENSE).
