# Meshwatch -- Real-Time Transaction Fraud Detection with GNNs + Agentic AI

> Graph-native fraud detection in real time. A heterogeneous GNN (PyTorch
> Geometric) + XGBoost ensemble catches collusion rings tabular ML misses,
> streams transactions via Kafka, scores in <50 ms through FastAPI + Ray
> Serve, and auto-investigates alerts with a LangGraph agent. A React
> dashboard visualises the fraud network. Built on IEEE-CIS (590K transactions).

[![Phase](https://img.shields.io/badge/phase-2%2F8-blue)](./Fraud_Detection_GNN_Implementation_Plan.pdf)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()
[![Tests](https://img.shields.io/badge/tests-85%20passing-success)]()

Production-grade fraud detection on the **IEEE-CIS** dataset (590,540 transactions, 3.5% fraud rate)
combining a **heterogeneous GNN** (PyTorch Geometric) with an **XGBoost ensemble**, served in
real time (<50ms P95) via FastAPI + Ray Serve + Kafka, and investigated automatically by a
**LangGraph agent** backed by a local Ollama LLM. Results surface in a React.js dashboard.

> This repository is currently at **Phase 2: Graph & Features (`v0.2.0-graph-engine`)**.
> See [Fraud_Detection_GNN_Implementation_Plan.pdf](./Fraud_Detection_GNN_Implementation_Plan.pdf)
> for the complete 8-phase roadmap.

---

## Phase 1 -- Foundation & Data Pipeline (`v0.1.0-data-foundation`)

| Area | File | Purpose |
| --- | --- | --- |
| Scaffolding | `pyproject.toml`, `Makefile`, `.gitignore`, `.env.example` | Reproducible environment, hatchling build with `dev`/`train`/`serve`/`agent` groups |
| Config | [`src/fraud_detection/utils/config.py`](src/fraud_detection/utils/config.py) | Pydantic `AppConfig` loading YAML + env vars |
| Logging | [`src/fraud_detection/utils/logging.py`](src/fraud_detection/utils/logging.py) | Structured JSON logs via `structlog` |
| Download | [`src/fraud_detection/data/download.py`](src/fraud_detection/data/download.py) | Kaggle downloader supporting both legacy `KAGGLE_USERNAME`/`KAGGLE_KEY` auth and the new `KAGGLE_API_TOKEN` bearer format |
| Preprocess | [`src/fraud_detection/data/preprocessing.py`](src/fraud_detection/data/preprocessing.py) | `IEEECISPreprocessor` -- per-group missing-value strategy (V/D/C/M/id/email), fit/transform, pickle persist |
| Splits | [`src/fraud_detection/data/splits.py`](src/fraud_detection/data/splits.py) | Chronological `TemporalSplitter` (60/20/20, non-overlap assertion) |
| EDA | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Class balance, missingness heatmap, amount distributions, temporal fraud rate |

## Phase 2 -- Graph & Features (`v0.2.0-graph-engine`)

| Area | File | Purpose |
| --- | --- | --- |
| **Heterograph** (7 nodes / 8 edges) | [`src/fraud_detection/data/graph_builder.py`](src/fraud_detection/data/graph_builder.py) | `HeteroGraphBuilder` -- PyTorch Geometric `HeteroData` with transaction / card / address / email / device / ip_address / merchant nodes; 6 transaction->entity edges + card-card `shared_address`/`shared_device` edges |
| Temporal + amount features (39) | [`src/fraud_detection/features/temporal.py`](src/fraud_detection/features/temporal.py) | Cyclical hour/dow, velocity over 1h/24h/7d windows, interarrival stats, cumulants, acceleration, jerk, percentile/z-score |
| Aggregated + identity features (52) | [`src/fraud_detection/features/aggregated.py`](src/fraud_detection/features/aggregated.py) | Per-entity mean/median/std/count/fraud-rate + diversity counts + C/M-family aggregates + id risk bins + device OS / proxy / email-mismatch / new-device / multi-card flags |
| Graph-structural features (28) | [`src/fraud_detection/features/graph_features.py`](src/fraud_detection/features/graph_features.py) | Degrees, PageRank, approximate betweenness + closeness on the card-card projection, connected-component size, 1/2-hop neighbor fraud rates, ring membership + size, avg neighbor degree |
| Pipeline orchestrator | [`src/fraud_detection/features/pipeline.py`](src/fraud_detection/features/pipeline.py) | `FeaturePipeline.fit_transform` / `transform` / `save` / `load` -- emits 119 total engineered columns |
| Feast feature store | [`configs/feast/`](configs/feast/) | 3 feature views (temporal / aggregated / graph) + SQLite online store; `feast apply` succeeds |
| Graph EDA | [`notebooks/02_graph.ipynb`](notebooks/02_graph.ipynb) | HeteroData stats, fraud rate by split, top-|corr| features, graph-feature distributions, ring-vs-fraud |

### Graph topology

```
          ┌─────────────────────────────────────────────────────────────┐
          │                                                              │
          │              card  ◄──shared_address──►  card               │
          │                │                                              │
          │                │ uses_card                                   │
          │                ▼                                              │
┌──────┐  │   ┌──── transaction ───┐                                     │
│ txn  │──┼──►│                    │──► address / email / device / ip /  │
└──────┘  │   └───── at_merchant ──┘    merchant (V-cluster)             │
          │                                                              │
          │              card  ◄──shared_device──►  card                 │
          └─────────────────────────────────────────────────────────────┘
```

**Acceptance (all met):** HeteroData loads in PyG with correct node/edge counts · All 119
features computed without errors · Graph construction <30 min on 16GB RAM
(benchmarked: 10 s on 50K → ~2 min on 590K) · Feast feature store `apply` succeeds.

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
│   ├── 01_eda.ipynb          # Phase 1 -- raw-data EDA
│   └── 02_graph.ipynb        # Phase 2 -- graph + features EDA
├── scripts/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── split_data.py
│   └── build_graph.py        # Phase 2: heterograph + features
├── src/fraud_detection/
│   ├── data/                 # download, preprocessing, splits, graph_builder
│   ├── features/             # temporal, aggregated, graph_features, pipeline
│   ├── models/               # hetero_gnn, xgboost_model, ensemble, losses  (Phase 3)
│   ├── training/             # trainer, callbacks, evaluator, hyperopt      (Phase 3)
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

```bash
make lint         # ruff check
make format       # ruff format + autofix
make typecheck    # mypy
make test         # pytest unit tests (85 tests, ~30 s)
make test-cov     # unit tests with coverage report
```

CI runs the full matrix (Python 3.10 / 3.11 / 3.12) on every push + PR via
`.github/workflows/ci.yml`.

---

## Roadmap

| Phase | Tag | Status |
| :-- | :-- | :-- |
| 1. Foundation & Data Pipeline | `v0.1.0-data-foundation` | ✅ Complete |
| 2. Graph & Features | `v0.2.0-graph-engine` | ✅ Complete |
| 3. GNN Model & Training | `v0.3.0-gnn-model` | 🚧 Up next |
| 4. Real-Time Serving | `v0.4.0-serving-pipeline` | ⏳ Planned |
| 5. Agentic Investigator | `v0.5.0-agent-investigator` | ⏳ Planned |
| 6. React Dashboard | `v0.6.0-dashboard` | ⏳ Planned |
| 7. MLOps & Monitoring | `v0.7.0-mlops` | ⏳ Planned |
| 8. Docs, Demo & Polish | `v1.0.0-release` | ⏳ Planned |

## License

MIT -- see [LICENSE](LICENSE).
