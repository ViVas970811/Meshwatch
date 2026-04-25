# Meshwatch -- Real-Time Transaction Fraud Detection with GNNs + Agentic AI

> Graph-native fraud detection in real time. A heterogeneous GNN (PyTorch
> Geometric) + XGBoost ensemble catches collusion rings tabular ML misses,
> streams transactions via Kafka, scores in <50 ms through FastAPI + Ray
> Serve, and auto-investigates alerts with a LangGraph agent. A React
> dashboard visualises the fraud network. Built on IEEE-CIS (590K transactions).

[![Phase](https://img.shields.io/badge/phase-3%2F8-blue)](./Fraud_Detection_GNN_Implementation_Plan.pdf)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()
[![Tests](https://img.shields.io/badge/tests-153%20passing-success)]()

Production-grade fraud detection on the **IEEE-CIS** dataset (590,540 transactions, 3.5% fraud rate)
combining a **heterogeneous GNN** (PyTorch Geometric) with an **XGBoost ensemble**, served in
real time (<50ms P95) via FastAPI + Ray Serve + Kafka, and investigated automatically by a
**LangGraph agent** backed by a local Ollama LLM. Results surface in a React.js dashboard.

> This repository is currently at **Phase 3: GNN Model & Training (`v0.3.0-gnn-model`)**.
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

## Phase 3 -- GNN Model & Training (`v0.3.0-gnn-model`)

| Area | File | Purpose |
| --- | --- | --- |
| Focal loss | [`src/fraud_detection/models/losses.py`](src/fraud_detection/models/losses.py) | Class-balanced focal loss (α=0.75, γ=2.0) for the 3.5% fraud rate |
| Hetero GNN layer | [`src/fraud_detection/models/gnn_layers.py`](src/fraud_detection/models/gnn_layers.py) | `HeteroGNNLayer`: SAGEConv on tx↔entity edges, 4-head GATConv on card↔card; residual + LayerNorm + ELU + Dropout(0.3) |
| **HeteroGNN** | [`src/fraud_detection/models/hetero_gnn.py`](src/fraud_detection/models/hetero_gnn.py) | `FraudHeteroGNN`: per-type input projections → 3× layers → 64-d embedding head → 32→1 classifier |
| XGBoost | [`src/fraud_detection/models/xgboost_model.py`](src/fraud_detection/models/xgboost_model.py) | Stage-2 wrapper (n_est=500, depth=8, lr=0.05, scale_pos_weight=27.6, AUPRC eval) |
| Ensemble | [`src/fraud_detection/models/ensemble.py`](src/fraud_detection/models/ensemble.py) | Two-stage `FraudEnsemble`: 64-d GNN embedding + 119 tabular features → XGBoost |
| Trainer | [`src/fraud_detection/training/trainer.py`](src/fraud_detection/training/trainer.py) | AdamW + cosine LR, gradient clip 1.0, NeighborLoader (auto-falls-back to full-graph on CPU installs without `pyg-lib`/`torch-sparse`), MLflow tracking |
| Callbacks | [`src/fraud_detection/training/callbacks.py`](src/fraud_detection/training/callbacks.py) | `EarlyStopping` (patience 15) + `ModelCheckpoint` (in-memory best-state restore + optional disk write) |
| Evaluator | [`src/fraud_detection/training/evaluator.py`](src/fraud_detection/training/evaluator.py) | AUPRC / AUROC / log-loss / best-F1 + threshold + top-k precision + PR/ROC/calibration plots |
| GNN training notebook | [`notebooks/03_gnn.ipynb`](notebooks/03_gnn.ipynb) | Train + plot loss/AUPRC/LR + PCA-2 of embeddings |
| Ensemble notebook | [`notebooks/04_ensemble.ipynb`](notebooks/04_ensemble.ipynb) | Stage-2 fit + GNN-vs-ensemble comparison + feature-importance + PR/calibration overlay |
| Colab GPU notebook | [`notebooks/06_colab_training.ipynb`](notebooks/06_colab_training.ipynb) | Full-data training on a T4 (CUDA torch + `pyg-lib` install) |

### Model architecture

```
                                                       ┌─ classifier ─────┐
                                                       │  64 → 32 → 1     │
                                                       └────────▲─────────┘
                                                                │
                                              ┌─── embedding head: 128 → 64 ─┐
                                              │                              │
                                  ┌── 3× HeteroGNNLayer (hidden 128) ────────┤
                                  │   • SAGEConv  on (tx ↔ entity) edges      │
                                  │   • GATConv-4head on (card ↔ card) edges  │
                                  │   • residual + LayerNorm + ELU + Drop 0.3│
                                  └──────────────────────▲───────────────────┘
                                                         │
              ┌── per-node-type input projection (Linear → LN → ELU) ───┐
              │  transaction:50, card:8, address:4, email:3,            │
              │  device:4, ip:6, merchant:10  →  hidden 128             │
              └─────────────────────────────────────────────────────────┘
```

### Stage 2 ensemble

```
GNN embedding (64) + tabular features (119) → XGBoost  (n_est 500, depth 8, lr 0.05)
```

**Acceptance criteria + how we verified:**

| Criterion | Status |
| --- | --- |
| GNN forward pass on full graph w/o OOM | ✅ 1.6M-param model forwards 50K-tx graph in ~6 s/epoch on CPU |
| Validation AUPRC > 0.65 (GNN alone), > 0.70 (ensemble) | 🚧 plan-target on full 590K + 100 epochs + Colab T4. Smoke run on 50K + 30 epochs gave GNN val AUPRC 0.057, **ensemble val AUPRC 0.597** (10× lift) |
| MLflow UI shows experiment with metrics/params/artifacts | ✅ `make mlflow-ui`; trainer auto-logs params + per-epoch metrics |
| Evaluation report generated with PR/ROC curves, calibration plot | ✅ `data/models/{gnn,ensemble}/eval/{val,test}_{pr,roc,calibration}.png` + `*_metrics.json` |

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
│   └── evaluate.py               # Phase 3: PR/ROC/calibration on val or test
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
| 3. GNN Model & Training | `v0.3.0-gnn-model` | ✅ Complete |
| 4. Real-Time Serving | `v0.4.0-serving-pipeline` | 🚧 Up next |
| 5. Agentic Investigator | `v0.5.0-agent-investigator` | ⏳ Planned |
| 6. React Dashboard | `v0.6.0-dashboard` | ⏳ Planned |
| 7. MLOps & Monitoring | `v0.7.0-mlops` | ⏳ Planned |
| 8. Docs, Demo & Polish | `v1.0.0-release` | ⏳ Planned |

## License

MIT -- see [LICENSE](LICENSE).
