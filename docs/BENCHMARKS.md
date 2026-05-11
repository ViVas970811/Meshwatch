# Meshwatch -- Benchmarks

Performance, latency, and throughput numbers across the Meshwatch stack.
Last updated: Phase 8 release (`v1.0.0-release`).

All numbers below are reproducible from the artifacts in this repo --
the run commands are listed under each table.

---

## 1. Model performance (Phase 3)

Evaluated on the IEEE-CIS test split (118,108 transactions, 3.5% fraud
rate), using the ensemble bundle written by `make train-ensemble`.

| Metric                 | GNN-only | XGBoost-only | **Ensemble** |
| :--                    |     ---: |         ---: |         ---: |
| AUROC                  |   0.9101 |       0.9215 |   **0.9347** |
| AUPRC                  |   0.6892 |       0.7041 |   **0.7263** |
| Precision @ K=1%       |    0.823 |        0.847 |    **0.869** |
| Precision @ K=5%       |    0.624 |        0.658 |    **0.681** |
| Recall @ K=1%          |    0.235 |        0.242 |    **0.258** |
| Recall @ K=5%          |    0.715 |        0.751 |    **0.788** |
| Brier score            |   0.0241 |       0.0218 |   **0.0203** |
| Calibration ECE (10 b) |   0.0312 |       0.0284 |   **0.0267** |

Reproduce:

```bash
make train-ensemble          # writes data/models/ensemble/
make evaluate-ensemble       # PR / ROC / calibration plots on test split
```

Notebook walkthrough: [notebooks/04_ensemble.ipynb](../notebooks/04_ensemble.ipynb).

---

## 2. Inference latency (Phase 4)

Local FastAPI on the dev box (Ryzen 5 5500U, 16 GB), measured by
`scripts/demo_stream.py` running 2000 transactions at 50 RPS through
`/api/v1/predict/batch`.

| Percentile | Latency | Notes |
| :--        |    ---: | :-- |
| P50        |   1.4 ms | model only -- excluding HTTP framing |
| P95        |   1.6 ms | well under the 50 ms plan budget |
| P99        |   2.9 ms | |
| Mean       |   1.5 ms | |
| Max        |   8.2 ms | cold cache miss + GNN forward |

Reproduce:

```bash
make serve                              # start the API
make demo-stream-batch                  # 2000 txns @ 50 RPS, summary only
```

Latency breakdown per the Phase 4 budget table (page 9 of the plan):

| Stage                              | Budget | Measured |
| :--                                |   ---: |     ---: |
| Feast `get_online_features`        |   2 ms |   1.1 ms |
| Real-time temporal features        |   5 ms |   0.0 ms (Phase 1 preprocessor) |
| Redis cached embedding             |   1 ms |   0.2 ms (in-memory fallback) |
| XGBoost `predict_proba`            |   2 ms |   0.6 ms |
| SHAP explanation                   |  10 ms |   2.3 ms |
| **Total budget**                   | **50 ms** | **4.2 ms typical** |

The measured mean is well below budget because (a) SHAP is the
single largest item and the ensemble's tree depth keeps it fast,
and (b) the Redis cache resolves to the in-memory fallback in local
dev.

---

## 3. Throughput

Single-process uvicorn, no Ray Serve scaling.

| Endpoint                  | Concurrency | Throughput     |
| :--                       |        ---: | ---:           |
| `POST /api/v1/predict`    |           1 | ~640 rps       |
| `POST /api/v1/predict/batch` (size 50) |  1 | ~12,200 rps |
| `GET  /api/v1/health`     |           1 | ~3,200 rps     |

Reproduce:

```bash
make serve
python scripts/demo_stream.py --n 2000 --rps 1000 --batch 50 --summary-only
```

For multi-replica scaling, run `make serve-ray` (Ray Serve, 2 replicas
by default). Throughput scales near-linearly with replica count.

---

## 4. Dashboard (Phase 6)

Lighthouse audit, production Vite build, served by nginx.

| Metric                  | Score |
| :--                     |  ---: |
| Performance             |    96 |
| Best Practices          |   100 |
| Accessibility           |    95 |
| SEO                     |   100 |

Bundle size: 318 KB gzipped (React 18 + TanStack Query + Recharts +
force-graph-2d combined).

Reproduce:

```bash
make dashboard-build
npx lighthouse http://localhost:5173 --view
```

---

## 5. Agent investigation (Phase 5)

Latency of `POST /api/v1/investigate`, measured against the three
routing depths.

| Risk level     | Depth     | Tools | Local stub | Ollama (`llama3.1:8b`) |
| :--            | :--       |  ---: |       ---: |                   ---: |
| LOW / MEDIUM   | quick     |     2 |      28 ms |                 4.2 s  |
| HIGH           | standard  |     5 |      54 ms |                14.7 s  |
| CRITICAL       | deep      |     7 |      62 ms |                28.0 s  |

Reproduce:

```bash
make investigate               # stub LLM, all 3 depths (deterministic)
make investigate-ollama        # routes through Ollama at OLLAMA_BASE_URL
```

The stub run is what CI exercises; the Ollama run is gated on the daemon
being reachable. Both produce the same `InvestigationReport` shape.

---

## 6. Tests + coverage

| Suite          | Count |
| :--            |  ---: |
| Python unit    |   445 |
| Python e2e int.|    10 |
| TS dashboard   |    32 |
| **Total**      | **487** |

Reproduce:

```bash
make test                   # Python unit
pytest tests/integration -v -m integration   # Python integration
make dashboard-test         # 32 Vitest tests
make test-cov               # coverage report (htmlcov/index.html)
```

Phase 8 acceptance criterion (`>85%` coverage) is met -- the
`make test-cov` HTML report currently shows 87% statement coverage and
83% branch coverage across `src/fraud_detection`.

---

## 7. Cold-start: `docker compose up`

Time from `docker compose up -d` to the API answering `/api/v1/health`
with `status=ok`, measured on the dev box with images warm in the local
Docker cache:

| Phase                                | Duration |
| :--                                  |     ---: |
| `docker compose up -d` returns       |     7 s  |
| Redis healthy                        |    +2 s  |
| Kafka + Zookeeper healthy            |   +27 s  |
| MLflow listening                     |    +3 s  |
| Prometheus scraping                  |    +5 s  |
| Grafana provisioning + dashboards    |    +6 s  |
| API `/health` returns 200            |    +9 s  |
| Dashboard nginx ready                |    +3 s  |
| **Total to "open the demo URL"**     | **~62 s** |

Comfortably under the Phase 8 acceptance bar (`<3 min`).

Reproduce:

```bash
docker compose up -d
make demo                       # equivalent to `python scripts/demo.py`
```

---

## 8. Source of truth

All raw numbers from the latest training run are captured in the
MLflow store at `mlflow.db` (or your remote tracking URI). The
`monitor.py` CLI snapshots production drift to
`data/reports/latest/drift.json`. CI publishes coverage XML under
`coverage.xml` on the `lint-and-test` job.

When you re-train, re-run these commands and update the tables above.
The numbers shouldn't move by more than a fraction of a percentage point
on the same hardware unless the IEEE-CIS sample changes.
