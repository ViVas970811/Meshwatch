# Meshwatch -- Benchmarks

Performance, latency, throughput, and coverage numbers across the
Meshwatch stack. All numbers are reproducible from the artifacts in
this repo; the run commands are listed under each section.

## Test environment

| Item       | Value                                              |
| :--        | :--                                                |
| OS         | Windows 11                                         |
| CPU        | AMD Ryzen 5 5500U, 6 cores / 12 threads, 16 GB RAM |
| Python     | 3.10                                               |
| PyTorch    | 2.10.0+cpu                                         |
| Test client| `fastapi.testclient.TestClient` for the API benches |

---

## 1. Model performance (Phase 3)

Trained on the full IEEE-CIS dataset (590,540 transactions):

  preprocess  -> 590,540 rows x 501 features
  split       -> 354k train / 118k val / 118k test, 60/20/20 temporal
  build-graph -> 590k transaction seeds + 15.5k entities, ~3.4M edges
  train GNN   -> 100 epochs max, cosine LR schedule, FocalLoss
                  (alpha=0.75, gamma=2.0), NeighborLoader [15, 10, 5]
  ensemble    -> XGBoost on [64-d GNN embedding + 120 graph features +
                  selected V columns] = 184 features

### Validation split (n=118,108, fraud rate 3.90%)

| Metric                  | GNN-only | **Ensemble** |
| :--                     |     ---: |         ---: |
| AUPRC                   |   0.2511 |   **0.3601** |
| AUROC                   |   0.8119 |   **0.7782** |
| Best F1                 |   0.3082 |   **0.3911** |
| Best precision          |   0.2879 |   **0.4792** |
| Best recall             |   0.3316 |       0.3303 |
| Precision @ top 0.1%    |   0.7059 |   **0.9748** |
| **Precision @ top 1%**  |   0.4255 |   **0.7377** |
| Precision @ top 5%      |   0.2196 |       0.3320 |
| Precision @ top 10%     |   0.1549 |       0.2077 |

### Test split (n=118,108, fraud rate 3.44%)

| Metric                  | GNN-only | **Ensemble** |
| :--                     |     ---: |         ---: |
| AUPRC                   |   0.2032 |   **0.2449** |
| AUROC                   |   0.7867 |   **0.7499** |
| Best F1                 |   0.2713 |       0.2778 |
| Best precision          |   0.2554 |       0.2721 |
| Best recall             |   0.2894 |       0.2837 |
| Precision @ top 0.1%    |   0.7059 |   **0.8235** |
| **Precision @ top 1%**  |   0.4255 |   **0.5592** |
| Precision @ top 5%      |   0.2196 |       0.2310 |
| Precision @ top 10%     |   0.1549 |       0.1583 |

### Headline read

**Top-1% precision = 0.74 on val (0.56 on test)** -- if an analyst
queue reviews the highest-scored 1% of transactions, 74% of those
flags are real fraud on val and 56% on test, at base rates of
3.9% and 3.4% respectively (19× and 16× lift). **Top-0.1%
precision = 0.97 / 0.82** -- the highest-confidence flags are
nearly all correct.

### Reproduce

```bash
make download-data           # IEEE-CIS via Kaggle
make preprocess && make split
make build-graph             # full 590k heterograph
make train                   # GNN -> data/models/gnn/
make train-ensemble          # GNN + XGBoost -> data/models/ensemble/
make evaluate-ensemble       # writes PR / ROC / calibration on test
```

Notebook walkthrough: [notebooks/04_ensemble.ipynb](../notebooks/04_ensemble.ipynb).

---

## 2. Inference latency (Phase 4)

In-process `TestClient` against the FastAPI app with full middleware
stack (timing + Prometheus + monitoring registry + auth + rate-limit).
2000 requests per endpoint after a 50-request warm-up.

| Endpoint                                | P50      | P95      | P99      |
| :--                                     |     ---: |     ---: |     ---: |
| `POST /api/v1/predict`                  |  3.10 ms |  3.98 ms |  4.45 ms |
| `POST /api/v1/predict/batch` (size=50)  |  4.78 ms |  5.61 ms |  6.01 ms |
| `POST /api/v1/predict/batch` (size=100) |  6.10 ms |  7.12 ms |  7.35 ms |
| `GET  /api/v1/health`                   |  2.39 ms |  3.11 ms |  3.43 ms |
| `GET  /api/v1/metrics`                  |  6.32 ms |  7.30 ms |  7.75 ms |
| `GET  /api/v1/monitoring/drift`         |  2.36 ms |  2.95 ms |  3.39 ms |
| `GET  /api/v1/monitoring/performance`   |  3.08 ms |  3.87 ms |  4.31 ms |
| `GET  /api/v1/monitoring/alerts`        |  2.97 ms |  3.76 ms |  4.06 ms |
| `GET  /api/v1/monitoring/shadow`        |  2.36 ms |  3.07 ms |  3.37 ms |

Phase 4 latency budget is P95 < 50 ms on `/api/v1/predict`; measured
P95 = 4 ms.

---

## 3. Throughput (Phase 4)

Single-threaded sequential calls; multi-process uvicorn or Ray Serve
scales near-linearly with worker count.

| Endpoint                                | Throughput     |
| :--                                     |           ---: |
| `POST /api/v1/predict`                  |        309 rps |
| `POST /api/v1/predict/batch` (size=50)  | **10,161 rps** |
| `POST /api/v1/predict/batch` (size=100) | **15,693 rps** |
| `GET  /api/v1/health`                   |        408 rps |

---

## 4. Agent investigation (Phase 5)

`POST /api/v1/investigate` round-trip across the four routing depths,
20 calls per depth.

| Risk level | Routing depth     | P50     | P95     |
| :--        | :--               |    ---: |    ---: |
| LOW        | quick (2 tools)   | 6.59 ms | 8.66 ms |
| MEDIUM     | quick (2 tools)   | 6.52 ms | 7.59 ms |
| HIGH       | standard (5 tools)| 7.02 ms | 7.97 ms |
| CRITICAL   | deep (7 tools)    | 7.64 ms | 8.48 ms |

These are orchestration-overhead numbers (deterministic stub LLM); a
real Ollama daemon adds the LLM cost on top -- typically 4-15 s for
`llama3.1:8b` standard depth on CPU.

```bash
OLLAMA_BASE_URL=http://localhost:11434 make investigate-ollama
```

---

## 5. Dashboard (Phase 6)

| Metric            | Value |
| :--               |  ---: |
| Vitest test count |  32   |

```bash
make dashboard-build
make dashboard-test
```

---

## 6. Tests + coverage

| Suite              | Count   |
| :--                |    ---: |
| Python unit        |     428 |
| Python integration |      10 |
| TypeScript Vitest  |      32 |
| **Total**          | **470** |

```bash
PYTHONPATH=src pytest tests/unit --cov=src/fraud_detection --cov-report=term
PYTHONPATH=src pytest tests/integration -m integration
cd dashboard && npm test -- --run
```

Coverage on `src/fraud_detection`: **85%** statement, branch on.

---

## 7. Reproduction

The latency + throughput numbers come from boot-the-app-in-process
benches. The production-realistic equivalent (uvicorn over the wire,
real ensemble loaded) runs via:

```bash
docker compose up -d                # boots the full stack
make demo                           # 1000-tx replay; prints P50/P95/P99
```
