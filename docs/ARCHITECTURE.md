# Meshwatch -- Architecture

A guided tour of the Meshwatch end-to-end system, phase by phase. This
document complements the high-level diagram in the project [README](../README.md)
and the detailed feature breakdowns inside each phase's notebook.

## System diagram

```mermaid
flowchart LR
    subgraph Ingest["Ingest"]
        K[Kafka topic]
        D[scripts/demo.py]
    end

    subgraph Serve["Real-time scoring (Phase 4)"]
        API[FastAPI + Ray Serve]
        R[(Redis<br/>embedding cache)]
        F[(Feast<br/>online features)]
    end

    subgraph Model["Model (Phase 3)"]
        GNN[Heterogeneous GNN<br/>SAGEConv + GAT]
        XGB[XGBoost<br/>ensemble]
    end

    subgraph Agent["Investigator (Phase 5)"]
        LG[LangGraph<br/>state machine]
        T[8 tools<br/>incl. GraphRAG]
        OLL[Ollama LLM<br/>llama3.1:8b]
    end

    subgraph Surfaces["Surfaces"]
        DASH[React dashboard<br/>Vite + force-graph]
        WS[/ws/alerts/]
        GRAF[Grafana<br/>4 dashboards]
        MLF[MLflow]
    end

    subgraph Ops["MLOps (Phase 7)"]
        DRIFT[DriftDetector<br/>PSI/KS/JSD]
        SHADOW[ShadowDeployment<br/>champion/challenger]
        PROM[Prometheus<br/>+ alert rules]
    end

    K -->|alert| API
    D -.replay 1k txns.-> API
    API --> F
    API --> R
    R --> XGB
    F --> XGB
    GNN -->|64-d embedding| XGB
    XGB -->|FraudPrediction| API
    API -->|alert >= 0.7| WS
    API -->|alert| LG
    LG --> T
    T --> OLL
    OLL -->|narrative| LG
    LG -->|InvestigationReport| API

    API --> DASH
    WS --> DASH
    DASH --> GRAF
    API -->|/metrics| PROM
    PROM --> GRAF
    GNN --- MLF
    XGB --- MLF
    API --> DRIFT
    API --> SHADOW
    SHADOW --> PROM
    DRIFT --> PROM
```

## Request walk-through

The 50 ms P95 hot path:

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant API as FastAPI + middleware
    participant Sec as ApiKey + RateLimit
    participant Pred as FraudPredictor
    participant Redis
    participant XGB as XGBoost
    participant Kafka

    Client->>API: POST /api/v1/predict
    API->>Sec: auth + rate-limit (Phase 8)
    Sec-->>API: ok (or 401/403/429)
    API->>Pred: TransactionRequest
    Pred->>Redis: GET emb:{card1}
    alt cache hit
        Redis-->>Pred: 64-d embedding
    else miss
        Pred->>Pred: GNN forward(graph_data)
        Pred->>Redis: SETEX emb:{card1}
    end
    Pred->>XGB: [embedding(64) + tabular(119)]
    XGB-->>Pred: fraud_proba
    Pred-->>API: FraudPrediction
    API->>Kafka: publish if score >= 0.7
    API-->>Client: 200 FraudPrediction
```

## Graph schema (Phase 2)

7 node types, 8 edge types, fed by the Phase 1 IEEE-CIS preprocessor:

```mermaid
graph TB
    txn((transaction<br/>50 dims))
    card[(card<br/>8 dims)]
    addr[(address<br/>4 dims)]
    email[(email<br/>3 dims)]
    device[(device<br/>4 dims)]
    ip[(ip_address<br/>6 dims)]
    merch[(merchant<br/>10 dims)]

    txn -->|uses_card| card
    txn -->|from_address| addr
    txn -->|from_email| email
    txn -->|from_device| device
    txn -->|from_ip| ip
    txn -->|at_merchant| merch
    card ---|shared_address| card
    card ---|shared_device| card
```

The `shared_*` edges are the core innovation: collusion rings show up as
cliques on those edges that flat-table models can't see.

## Investigation routing (Phase 5)

```mermaid
flowchart TD
    Start([alert in]) --> Route{risk_level?}
    Route -->|LOW/MEDIUM| Quick[Quick scan<br/>2 tools]
    Route -->|HIGH| Std[Gather context<br/>+ analyze patterns<br/>5 tools]
    Route -->|CRITICAL| Deep[Full graph traversal<br/>+ cross-entity<br/>7 tools]

    Quick --> Gen[generate_investigation_report]
    Std --> Gen
    Deep --> Gen
    Gen --> Review{requires<br/>human review?}
    Review -->|no| Done([InvestigationReport])
    Review -->|yes| HR[flag for analyst]
    HR --> Done
```

## Monitoring surface (Phase 7)

```mermaid
flowchart LR
    subgraph "Hot path (per request)"
        Predict[/api/v1/predict/] --> Track[PerformanceTracker]
        Predict --> Shadow[ShadowDeployment]
    end

    subgraph "Periodic (scripts/monitor.py)"
        Train[(training split)] --> DD[DriftDetector]
        Prod[(recent prod)] --> DD
        DD --> Report[DriftReport<br/>JSON + HTML]
    end

    subgraph "Exposed"
        Track --> MEP[/monitoring/performance/]
        DD --> MED[/monitoring/drift/]
        Shadow --> MES[/monitoring/shadow/]
        Track --> Prom[Prometheus gauges]
        DD --> Prom
        Shadow --> Prom
        Prom --> Graf[Grafana<br/>4 dashboards]
        Prom --> Rules[Alert rules<br/>configs/prometheus_rules.yml]
    end
```

## Phase / component map

| Phase | Source                                              | Public surface                                  |
| :--   | :--                                                 | :--                                             |
| 1     | `src/fraud_detection/data/`                         | `make download-data preprocess split`           |
| 2     | `src/fraud_detection/data/graph_builder.py`         | `make build-graph`, `configs/feast/`            |
| 3     | `src/fraud_detection/{models,training}/`            | `make train train-ensemble evaluate-ensemble`   |
| 4     | `src/fraud_detection/{serving,streaming}/`          | `POST /api/v1/predict`, `make serve compose-up` |
| 5     | `src/fraud_detection/agent/`                        | `POST /api/v1/investigate`, `make investigate`  |
| 6     | `dashboard/`                                        | `http://localhost:5173`                         |
| 7     | `src/fraud_detection/monitoring/`, `configs/grafana/`, `configs/prometheus_rules.yml` | `/api/v1/monitoring/*`, Grafana :3000 |
| 8     | `src/fraud_detection/serving/security.py`, `scripts/demo.py`, `tests/integration/` | `make demo`, API key + rate limit |

## Deployment topology (Docker Compose)

```mermaid
flowchart LR
    user((operator)) -->|:5173| dash
    user -.->|:8000| api
    user -.->|:3000| graf
    user -.->|:9090| prom
    user -.->|:5000| mlf

    subgraph host["docker-compose.yml"]
        dash[meshwatch-dashboard<br/>nginx + Vite build]
        api[meshwatch-api<br/>uvicorn]
        redis[(meshwatch-redis)]
        kafka[meshwatch-kafka]
        zk[meshwatch-zookeeper]
        prom[meshwatch-prometheus]
        graf[meshwatch-grafana]
        mlf[meshwatch-mlflow]
    end

    dash -->|/api| api
    dash -.->|/ws| api
    api --> redis
    api --> kafka
    kafka --> zk
    prom -.->|scrape /metrics| api
    graf -.-> prom
```

All services are configured to degrade gracefully -- if Kafka, Redis, or
Neo4j are unreachable the API stays up with in-memory fallbacks.

## Coding conventions

- Python 3.10+, ruff for lint + format, mypy for type-checking (best-
  effort, not strict).
- Pydantic v2 for request/response schemas (`src/fraud_detection/serving/schemas.py`).
- structlog for logging -- every record is a JSON line in production.
- Tests in `tests/unit` are pure-Python (no docker required). Tests in
  `tests/integration` boot the FastAPI app in-process via `TestClient`.
  Marked with `pytest.mark.integration` so they're skipped by `make
  test`.
