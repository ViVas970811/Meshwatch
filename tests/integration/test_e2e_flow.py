"""End-to-end integration test (Phase 8).

Boots the FastAPI app in-process (no docker), exercises every public
endpoint in the order an operator would touch them during the demo, and
verifies the cross-phase interactions:

* Phase 4  predict (single + batch) + WebSocket fan-out
* Phase 5  agent investigation
* Phase 6  /recent buffer feeding the dashboard's first paint
* Phase 7  drift seeded + monitoring endpoints + shadow lane
* Phase 8  security middleware (API key + rate limit)

We bypass the real ensemble by swapping a stub predictor into the
lifespan-provisioned ``AppState`` -- the goal here is to validate the
**wiring**, not retrain the model.
"""

from __future__ import annotations

import time
from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud_detection.monitoring import (
    DriftDetector,
    PerformanceTracker,
    ShadowDeployment,
    reset_state,
)
from fraud_detection.serving.app import AppState, create_app
from fraud_detection.serving.redis_cache import EmbeddingCache
from fraud_detection.serving.schemas import (
    FraudPrediction,
    TransactionRequest,
    risk_level,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Stub predictor (mirrors the one in tests/unit/test_serving_app.py)
# ---------------------------------------------------------------------------


class _StubPredictor:
    threshold = 0.7
    model_version = "v1.0.0-e2e"
    embedding_dim = 0
    feature_columns: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self.cache = EmbeddingCache(url=None)
        self.cache.connect()
        self.ensemble = MagicMock()
        self.ensemble.gnn = MagicMock()
        self.ensemble.gnn.n_parameters.return_value = 1_600_000
        self.ensemble.gnn.embedding_dim = 64
        self.ensemble.gnn.edge_types = [("transaction", "uses_card", "card")]
        self.ensemble.gnn.node_types = ["transaction", "card"]
        self.ensemble.xgb.feature_importance.return_value = {"TransactionAmt": 1.0}

    @staticmethod
    def _score(amt: float) -> float:
        if amt >= 2000:
            return 0.95
        if amt >= 500:
            return 0.75
        if amt >= 100:
            return 0.50
        return 0.20

    def predict_one(self, req: TransactionRequest) -> FraudPrediction:
        score = self._score(req.transaction_amt)
        return FraudPrediction(
            transaction_id=req.transaction_id,
            fraud_probability=score,
            fraud_score=score,
            risk_level=risk_level(score),
            is_fraud_predicted=score >= self.threshold,
            threshold=self.threshold,
            top_features=[],
            latency_ms={"total_ms": 0.5},
            model_version=self.model_version,
        )

    def predict_batch(self, reqs):
        return [self.predict_one(r) for r in reqs]

    def info(self) -> dict:
        return {"model_version": self.model_version}


# ---------------------------------------------------------------------------
# Fixture: a stub-backed app with everything wired up
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FRAUD_ENSEMBLE_DIR", "/nonexistent/path/skip-load")
    monkeypatch.setenv("FRAUD_AGENT_DISABLED", "false")  # keep agent on for this flow
    monkeypatch.delenv("FRAUD_API_KEYS", raising=False)
    monkeypatch.setenv("FRAUD_RATE_LIMIT_DISABLED", "true")
    monkeypatch.setenv("FRAUD_TEST_FAST", "true")
    reset_state()
    app = create_app()
    with TestClient(app) as c:
        state: AppState = app.state.fraud_app
        state.predictor = _StubPredictor()  # type: ignore[assignment]
        state.monitoring.performance = PerformanceTracker(threshold=0.5)
        yield c, app, state


def _tx(tx_id: str, amt: float) -> dict:
    return {
        "transaction_id": tx_id,
        "transaction_dt": 1_700_000_000,
        "transaction_amt": amt,
        "product_cd": "W",
        "card1": 12345,
    }


# ---------------------------------------------------------------------------
# 1. Health / model info
# ---------------------------------------------------------------------------


def test_health_reports_ok_when_model_loaded(client):
    c, _app, _state = client
    resp = c.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_returns_metadata(client):
    c, _app, _state = client
    resp = c.get("/api/v1/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "v1.0.0-e2e"
    assert body["n_parameters"] == 1_600_000


# ---------------------------------------------------------------------------
# 2. Predict path -- single + batch + recent buffer + risk distribution
# ---------------------------------------------------------------------------


def test_predict_pipeline_records_history_and_buckets_by_risk(client):
    c, _app, state = client
    # 30 transactions spanning every risk bucket
    rng = np.random.default_rng(42)
    amounts = rng.choice([25, 75, 250, 800, 3000], size=30, replace=True)

    for i, amt in enumerate(amounts):
        resp = c.post("/api/v1/predict", json=_tx(f"tx_{i:03d}", float(amt)))
        assert resp.status_code == 200
        body = resp.json()
        assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    assert state.monitoring.performance.n_records == 30
    assert len(state.recent_predictions) == 30

    # /api/v1/recent should surface them.
    recent = c.get("/api/v1/recent?limit=30").json()
    assert len(recent["predictions"]) == 30


def test_predict_batch_processes_all_transactions(client):
    c, _app, _state = client
    payload = {"transactions": [_tx(f"batch_{i}", 100 + i * 10) for i in range(10)]}
    resp = c.post("/api/v1/predict/batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_processed"] == 10
    assert body["elapsed_ms"] > 0
    assert all(0 <= p["fraud_score"] <= 1 for p in body["predictions"])


# ---------------------------------------------------------------------------
# 3. Investigation (Phase 5)
# ---------------------------------------------------------------------------


def test_investigate_runs_against_predicted_alert(client):
    c, _app, _state = client
    # Score a hot tx so the agent sees a HIGH/CRITICAL risk.
    pred = c.post("/api/v1/predict", json=_tx("hot_tx", 3500.0)).json()
    assert pred["risk_level"] in {"HIGH", "CRITICAL"}

    inv = c.post(
        "/api/v1/investigate",
        json={"prediction": pred, "alert_id": "e2e-test"},
    )
    assert inv.status_code == 200
    report = inv.json()
    assert report["alert_id"] == "e2e-test"
    assert report["risk_level"] in {"HIGH", "CRITICAL"}
    assert report["recommended_action"] in {"approve", "review", "decline", "escalate"}
    assert "summary" in report
    assert len(report["tools_used"]) >= 1


# ---------------------------------------------------------------------------
# 4. Monitoring surface (Phase 7)
# ---------------------------------------------------------------------------


def test_monitoring_endpoints_round_trip(client):
    c, _app, state = client

    # Seed enough labelled predictions for the snapshot to be informative.
    for i in range(10):
        amt = 3000 if i < 5 else 30
        pred = c.post("/api/v1/predict", json=_tx(f"lbl_{i}", float(amt))).json()
        c.post(
            "/api/v1/monitoring/label",
            json={"transaction_id": pred["transaction_id"], "label": 1 if i < 5 else 0},
        )

    perf = c.get("/api/v1/monitoring/performance").json()
    assert perf["n_labelled"] == 10
    assert 0 <= perf["precision"] <= 1
    assert 0 <= perf["recall"] <= 1

    # Drift -- seed a real report through the state and check the endpoint.
    rng = np.random.default_rng(7)
    ref = {"amount": rng.normal(0, 1, size=500).tolist()}
    cur = {"amount": rng.normal(5, 1, size=500).tolist()}
    state.monitoring.last_drift_report = DriftDetector(ref).detect(cur)

    drift = c.get("/api/v1/monitoring/drift").json()
    assert drift["severity"] == "severe"

    alerts = c.get("/api/v1/monitoring/alerts").json()
    # Severe drift should fire DataDriftSevere; the predictor is loaded so
    # ModelNotLoaded should not fire.
    names = {a["name"] for a in alerts["alerts"]}
    assert "DataDriftSevere" in names
    assert "ModelNotLoaded" not in names


def test_monitoring_metrics_exposition_includes_meshwatch_namespace(client):
    c, _app, _state = client
    # Drive at least one request through so counters fire.
    c.post("/api/v1/predict", json=_tx("metrics_seed", 50.0))
    body = c.get("/api/v1/metrics").text
    # The prometheus extra may or may not be installed.
    assert "meshwatch_http_requests_total" in body or "prometheus_client not installed" in body


# ---------------------------------------------------------------------------
# 5. Shadow deployment (Phase 7)
# ---------------------------------------------------------------------------


def test_shadow_deployment_logs_decisions(client):
    c, _app, state = client

    # Build a shadow that disagrees with the stub on borderline txs.
    class _Challenger(_StubPredictor):
        model_version = "v1.0.0-challenger"

        @staticmethod
        def _score(amt: float) -> float:  # always more conservative
            return min(0.99, _StubPredictor._score(amt) + 0.15)

    deployment = ShadowDeployment(
        champion=state.predictor,  # type: ignore[arg-type]
        challenger=_Challenger(),  # type: ignore[arg-type]
        max_records=50,
    )
    state.monitoring.shadow = deployment

    try:
        for i in range(10):
            c.post("/api/v1/predict", json=_tx(f"shadow_{i}", 600.0 + i * 10))
        # Allow the background thread to drain.
        for _ in range(50):
            if deployment.n_records >= 10:
                break
            time.sleep(0.02)

        summary = c.get("/api/v1/monitoring/shadow").json()
        assert summary["n_total"] >= 10
        assert summary["champion_model"] == "v1.0.0-e2e"
        assert summary["challenger_model"] == "v1.0.0-challenger"
    finally:
        deployment.shutdown(wait=True)
        state.monitoring.shadow = None


# ---------------------------------------------------------------------------
# 6. Security middleware (Phase 8) -- exercised in a separate app so we
#    can flip the env without leaking into the rest of the suite.
# ---------------------------------------------------------------------------


def test_full_security_stack_blocks_then_allows(monkeypatch):
    monkeypatch.setenv("FRAUD_ENSEMBLE_DIR", "/nonexistent/path/skip-load")
    monkeypatch.setenv("FRAUD_AGENT_DISABLED", "true")
    monkeypatch.setenv("FRAUD_API_KEYS", "secret-1")
    monkeypatch.setenv("FRAUD_RATE_LIMIT_DISABLED", "true")
    reset_state()
    app = create_app()
    with TestClient(app) as c:
        state: AppState = app.state.fraud_app
        state.predictor = _StubPredictor()  # type: ignore[assignment]

        # No key -> 401
        assert c.post("/api/v1/predict", json=_tx("t1", 100.0)).status_code == 401
        # Bad key -> 403
        assert (
            c.post(
                "/api/v1/predict",
                json=_tx("t2", 100.0),
                headers={"X-API-Key": "wrong"},
            ).status_code
            == 403
        )
        # Good key -> 200
        assert (
            c.post(
                "/api/v1/predict",
                json=_tx("t3", 100.0),
                headers={"X-API-Key": "secret-1"},
            ).status_code
            == 200
        )
        # Health remains exempt.
        assert c.get("/api/v1/health").status_code == 200


# ---------------------------------------------------------------------------
# 7. Latency budget (Phase 4 acceptance criterion, p. 9)
# ---------------------------------------------------------------------------


def test_predict_latency_stays_under_budget(client):
    c, _app, _state = client
    latencies_ms: list[float] = []
    for i in range(200):
        t0 = time.perf_counter()
        resp = c.post("/api/v1/predict", json=_tx(f"lat_{i}", 50.0 + i))
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        assert resp.status_code == 200
    arr = np.array(latencies_ms)
    p95 = float(np.percentile(arr, 95))
    # The stub is trivial -- P95 of in-process FastAPI calls should be
    # comfortably under the 50ms budget on any developer machine.
    assert p95 < 50.0, f"P95 latency {p95:.2f}ms exceeds the 50ms budget"
