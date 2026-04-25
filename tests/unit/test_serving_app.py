"""Integration tests for the FastAPI app using TestClient.

We don't bring up Redis/Kafka -- the app's graceful fallbacks let the
endpoints run end-to-end with a mock predictor + in-memory streaming.
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from fraud_detection.serving.app import AppState, create_app
from fraud_detection.serving.redis_cache import EmbeddingCache
from fraud_detection.serving.schemas import (
    BatchPredictResponse,
    FraudPrediction,
    HealthStatus,
    ModelInfoResponse,
    TransactionRequest,
    risk_level,
)


class _StubPredictor:
    """A minimal fake predictor to avoid loading a real model.

    Score is a deterministic function of `transaction_amt` so we can
    push individual requests above / below the alert threshold at will.
    """

    threshold = 0.7
    model_version = "v0.4.0-test"
    embedding_dim = 0
    feature_columns: ClassVar[list[str]] = []

    def __init__(self) -> None:
        # Stub objects to satisfy the model_info endpoint.
        self.cache = EmbeddingCache(url=None)
        self.cache.connect()
        self.ensemble = MagicMock()
        self.ensemble.gnn = MagicMock()
        self.ensemble.gnn.n_parameters.return_value = 1234
        self.ensemble.gnn.embedding_dim = 64
        self.ensemble.gnn.edge_types = [("transaction", "uses_card", "card")]
        self.ensemble.gnn.node_types = ["transaction", "card"]
        self.ensemble.xgb.feature_importance.return_value = {
            "gnn_emb_000": 0.5,
            "TransactionAmt": 0.3,
        }

    def _score(self, amt: float) -> float:
        # Map any amount > 100 to a fraud score >= 0.7 (alert threshold);
        # anything below stays below.
        if amt >= 100:
            return min(0.99, 0.5 + amt / 1000.0)
        return max(0.01, amt / 200.0)

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

    def info(self) -> dict[str, Any]:
        return {"model_version": self.model_version}


@pytest.fixture
def app_with_stub_predictor(monkeypatch):
    """Start the app and replace the lifespan-provisioned predictor with a stub.

    Points the lifespan at a non-existent ensemble dir to skip the (~10s)
    real-model load. Producer + consumer are kept as the lifespan-provisioned
    in-memory instances so the running consume_async task sees the same
    queue our predict endpoint writes to.
    """
    monkeypatch.setenv("FRAUD_ENSEMBLE_DIR", "/nonexistent/path/skip-load")
    # Tighten the consumer poll so WS tests don't hang waiting on the loop.
    monkeypatch.setenv("FRAUD_TEST_FAST", "true")
    app = create_app()

    with TestClient(app) as client:
        state: AppState = app.state.fraud_app
        # No model was loaded -- substitute the stub.
        state.predictor = _StubPredictor()  # type: ignore[assignment]
        # Keep state.producer / state.consumer as the lifespan provisioned them
        # so the consume_async task fans out to the broadcaster correctly.
        # Tighten the consumer's poll cadence for the WS test.
        if state.consumer is not None:
            state.consumer.poll_timeout_seconds = 0.05
        yield client, state


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_when_model_loaded(app_with_stub_predictor):
    client, _ = app_with_stub_predictor
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = HealthStatus.model_validate(r.json())
    assert body.status == "ok"
    assert body.model_loaded
    assert body.uptime_seconds >= 0


def test_health_degraded_when_model_missing(monkeypatch):
    """Without a stub predictor the lifespan still starts but model_loaded is False."""
    monkeypatch.setenv("FRAUD_ENSEMBLE_DIR", "/nonexistent/path/skip-load")
    app = create_app()
    with TestClient(app) as client:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = HealthStatus.model_validate(r.json())
        # No model loaded -> degraded
        assert body.status == "degraded"
        assert not body.model_loaded


# ---------------------------------------------------------------------------
# Predict (single)
# ---------------------------------------------------------------------------


def test_predict_below_threshold_no_alert(app_with_stub_predictor):
    client, state = app_with_stub_predictor
    payload = {"transaction_id": 1, "transaction_dt": 100, "transaction_amt": 25.0}
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 200
    body = FraudPrediction.model_validate(r.json())
    assert body.transaction_id == 1
    assert not body.is_fraud_predicted
    assert body.risk_level == "LOW"
    # No alert was published because score < threshold
    assert len(state.producer.drain_in_memory()) == 0


def test_predict_above_threshold_publishes_alert(app_with_stub_predictor):
    client, state = app_with_stub_predictor
    payload = {"transaction_id": 99, "transaction_dt": 100, "transaction_amt": 500.0}
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 200
    body = FraudPrediction.model_validate(r.json())
    assert body.is_fraud_predicted
    assert body.risk_level in ("HIGH", "CRITICAL")
    # Alert was published to in-memory producer
    drained = state.producer.drain_in_memory()
    assert len(drained) == 1
    assert drained[0].transaction_id == 99


def test_predict_validates_payload(app_with_stub_predictor):
    client, _ = app_with_stub_predictor
    r = client.post("/api/v1/predict", json={"transaction_id": 1})  # missing fields
    assert r.status_code == 422  # Pydantic validation error


def test_predict_503_when_no_model(monkeypatch):
    """If predictor is None, /predict returns 503."""
    monkeypatch.setenv("FRAUD_ENSEMBLE_DIR", "/nonexistent/path/skip-load")
    app = create_app()
    with TestClient(app) as client:
        state: AppState = app.state.fraud_app
        state.predictor = None  # force unloaded
        r = client.post(
            "/api/v1/predict",
            json={"transaction_id": 1, "transaction_dt": 1, "transaction_amt": 10.0},
        )
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# Predict (batch)
# ---------------------------------------------------------------------------


def test_predict_batch(app_with_stub_predictor):
    client, _ = app_with_stub_predictor
    txs = [
        {"transaction_id": i, "transaction_dt": i, "transaction_amt": 10.0 + i * 100}
        for i in range(5)
    ]
    r = client.post("/api/v1/predict/batch", json={"transactions": txs})
    assert r.status_code == 200
    body = BatchPredictResponse.model_validate(r.json())
    assert body.n_processed == 5
    assert body.n_alerts >= 1  # at least the higher-amount ones
    assert body.elapsed_ms >= 0


def test_predict_batch_validates_max_size(app_with_stub_predictor):
    client, _ = app_with_stub_predictor
    txs = [{"transaction_id": i, "transaction_dt": i, "transaction_amt": 10.0} for i in range(101)]
    r = client.post("/api/v1/predict/batch", json={"transactions": txs})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------


def test_model_info(app_with_stub_predictor):
    client, _ = app_with_stub_predictor
    r = client.get("/api/v1/model/info")
    assert r.status_code == 200
    body = ModelInfoResponse.model_validate(r.json())
    assert body.n_parameters == 1234
    assert "uses_card" in body.edge_types[0]
    assert "transaction" in body.node_types
    assert body.feature_importance_top_k


def test_model_info_503_when_no_model(monkeypatch):
    monkeypatch.setenv("FRAUD_ENSEMBLE_DIR", "/nonexistent/path/skip-load")
    app = create_app()
    with TestClient(app) as client:
        state: AppState = app.state.fraud_app
        state.predictor = None
        r = client.get("/api/v1/model/info")
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------


def test_metrics_endpoint_returns_prometheus_text(app_with_stub_predictor):
    client, _ = app_with_stub_predictor
    # Make a request to populate counters
    client.get("/api/v1/health")
    r = client.get("/api/v1/metrics")
    assert r.status_code == 200
    body = r.text
    # We expect at least one of our custom metrics to show up.
    assert "meshwatch_http_requests_total" in body or "meshwatch_http_request_latency" in body


# ---------------------------------------------------------------------------
# Root convenience route
# ---------------------------------------------------------------------------


def test_root(app_with_stub_predictor):
    client, _ = app_with_stub_predictor
    r = client.get("/")
    assert r.status_code == 200
    j = r.json()
    assert j["service"] == "meshwatch"


# ---------------------------------------------------------------------------
# Middleware (X-Request-ID + X-Response-Time-MS)
# ---------------------------------------------------------------------------


def test_request_id_header_set(app_with_stub_predictor):
    client, _ = app_with_stub_predictor
    r = client.get("/api/v1/health")
    assert "x-request-id" in r.headers
    assert "x-response-time-ms" in r.headers
    assert float(r.headers["x-response-time-ms"]) >= 0


def test_request_id_passes_through(app_with_stub_predictor):
    client, _ = app_with_stub_predictor
    r = client.get("/api/v1/health", headers={"x-request-id": "test-12345"})
    assert r.headers["x-request-id"] == "test-12345"


# ---------------------------------------------------------------------------
# WebSocket /ws/alerts
# ---------------------------------------------------------------------------


def test_websocket_receives_alert(app_with_stub_predictor):
    """End-to-end WS path: high-score POST -> producer -> consumer -> broadcaster -> ws.

    The full pipeline is async; we POST a transaction whose score crosses
    the alert threshold and then wait for the WS message. The
    in-memory consumer drains every ~1s by default, but we drop the
    poll interval so the test stays fast.
    """
    client, state = app_with_stub_predictor
    state.consumer.poll_timeout_seconds = 0.05  # tighten poll loop for the test
    with client.websocket_connect("/ws/alerts") as ws:
        assert state.broadcaster.n_clients == 1
        # Score = 0.99 (transaction_amt 500 -> stub_score = min(0.99, 0.5 + 0.5))
        r = client.post(
            "/api/v1/predict",
            json={"transaction_id": "ws-1", "transaction_dt": 1, "transaction_amt": 500.0},
        )
        assert r.status_code == 200
        msg = ws.receive_json()
        assert msg["transaction_id"] == "ws-1"
        assert msg["risk_level"] in ("HIGH", "CRITICAL")
