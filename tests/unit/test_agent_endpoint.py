"""Integration tests for the /api/v1/investigate endpoint (Phase 5)."""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from fastapi.testclient import TestClient

from fraud_detection.serving.app import AppState, create_app


@pytest.fixture
def app_no_predictor(monkeypatch):
    """Bring up the app without a real model. The agent endpoint should
    still work as long as the caller supplies a prediction inline."""
    monkeypatch.setenv("FRAUD_ENSEMBLE_DIR", "/nonexistent/path/skip-load")
    app = create_app()
    with TestClient(app) as client:
        state: AppState = app.state.fraud_app
        yield client, state


def _payload(score: float = 0.95, risk: str = "CRITICAL") -> dict:
    return {
        "transaction": {
            "transaction_id": 42,
            "transaction_dt": 100,
            "transaction_amt": 4210.0,
            "card1": 9999,
        },
        "prediction": {
            "transaction_id": 42,
            "fraud_probability": score,
            "fraud_score": score,
            "risk_level": risk,
            "is_fraud_predicted": score >= 0.7,
            "threshold": 0.7,
            "top_features": [],
            "latency_ms": {},
            "model_version": "v0.3.0-gnn-model",
        },
        "alert_id": "inv-test-1",
    }


def test_investigate_returns_structured_report(app_no_predictor) -> None:
    client, state = app_no_predictor
    if state.agent_compiled is None:
        pytest.skip("agent extras not loaded in this environment")

    r = client.post("/api/v1/investigate", json=_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["alert_id"] == "inv-test-1"
    assert body["risk_level"] == "CRITICAL"
    assert body["depth"] == "deep"
    assert body["recommended_action"] in ("approve", "review", "decline", "escalate")
    assert body["confidence"] >= 0
    assert body["requires_human_review"] is True
    assert isinstance(body["entity_risks"], list)
    assert len(body["entity_risks"]) == 5
    assert "tools_used" in body


def test_investigate_low_risk_path(app_no_predictor) -> None:
    client, state = app_no_predictor
    if state.agent_compiled is None:
        pytest.skip("agent extras not loaded")
    r = client.post("/api/v1/investigate", json=_payload(score=0.1, risk="LOW"))
    assert r.status_code == 200
    body = r.json()
    assert body["depth"] == "quick"
    assert body["requires_human_review"] is False


def test_investigate_503_when_agent_disabled(monkeypatch) -> None:
    monkeypatch.setenv("FRAUD_ENSEMBLE_DIR", "/nonexistent/path/skip-load")
    monkeypatch.setenv("FRAUD_AGENT_DISABLED", "true")
    app = create_app()
    with TestClient(app) as client:
        r = client.post("/api/v1/investigate", json=_payload())
        assert r.status_code == 503


def test_investigate_400_when_neither_transaction_nor_prediction(app_no_predictor) -> None:
    client, state = app_no_predictor
    if state.agent_compiled is None:
        pytest.skip("agent extras not loaded")
    # Empty body -- pydantic validator rejects.
    r = client.post("/api/v1/investigate", json={})
    assert r.status_code == 422


def test_investigate_400_when_only_transaction_and_no_predictor(app_no_predictor) -> None:
    client, state = app_no_predictor
    if state.agent_compiled is None:
        pytest.skip("agent extras not loaded")
    payload = {
        "transaction": {
            "transaction_id": 1,
            "transaction_dt": 1,
            "transaction_amt": 10.0,
        }
    }
    r = client.post("/api/v1/investigate", json=payload)
    # No model loaded + no prediction inline -> 400.
    assert r.status_code == 400
