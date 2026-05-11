"""Integration tests for the Phase 7 monitoring endpoints."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud_detection.monitoring import (
    DriftDetector,
    DriftReport,
    PerformanceTracker,
    reset_state,
)
from fraud_detection.serving.app import AppState, create_app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FRAUD_ENSEMBLE_DIR", "/nonexistent/path/skip-load")
    monkeypatch.setenv("FRAUD_AGENT_DISABLED", "true")
    reset_state()  # isolate the monitoring singleton between tests
    app = create_app()
    with TestClient(app) as c:
        # Give the app a fresh, empty tracker so we can record clean states.
        state: AppState = app.state.fraud_app
        state.monitoring.performance = PerformanceTracker(threshold=0.5)
        yield c, state


def _seed_drift_report(state: AppState) -> DriftReport:
    rng = np.random.default_rng(3)
    ref = {
        "amount": rng.normal(loc=0, scale=1, size=500).tolist(),
        "country": rng.choice(["US", "GB"], size=500).tolist(),
    }
    cur = {
        "amount": rng.normal(loc=4, scale=1, size=500).tolist(),
        "country": rng.choice(["US", "GB"], size=500).tolist(),
    }
    report = DriftDetector(ref).detect(cur)
    state.monitoring.last_drift_report = report
    state.monitoring.drift_detector = DriftDetector(ref)
    return report


class TestDriftEndpoints:
    def test_drift_endpoint_returns_no_report_initially(self, client):
        c, _state = client
        resp = c.get("/api/v1/monitoring/drift")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "no_report"

    def test_drift_endpoint_returns_report_when_seeded(self, client):
        c, state = client
        seeded = _seed_drift_report(state)
        resp = c.get("/api/v1/monitoring/drift")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_features"] == seeded.n_features
        assert body["severity"] in {"none", "moderate", "severe"}

    def test_drift_html_endpoint_returns_html(self, client):
        c, state = client
        _seed_drift_report(state)
        resp = c.get("/api/v1/monitoring/drift.html")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert resp.text.startswith("<!doctype html>")
        assert "Overall PSI" in resp.text

    def test_drift_html_endpoint_returns_placeholder_when_empty(self, client):
        c, _state = client
        resp = c.get("/api/v1/monitoring/drift.html")
        assert resp.status_code == 200
        assert "No drift report" in resp.text


class TestPerformanceEndpoint:
    def test_performance_endpoint_returns_empty_snapshot(self, client):
        c, _state = client
        resp = c.get("/api/v1/monitoring/performance")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_labelled"] == 0
        assert body["n_total"] == 0

    def test_performance_endpoint_reports_metrics_after_labels(self, client):
        c, state = client
        # Manually populate the tracker.
        state.monitoring.performance.record_prediction("tx_1", 0.9, label=1)
        state.monitoring.performance.record_prediction("tx_2", 0.2, label=0)
        state.monitoring.performance.record_prediction("tx_3", 0.6, label=1)
        resp = c.get("/api/v1/monitoring/performance")
        body = resp.json()
        assert body["n_labelled"] == 3
        assert body["precision"] > 0
        assert body["recall"] > 0


class TestLabelEndpoint:
    def test_label_endpoint_attaches_label(self, client):
        c, state = client
        state.monitoring.performance.record_prediction("tx_lbl", 0.9)
        resp = c.post(
            "/api/v1/monitoring/label",
            json={"transaction_id": "tx_lbl", "label": 1},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["found"] is True
        assert body["label"] == 1
        assert state.monitoring.performance.n_labelled == 1

    def test_label_endpoint_400_when_missing_fields(self, client):
        c, _state = client
        resp = c.post("/api/v1/monitoring/label", json={"transaction_id": "x"})
        assert resp.status_code == 400

    def test_label_endpoint_rejects_invalid_label(self, client):
        c, _state = client
        resp = c.post(
            "/api/v1/monitoring/label",
            json={"transaction_id": "x", "label": 2},
        )
        assert resp.status_code == 400

    def test_label_endpoint_handles_unknown_transaction(self, client):
        c, _state = client
        resp = c.post(
            "/api/v1/monitoring/label",
            json={"transaction_id": "never_existed", "label": 1},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["found"] is False


class TestAlertsEndpoint:
    def test_alerts_endpoint_returns_empty_on_healthy_state(self, client):
        c, state = client
        state.predictor = object()  # type: ignore[assignment] -- truthy stand-in
        resp = c.get("/api/v1/monitoring/alerts")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_active"] == 0
        assert body["alerts"] == []

    def test_alerts_endpoint_fires_on_model_unloaded(self, client):
        c, state = client
        state.predictor = None  # type: ignore[assignment]
        resp = c.get("/api/v1/monitoring/alerts")
        body = resp.json()
        names = {a["name"] for a in body["alerts"]}
        assert "ModelNotLoaded" in names

    def test_alerts_endpoint_includes_drift_alert_when_severe(self, client):
        c, state = client
        state.predictor = object()  # type: ignore[assignment]
        _seed_drift_report(state)  # this builds a severe-drift report
        resp = c.get("/api/v1/monitoring/alerts")
        body = resp.json()
        names = {a["name"] for a in body["alerts"]}
        assert "DataDriftSevere" in names


class TestPrometheusExposition:
    def test_metrics_endpoint_includes_meshwatch_namespace(self, client):
        c, _state = client
        resp = c.get("/api/v1/metrics")
        assert resp.status_code == 200
        # Without prometheus_client the endpoint returns a no-op placeholder;
        # with it installed, we get the meshwatch_* namespace.
        body = resp.text
        assert "meshwatch_http_requests_total" in body or "prometheus_client not installed" in body


class TestShadowEndpoints:
    def test_shadow_summary_disabled_when_not_attached(self, client):
        c, _state = client
        resp = c.get("/api/v1/monitoring/shadow")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "disabled"

    def test_shadow_summary_returns_aggregates_when_attached(self, client):
        from fraud_detection.monitoring import ShadowDeployment

        c, state = client

        class _Pred:
            model_version = "test-v1"
            threshold = 0.7

            def predict_one(self, req):
                from fraud_detection.serving.schemas import FraudPrediction, risk_level

                score = min(0.99, req.transaction_amt / 1000.0)
                return FraudPrediction(
                    transaction_id=req.transaction_id,
                    fraud_probability=score,
                    fraud_score=score,
                    risk_level=risk_level(score),
                    is_fraud_predicted=score >= self.threshold,
                    threshold=self.threshold,
                    top_features=[],
                    latency_ms={"total_ms": 0.1},
                    model_version=self.model_version,
                )

        deployment = ShadowDeployment(champion=_Pred(), challenger=_Pred(), max_records=10)  # type: ignore[arg-type]
        try:
            state.monitoring.shadow = deployment
            resp = c.get("/api/v1/monitoring/shadow")
            assert resp.status_code == 200
            body = resp.json()
            assert "champion_model" in body
            assert body["champion_model"] == "test-v1"

            resp = c.get("/api/v1/monitoring/shadow/recent?limit=5")
            assert resp.status_code == 200
            body = resp.json()
            assert body["n"] == 0
        finally:
            deployment.shutdown(wait=True)
            state.monitoring.shadow = None
