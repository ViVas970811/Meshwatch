"""Unit tests for Phase 8 security middleware."""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fraud_detection.serving.security import (
    ApiKeyAuthMiddleware,
    RateLimitMiddleware,
)


def _build_app(*middlewares) -> FastAPI:
    app = FastAPI()
    for mw, kwargs in middlewares:
        app.add_middleware(mw, **kwargs)

    @app.get("/api/v1/predict")
    async def predict() -> dict:
        return {"ok": True}

    @app.get("/api/v1/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/v1/metrics")
    async def metrics() -> dict:
        return {"metrics": "ok"}

    return app


# ---------------------------------------------------------------------------
# ApiKeyAuthMiddleware
# ---------------------------------------------------------------------------


class TestApiKeyAuth:
    def test_no_keys_configured_is_a_no_op(self):
        app = _build_app((ApiKeyAuthMiddleware, {"api_keys": []}))
        with TestClient(app) as c:
            assert c.get("/api/v1/predict").status_code == 200

    def test_missing_key_returns_401(self):
        app = _build_app((ApiKeyAuthMiddleware, {"api_keys": ["secret-1"]}))
        with TestClient(app) as c:
            resp = c.get("/api/v1/predict")
            assert resp.status_code == 401
            assert resp.json()["detail"] == "Missing API key"

    def test_invalid_key_returns_403(self):
        app = _build_app((ApiKeyAuthMiddleware, {"api_keys": ["secret-1"]}))
        with TestClient(app) as c:
            resp = c.get("/api/v1/predict", headers={"X-API-Key": "wrong"})
            assert resp.status_code == 403
            assert resp.json()["detail"] == "Invalid API key"

    def test_valid_x_api_key_header_allows_request(self):
        app = _build_app((ApiKeyAuthMiddleware, {"api_keys": ["secret-1"]}))
        with TestClient(app) as c:
            resp = c.get("/api/v1/predict", headers={"X-API-Key": "secret-1"})
            assert resp.status_code == 200

    def test_bearer_authorization_is_accepted(self):
        app = _build_app((ApiKeyAuthMiddleware, {"api_keys": ["secret-1"]}))
        with TestClient(app) as c:
            resp = c.get("/api/v1/predict", headers={"Authorization": "Bearer secret-1"})
            assert resp.status_code == 200

    def test_health_and_metrics_are_exempt(self):
        app = _build_app((ApiKeyAuthMiddleware, {"api_keys": ["secret-1"]}))
        with TestClient(app) as c:
            assert c.get("/api/v1/health").status_code == 200
            assert c.get("/api/v1/metrics").status_code == 200

    def test_custom_exempt_paths(self):
        app = _build_app(
            (
                ApiKeyAuthMiddleware,
                {"api_keys": ["secret-1"], "exempt_paths": ["/api/v1/predict"]},
            )
        )
        with TestClient(app) as c:
            # Predict is now exempt, no key required.
            assert c.get("/api/v1/predict").status_code == 200

    def test_case_insensitive_header_match(self):
        app = _build_app((ApiKeyAuthMiddleware, {"api_keys": ["secret-1"]}))
        with TestClient(app) as c:
            resp = c.get("/api/v1/predict", headers={"x-api-key": "secret-1"})
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# RateLimitMiddleware
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_under_limit_passes_through(self):
        app = _build_app((RateLimitMiddleware, {"rate": 100.0, "burst": 100}))
        with TestClient(app) as c:
            for _ in range(50):
                assert c.get("/api/v1/predict").status_code == 200

    def test_burst_capacity_is_enforced(self):
        # rate=1, burst=3 -- 3 fast requests succeed, the 4th is throttled.
        app = _build_app((RateLimitMiddleware, {"rate": 1.0, "burst": 3}))
        with TestClient(app) as c:
            statuses = [c.get("/api/v1/predict").status_code for _ in range(5)]
            assert statuses[:3] == [200, 200, 200]
            assert 429 in statuses[3:]

    def test_throttled_response_includes_retry_after_header(self):
        app = _build_app((RateLimitMiddleware, {"rate": 0.5, "burst": 1}))
        with TestClient(app) as c:
            assert c.get("/api/v1/predict").status_code == 200
            resp = c.get("/api/v1/predict")
            assert resp.status_code == 429
            assert "retry-after" in {k.lower() for k in resp.headers}

    def test_tokens_refill_over_time(self):
        # rate=20/s, burst=1 -> after one denial, ~50ms later we should pass.
        app = _build_app((RateLimitMiddleware, {"rate": 20.0, "burst": 1}))
        with TestClient(app) as c:
            assert c.get("/api/v1/predict").status_code == 200
            assert c.get("/api/v1/predict").status_code == 429
            time.sleep(0.15)  # 3x the regen time, safe margin
            assert c.get("/api/v1/predict").status_code == 200

    def test_health_and_metrics_are_exempt(self):
        app = _build_app((RateLimitMiddleware, {"rate": 1.0, "burst": 1}))
        with TestClient(app) as c:
            for _ in range(20):
                assert c.get("/api/v1/health").status_code == 200
                assert c.get("/api/v1/metrics").status_code == 200

    def test_max_buckets_evicts_oldest(self):
        mw = RateLimitMiddleware(app=lambda *_: None, rate=10.0, burst=10, max_buckets=2)
        b1 = mw._bucket_for("ip:1.1.1.1")
        b2 = mw._bucket_for("ip:2.2.2.2")
        b3 = mw._bucket_for("ip:3.3.3.3")
        # The oldest (1.1.1.1) should have been evicted.
        assert "ip:1.1.1.1" not in mw._buckets
        assert "ip:2.2.2.2" in mw._buckets
        assert "ip:3.3.3.3" in mw._buckets
        # New look-up for the evicted ip is allowed to create a fresh bucket.
        b1_new = mw._bucket_for("ip:1.1.1.1")
        assert b1_new is not b1
        assert b2 is not None and b3 is not None  # silence unused-locals


# ---------------------------------------------------------------------------
# Auth + rate limit interaction
# ---------------------------------------------------------------------------


class TestSecurityStack:
    def test_api_key_identity_buckets_separately_from_ip(self):
        """Two keys should each get their own bucket -- one being throttled
        must not throttle the other."""
        # Middleware order: auth runs first (adds scope["api_key"]),
        # rate limiter keys on that.
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, rate=0.5, burst=1)
        app.add_middleware(ApiKeyAuthMiddleware, api_keys=["key-a", "key-b"])

        @app.get("/api/v1/predict")
        async def predict() -> dict:
            return {"ok": True}

        with TestClient(app) as c:
            # Burn key-a's bucket.
            assert c.get("/api/v1/predict", headers={"X-API-Key": "key-a"}).status_code == 200
            assert c.get("/api/v1/predict", headers={"X-API-Key": "key-a"}).status_code == 429
            # key-b still has tokens.
            assert c.get("/api/v1/predict", headers={"X-API-Key": "key-b"}).status_code == 200

    def test_configure_security_reads_env(self, monkeypatch):
        from fraud_detection.serving.security import configure_security

        monkeypatch.setenv("FRAUD_API_KEYS", "k1,k2")
        monkeypatch.setenv("FRAUD_RATE_LIMIT_RPS", "5")
        monkeypatch.setenv("FRAUD_RATE_LIMIT_BURST", "5")
        app = FastAPI()

        @app.get("/api/v1/predict")
        async def predict() -> dict:
            return {"ok": True}

        configure_security(app)
        with TestClient(app) as c:
            assert c.get("/api/v1/predict").status_code == 401  # no key
            assert c.get("/api/v1/predict", headers={"X-API-Key": "k1"}).status_code == 200

    def test_configure_security_disabled_rate_limit(self, monkeypatch):
        from fraud_detection.serving.security import configure_security

        monkeypatch.setenv("FRAUD_API_KEYS", "k1")
        monkeypatch.setenv("FRAUD_RATE_LIMIT_DISABLED", "true")
        app = FastAPI()

        @app.get("/api/v1/predict")
        async def predict() -> dict:
            return {"ok": True}

        configure_security(app)
        # Should still require API key but never throttle.
        with TestClient(app) as c:
            for _ in range(50):
                assert c.get("/api/v1/predict", headers={"X-API-Key": "k1"}).status_code == 200


# ---------------------------------------------------------------------------
# Token bucket internals
# ---------------------------------------------------------------------------


class TestTokenBucket:
    def test_token_bucket_take_returns_allowed_and_retry(self):
        from fraud_detection.serving.security import _TokenBucket

        b = _TokenBucket(capacity=2, rate=1.0)
        allowed, retry = b.take(now=0.0)
        assert allowed is True
        assert retry == 0.0
        b.take(now=0.0)  # use the second token
        allowed, retry = b.take(now=0.0)
        assert allowed is False
        assert retry > 0.0

    def test_token_bucket_refills_with_elapsed_time(self):
        from fraud_detection.serving.security import _TokenBucket

        b = _TokenBucket(capacity=1, rate=2.0)  # 0.5s per token
        b.take(now=0.0)  # empty
        allowed, _ = b.take(now=0.0)
        assert allowed is False
        # 0.5s later -- one token regenerated.
        allowed, _ = b.take(now=0.5)
        assert allowed is True


# ---------------------------------------------------------------------------
# Wired into the real app (smoke -- no auth, no limit by default)
# ---------------------------------------------------------------------------


@pytest.fixture
def fraud_app(monkeypatch):
    monkeypatch.setenv("FRAUD_ENSEMBLE_DIR", "/nonexistent/path/skip-load")
    monkeypatch.setenv("FRAUD_AGENT_DISABLED", "true")
    monkeypatch.delenv("FRAUD_API_KEYS", raising=False)
    monkeypatch.setenv("FRAUD_RATE_LIMIT_DISABLED", "true")
    from fraud_detection.monitoring import reset_state
    from fraud_detection.serving.app import create_app

    reset_state()
    return create_app()


def test_health_is_reachable_without_keys(fraud_app):
    with TestClient(fraud_app) as c:
        assert c.get("/api/v1/health").status_code == 200


def test_security_locks_predict_when_keys_configured(monkeypatch):
    monkeypatch.setenv("FRAUD_ENSEMBLE_DIR", "/nonexistent/path/skip-load")
    monkeypatch.setenv("FRAUD_AGENT_DISABLED", "true")
    monkeypatch.setenv("FRAUD_API_KEYS", "production-key")
    monkeypatch.setenv("FRAUD_RATE_LIMIT_DISABLED", "true")
    from fraud_detection.monitoring import reset_state
    from fraud_detection.serving.app import create_app

    reset_state()
    app = create_app()
    with TestClient(app) as c:
        # Health stays exempt.
        assert c.get("/api/v1/health").status_code == 200
        # Predict requires the key.
        resp = c.post("/api/v1/predict", json={"transaction_id": "tx_1"})
        assert resp.status_code == 401
