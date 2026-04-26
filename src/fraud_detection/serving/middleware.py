"""FastAPI middleware: request timing, structured-logging access logs, Prometheus metrics.

Designed so the metrics module is importable without breaking the app
when ``prometheus-client`` is missing (Phase 7 makes it mandatory; for
Phase 4 we keep it optional).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prometheus integration -- optional
# ---------------------------------------------------------------------------


class _MetricsRegistry:
    """Thin wrapper around prometheus_client that no-ops when the lib is missing.

    We expose ``request_count``, ``request_latency``, ``in_flight``,
    ``predictions_total``, ``alerts_total`` as instance attributes whose
    ``.labels(...).inc()`` etc. methods are either real Prometheus
    objects or harmless stubs.
    """

    def __init__(self) -> None:
        try:
            from prometheus_client import (
                CONTENT_TYPE_LATEST,
                CollectorRegistry,
                Counter,
                Gauge,
                Histogram,
                generate_latest,
            )

            self._enabled = True
            self.registry = CollectorRegistry()
            self.request_count = Counter(
                "meshwatch_http_requests_total",
                "HTTP requests",
                ["method", "path", "status"],
                registry=self.registry,
            )
            self.request_latency = Histogram(
                "meshwatch_http_request_latency_seconds",
                "HTTP request latency",
                ["method", "path"],
                buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
                registry=self.registry,
            )
            self.in_flight = Gauge(
                "meshwatch_http_in_flight_requests",
                "In-flight HTTP requests",
                registry=self.registry,
            )
            self.predictions_total = Counter(
                "meshwatch_predictions_total",
                "Predictions served",
                ["risk_level"],
                registry=self.registry,
            )
            self.alerts_total = Counter(
                "meshwatch_alerts_published_total",
                "Fraud alerts published to Kafka / WebSocket",
                ["channel"],
                registry=self.registry,
            )
            self._generate_latest = generate_latest
            self.content_type = CONTENT_TYPE_LATEST
        except ImportError:
            self._enabled = False
            self._generate_latest = None
            self.content_type = "text/plain; version=0.0.4; charset=utf-8"
            self.request_count = _NoopMetric()
            self.request_latency = _NoopMetric()
            self.in_flight = _NoopMetric()
            self.predictions_total = _NoopMetric()
            self.alerts_total = _NoopMetric()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def render(self) -> bytes:
        if self._enabled and self._generate_latest is not None:
            return self._generate_latest(self.registry)
        return b"# prometheus_client not installed\n"


class _NoopMetric:
    """Stand-in for Prometheus metrics when the library isn't installed."""

    def labels(self, *args: Any, **kwargs: Any) -> _NoopMetric:
        return self

    def inc(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def dec(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def observe(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set(self, *_args: Any, **_kwargs: Any) -> None:
        return None


# Module-level singleton -- the FastAPI app reaches into this for
# /api/v1/metrics rendering.
metrics = _MetricsRegistry()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Add ``X-Request-ID`` + ``X-Response-Time-MS`` headers and log every request."""

    HEADER_REQUEST_ID = "x-request-id"
    HEADER_RESPONSE_TIME = "x-response-time-ms"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(self.HEADER_REQUEST_ID) or uuid.uuid4().hex
        start = time.perf_counter()
        path_template = request.url.path
        method = request.method

        metrics.in_flight.inc()
        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.exception(
                "http_request_failed",
                request_id=request_id,
                method=method,
                path=path_template,
                error=str(exc),
                elapsed_ms=elapsed_ms,
            )
            metrics.request_count.labels(method, path_template, "500").inc()
            metrics.request_latency.labels(method, path_template).observe(elapsed_ms / 1000)
            raise
        finally:
            metrics.in_flight.dec()

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers[self.HEADER_REQUEST_ID] = request_id
        response.headers[self.HEADER_RESPONSE_TIME] = f"{elapsed_ms:.2f}"

        metrics.request_count.labels(method, path_template, str(response.status_code)).inc()
        metrics.request_latency.labels(method, path_template).observe(elapsed_ms / 1000)

        log.info(
            "http_request_complete",
            request_id=request_id,
            method=method,
            path=path_template,
            status=response.status_code,
            elapsed_ms=round(elapsed_ms, 3),
        )
        return response


__all__ = ["RequestTimingMiddleware", "metrics"]
