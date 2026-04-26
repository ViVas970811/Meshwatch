"""FastAPI + Ray Serve serving stack (Phase 4).

Public surface:

* :class:`FraudPredictor` -- core scoring path (Feast + cache + ensemble + SHAP).
* :class:`EmbeddingCache` -- Redis-backed cache with in-memory fallback.
* :class:`AlertBroadcaster` -- WebSocket fan-out for ``/ws/alerts``.
* :class:`RequestTimingMiddleware` -- structured-log access logs + Prometheus metrics.
* :func:`create_app` -- FastAPI factory.
* :func:`build_deployment` -- optional Ray Serve wrapper.
* Pydantic schemas for request/response/streaming.
"""

from fraud_detection.serving.app import (
    AlertBroadcaster,
    AppState,
    app,
    create_app,
)
from fraud_detection.serving.middleware import RequestTimingMiddleware, metrics
from fraud_detection.serving.predictor import FraudPredictor, load_predictor
from fraud_detection.serving.redis_cache import EmbeddingCache
from fraud_detection.serving.schemas import (
    ALERT_THRESHOLD,
    BatchPredictRequest,
    BatchPredictResponse,
    FeatureContribution,
    FraudAlert,
    FraudPrediction,
    HealthStatus,
    ModelInfoResponse,
    TransactionRequest,
    risk_level,
)

__all__ = [
    "ALERT_THRESHOLD",
    "AlertBroadcaster",
    "AppState",
    "BatchPredictRequest",
    "BatchPredictResponse",
    "EmbeddingCache",
    "FeatureContribution",
    "FraudAlert",
    "FraudPrediction",
    "FraudPredictor",
    "HealthStatus",
    "ModelInfoResponse",
    "RequestTimingMiddleware",
    "TransactionRequest",
    "app",
    "create_app",
    "load_predictor",
    "metrics",
    "risk_level",
]
