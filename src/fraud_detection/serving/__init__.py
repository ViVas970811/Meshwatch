"""FastAPI + Ray Serve serving stack (Phase 4).

Public surface (resolved lazily via PEP 562 ``__getattr__``):

* :class:`FraudPredictor` -- core scoring path (Feast + cache + ensemble + SHAP).
* :class:`EmbeddingCache` -- Redis-backed cache with in-memory fallback.
* :class:`AlertBroadcaster` -- WebSocket fan-out for ``/ws/alerts``.
* :class:`RequestTimingMiddleware` -- structured-log access logs + Prometheus metrics.
* :func:`create_app` -- FastAPI factory.
* Pydantic schemas for request/response/streaming.

We resolve attributes lazily so that callers wanting only a single piece
(say :data:`fraud_detection.serving.middleware.metrics`) don't have to
pull in the whole app graph -- which would otherwise create a cycle with
:mod:`fraud_detection.monitoring`.
"""

from __future__ import annotations

from typing import Any

_ATTR_MAP: dict[str, tuple[str, str]] = {
    # name -> (submodule, attribute)
    "AlertBroadcaster": ("fraud_detection.serving.app", "AlertBroadcaster"),
    "AppState": ("fraud_detection.serving.app", "AppState"),
    "app": ("fraud_detection.serving.app", "app"),
    "create_app": ("fraud_detection.serving.app", "create_app"),
    "RequestTimingMiddleware": ("fraud_detection.serving.middleware", "RequestTimingMiddleware"),
    "metrics": ("fraud_detection.serving.middleware", "metrics"),
    "FraudPredictor": ("fraud_detection.serving.predictor", "FraudPredictor"),
    "load_predictor": ("fraud_detection.serving.predictor", "load_predictor"),
    "EmbeddingCache": ("fraud_detection.serving.redis_cache", "EmbeddingCache"),
    "ALERT_THRESHOLD": ("fraud_detection.serving.schemas", "ALERT_THRESHOLD"),
    "BatchPredictRequest": ("fraud_detection.serving.schemas", "BatchPredictRequest"),
    "BatchPredictResponse": ("fraud_detection.serving.schemas", "BatchPredictResponse"),
    "FeatureContribution": ("fraud_detection.serving.schemas", "FeatureContribution"),
    "FraudAlert": ("fraud_detection.serving.schemas", "FraudAlert"),
    "FraudPrediction": ("fraud_detection.serving.schemas", "FraudPrediction"),
    "HealthStatus": ("fraud_detection.serving.schemas", "HealthStatus"),
    "ModelInfoResponse": ("fraud_detection.serving.schemas", "ModelInfoResponse"),
    "TransactionRequest": ("fraud_detection.serving.schemas", "TransactionRequest"),
    "risk_level": ("fraud_detection.serving.schemas", "risk_level"),
}


def __getattr__(name: str) -> Any:
    target = _ATTR_MAP.get(name)
    if target is None:
        raise AttributeError(f"module 'fraud_detection.serving' has no attribute {name!r}")
    module_name, attr_name = target
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value  # cache so __getattr__ isn't called again
    return value


__all__ = sorted(_ATTR_MAP.keys())
