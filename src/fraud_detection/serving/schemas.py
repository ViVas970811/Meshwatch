"""Pydantic request/response schemas for the fraud-detection serving API.

These mirror the public contract documented in README/Phase 4 of the plan.
The shapes here drive both FastAPI's auto-generated OpenAPI docs and the
TypeScript client used by the dashboard in Phase 6.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class TransactionRequest(BaseModel):
    """A single transaction submitted for fraud scoring.

    All optional fields mirror IEEE-CIS feature names. Anything unknown at
    request time is filled in by the upstream pre-processing pipeline.
    """

    model_config = ConfigDict(extra="allow")  # tolerate extra V*/D*/M*/id_* fields

    # Mandatory identifiers
    transaction_id: int | str = Field(..., description="Unique transaction identifier")
    transaction_dt: int = Field(..., ge=0, description="Seconds since the IEEE-CIS reference epoch")
    transaction_amt: float = Field(..., ge=0.0, description="Transaction amount in USD")
    product_cd: str = Field("W", description="Product type (W/C/H/S/R)")

    # Card / address
    card1: int | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    dist2: float | None = None

    # Email
    p_emaildomain: str | None = Field(default=None, alias="P_emaildomain")
    r_emaildomain: str | None = Field(default=None, alias="R_emaildomain")

    # Identity (subset)
    device_type: str | None = Field(default=None, alias="DeviceType")
    device_info: str | None = Field(default=None, alias="DeviceInfo")

    # Optional: client-supplied wall-clock timestamp (ISO-8601). Used for
    # logging + temporal-feature freshness sanity checks.
    received_at: datetime | None = None

    @field_validator("transaction_id", mode="before")
    @classmethod
    def _normalise_id(cls, v: Any) -> int | str:
        # Accept ints from old IEEE-CIS dumps and strings (UUIDs) from
        # production producers.
        if isinstance(v, (int, str)):
            return v
        return str(v)


class BatchPredictRequest(BaseModel):
    """Up to ``MAX_BATCH_SIZE`` transactions in one round-trip."""

    MAX_BATCH_SIZE: int = Field(default=100, exclude=True)
    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=100)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class FeatureContribution(BaseModel):
    """One row of the SHAP-style explanation table."""

    feature: str
    value: float
    contribution: float


class FraudPrediction(BaseModel):
    """Scoring result for one transaction."""

    transaction_id: int | str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    is_fraud_predicted: bool
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Optional explanation -- populated when SHAP is available.
    top_features: list[FeatureContribution] = Field(default_factory=list)

    # Latency breakdown in milliseconds (per the Phase 4 budget table).
    latency_ms: dict[str, float] = Field(default_factory=dict)

    # Model + serving metadata.
    model_version: str = "v0.3.0-gnn-model"
    served_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchPredictResponse(BaseModel):
    predictions: list[FraudPrediction]
    n_processed: int
    n_alerts: int = Field(0, description="Predictions exceeding the alert threshold")
    elapsed_ms: float


# ---------------------------------------------------------------------------
# System-status models
# ---------------------------------------------------------------------------


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded", "down"] = "ok"
    model_loaded: bool = False
    redis_connected: bool = False
    kafka_connected: bool = False
    ray_serve_active: bool = False
    uptime_seconds: float = 0.0
    version: str = "v0.4.0-serving-pipeline"


class ModelInfoResponse(BaseModel):
    """Returned by ``GET /api/v1/model/info``."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    n_parameters: int
    embedding_dim: int
    n_features: int
    feature_columns: list[str]
    edge_types: list[str]
    node_types: list[str]
    train_metrics: dict[str, float] = Field(default_factory=dict)
    feature_importance_top_k: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Streaming / WebSocket
# ---------------------------------------------------------------------------


class FraudAlert(BaseModel):
    """Pushed to the ``fraud_alerts`` Kafka topic and the ``/ws/alerts``
    WebSocket whenever a transaction crosses the alert threshold."""

    transaction_id: int | str
    fraud_score: float
    risk_level: Literal["MEDIUM", "HIGH", "CRITICAL"]
    transaction_amt: float
    card_id: str | int | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    top_features: list[FeatureContribution] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALERT_THRESHOLD: float = 0.7
"""Predictions at or above this score trigger a Kafka alert + WS push."""


def risk_level(score: float) -> Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
    """Bin a fraud score into the analyst-facing 4-level scale."""
    if score >= 0.9:
        return "CRITICAL"
    if score >= 0.7:
        return "HIGH"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Agent investigation (Phase 5)
# ---------------------------------------------------------------------------


class InvestigationRequest(BaseModel):
    """Trigger an agent investigation for a transaction.

    Either supply a fully-formed :class:`TransactionRequest` (the API
    will re-score it under the hood) or supply an existing
    :class:`FraudPrediction` to skip re-scoring.
    """

    transaction: TransactionRequest | None = None
    prediction: FraudPrediction | None = None
    alert_id: str | None = Field(
        default=None,
        description="Optional caller-supplied alert id (defaults to ``inv-{transaction_id}``).",
    )

    @model_validator(mode="after")
    def _require_one_of_transaction_or_prediction(self) -> InvestigationRequest:
        if self.transaction is None and self.prediction is None:
            raise ValueError("Provide either 'transaction' or 'prediction' (or both).")
        return self


__all__ = [
    "ALERT_THRESHOLD",
    "BatchPredictRequest",
    "BatchPredictResponse",
    "FeatureContribution",
    "FraudAlert",
    "FraudPrediction",
    "HealthStatus",
    "InvestigationRequest",
    "ModelInfoResponse",
    "TransactionRequest",
    "risk_level",
]
