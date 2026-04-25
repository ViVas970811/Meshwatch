"""Tests for ``fraud_detection.serving.schemas``."""

from __future__ import annotations

from datetime import datetime

import pytest

from fraud_detection.serving.schemas import (
    ALERT_THRESHOLD,
    BatchPredictRequest,
    FeatureContribution,
    FraudAlert,
    FraudPrediction,
    HealthStatus,
    ModelInfoResponse,
    TransactionRequest,
    risk_level,
)

# ---------------------------------------------------------------------------
# TransactionRequest
# ---------------------------------------------------------------------------


def test_minimal_valid_request():
    req = TransactionRequest(
        transaction_id=1,
        transaction_dt=12345,
        transaction_amt=42.5,
    )
    assert req.transaction_id == 1
    assert req.product_cd == "W"  # default


def test_request_accepts_string_id():
    req = TransactionRequest(
        transaction_id="abc-uuid-123",
        transaction_dt=1,
        transaction_amt=10.0,
    )
    assert req.transaction_id == "abc-uuid-123"


def test_request_accepts_extra_v_features():
    req = TransactionRequest(
        transaction_id=1,
        transaction_dt=1,
        transaction_amt=10.0,
        V1=0.5,  # extra V-feature passes through
        V99=-2.0,
        unknown_field="banana",
    )
    dump = req.model_dump()
    assert dump["V1"] == 0.5
    assert dump["unknown_field"] == "banana"


def test_request_validates_negative_amount():
    with pytest.raises(ValueError):
        TransactionRequest(transaction_id=1, transaction_dt=1, transaction_amt=-1.0)


def test_request_validates_negative_dt():
    with pytest.raises(ValueError):
        TransactionRequest(transaction_id=1, transaction_dt=-1, transaction_amt=10.0)


def test_request_aliases_email_fields():
    req = TransactionRequest(
        transaction_id=1,
        transaction_dt=1,
        transaction_amt=10.0,
        P_emaildomain="gmail.com",  # alias
        DeviceType="desktop",
    )
    assert req.p_emaildomain == "gmail.com"
    assert req.device_type == "desktop"


# ---------------------------------------------------------------------------
# BatchPredictRequest
# ---------------------------------------------------------------------------


def _stub_request(i: int = 1) -> TransactionRequest:
    return TransactionRequest(transaction_id=i, transaction_dt=i, transaction_amt=10.0)


def test_batch_request_min_size_one():
    req = BatchPredictRequest(transactions=[_stub_request(1)])
    assert len(req.transactions) == 1


def test_batch_request_max_size_100():
    big = [_stub_request(i) for i in range(100)]
    BatchPredictRequest(transactions=big)  # ok
    with pytest.raises(ValueError):
        BatchPredictRequest(transactions=[*big, _stub_request(101)])


def test_batch_request_empty_rejected():
    with pytest.raises(ValueError):
        BatchPredictRequest(transactions=[])


# ---------------------------------------------------------------------------
# FraudPrediction
# ---------------------------------------------------------------------------


def test_prediction_round_trip_json():
    pred = FraudPrediction(
        transaction_id=42,
        fraud_probability=0.85,
        fraud_score=0.85,
        risk_level="HIGH",
        is_fraud_predicted=True,
        threshold=0.7,
        top_features=[FeatureContribution(feature="V1", value=0.5, contribution=0.2)],
        latency_ms={"total_ms": 12.3},
    )
    s = pred.model_dump_json()
    parsed = FraudPrediction.model_validate_json(s)
    assert parsed.transaction_id == 42
    assert parsed.fraud_score == pytest.approx(0.85)
    assert parsed.top_features[0].feature == "V1"


def test_prediction_validates_proba_range():
    with pytest.raises(ValueError):
        FraudPrediction(
            transaction_id=1,
            fraud_probability=1.5,
            fraud_score=0.5,
            risk_level="HIGH",
            is_fraud_predicted=True,
        )


# ---------------------------------------------------------------------------
# risk_level
# ---------------------------------------------------------------------------


def test_risk_level_buckets():
    assert risk_level(0.0) == "LOW"
    assert risk_level(0.39) == "LOW"
    assert risk_level(0.4) == "MEDIUM"
    assert risk_level(0.69) == "MEDIUM"
    assert risk_level(0.7) == "HIGH"
    assert risk_level(0.89) == "HIGH"
    assert risk_level(0.9) == "CRITICAL"
    assert risk_level(1.0) == "CRITICAL"


def test_alert_threshold_constant():
    assert 0.5 < ALERT_THRESHOLD < 1.0


# ---------------------------------------------------------------------------
# FraudAlert + HealthStatus + ModelInfo
# ---------------------------------------------------------------------------


def test_fraud_alert_round_trip():
    a = FraudAlert(
        transaction_id="abc",
        fraud_score=0.92,
        risk_level="CRITICAL",
        transaction_amt=500.0,
        card_id=12345,
    )
    s = a.model_dump_json()
    parsed = FraudAlert.model_validate_json(s)
    assert parsed.fraud_score == pytest.approx(0.92)
    assert parsed.risk_level == "CRITICAL"
    assert isinstance(parsed.timestamp, datetime)


def test_health_status_defaults():
    h = HealthStatus()
    assert h.status == "ok"
    assert h.uptime_seconds == 0.0


def test_model_info_excludes_protected_namespaces():
    """ModelInfoResponse uses ``protected_namespaces=()`` so ``model_*`` fields work."""
    info = ModelInfoResponse(
        model_version="vX",
        n_parameters=1,
        embedding_dim=64,
        n_features=10,
        feature_columns=["a", "b"],
        edge_types=["x"],
        node_types=["y"],
    )
    assert info.model_version == "vX"
