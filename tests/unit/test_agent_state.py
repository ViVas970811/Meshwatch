"""Tests for the AgentState + InvestigationReport schemas (Phase 5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fraud_detection.agent.state import (
    EntityRisk,
    FraudPattern,
    InvestigationReport,
    SimilarCase,
    new_state,
)
from fraud_detection.serving.schemas import FraudPrediction


def _pred(score: float = 0.5, risk: str = "MEDIUM") -> FraudPrediction:
    return FraudPrediction(
        transaction_id=42,
        fraud_probability=score,
        fraud_score=score,
        risk_level=risk,  # type: ignore[arg-type]
        is_fraud_predicted=score >= 0.7,
        threshold=0.7,
    )


# ---------------------------------------------------------------------------
# new_state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("risk", "expected_depth"),
    [
        ("LOW", "quick"),
        ("MEDIUM", "quick"),
        ("HIGH", "standard"),
        ("CRITICAL", "deep"),
    ],
)
def test_new_state_picks_depth_from_risk(risk: str, expected_depth: str) -> None:
    state = new_state(transaction_id=42, prediction=_pred(0.55, risk))
    assert state["depth"] == expected_depth


def test_new_state_high_or_critical_requires_human_review() -> None:
    high = new_state(transaction_id=1, prediction=_pred(0.8, "HIGH"))
    crit = new_state(transaction_id=2, prediction=_pred(0.95, "CRITICAL"))
    low = new_state(transaction_id=3, prediction=_pred(0.1, "LOW"))
    assert high["requires_human_review"]
    assert crit["requires_human_review"]
    assert not low["requires_human_review"]


def test_new_state_initialises_evidence_and_tools() -> None:
    state = new_state(transaction_id=7, prediction=_pred(0.8, "HIGH"))
    assert state["evidence"] == {}
    assert state["tool_calls"] == []
    assert state["report"] == {}
    assert state["errors"] == []


def test_new_state_serialises_prediction() -> None:
    pred = _pred(0.8, "HIGH")
    state = new_state(transaction_id="t-1", prediction=pred)
    assert state["prediction"]["fraud_score"] == pytest.approx(0.8)
    assert state["prediction"]["risk_level"] == "HIGH"


def test_new_state_alert_id_default_and_override() -> None:
    state = new_state(transaction_id=99, prediction=_pred(0.8, "HIGH"))
    assert state["alert_id"] == "inv-99"
    state2 = new_state(transaction_id=99, prediction=_pred(0.8, "HIGH"), alert_id="custom-1")
    assert state2["alert_id"] == "custom-1"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


def test_entity_risk_validates_score_range() -> None:
    EntityRisk(entity_type="card", entity_id="c1", risk_score=0.0)
    EntityRisk(entity_type="card", entity_id="c1", risk_score=1.0)
    with pytest.raises(ValidationError):
        EntityRisk(entity_type="card", entity_id="c1", risk_score=1.5)


def test_fraud_pattern_rejects_unknown_name() -> None:
    FraudPattern(name="card_testing", confidence=0.5)
    with pytest.raises(ValidationError):
        FraudPattern(name="totally_invented", confidence=0.5)  # type: ignore[arg-type]


def test_similar_case_clamps_similarity() -> None:
    s = SimilarCase(case_id="C-001", similarity=0.5, pattern="ring", summary="x")
    assert 0.0 <= s.similarity <= 1.0


def test_investigation_report_minimal_construct() -> None:
    r = InvestigationReport(
        alert_id="inv-1",
        transaction_id=1,
        risk_level="LOW",
        depth="quick",
        fraud_score=0.1,
        summary="x",
        narrative="y",
        recommended_action="approve",
    )
    assert r.recommended_action == "approve"
    assert r.confidence == 0.5
    assert r.matched_patterns == []
