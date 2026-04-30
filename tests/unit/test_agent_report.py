"""Tests for the report builder (Phase 5)."""

from __future__ import annotations

from fraud_detection.agent.report import build_report
from fraud_detection.agent.state import new_state
from fraud_detection.serving.schemas import FraudPrediction


def _pred(score: float = 0.85, risk: str = "HIGH") -> FraudPrediction:
    return FraudPrediction(
        transaction_id=99,
        fraud_probability=score,
        fraud_score=score,
        risk_level=risk,  # type: ignore[arg-type]
        is_fraud_predicted=score >= 0.7,
        threshold=0.7,
    )


def test_build_report_uses_llm_evidence_when_present() -> None:
    state = new_state(transaction_id=99, prediction=_pred(0.9, "HIGH"))
    state["evidence"] = {  # type: ignore[index]
        "generate_investigation_report": {
            "summary": "S",
            "narrative": "N",
            "recommended_action": "decline",
            "confidence": 0.8,
            "model": "llama3.1:8b",
            "matched_patterns": [{"name": "card_testing", "confidence": 0.7, "rationale": "..."}],
        }
    }
    report = build_report(state)
    assert report.summary == "S"
    assert report.narrative == "N"
    assert report.recommended_action == "decline"
    assert report.confidence == 0.8
    assert report.model == "llama3.1:8b"
    assert report.matched_patterns[0].name == "card_testing"


def test_build_report_falls_back_when_no_llm_call() -> None:
    state = new_state(transaction_id=99, prediction=_pred(0.05, "LOW"))
    state["evidence"] = {}  # type: ignore[index]
    report = build_report(state)
    assert report.recommended_action == "approve"
    assert "LOW" in report.summary


def test_build_report_normalises_invalid_action() -> None:
    state = new_state(transaction_id=99, prediction=_pred(0.5, "MEDIUM"))
    state["evidence"] = {  # type: ignore[index]
        "generate_investigation_report": {
            "summary": "S",
            "narrative": "N",
            "recommended_action": "totally_invented",
            "confidence": 0.5,
        }
    }
    report = build_report(state)
    assert report.recommended_action == "review"


def test_build_report_clamps_confidence() -> None:
    state = new_state(transaction_id=99, prediction=_pred(0.5, "MEDIUM"))
    state["evidence"] = {  # type: ignore[index]
        "generate_investigation_report": {
            "summary": "x",
            "narrative": "y",
            "recommended_action": "review",
            "confidence": 99.0,
        }
    }
    report = build_report(state)
    assert 0.0 <= report.confidence <= 1.0


def test_build_report_pulls_entity_risks_and_similar_cases_from_evidence() -> None:
    state = new_state(transaction_id=99, prediction=_pred(0.95, "CRITICAL"))
    state["evidence"] = {  # type: ignore[index]
        "compute_cross_entity_risk": {
            "entity_risks": [
                {
                    "entity_type": "card",
                    "entity_id": "c1",
                    "risk_score": 0.8,
                    "contributing_factors": [],
                }
            ]
        },
        "retrieve_similar_cases": {
            "similar_cases": [
                {"case_id": "C-1", "similarity": 0.9, "pattern": "ring", "summary": "x"}
            ]
        },
    }
    report = build_report(state)
    assert len(report.entity_risks) == 1
    assert report.entity_risks[0].entity_type == "card"
    assert len(report.similar_cases) == 1
    assert report.similar_cases[0].case_id == "C-1"


def test_build_report_carries_through_human_review_flag() -> None:
    s_low = new_state(transaction_id=1, prediction=_pred(0.1, "LOW"))
    s_critical = new_state(transaction_id=2, prediction=_pred(0.95, "CRITICAL"))
    assert build_report(s_low).requires_human_review is False
    assert build_report(s_critical).requires_human_review is True
