"""Materialise an :class:`InvestigationReport` from the agent state.

Separated from ``graph.py`` so it can be unit-tested without spinning up
a LangGraph app.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from fraud_detection.agent.state import (
    AgentState,
    EntityRisk,
    FraudPattern,
    InvestigationReport,
    SimilarCase,
)
from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


def build_report(state: AgentState, *, t_started: float | None = None) -> InvestigationReport:
    """Translate the (mutable) agent state into a Pydantic report.

    The state is expected to have already had a
    ``generate_investigation_report`` tool call run; if it hasn't, we
    fall back to a deterministic "no LLM" narrative built from the
    structured evidence.
    """
    pred = state.get("prediction", {}) or {}
    evidence: dict[str, Any] = state.get("evidence", {}) or {}
    report_evidence = evidence.get("generate_investigation_report", {}) or {}

    summary = report_evidence.get("summary") or _fallback_summary(state)
    narrative = report_evidence.get("narrative") or _fallback_narrative(state)
    action = (report_evidence.get("recommended_action") or _fallback_action(state)).lower()
    confidence = float(report_evidence.get("confidence") or 0.5)
    model = report_evidence.get("model") or "stub"

    matched = [FraudPattern(**p) for p in (report_evidence.get("matched_patterns") or [])]
    if not matched:
        # Pull from match_fraud_patterns evidence if no LLM patterns came through.
        matched = [
            FraudPattern(**p)
            for p in (evidence.get("match_fraud_patterns", {}) or {}).get("matched_patterns", [])
        ]

    entities = [
        EntityRisk(**e)
        for e in (evidence.get("compute_cross_entity_risk", {}) or {}).get("entity_risks", [])
    ]
    similar = [
        SimilarCase(**c)
        for c in (evidence.get("retrieve_similar_cases", {}) or {}).get("similar_cases", [])
    ]

    elapsed_ms = (time.perf_counter() - t_started) * 1000 if t_started is not None else 0.0
    return InvestigationReport(
        alert_id=state.get("alert_id", "inv-unknown"),
        transaction_id=state.get("transaction_id", "?"),
        risk_level=state.get("risk_level", pred.get("risk_level", "LOW")),
        depth=state.get("depth", "quick"),
        fraud_score=float(pred.get("fraud_score") or 0.0),
        summary=summary,
        narrative=narrative,
        recommended_action=_normalise_action(action),
        confidence=max(0.0, min(1.0, confidence)),
        requires_human_review=bool(state.get("requires_human_review")),
        entity_risks=entities,
        matched_patterns=matched,
        similar_cases=similar,
        tools_used=list(evidence.keys()),
        tool_calls=list(state.get("tool_calls") or []),
        elapsed_ms=elapsed_ms,
        model=str(model),
    )


# ---------------------------------------------------------------------------
# Fallbacks (used when the LLM tool call didn't run)
# ---------------------------------------------------------------------------


def _fallback_summary(state: AgentState) -> str:
    pred = state.get("prediction", {}) or {}
    risk = pred.get("risk_level", "LOW")
    score = float(pred.get("fraud_score") or 0.0)
    return f"{risk} risk on transaction {state.get('transaction_id', '?')} (score {score:.2f})."


def _fallback_narrative(state: AgentState) -> str:
    evidence = state.get("evidence", {}) or {}
    parts: list[str] = [_fallback_summary(state)]
    h = evidence.get("analyze_card_history") or {}
    if h.get("status") == "ok":
        parts.append(
            f"Card history: {h.get('n_transactions', 0)} txns, "
            f"30d spend ${h.get('total_spend', 0):.0f}, "
            f"avg ${h.get('avg_amount', 0):.2f}, "
            f"velocity {h.get('velocity_per_hour', 0):.1f}/h, "
            f"fraud_rate {h.get('fraud_rate', 0):.2%}."
        )
    n = evidence.get("explore_graph_neighborhood") or {}
    if n.get("status") == "ok" and (n.get("n_unique_neighbors") or 0):
        parts.append(
            f"Card connected to {n.get('n_unique_neighbors')} peer cards via shared device/address."
        )
    return " ".join(parts)


def _fallback_action(state: AgentState) -> str:
    pred = state.get("prediction", {}) or {}
    score = float(pred.get("fraud_score") or 0.0)
    risk = pred.get("risk_level", "LOW")
    if risk == "CRITICAL" or score >= 0.9:
        return "decline"
    if risk == "HIGH":
        return "escalate"
    if risk == "MEDIUM":
        return "review"
    return "approve"


_VALID_ACTIONS = ("approve", "review", "decline", "escalate")


def _normalise_action(action: str) -> str:
    a = (action or "review").strip().lower()
    return a if a in _VALID_ACTIONS else "review"


__all__ = ["build_report"]


# ---------------------------------------------------------------------------
# Re-export Mapping for typing convenience -- silence ruff F401
# ---------------------------------------------------------------------------

_ = Mapping
