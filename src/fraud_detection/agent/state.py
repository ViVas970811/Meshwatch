"""State + result schemas for the LangGraph fraud-investigation agent (Phase 5).

The agent is built around an :class:`AgentState` ``TypedDict`` (LangGraph
convention) -- nodes read from / write to it as the graph executes. The
final output is materialised as :class:`InvestigationReport`, a Pydantic
model that's safe to expose over the FastAPI ``/api/v1/investigate``
endpoint.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from fraud_detection.serving.schemas import FraudPrediction

InvestigationDepth = Literal["quick", "standard", "deep"]
"""The agent picks one of these depths from the alert's ``risk_level``.

* ``quick``    -- LOW / MEDIUM   -> 2 tool calls, no human review
* ``standard`` -- HIGH           -> 5 tool calls, flagged for human review
* ``deep``     -- CRITICAL       -> 7 tool calls + cross-entity scan, human review
"""


# ---------------------------------------------------------------------------
# AgentState (mutable, LangGraph carries this through every node)
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """The blackboard the LangGraph nodes share.

    ``total=False`` so each node only has to populate the keys it touches;
    we initialise the dict with sane defaults inside :func:`new_state`.
    """

    # --- inputs ----------------------------------------------------------------
    alert_id: str
    transaction_id: int | str
    prediction: dict[str, Any]  # serialised FraudPrediction
    request: dict[str, Any]  # serialised TransactionRequest
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    depth: InvestigationDepth

    # --- evidence (each tool appends a chunk keyed by its name) ----------------
    evidence: dict[str, Any]
    tool_calls: list[dict[str, Any]]  # audit log: name, latency_ms, status

    # --- outputs ---------------------------------------------------------------
    report: dict[str, Any]  # serialised InvestigationReport
    requires_human_review: bool
    errors: list[str]


def new_state(
    *,
    transaction_id: int | str,
    prediction: FraudPrediction,
    request: dict[str, Any] | None = None,
    alert_id: str | None = None,
) -> AgentState:
    """Build a fresh :class:`AgentState` from a scoring decision."""
    risk_level = prediction.risk_level
    depth: InvestigationDepth = (
        "deep" if risk_level == "CRITICAL" else "standard" if risk_level == "HIGH" else "quick"
    )
    return AgentState(  # type: ignore[typeddict-item]
        alert_id=alert_id or f"inv-{transaction_id}",
        transaction_id=transaction_id,
        prediction=prediction.model_dump(mode="json"),
        request=request or {},
        risk_level=risk_level,
        depth=depth,
        evidence={},
        tool_calls=[],
        report={},
        requires_human_review=risk_level in ("HIGH", "CRITICAL"),
        errors=[],
    )


# ---------------------------------------------------------------------------
# InvestigationReport (the agent's final, user-facing output)
# ---------------------------------------------------------------------------


class EntityRisk(BaseModel):
    """Risk factor breakdown for a single entity (card / device / email / ...)."""

    entity_type: Literal["card", "device", "email", "ip", "merchant"]
    entity_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    contributing_factors: list[str] = Field(default_factory=list)


class FraudPattern(BaseModel):
    """A canonical fraud pattern matched against the evidence."""

    name: Literal[
        "card_testing",
        "account_takeover",
        "collusion_ring",
        "velocity_spike",
        "geo_anomaly",
        "none",
    ]
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = ""


class SimilarCase(BaseModel):
    """One row of the GraphRAG similar-case table."""

    case_id: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    pattern: str = "unknown"
    summary: str = ""


class InvestigationReport(BaseModel):
    """Structured narrative the agent emits at the end of every run.

    Designed to be both machine-readable (for the dashboard) *and*
    analyst-readable (the ``narrative`` field is a paragraph of prose).
    """

    model_config = ConfigDict(protected_namespaces=())

    alert_id: str
    transaction_id: int | str
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    depth: InvestigationDepth
    fraud_score: float = Field(..., ge=0.0, le=1.0)

    # core narrative
    summary: str
    narrative: str
    recommended_action: Literal[
        "approve",
        "review",
        "decline",
        "escalate",
    ]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    requires_human_review: bool = False

    # structured evidence
    entity_risks: list[EntityRisk] = Field(default_factory=list)
    matched_patterns: list[FraudPattern] = Field(default_factory=list)
    similar_cases: list[SimilarCase] = Field(default_factory=list)

    # bookkeeping
    tools_used: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str = "stub"  # set to the LLM model id when an LLM was used


__all__ = [
    "AgentState",
    "EntityRisk",
    "FraudPattern",
    "InvestigationDepth",
    "InvestigationReport",
    "SimilarCase",
    "new_state",
]
