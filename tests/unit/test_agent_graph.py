"""End-to-end tests for the LangGraph fraud-investigation graph (Phase 5)."""

from __future__ import annotations

import pytest

# The whole graph requires LangGraph; if it isn't installed we skip.
pytest.importorskip("langgraph")

from fraud_detection.agent.graph import (
    AgentDeps,
    investigate,
    route_by_risk_level,
)
from fraud_detection.agent.llm import StubProvider
from fraud_detection.agent.state import new_state
from fraud_detection.agent.tools import CardHistoryStore, HistoricalTransaction
from fraud_detection.serving.schemas import FraudPrediction


def _pred(score: float, risk: str) -> FraudPrediction:
    return FraudPrediction(
        transaction_id=42,
        fraud_probability=score,
        fraud_score=score,
        risk_level=risk,  # type: ignore[arg-type]
        is_fraud_predicted=score >= 0.7,
        threshold=0.7,
    )


def _request(card: int = 99, dt: int = 1_000_000, amt: float = 100.0) -> dict:
    return {
        "transaction_id": 42,
        "transaction_dt": dt,
        "transaction_amt": amt,
        "card1": card,
        "product_cd": "W",
    }


def _populated_history(card: int = 99) -> CardHistoryStore:
    store = CardHistoryStore()
    for i in range(20):
        store.add(
            HistoricalTransaction(
                transaction_id=i,
                transaction_dt=1_000_000 - i * 600,
                transaction_amt=15.0 + i,
                is_fraud=int(i in {3, 5}),
                card_id=card,
            )
        )
    return store


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("risk", "branch"),
    [
        ("LOW", "quick_scan"),
        ("MEDIUM", "quick_scan"),
        ("HIGH", "gather_context"),
        ("CRITICAL", "full_traversal"),
    ],
)
def test_router_picks_correct_branch(risk: str, branch: str) -> None:
    state = new_state(transaction_id=1, prediction=_pred(0.5, risk), request=_request())
    assert route_by_risk_level(state) == branch


# ---------------------------------------------------------------------------
# End-to-end runs
# ---------------------------------------------------------------------------


def test_low_risk_runs_only_quick_scan_path() -> None:
    state = new_state(transaction_id=1, prediction=_pred(0.1, "LOW"), request=_request())
    deps = AgentDeps(llm=StubProvider())
    report = investigate(state, deps=deps)
    # quick_scan + generate_report -> 3 tools (2 + 1 report).
    expected = {"get_transaction_details", "analyze_card_history", "generate_investigation_report"}
    assert expected <= set(report.tools_used)
    # No graph traversal on the LOW path.
    assert "explore_graph_neighborhood" not in report.tools_used
    assert report.depth == "quick"
    assert report.recommended_action in ("approve", "review")


def test_high_risk_runs_gather_then_analyze() -> None:
    state = new_state(transaction_id=2, prediction=_pred(0.8, "HIGH"), request=_request())
    deps = AgentDeps(llm=StubProvider(), history=_populated_history())
    report = investigate(state, deps=deps)
    expected = {
        "get_transaction_details",
        "analyze_card_history",
        "analyze_velocity",
        "match_fraud_patterns",
        "retrieve_similar_cases",
        "generate_investigation_report",
    }
    assert expected <= set(report.tools_used)
    assert report.depth == "standard"


def test_critical_risk_runs_full_pipeline() -> None:
    class _G:
        def neighbors(self, n):
            return [n + 1, n + 2, n + 3, n + 4, n + 5]

    state = new_state(transaction_id=3, prediction=_pred(0.95, "CRITICAL"), request=_request())
    deps = AgentDeps(llm=StubProvider(), history=_populated_history(), graph=_G())
    report = investigate(state, deps=deps)
    # All 8 tools should have run.
    assert len(report.tools_used) == 8
    assert "compute_cross_entity_risk" in report.tools_used
    assert "explore_graph_neighborhood" in report.tools_used
    assert report.depth == "deep"
    assert report.requires_human_review


def test_investigate_completes_under_30_seconds() -> None:
    """Plan acceptance (page 10): standard-depth investigation < 30 s."""
    state = new_state(transaction_id=4, prediction=_pred(0.8, "HIGH"), request=_request())
    deps = AgentDeps(llm=StubProvider(), history=_populated_history())
    report = investigate(state, deps=deps)
    assert report.elapsed_ms < 30_000


def test_investigate_records_tool_calls_audit_log() -> None:
    state = new_state(transaction_id=5, prediction=_pred(0.5, "MEDIUM"), request=_request())
    deps = AgentDeps(llm=StubProvider())
    report = investigate(state, deps=deps)
    # tool_calls list should include every fired tool with a status + latency.
    assert report.tool_calls
    for entry in report.tool_calls:
        assert "name" in entry
        assert "status" in entry
        assert "elapsed_ms" in entry


def test_compiled_graph_is_reusable() -> None:
    from fraud_detection.agent.graph import build_graph

    deps = AgentDeps(llm=StubProvider())
    compiled = build_graph(deps)
    s1 = new_state(transaction_id=10, prediction=_pred(0.6, "MEDIUM"), request=_request())
    s2 = new_state(transaction_id=11, prediction=_pred(0.95, "CRITICAL"), request=_request())
    # Reusing the compiled graph for two different alerts must not raise.
    r1 = investigate(s1, deps=deps, compiled=compiled)
    r2 = investigate(s2, deps=deps, compiled=compiled)
    assert r1.depth == "quick"
    assert r2.depth == "deep"
