"""LangGraph StateGraph orchestrating the 8 tools (Phase 5).

Plan flow (page 10):

    START (alert) -> Route by Risk Level
    LOW/MEDIUM   -> quick_scan -> generate_report -> END
    HIGH         -> gather_context -> analyze_patterns -> generate_report -> END (human review flag)
    CRITICAL     -> full_traversal -> pattern_matching -> cross_entity_analysis
                    -> generate_report -> END (human review flag)

We expose a single :func:`investigate` entry point that builds the graph
once and runs it for an :class:`AgentState`. The graph is also cached so
repeated calls are cheap.
"""

from __future__ import annotations

import functools
import time
from typing import Any

from fraud_detection.agent import tools as T  # noqa: N812 -- terse alias used heavily below
from fraud_detection.agent.case_bank import CaseBank
from fraud_detection.agent.llm import StubProvider, get_llm
from fraud_detection.agent.report import build_report
from fraud_detection.agent.state import AgentState, InvestigationReport
from fraud_detection.agent.tracing import AgentTracer
from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# AgentDeps -- everything the nodes need that's not in the state.
# ---------------------------------------------------------------------------


class AgentDeps:
    """Dependency container so node closures stay easy to test."""

    def __init__(
        self,
        *,
        llm: Any | None = None,
        history: T.CardHistoryStore | None = None,
        graph: Any | None = None,
        case_bank: CaseBank | None = None,
        embedding_lookup: dict[Any, Any] | None = None,
        tracer: AgentTracer | None = None,
    ) -> None:
        self.llm = llm or get_llm(prefer_ollama=False)  # default: stub
        self.history = history
        self.graph = graph
        self.case_bank = case_bank or CaseBank.with_seed()
        self.embedding_lookup = embedding_lookup or {}
        self.tracer = tracer or AgentTracer()


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def _record_tool(state: AgentState, name: str, payload: dict[str, Any]) -> None:
    state.setdefault("evidence", {})[name] = payload  # type: ignore[index]
    state.setdefault("tool_calls", []).append(  # type: ignore[arg-type]
        {
            "name": name,
            "status": payload.get("status", "ok"),
            "elapsed_ms": float(payload.get("elapsed_ms") or 0.0),
        }
    )


def node_quick_scan(state: AgentState, deps: AgentDeps) -> AgentState:
    """LOW / MEDIUM path -- 2 tools, no graph walks."""
    with deps.tracer.span("quick_scan", risk=state.get("risk_level")):
        _record_tool(
            state,
            "get_transaction_details",
            T.get_transaction_details(
                transaction=state.get("request", {}),
                prediction=state.get("prediction", {}),
            ),
        )
        _record_tool(
            state,
            "analyze_card_history",
            T.analyze_card_history(
                card_id=state.get("request", {}).get("card1"),
                as_of_dt=state.get("request", {}).get("transaction_dt"),
                history=deps.history,
            ),
        )
    return state


def node_gather_context(state: AgentState, deps: AgentDeps) -> AgentState:
    """HIGH path -- collect transaction details, card history, velocity."""
    with deps.tracer.span("gather_context", risk=state.get("risk_level")):
        _record_tool(
            state,
            "get_transaction_details",
            T.get_transaction_details(
                transaction=state.get("request", {}),
                prediction=state.get("prediction", {}),
            ),
        )
        _record_tool(
            state,
            "analyze_card_history",
            T.analyze_card_history(
                card_id=state.get("request", {}).get("card1"),
                as_of_dt=state.get("request", {}).get("transaction_dt"),
                history=deps.history,
            ),
        )
        _record_tool(
            state,
            "analyze_velocity",
            T.analyze_velocity(
                card_id=state.get("request", {}).get("card1"),
                as_of_dt=state.get("request", {}).get("transaction_dt"),
                history=deps.history,
            ),
        )
    return state


def node_analyze_patterns(state: AgentState, deps: AgentDeps) -> AgentState:
    """HIGH path -- match patterns + retrieve similar cases."""
    with deps.tracer.span("analyze_patterns", risk=state.get("risk_level")):
        evidence = state.get("evidence", {}) or {}
        history_summary = evidence.get("analyze_card_history") or {}
        velocity_summary = evidence.get("analyze_velocity") or {}
        neighborhood_summary = evidence.get("explore_graph_neighborhood") or {}
        fraud_score = float((state.get("prediction") or {}).get("fraud_score") or 0.0)

        _record_tool(
            state,
            "match_fraud_patterns",
            T.match_fraud_patterns(
                history_summary=history_summary,
                velocity_summary=velocity_summary,
                neighborhood_summary=neighborhood_summary,
                fraud_score=fraud_score,
            ),
        )
        emb = _resolve_embedding(state, deps)
        _record_tool(
            state,
            "retrieve_similar_cases",
            T.retrieve_similar_cases(embedding=emb, case_bank=deps.case_bank, k=3),
        )
    return state


def node_full_traversal(state: AgentState, deps: AgentDeps) -> AgentState:
    """CRITICAL path -- N-hop neighborhood, deeper card history, velocity."""
    with deps.tracer.span("full_traversal", risk=state.get("risk_level")):
        _record_tool(
            state,
            "get_transaction_details",
            T.get_transaction_details(
                transaction=state.get("request", {}),
                prediction=state.get("prediction", {}),
            ),
        )
        _record_tool(
            state,
            "analyze_card_history",
            T.analyze_card_history(
                card_id=state.get("request", {}).get("card1"),
                as_of_dt=state.get("request", {}).get("transaction_dt"),
                history=deps.history,
            ),
        )
        _record_tool(
            state,
            "analyze_velocity",
            T.analyze_velocity(
                card_id=state.get("request", {}).get("card1"),
                as_of_dt=state.get("request", {}).get("transaction_dt"),
                history=deps.history,
            ),
        )
        _record_tool(
            state,
            "explore_graph_neighborhood",
            T.explore_graph_neighborhood(
                transaction_id=state.get("transaction_id"),
                card_id=state.get("request", {}).get("card1"),
                graph=deps.graph,
                n_hops=2,
            ),
        )
    return state


def node_pattern_matching(state: AgentState, deps: AgentDeps) -> AgentState:
    """CRITICAL path -- pattern matcher + GraphRAG similar cases."""
    with deps.tracer.span("pattern_matching", risk=state.get("risk_level")):
        evidence = state.get("evidence", {}) or {}
        fraud_score = float((state.get("prediction") or {}).get("fraud_score") or 0.0)
        _record_tool(
            state,
            "match_fraud_patterns",
            T.match_fraud_patterns(
                history_summary=evidence.get("analyze_card_history") or {},
                velocity_summary=evidence.get("analyze_velocity") or {},
                neighborhood_summary=evidence.get("explore_graph_neighborhood") or {},
                fraud_score=fraud_score,
            ),
        )
        emb = _resolve_embedding(state, deps)
        _record_tool(
            state,
            "retrieve_similar_cases",
            T.retrieve_similar_cases(embedding=emb, case_bank=deps.case_bank, k=5),
        )
    return state


def node_cross_entity(state: AgentState, deps: AgentDeps) -> AgentState:
    """CRITICAL path -- per-entity-type risk decomposition."""
    with deps.tracer.span("cross_entity", risk=state.get("risk_level")):
        evidence = state.get("evidence", {}) or {}
        fraud_score = float((state.get("prediction") or {}).get("fraud_score") or 0.0)
        _record_tool(
            state,
            "compute_cross_entity_risk",
            T.compute_cross_entity_risk(
                transaction=state.get("request", {}),
                history_summary=evidence.get("analyze_card_history") or {},
                neighborhood_summary=evidence.get("explore_graph_neighborhood") or {},
                fraud_score=fraud_score,
            ),
        )
    return state


def node_generate_report(state: AgentState, deps: AgentDeps) -> AgentState:
    """Final node -- LLM synthesises the structured narrative."""
    with deps.tracer.span("generate_report", risk=state.get("risk_level")):
        _record_tool(
            state,
            "generate_investigation_report",
            T.generate_investigation_report(
                state_evidence=state.get("evidence", {}) or {},
                prediction=state.get("prediction", {}) or {},
                transaction_id=state.get("transaction_id", "?"),
                alert_id=state.get("alert_id", "inv-unknown"),
                depth=state.get("depth", "quick"),
                llm=deps.llm,
            ),
        )
    return state


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def route_by_risk_level(state: AgentState) -> str:
    """Decide which downstream branch to take from the START router."""
    risk = state.get("risk_level", "LOW")
    if risk == "CRITICAL":
        return "full_traversal"
    if risk == "HIGH":
        return "gather_context"
    return "quick_scan"  # LOW / MEDIUM


# ---------------------------------------------------------------------------
# Graph builder (cached)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _import_langgraph() -> tuple[Any, Any, Any]:
    from langgraph.graph import END, START, StateGraph

    return StateGraph, START, END


def build_graph(deps: AgentDeps) -> Any:
    """Compile the StateGraph. Returns a runnable.

    The compiled graph is *not* cached at module level because it closes
    over ``deps`` -- different deployments (different LLMs, history
    stores, ...) need their own compiled graph. Within a single
    deployment, callers should hold onto the compiled instance.
    """
    StateGraph, START, END = _import_langgraph()

    workflow: Any = StateGraph(AgentState)

    # Bind deps via partials so the StateGraph just sees state -> state.
    workflow.add_node("quick_scan", functools.partial(node_quick_scan, deps=deps))
    workflow.add_node("gather_context", functools.partial(node_gather_context, deps=deps))
    workflow.add_node("analyze_patterns", functools.partial(node_analyze_patterns, deps=deps))
    workflow.add_node("full_traversal", functools.partial(node_full_traversal, deps=deps))
    workflow.add_node("pattern_matching", functools.partial(node_pattern_matching, deps=deps))
    workflow.add_node("cross_entity", functools.partial(node_cross_entity, deps=deps))
    workflow.add_node("generate_report", functools.partial(node_generate_report, deps=deps))

    workflow.add_conditional_edges(
        START,
        route_by_risk_level,
        {
            "quick_scan": "quick_scan",
            "gather_context": "gather_context",
            "full_traversal": "full_traversal",
        },
    )
    workflow.add_edge("quick_scan", "generate_report")
    workflow.add_edge("gather_context", "analyze_patterns")
    workflow.add_edge("analyze_patterns", "generate_report")
    workflow.add_edge("full_traversal", "pattern_matching")
    workflow.add_edge("pattern_matching", "cross_entity")
    workflow.add_edge("cross_entity", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def investigate(
    state: AgentState,
    *,
    deps: AgentDeps | None = None,
    compiled: Any | None = None,
) -> InvestigationReport:
    """Run the agent end-to-end and return a structured report.

    Reuses ``compiled`` (the output of :func:`build_graph`) if supplied
    so callers in a hot path don't pay the compile cost per call.
    """
    deps = deps or AgentDeps(llm=StubProvider())
    runnable = compiled or build_graph(deps)

    t0 = time.perf_counter()
    log.info(
        "agent_investigate_start",
        alert_id=state.get("alert_id"),
        risk_level=state.get("risk_level"),
        depth=state.get("depth"),
    )
    final_state: AgentState = runnable.invoke(state)
    report = build_report(final_state, t_started=t0)
    log.info(
        "agent_investigate_done",
        alert_id=report.alert_id,
        depth=report.depth,
        elapsed_ms=round(report.elapsed_ms, 2),
        recommended_action=report.recommended_action,
        n_tools=len(report.tools_used),
    )
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_embedding(state: AgentState, deps: AgentDeps) -> Any | None:
    """Pick the right embedding for GraphRAG retrieval.

    Strategy:
    1. Try ``deps.embedding_lookup[card_id]`` -- typically populated from
       the predictor's Redis cache.
    2. Otherwise, look at the prediction's ``top_features`` and use the
       contributions vector as a low-rank surrogate.
    3. Otherwise return None and let the tool short-circuit.
    """
    card_id = (state.get("request") or {}).get("card1")
    if card_id is not None and card_id in deps.embedding_lookup:
        return deps.embedding_lookup[card_id]

    pred = state.get("prediction") or {}
    contribs = pred.get("top_features") or []
    if contribs:
        return [float(c.get("contribution") or 0.0) for c in contribs]
    return None


__all__ = [
    "AgentDeps",
    "build_graph",
    "investigate",
    "node_analyze_patterns",
    "node_cross_entity",
    "node_full_traversal",
    "node_gather_context",
    "node_generate_report",
    "node_pattern_matching",
    "node_quick_scan",
    "route_by_risk_level",
]
