"""LangGraph agentic investigator (Phase 5).

Public surface
--------------

* :class:`AgentDeps`            -- DI container for nodes (LLM, stores, graph, ...).
* :class:`AgentState`           -- mutable blackboard the LangGraph nodes share.
* :class:`InvestigationReport`  -- the structured analyst-facing output.
* :func:`new_state`             -- build an initial state from a :class:`FraudPrediction`.
* :func:`investigate`           -- run the full graph end-to-end.
* :func:`build_graph`           -- compile a reusable LangGraph runnable.
"""

from __future__ import annotations

from fraud_detection.agent.case_bank import CaseBank, CaseRecord
from fraud_detection.agent.graph import (
    AgentDeps,
    build_graph,
    investigate,
    route_by_risk_level,
)
from fraud_detection.agent.llm import LLMResponse, OllamaProvider, StubProvider, get_llm
from fraud_detection.agent.neo4j_adapter import Neo4jGraphAdapter
from fraud_detection.agent.report import build_report
from fraud_detection.agent.state import (
    AgentState,
    EntityRisk,
    FraudPattern,
    InvestigationDepth,
    InvestigationReport,
    SimilarCase,
    new_state,
)
from fraud_detection.agent.tracing import AgentTracer

__all__ = [
    "AgentDeps",
    "AgentState",
    "AgentTracer",
    "CaseBank",
    "CaseRecord",
    "EntityRisk",
    "FraudPattern",
    "InvestigationDepth",
    "InvestigationReport",
    "LLMResponse",
    "Neo4jGraphAdapter",
    "OllamaProvider",
    "SimilarCase",
    "StubProvider",
    "build_graph",
    "build_report",
    "get_llm",
    "investigate",
    "new_state",
    "route_by_risk_level",
]
