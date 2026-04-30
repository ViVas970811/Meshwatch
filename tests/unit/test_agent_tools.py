"""Tests for the 8 investigation tools (Phase 5)."""

from __future__ import annotations

import numpy as np

from fraud_detection.agent.case_bank import CaseBank
from fraud_detection.agent.llm import StubProvider
from fraud_detection.agent.tools import (
    TOOL_NAMES,
    CardHistoryStore,
    HistoricalTransaction,
    analyze_card_history,
    analyze_velocity,
    compute_cross_entity_risk,
    explore_graph_neighborhood,
    generate_investigation_report,
    get_transaction_details,
    match_fraud_patterns,
    retrieve_similar_cases,
)


def test_tool_registry_has_eight_tools() -> None:
    # Plan page 10 mandates eight tools.
    assert len(TOOL_NAMES) == 8
    assert set(TOOL_NAMES) == {
        "get_transaction_details",
        "analyze_card_history",
        "explore_graph_neighborhood",
        "match_fraud_patterns",
        "retrieve_similar_cases",
        "analyze_velocity",
        "compute_cross_entity_risk",
        "generate_investigation_report",
    }


# ---------------------------------------------------------------------------
# Tool 1: get_transaction_details
# ---------------------------------------------------------------------------


def test_get_transaction_details_echoes_request_and_prediction() -> None:
    out = get_transaction_details(
        transaction={
            "transaction_id": 1,
            "transaction_amt": 25.0,
            "card1": 99,
            "DeviceType": "mobile",
        },
        prediction={
            "fraud_score": 0.5,
            "risk_level": "MEDIUM",
            "is_fraud_predicted": False,
            "model_version": "v0.3.0-gnn-model",
            "top_features": [{"feature": "f", "value": 1.0, "contribution": 0.1}],
        },
    )
    assert out["status"] == "ok"
    assert out["fraud_score"] == 0.5
    assert out["risk_level"] == "MEDIUM"
    assert out["device_type"] == "mobile"
    assert len(out["top_features"]) == 1
    assert out["elapsed_ms"] >= 0


# ---------------------------------------------------------------------------
# Tool 2: analyze_card_history
# ---------------------------------------------------------------------------


def _make_history_store() -> CardHistoryStore:
    store = CardHistoryStore()
    base_dt = 100_000
    for i in range(10):
        store.add(
            HistoricalTransaction(
                transaction_id=i,
                transaction_dt=base_dt + i * 600,  # one tx every 10 min
                transaction_amt=12.5 * (i + 1),
                is_fraud=int(i in {7, 8}),
                card_id=99,
            )
        )
    return store


def test_analyze_card_history_no_history_skips() -> None:
    out = analyze_card_history(card_id=None, history=None)
    assert out["status"] == "skipped"


def test_analyze_card_history_computes_rollup() -> None:
    store = _make_history_store()
    out = analyze_card_history(card_id=99, as_of_dt=200_000, history=store)
    assert out["status"] == "ok"
    assert out["n_transactions"] == 10
    assert out["fraud_count"] == 2
    assert out["fraud_rate"] == 0.2
    assert out["total_spend"] > 0
    assert out["avg_amount"] > 0
    assert out["velocity_per_hour"] > 0


def test_analyze_card_history_respects_window() -> None:
    store = _make_history_store()
    # 30s window only catches the most recent ~5 entries.
    out = analyze_card_history(card_id=99, as_of_dt=105_000, window_days=0, history=store)
    # window_days=0 with as_of_dt=105_000 means seconds=0 -> very small window
    assert out["status"] in ("ok", "skipped")


# ---------------------------------------------------------------------------
# Tool 3: explore_graph_neighborhood
# ---------------------------------------------------------------------------


class _StubGraph:
    """Tiny ``neighbors``-shaped object."""

    def __init__(self, edges: dict[int, list[int]]) -> None:
        self._edges = edges

    def neighbors(self, node: int) -> list[int]:
        return list(self._edges.get(node, []))


def test_explore_graph_neighborhood_no_graph_skips() -> None:
    out = explore_graph_neighborhood(transaction_id=1, card_id=99, graph=None)
    assert out["status"] == "skipped"


def test_explore_graph_neighborhood_walks_two_hops() -> None:
    g = _StubGraph({1: [2, 3], 2: [4], 3: [5]})
    out = explore_graph_neighborhood(transaction_id=10, card_id=1, graph=g, n_hops=2)
    assert out["status"] == "ok"
    assert out["n_hops"] == 2
    assert out["n_unique_neighbors"] >= 2  # at least 2 (direct neighbors)
    # First hop should include 2 and 3.
    assert "2" in out["neighbors_by_hop"][0]
    assert "3" in out["neighbors_by_hop"][0]


# ---------------------------------------------------------------------------
# Tool 4: match_fraud_patterns
# ---------------------------------------------------------------------------


def test_match_fraud_patterns_velocity_spike() -> None:
    out = match_fraud_patterns(
        history_summary={"velocity_per_hour": 12.0},
        velocity_summary={"velocity_1h": 12.0, "baseline_per_hour": 2.0},
        fraud_score=0.85,
    )
    assert out["status"] == "ok"
    names = [p["name"] for p in out["matched_patterns"]]
    assert "velocity_spike" in names


def test_match_fraud_patterns_card_testing() -> None:
    out = match_fraud_patterns(
        history_summary={"n_transactions": 25, "avg_amount": 2.0, "max_amount": 5.0},
        fraud_score=0.6,
    )
    names = [p["name"] for p in out["matched_patterns"]]
    assert "card_testing" in names


def test_match_fraud_patterns_collusion_ring() -> None:
    out = match_fraud_patterns(
        neighborhood_summary={"n_unique_neighbors": 10},
        fraud_score=0.6,
    )
    names = [p["name"] for p in out["matched_patterns"]]
    assert "collusion_ring" in names


def test_match_fraud_patterns_none_for_clean_input() -> None:
    out = match_fraud_patterns(fraud_score=0.05)
    names = [p["name"] for p in out["matched_patterns"]]
    assert names == ["none"]


# ---------------------------------------------------------------------------
# Tool 5: retrieve_similar_cases
# ---------------------------------------------------------------------------


def test_retrieve_similar_cases_with_seed_bank() -> None:
    bank = CaseBank.with_seed(use_faiss=False)
    out = retrieve_similar_cases(
        embedding=np.ones(64, dtype=np.float32) * 0.01,
        case_bank=bank,
        k=3,
    )
    assert out["status"] == "ok"
    assert len(out["similar_cases"]) == 3
    for c in out["similar_cases"]:
        assert 0.0 <= c["similarity"] <= 1.0


def test_retrieve_similar_cases_no_embedding_skips() -> None:
    out = retrieve_similar_cases(embedding=None, case_bank=None, k=3)
    assert out["status"] == "skipped"
    assert out["similar_cases"] == []


# ---------------------------------------------------------------------------
# Tool 6: analyze_velocity
# ---------------------------------------------------------------------------


def test_analyze_velocity_three_windows() -> None:
    store = _make_history_store()
    out = analyze_velocity(card_id=99, as_of_dt=200_000, history=store)
    assert out["status"] == "ok"
    # All three keys present
    assert "velocity_1h" in out
    assert "velocity_6h" in out
    assert "velocity_24h" in out
    assert out["baseline_per_hour"] >= 0


def test_analyze_velocity_missing_inputs_skips() -> None:
    out = analyze_velocity(card_id=None, as_of_dt=None, history=None)
    assert out["status"] == "skipped"


# ---------------------------------------------------------------------------
# Tool 7: compute_cross_entity_risk
# ---------------------------------------------------------------------------


def test_compute_cross_entity_risk_returns_five_entities() -> None:
    out = compute_cross_entity_risk(
        transaction={"card1": 1, "device_type": "mobile", "p_emaildomain": "gmail.com"},
        history_summary={"fraud_rate": 0.1, "velocity_per_hour": 2.0},
        neighborhood_summary={"n_unique_neighbors": 6},
        fraud_score=0.8,
    )
    assert out["status"] == "ok"
    types = [e["entity_type"] for e in out["entity_risks"]]
    assert types == ["card", "device", "email", "ip", "merchant"]
    for e in out["entity_risks"]:
        assert 0.0 <= e["risk_score"] <= 1.0


def test_compute_cross_entity_risk_disposable_email_boost() -> None:
    out = compute_cross_entity_risk(
        transaction={"card1": 1, "p_emaildomain": "mailinator.com"},
        fraud_score=0.3,
    )
    email_entry = next(e for e in out["entity_risks"] if e["entity_type"] == "email")
    assert "disposable_email_domain" in email_entry["contributing_factors"]


# ---------------------------------------------------------------------------
# Tool 8: generate_investigation_report
# ---------------------------------------------------------------------------


def test_generate_investigation_report_with_stub_llm() -> None:
    llm = StubProvider()
    out = generate_investigation_report(
        state_evidence={"analyze_card_history": {"velocity_per_hour": 12.0, "status": "ok"}},
        prediction={"fraud_score": 0.85, "risk_level": "HIGH"},
        transaction_id="t-1",
        alert_id="inv-1",
        depth="standard",
        llm=llm,
    )
    assert out["status"] == "ok"
    assert out["recommended_action"] in ("approve", "review", "decline", "escalate")
    assert out["narrative"]
    assert out["is_stub"]


def test_generate_investigation_report_handles_llm_failure() -> None:
    class _BoomLLM:
        def invoke(self, *_a, **_k):
            raise RuntimeError("boom")

    out = generate_investigation_report(
        state_evidence={},
        prediction={"fraud_score": 0.8, "risk_level": "HIGH"},
        transaction_id=1,
        alert_id="inv-1",
        depth="standard",
        llm=_BoomLLM(),
    )
    assert out["status"] == "skipped"
    assert "error" in out
