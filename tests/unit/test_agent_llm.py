"""Tests for the LLM provider abstraction (Phase 5)."""

from __future__ import annotations

import json

from fraud_detection.agent.llm import StubProvider, _try_parse_json, get_llm
from fraud_detection.agent.prompts import REPORT_TEMPLATE, SYSTEM_PROMPT


def _build_user_prompt(score: float, risk: str = "HIGH") -> str:
    return REPORT_TEMPLATE.substitute(
        alert_id="inv-1",
        transaction_id="t-1",
        risk_level=risk,
        fraud_score=f"{score:.3f}",
        depth="standard",
        evidence_block="- [analyze_card_history] (ok)  velocity_per_hour=12.5",
    )


# ---------------------------------------------------------------------------
# StubProvider
# ---------------------------------------------------------------------------


def test_stub_provider_returns_structured_json() -> None:
    p = StubProvider()
    resp = p.invoke(SYSTEM_PROMPT, _build_user_prompt(0.95, "CRITICAL"))
    assert resp.is_stub
    assert resp.parsed
    assert "summary" in resp.parsed
    assert "narrative" in resp.parsed
    assert "recommended_action" in resp.parsed


def test_stub_provider_action_rule_critical() -> None:
    p = StubProvider()
    resp = p.invoke(SYSTEM_PROMPT, _build_user_prompt(0.95, "CRITICAL"))
    assert resp.parsed["recommended_action"] == "decline"


def test_stub_provider_action_rule_high() -> None:
    p = StubProvider()
    resp = p.invoke(SYSTEM_PROMPT, _build_user_prompt(0.75, "HIGH"))
    assert resp.parsed["recommended_action"] == "escalate"


def test_stub_provider_action_rule_medium_low() -> None:
    p = StubProvider()
    medium = p.invoke(SYSTEM_PROMPT, _build_user_prompt(0.5, "MEDIUM"))
    low = p.invoke(SYSTEM_PROMPT, _build_user_prompt(0.05, "LOW"))
    assert medium.parsed["recommended_action"] == "review"
    assert low.parsed["recommended_action"] == "approve"


def test_stub_provider_matches_velocity_pattern_from_evidence() -> None:
    p = StubProvider()
    resp = p.invoke(SYSTEM_PROMPT, _build_user_prompt(0.85, "HIGH"))
    pattern_names = [p["name"] for p in resp.parsed["matched_patterns"]]
    assert "velocity_spike" in pattern_names


def test_stub_provider_confidence_in_unit_interval() -> None:
    p = StubProvider()
    resp = p.invoke(SYSTEM_PROMPT, _build_user_prompt(0.85, "HIGH"))
    assert 0.0 <= float(resp.parsed["confidence"]) <= 1.0


# ---------------------------------------------------------------------------
# get_llm factory
# ---------------------------------------------------------------------------


def test_get_llm_returns_stub_when_prefer_ollama_false() -> None:
    llm = get_llm(prefer_ollama=False)
    assert isinstance(llm, StubProvider)


def test_get_llm_respects_env_override(monkeypatch) -> None:
    monkeypatch.setenv("FRAUD_AGENT_LLM", "stub")
    llm = get_llm()
    assert isinstance(llm, StubProvider)


# ---------------------------------------------------------------------------
# JSON parser robustness
# ---------------------------------------------------------------------------


def test_parse_json_handles_fenced_block() -> None:
    text = "```json\n" + json.dumps({"a": 1}) + "\n```"
    assert _try_parse_json(text) == {"a": 1}


def test_parse_json_handles_prose_wrapping() -> None:
    text = 'Here is the report: {"a": 1, "b": "x"} thanks.'
    assert _try_parse_json(text) == {"a": 1, "b": "x"}


def test_parse_json_returns_empty_on_garbage() -> None:
    assert _try_parse_json("not json") == {}
    assert _try_parse_json("") == {}
