"""LLM provider for the fraud-investigation agent (Phase 5).

The plan specifies *Ollama* (local LLM) via ``langchain-ollama``. We don't
want a hard dep on a running Ollama daemon though -- tests, CI, and most
laptops won't have one -- so this module ships two providers:

* :class:`OllamaProvider`  -- real ChatOllama, used when the daemon answers.
* :class:`StubProvider`    -- deterministic, evidence-driven narrative.

:func:`get_llm` returns the first provider that connects. Callers always
get something with the same ``invoke(system, user, **kwargs)`` shape.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public response shape
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """What every provider returns. ``parsed`` is best-effort JSON."""

    content: str
    parsed: dict[str, Any] = field(default_factory=dict)
    model: str = "unknown"
    elapsed_ms: float = 0.0
    is_stub: bool = False


# ---------------------------------------------------------------------------
# Stub provider (deterministic, no network)
# ---------------------------------------------------------------------------


class StubProvider:
    """Generates a structured report from evidence without an LLM.

    The output JSON shape matches what the real Ollama provider returns
    so the rest of the agent (and the report generator) can stay model-
    agnostic. Used in tests + on laptops without a running Ollama daemon.
    """

    name = "stub"

    def __init__(self, *, action_rule: Callable[[float, str], str] | None = None) -> None:
        self.action_rule = action_rule or _default_action_rule

    def invoke(self, system: str, user: str, **_: Any) -> LLMResponse:
        # Pull a few hints from the rendered user prompt so the stub feels
        # responsive to the evidence. The prompt is built by
        # :data:`agent.prompts.REPORT_TEMPLATE` so we can grep for the
        # structured fields it interpolates.
        score = _grep_float(user, r"Fraud score:\s*([0-9.]+)") or 0.0
        risk = _grep_str(user, r"Risk level:\s*(LOW|MEDIUM|HIGH|CRITICAL)") or "LOW"
        txn = _grep_str(user, r"transaction\s+([A-Za-z0-9_-]+)") or "?"

        action = self.action_rule(score, risk)
        patterns = _stub_match_patterns(user, score)
        evidence_summary = _summarise_evidence(user)

        narrative = (
            f"Transaction {txn} scored {score:.3f} (risk {risk}). "
            f"{evidence_summary} "
            f"Recommended action: {action}."
        ).strip()

        payload: dict[str, Any] = {
            "summary": f"{risk} risk on transaction {txn} (score {score:.2f}).",
            "narrative": narrative,
            "recommended_action": action,
            "confidence": min(0.95, max(0.4, score if score > 0 else 0.5)),
            "matched_patterns": patterns,
        }
        return LLMResponse(
            content=json.dumps(payload),
            parsed=payload,
            model=self.name,
            is_stub=True,
        )


# ---------------------------------------------------------------------------
# Ollama provider (real langchain-ollama)
# ---------------------------------------------------------------------------


class OllamaProvider:
    """Calls a local Ollama server via ``langchain-ollama``.

    The provider is constructed lazily so importing this module doesn't
    require ``langchain-ollama`` to be installed.
    """

    def __init__(
        self,
        *,
        model: str = "llama3.1:8b",
        base_url: str | None = None,
        temperature: float = 0.1,
        request_timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.temperature = temperature
        self.request_timeout = request_timeout
        self._client: Any | None = None

    @property
    def name(self) -> str:
        return self.model

    def connect(self) -> bool:
        try:
            from langchain_ollama import ChatOllama  # type: ignore[import-untyped]
        except Exception as exc:
            log.warning("ollama_module_unavailable", error=str(exc))
            return False
        try:
            self._client = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=512,
                timeout=self.request_timeout,
            )
            log.info("ollama_provider_ready", model=self.model, base_url=self.base_url)
            return True
        except Exception as exc:
            log.warning("ollama_provider_init_failed", error=str(exc))
            self._client = None
            return False

    def invoke(self, system: str, user: str, **kwargs: Any) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("OllamaProvider not connected")
        # langchain-ollama's ChatOllama takes a list of (role, text) tuples.
        import time

        t0 = time.perf_counter()
        msg = self._client.invoke(  # type: ignore[union-attr]
            [
                ("system", system),
                ("human", user),
            ],
            **kwargs,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        content = getattr(msg, "content", str(msg))
        parsed = _try_parse_json(content)
        return LLMResponse(
            content=content,
            parsed=parsed,
            model=self.model,
            elapsed_ms=elapsed,
            is_stub=False,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_llm(
    *,
    prefer_ollama: bool | None = None,
    model: str | None = None,
) -> StubProvider | OllamaProvider:
    """Return the first available LLM provider.

    Resolution order:

    1. If ``FRAUD_AGENT_LLM=stub``  -> :class:`StubProvider`.
    2. Otherwise try Ollama (``OllamaProvider.connect()``).
    3. Fall back to :class:`StubProvider`.

    Setting ``prefer_ollama=False`` bypasses the network call (useful in
    tests).
    """
    explicit = os.environ.get("FRAUD_AGENT_LLM", "").lower()
    if explicit == "stub":
        log.info("agent_llm_stub_forced")
        return StubProvider()
    if prefer_ollama is False or explicit == "off":
        return StubProvider()

    provider = OllamaProvider(model=model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
    if provider.connect():
        return provider

    log.info("agent_llm_falling_back_to_stub")
    return StubProvider()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_parse_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    # The model sometimes wraps JSON in ```json ... ``` fences.
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    # Or it puts prose around it -- grab the outermost {...}.
    if not cleaned.startswith("{"):
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(0)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _grep_float(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def _grep_str(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text)
    return m.group(1) if m else None


def _default_action_rule(score: float, risk: str) -> str:
    if risk == "CRITICAL" or score >= 0.9:
        return "decline"
    if risk == "HIGH":
        return "escalate"
    if risk == "MEDIUM":
        return "review"
    return "approve"


_PATTERN_KEYWORDS: list[tuple[str, str]] = [
    ("velocity_spike", r"velocity[_ ]spike|spike|burst|tx_per_hour|velocity_1h"),
    ("card_testing", r"card[_ ]testing|small.+amount|micro[_ ]?charge"),
    ("collusion_ring", r"ring|collus|shared[_ ]device|shared[_ ]address"),
    ("account_takeover", r"new[_ ]device|account[_ ]takeover|password[_ ]reset"),
    ("geo_anomaly", r"geo[_ ]?anomaly|country[_ ]mismatch|distance"),
]


def _stub_match_patterns(prompt: str, score: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, kw in _PATTERN_KEYWORDS:
        if re.search(kw, prompt, re.IGNORECASE):
            out.append(
                {
                    "name": name,
                    "confidence": min(0.9, max(0.3, score if score > 0 else 0.5)),
                    "rationale": f"Matched keyword pattern '{name}' in evidence.",
                }
            )
    if not out and score >= 0.7:
        out.append(
            {
                "name": "velocity_spike",
                "confidence": float(score),
                "rationale": "High score with no specific keyword match -- defaulting to velocity_spike.",
            }
        )
    return out[:3]


def _summarise_evidence(prompt: str) -> str:
    """Pick a short prose summary out of an EVIDENCE block."""
    m = re.search(r"==== EVIDENCE ====\n(.*?)\n==== TASK ====", prompt, re.DOTALL)
    if not m:
        return "Evidence is sparse."
    block = m.group(1)
    # Take the first 3 non-empty lines for the stub prose.
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()][:3]
    if not lines:
        return "Evidence is sparse."
    return " ".join(lines)


__all__ = ["LLMResponse", "OllamaProvider", "StubProvider", "get_llm"]
