"""LLM prompts for the fraud-investigation agent (Phase 5).

The agent uses a small handful of prompts; we keep them in one module so
they can be unit-tested for the placeholder names + iterated on without
hunting through ``graph.py``.
"""

from __future__ import annotations

from string import Template

# ---------------------------------------------------------------------------
# System prompt -- pinned for every LLM call from the agent.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior fraud-investigation analyst at a payments company.
You receive automated alerts produced by a graph neural network + XGBoost ensemble.
Your job is to:

1. Read the structured evidence collected by the upstream tools.
2. Decide which (if any) canonical fraud pattern applies:
   card_testing, account_takeover, collusion_ring, velocity_spike, geo_anomaly, none.
3. Recommend exactly one action: approve, review, decline, escalate.
4. Write a concise, factual narrative (<=180 words) summarising the case.

Rules:
- Quote concrete numbers from the evidence (e.g. "spent $4,210 across 14 cards in 1h").
- Never invent fields or scores that aren't in the evidence.
- Use neutral, analyst-style prose; avoid hype words like "definitely" / "obvious".
- If evidence is thin or contradictory, recommend "review" and say so."""


# ---------------------------------------------------------------------------
# Templates -- caller uses ``Template.substitute(...)`` to interpolate.
# ---------------------------------------------------------------------------

REPORT_TEMPLATE = Template(
    """ALERT $alert_id  (transaction $transaction_id)
Risk level: $risk_level   Fraud score: $fraud_score   Depth: $depth

==== EVIDENCE ====
$evidence_block

==== TASK ====
Produce a JSON object with these keys (and nothing else):

{
  "summary":             "<=20 words describing what happened",
  "narrative":           "<=180 word analyst paragraph",
  "recommended_action":  one of [approve, review, decline, escalate],
  "confidence":          float in [0, 1],
  "matched_patterns":    [ { "name": "...", "confidence": 0.0, "rationale": "..." } ]
}

Return only the JSON. No markdown, no surrounding prose."""
)


# Simple text used by the deterministic stub LLM -- mirrors the JSON shape
# above so the call sites can rely on the same parser.
STUB_NARRATIVE_TEMPLATE = Template(
    "Score $fraud_score (risk $risk_level) on transaction $transaction_id. "
    "$evidence_summary "
    "Recommended action: $recommended_action."
)


__all__ = ["REPORT_TEMPLATE", "STUB_NARRATIVE_TEMPLATE", "SYSTEM_PROMPT"]
