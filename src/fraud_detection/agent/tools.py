"""The 8 investigation tools the agent uses (Phase 5).

The plan (page 10) lists them as:

    1. get_transaction_details       -- record + prediction explanation
    2. analyze_card_history          -- 30d count / spend / avg / fraud / velocity
    3. explore_graph_neighborhood    -- N-hop entity neighborhood (Neo4j or networkx)
    4. match_fraud_patterns          -- card_testing / takeover / ring / velocity / geo
    5. retrieve_similar_cases        -- GraphRAG: GNN embeddings + FAISS
    6. analyze_velocity              -- multi-window (1h/6h/24h) vs. baseline
    7. compute_cross_entity_risk     -- per-entity-type risk scores
    8. generate_investigation_report -- LLM synthesises everything

We keep them as plain Python functions so they're trivially unit-testable
and the LangGraph nodes can compose them however they like. Each tool:

* Accepts a small, typed kwargs payload (not the whole agent state).
* Returns a JSON-serialisable ``dict`` with a ``status`` field.
* Logs structured timings.
* Falls back gracefully when its real data source (CardHistoryStore,
  Neo4j, FAISS, ...) isn't wired up.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from fraud_detection.agent.case_bank import CaseBank
from fraud_detection.agent.state import EntityRisk, FraudPattern, SimilarCase
from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool registry -- name -> callable. Used by the graph nodes + tests.
# ---------------------------------------------------------------------------

TOOL_NAMES: tuple[str, ...] = (
    "get_transaction_details",
    "analyze_card_history",
    "explore_graph_neighborhood",
    "match_fraud_patterns",
    "retrieve_similar_cases",
    "analyze_velocity",
    "compute_cross_entity_risk",
    "generate_investigation_report",
)


# ---------------------------------------------------------------------------
# Optional data source (in-memory store) -- the real serving stack would
# wire these to Feast / Neo4j / Redis. Tests + offline runs use this.
# ---------------------------------------------------------------------------


@dataclass
class HistoricalTransaction:
    transaction_id: int | str
    transaction_dt: int  # seconds since IEEE-CIS epoch
    transaction_amt: float
    is_fraud: int = 0
    card_id: int | str | None = None
    device_id: str | None = None
    email: str | None = None
    ip: str | None = None
    merchant: str | None = None


class CardHistoryStore:
    """In-memory key/value mock for card-level history.

    The serving layer can replace this with a Feast online-store call
    without changing the tool signatures. Keys are ``card_id``; values
    are sorted lists of :class:`HistoricalTransaction`.
    """

    def __init__(self) -> None:
        self._by_card: dict[int | str, list[HistoricalTransaction]] = {}

    def add(self, txn: HistoricalTransaction) -> None:
        if txn.card_id is None:
            return
        self._by_card.setdefault(txn.card_id, []).append(txn)

    def card_history(
        self,
        card_id: int | str,
        *,
        as_of_dt: int | None = None,
        window_seconds: int | None = None,
    ) -> list[HistoricalTransaction]:
        rows = list(self._by_card.get(card_id, ()))
        if as_of_dt is not None:
            rows = [r for r in rows if r.transaction_dt <= as_of_dt]
            if window_seconds is not None:
                cutoff = as_of_dt - int(window_seconds)
                rows = [r for r in rows if r.transaction_dt >= cutoff]
        rows.sort(key=lambda r: r.transaction_dt)
        return rows


# ---------------------------------------------------------------------------
# Tool 1: get_transaction_details
# ---------------------------------------------------------------------------


def get_transaction_details(
    *,
    transaction: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    """Echo the transaction + the model's prediction explanation."""
    t0 = time.perf_counter()
    top_features = list(prediction.get("top_features") or [])
    out = {
        "status": "ok",
        "tool": "get_transaction_details",
        "transaction_id": transaction.get("transaction_id"),
        "transaction_amt": transaction.get("transaction_amt"),
        "transaction_dt": transaction.get("transaction_dt"),
        "product_cd": transaction.get("product_cd"),
        "card_id": transaction.get("card1"),
        "device_type": transaction.get("device_type") or transaction.get("DeviceType"),
        "p_emaildomain": transaction.get("p_emaildomain") or transaction.get("P_emaildomain"),
        "fraud_score": prediction.get("fraud_score"),
        "risk_level": prediction.get("risk_level"),
        "is_fraud_predicted": prediction.get("is_fraud_predicted"),
        "model_version": prediction.get("model_version"),
        "top_features": top_features[:5],
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
    }
    log.debug("tool_get_transaction_details", elapsed_ms=out["elapsed_ms"])
    return out


# ---------------------------------------------------------------------------
# Tool 2: analyze_card_history
# ---------------------------------------------------------------------------


def analyze_card_history(
    *,
    card_id: int | str | None,
    as_of_dt: int | None = None,
    window_days: int = 30,
    history: CardHistoryStore | None = None,
) -> dict[str, Any]:
    """30-day card-level rollup: count, spend, avg, fraud, velocity."""
    t0 = time.perf_counter()
    if card_id is None or history is None:
        return _empty(
            tool="analyze_card_history",
            reason="missing_card_id_or_store" if history is None else "missing_card_id",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    rows = history.card_history(
        card_id,
        as_of_dt=as_of_dt,
        window_seconds=window_days * 86400 if as_of_dt is not None else None,
    )
    if not rows:
        return _empty(
            tool="analyze_card_history",
            reason="no_history",
            card_id=str(card_id),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    amts = [r.transaction_amt for r in rows]
    fraud_count = sum(r.is_fraud for r in rows)
    velocity = _velocity_per_hour(rows, as_of_dt=as_of_dt)
    out = {
        "status": "ok",
        "tool": "analyze_card_history",
        "card_id": str(card_id),
        "n_transactions": len(rows),
        "total_spend": float(sum(amts)),
        "avg_amount": float(statistics.fmean(amts)) if amts else 0.0,
        "max_amount": float(max(amts)) if amts else 0.0,
        "fraud_count": int(fraud_count),
        "fraud_rate": float(fraud_count) / len(rows) if rows else 0.0,
        "velocity_per_hour": velocity,
        "window_days": window_days,
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
    }
    return out


# ---------------------------------------------------------------------------
# Tool 3: explore_graph_neighborhood
# ---------------------------------------------------------------------------


def explore_graph_neighborhood(
    *,
    transaction_id: int | str,
    card_id: int | str | None = None,
    graph: Any | None = None,
    n_hops: int = 2,
) -> dict[str, Any]:
    """Walk N hops over a card-card / shared-device graph.

    ``graph`` may be a ``networkx.Graph`` or anything exposing
    ``neighbors(node)``. When absent we return a stub describing what
    we'd have done.
    """
    t0 = time.perf_counter()
    if graph is None or card_id is None:
        return _empty(
            tool="explore_graph_neighborhood",
            reason="no_graph_or_card_id",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            n_hops=n_hops,
        )

    visited: set[Any] = set()
    frontier: set[Any] = {card_id}
    levels: list[set[Any]] = []
    for _ in range(max(1, n_hops)):
        next_frontier: set[Any] = set()
        for node in frontier:
            visited.add(node)
            try:
                neighbors = list(graph.neighbors(node))  # type: ignore[union-attr]
            except Exception:
                neighbors = []
            for nb in neighbors:
                if nb not in visited:
                    next_frontier.add(nb)
        levels.append(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    n_unique = sum(len(level) for level in levels)
    return {
        "status": "ok",
        "tool": "explore_graph_neighborhood",
        "transaction_id": transaction_id,
        "card_id": str(card_id),
        "n_hops": n_hops,
        "n_unique_neighbors": n_unique,
        "neighbors_by_hop": [[str(n) for n in level] for level in levels],
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
    }


# ---------------------------------------------------------------------------
# Tool 4: match_fraud_patterns
# ---------------------------------------------------------------------------


_VELOCITY_SPIKE_THRESHOLD = 5.0  # tx/hour
_CARD_TESTING_AVG_AMT = 5.0  # USD
_CARD_TESTING_MIN_TX = 10
_RING_NEIGHBORHOOD = 4  # n_unique_neighbors


def match_fraud_patterns(
    *,
    history_summary: Mapping[str, Any] | None = None,
    velocity_summary: Mapping[str, Any] | None = None,
    neighborhood_summary: Mapping[str, Any] | None = None,
    fraud_score: float = 0.0,
) -> dict[str, Any]:
    """Match the canonical fraud patterns against tool 2/3/6 evidence."""
    t0 = time.perf_counter()
    matched: list[FraudPattern] = []

    # velocity_spike
    velocity_per_hour = _f((history_summary or {}).get("velocity_per_hour")) or 0.0
    velocity_1h = _f((velocity_summary or {}).get("velocity_1h")) or 0.0
    velocity_baseline = _f((velocity_summary or {}).get("baseline_per_hour")) or 0.0
    if velocity_per_hour >= _VELOCITY_SPIKE_THRESHOLD or (
        velocity_1h >= 3.0 and velocity_baseline > 0 and velocity_1h >= 3 * velocity_baseline
    ):
        conf = min(0.95, max(0.5, velocity_per_hour / 20.0 + fraud_score / 2.0))
        matched.append(
            FraudPattern(
                name="velocity_spike",
                confidence=conf,
                rationale=f"Velocity {velocity_per_hour:.1f}/h vs baseline "
                f"{velocity_baseline:.1f}/h (1h window: {velocity_1h:.1f}).",
            )
        )

    # card_testing
    n_tx = int(_f((history_summary or {}).get("n_transactions")) or 0)
    avg_amt = _f((history_summary or {}).get("avg_amount")) or 0.0
    if n_tx >= _CARD_TESTING_MIN_TX and 0 < avg_amt <= _CARD_TESTING_AVG_AMT:
        matched.append(
            FraudPattern(
                name="card_testing",
                confidence=min(0.95, 0.5 + n_tx / 100.0),
                rationale=f"{n_tx} transactions averaging ${avg_amt:.2f} -- micro-charge pattern.",
            )
        )

    # collusion_ring
    n_neighbors = int(_f((neighborhood_summary or {}).get("n_unique_neighbors")) or 0)
    if n_neighbors >= _RING_NEIGHBORHOOD:
        matched.append(
            FraudPattern(
                name="collusion_ring",
                confidence=min(0.95, 0.5 + n_neighbors / 20.0),
                rationale=f"Card connected to {n_neighbors} peer cards via shared device/address.",
            )
        )

    # account_takeover
    if (
        history_summary
        and float(history_summary.get("max_amount") or 0)
        >= 4 * float(history_summary.get("avg_amount") or 1.0)
        and float(history_summary.get("avg_amount") or 0) > 0
    ):
        matched.append(
            FraudPattern(
                name="account_takeover",
                confidence=0.6,
                rationale=(
                    f"Max charge ${history_summary.get('max_amount', 0):.2f} is 4x+ the "
                    f"30d average -- consistent with account_takeover."
                ),
            )
        )

    if not matched and fraud_score >= 0.7:
        matched.append(
            FraudPattern(
                name="velocity_spike",
                confidence=fraud_score,
                rationale="High score with no specific pattern triggered; defaulting.",
            )
        )

    if not matched:
        matched.append(
            FraudPattern(
                name="none",
                confidence=1.0 - fraud_score,
                rationale="No canonical fraud pattern matched.",
            )
        )

    return {
        "status": "ok",
        "tool": "match_fraud_patterns",
        "matched_patterns": [p.model_dump() for p in matched],
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
    }


# ---------------------------------------------------------------------------
# Tool 5: retrieve_similar_cases (GraphRAG)
# ---------------------------------------------------------------------------


def retrieve_similar_cases(
    *,
    embedding: Iterable[float] | np.ndarray | None,
    case_bank: CaseBank | None = None,
    k: int = 3,
) -> dict[str, Any]:
    """Find the top-k cases whose GNN embedding is closest to ``embedding``.

    Falls back to an empty list if the embedding is missing. If
    ``case_bank`` is None we build one from the seed.
    """
    t0 = time.perf_counter()
    if embedding is None:
        return _empty(
            tool="retrieve_similar_cases",
            reason="no_embedding",
            similar_cases=[],
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    bank = case_bank or CaseBank.with_seed()
    q = np.asarray(list(embedding), dtype=np.float32)
    pairs = bank.search(q, k=k)
    similar = [
        SimilarCase(
            case_id=rec.case_id,
            similarity=max(0.0, min(1.0, (sim + 1.0) / 2.0)),  # cosine -> [0,1]
            pattern=rec.pattern,
            summary=rec.summary,
        )
        for rec, sim in pairs
    ]
    return {
        "status": "ok",
        "tool": "retrieve_similar_cases",
        "k": k,
        "similar_cases": [s.model_dump() for s in similar],
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
    }


# ---------------------------------------------------------------------------
# Tool 6: analyze_velocity (multi-window)
# ---------------------------------------------------------------------------


def analyze_velocity(
    *,
    card_id: int | str | None,
    as_of_dt: int | None,
    history: CardHistoryStore | None = None,
    windows_seconds: tuple[int, ...] = (3600, 21600, 86400),  # 1h, 6h, 24h
) -> dict[str, Any]:
    """Multi-window velocity (tx/hour) vs. 30-day baseline."""
    t0 = time.perf_counter()
    if card_id is None or as_of_dt is None or history is None:
        return _empty(
            tool="analyze_velocity",
            reason="missing_inputs",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            velocity_1h=0.0,
            velocity_6h=0.0,
            velocity_24h=0.0,
            baseline_per_hour=0.0,
        )

    out: dict[str, Any] = {"status": "ok", "tool": "analyze_velocity"}
    label_for = {3600: "velocity_1h", 21600: "velocity_6h", 86400: "velocity_24h"}
    for w in windows_seconds:
        rows = history.card_history(card_id, as_of_dt=as_of_dt, window_seconds=w)
        # Per-hour velocity within the window.
        hours = max(w / 3600.0, 1.0)
        out[label_for.get(w, f"velocity_{w}s")] = float(len(rows)) / hours

    baseline_rows = history.card_history(card_id, as_of_dt=as_of_dt, window_seconds=30 * 86400)
    out["baseline_per_hour"] = float(len(baseline_rows)) / (30 * 24) if baseline_rows else 0.0
    out["elapsed_ms"] = (time.perf_counter() - t0) * 1000
    return out


# ---------------------------------------------------------------------------
# Tool 7: compute_cross_entity_risk
# ---------------------------------------------------------------------------


def compute_cross_entity_risk(
    *,
    transaction: Mapping[str, Any],
    history_summary: Mapping[str, Any] | None = None,
    neighborhood_summary: Mapping[str, Any] | None = None,
    fraud_score: float = 0.0,
) -> dict[str, Any]:
    """Per-entity-type risk score, with contributing factors.

    Five entity types: card / device / email / ip / merchant. We blend
    the model's fraud_score with feature signals from prior tools and
    cap each entity's score at 1.0.
    """
    t0 = time.perf_counter()
    entities: list[EntityRisk] = []

    base = float(fraud_score or 0.0)
    n_neighbors = int(_f((neighborhood_summary or {}).get("n_unique_neighbors")) or 0)
    fraud_rate = float(_f((history_summary or {}).get("fraud_rate")) or 0.0)
    velocity = float(_f((history_summary or {}).get("velocity_per_hour")) or 0.0)

    # Card
    card_factors: list[str] = []
    card_score = base
    if fraud_rate > 0:
        card_score += 0.3 * min(1.0, fraud_rate * 5)
        card_factors.append(f"30d fraud_rate={fraud_rate:.2%}")
    if velocity >= _VELOCITY_SPIKE_THRESHOLD:
        card_score += 0.2
        card_factors.append(f"velocity={velocity:.1f}/h")
    if n_neighbors >= _RING_NEIGHBORHOOD:
        card_score += 0.2
        card_factors.append(f"n_peer_cards={n_neighbors}")
    entities.append(
        EntityRisk(
            entity_type="card",
            entity_id=str(transaction.get("card1") or "unknown"),
            risk_score=_clip01(card_score),
            contributing_factors=card_factors or ["no signal"],
        )
    )

    # Device
    device = transaction.get("device_type") or transaction.get("DeviceType") or "unknown"
    device_score = base * 0.7
    device_factors = ["model fraud_score"]
    if n_neighbors >= _RING_NEIGHBORHOOD:
        device_score += 0.25
        device_factors.append("device_shared_with_peer_cards")
    entities.append(
        EntityRisk(
            entity_type="device",
            entity_id=str(device),
            risk_score=_clip01(device_score),
            contributing_factors=device_factors,
        )
    )

    # Email
    email = transaction.get("p_emaildomain") or transaction.get("P_emaildomain") or "unknown"
    email_score = base * 0.5
    email_factors = ["model fraud_score"]
    if isinstance(email, str) and any(
        marker in email for marker in ("temp", "mailinator", "10min", "guerrilla")
    ):
        email_score += 0.4
        email_factors.append("disposable_email_domain")
    entities.append(
        EntityRisk(
            entity_type="email",
            entity_id=str(email),
            risk_score=_clip01(email_score),
            contributing_factors=email_factors,
        )
    )

    # IP (we only have id_02 bin in IEEE-CIS -- approximate)
    ip_id = str(transaction.get("id_02") or "unknown")
    entities.append(
        EntityRisk(
            entity_type="ip",
            entity_id=ip_id,
            risk_score=_clip01(base * 0.6 + (0.2 if fraud_rate > 0 else 0)),
            contributing_factors=["model fraud_score", f"30d fraud_rate={fraud_rate:.2%}"],
        )
    )

    # Merchant
    merchant_id = str(transaction.get("product_cd") or "W")
    entities.append(
        EntityRisk(
            entity_type="merchant",
            entity_id=merchant_id,
            risk_score=_clip01(base * 0.4),
            contributing_factors=["model fraud_score"],
        )
    )

    return {
        "status": "ok",
        "tool": "compute_cross_entity_risk",
        "entity_risks": [e.model_dump() for e in entities],
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
    }


# ---------------------------------------------------------------------------
# Tool 8: generate_investigation_report
# ---------------------------------------------------------------------------


def generate_investigation_report(
    *,
    state_evidence: Mapping[str, Any],
    prediction: Mapping[str, Any],
    transaction_id: int | str,
    alert_id: str,
    depth: str,
    llm: Any,
) -> dict[str, Any]:
    """Hand the evidence to the LLM and parse a structured response back."""
    from fraud_detection.agent.prompts import REPORT_TEMPLATE, SYSTEM_PROMPT

    t0 = time.perf_counter()
    evidence_block = _render_evidence(state_evidence)
    user_prompt = REPORT_TEMPLATE.substitute(
        alert_id=alert_id,
        transaction_id=transaction_id,
        risk_level=prediction.get("risk_level", "LOW"),
        fraud_score=f"{float(prediction.get('fraud_score') or 0):.3f}",
        depth=depth,
        evidence_block=evidence_block,
    )

    try:
        resp = llm.invoke(SYSTEM_PROMPT, user_prompt)
        parsed = resp.parsed or {}
        out: dict[str, Any] = {
            "status": "ok",
            "tool": "generate_investigation_report",
            "model": resp.model,
            "is_stub": getattr(resp, "is_stub", False),
            "summary": parsed.get("summary") or "",
            "narrative": parsed.get("narrative") or resp.content,
            "recommended_action": (parsed.get("recommended_action") or "review").lower(),
            "confidence": float(parsed.get("confidence") or 0.5),
            "matched_patterns": parsed.get("matched_patterns") or [],
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }
        return out
    except Exception as exc:
        log.exception("tool_generate_investigation_report_failed", error=str(exc))
        return _empty(
            tool="generate_investigation_report",
            reason="llm_invoke_failed",
            error=str(exc),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty(*, tool: str, reason: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "skipped",
        "tool": tool,
        "reason": reason,
    }
    out.update(extra)
    return out


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _velocity_per_hour(
    rows: list[HistoricalTransaction],
    *,
    as_of_dt: int | None = None,
) -> float:
    if not rows:
        return 0.0
    timestamps = [r.transaction_dt for r in rows]
    span = (max(timestamps) - min(timestamps)) if as_of_dt is None else (as_of_dt - min(timestamps))
    span = max(span, 1)
    hours = span / 3600.0
    return float(len(rows)) / max(hours, 1.0)


def _render_evidence(evidence: Mapping[str, Any]) -> str:
    """Render the evidence dict into a compact bulleted block for the LLM."""
    if not evidence:
        return "(no evidence collected)"
    lines: list[str] = []
    for tool_name, payload in evidence.items():
        if not isinstance(payload, Mapping):
            continue
        status = payload.get("status", "ok")
        prefix = f"- [{tool_name}] ({status})"
        # Pick out 4-6 of the most informative fields per tool.
        keep_keys = [
            k
            for k in (
                "n_transactions",
                "total_spend",
                "avg_amount",
                "max_amount",
                "fraud_count",
                "fraud_rate",
                "velocity_per_hour",
                "velocity_1h",
                "velocity_6h",
                "velocity_24h",
                "baseline_per_hour",
                "n_unique_neighbors",
                "matched_patterns",
                "similar_cases",
                "entity_risks",
                "fraud_score",
                "risk_level",
                "transaction_amt",
                "card_id",
                "reason",
            )
            if k in payload
        ]
        bullets = []
        for k in keep_keys:
            v = payload.get(k)
            if isinstance(v, list):
                v = f"[{len(v)} items]"
            elif isinstance(v, float):
                v = f"{v:.3f}"
            bullets.append(f"{k}={v}")
        if bullets:
            prefix += "  " + ", ".join(bullets)
        lines.append(prefix)
    return "\n".join(lines)


__all__ = [
    "TOOL_NAMES",
    "CardHistoryStore",
    "HistoricalTransaction",
    "analyze_card_history",
    "analyze_velocity",
    "compute_cross_entity_risk",
    "explore_graph_neighborhood",
    "generate_investigation_report",
    "get_transaction_details",
    "match_fraud_patterns",
    "retrieve_similar_cases",
]
