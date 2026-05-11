"""Shadow / champion-vs-challenger deployment (Phase 7).

The plan's MLOps story includes a "shadow deploy" lane: every production
request is served by the *champion* model, but a *challenger* model
scores the same request in the background. We never block on the
challenger -- it just logs its score, the delta vs. champion, and any
disagreement so MLOps engineers can decide whether to promote.

This module provides:

* :class:`ShadowDecision` -- the per-request log record.
* :class:`ShadowDeployment` -- holds a champion :class:`FraudPredictor`,
  an optional challenger predictor, and a bounded history. Exposes
  ``score()`` that always returns the champion's result and kicks off a
  best-effort challenger scoring in a background thread.
* Prometheus metrics: challenger run count, agreement rate,
  score-delta histogram.

Why threads (rather than asyncio)? The :class:`FraudPredictor` hot path
is mostly numpy + XGBoost -- CPU-bound, no event loop yield -- so we
push challenger inference off the request thread with a tiny
:class:`ThreadPoolExecutor` and never wait on the result. If the
challenger blows past a budget (default 100ms) we drop the record.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fraud_detection.utils.logging import get_logger

if TYPE_CHECKING:
    from fraud_detection.serving.predictor import FraudPredictor
    from fraud_detection.serving.schemas import FraudPrediction, TransactionRequest

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-request record
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ShadowDecision:
    """One champion/challenger comparison."""

    transaction_id: int | str
    champion_score: float
    challenger_score: float
    score_delta: float
    champion_label: bool
    challenger_label: bool
    agreement: bool
    champion_model: str
    challenger_model: str
    champion_latency_ms: float
    challenger_latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if isinstance(d["timestamp"], datetime):
            d["timestamp"] = d["timestamp"].isoformat()
        return d


@dataclass(slots=True)
class ShadowSummary:
    """Aggregate stats over a window of decisions."""

    n_total: int = 0
    n_agreement: int = 0
    n_disagreement: int = 0
    agreement_rate: float = 0.0
    mean_score_delta: float = 0.0
    max_score_delta: float = 0.0
    challenger_skipped: int = 0
    challenger_failed: int = 0
    champion_model: str = ""
    challenger_model: str = ""
    window_start: datetime | None = None
    window_end: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("window_start", "window_end"):
            v = d.get(k)
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        return d


# ---------------------------------------------------------------------------
# ShadowDeployment
# ---------------------------------------------------------------------------


class ShadowDeployment:
    """Champion/challenger orchestrator.

    Parameters
    ----------
    champion
        The :class:`FraudPredictor` that owns the production hot path.
        Every call to :meth:`score` returns its prediction unchanged.
    challenger
        Optional second :class:`FraudPredictor`. When supplied, every
        request is also scored by the challenger off the request thread.
        Setting ``None`` (or calling :meth:`detach_challenger`) disables
        the shadow lane.
    max_records
        Bounded history for :meth:`recent_decisions`.
    challenger_budget_ms
        Drop the challenger run if it exceeds this budget.
    threshold
        Decision threshold used to compute the per-request agreement
        bit. Defaults to the champion's threshold.
    """

    def __init__(
        self,
        *,
        champion: FraudPredictor,
        challenger: FraudPredictor | None = None,
        max_records: int = 1000,
        challenger_budget_ms: float = 100.0,
        threshold: float | None = None,
    ) -> None:
        self.champion = champion
        self.challenger = challenger
        self.threshold = float(threshold if threshold is not None else champion.threshold)
        self.challenger_budget_ms = float(challenger_budget_ms)

        self._records: deque[ShadowDecision] = deque(maxlen=max_records)
        self._skipped = 0
        self._failed = 0
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="shadow")

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    def attach_challenger(self, challenger: FraudPredictor) -> None:
        with self._lock:
            self.challenger = challenger

    def detach_challenger(self) -> None:
        with self._lock:
            self.challenger = None

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._skipped = 0
            self._failed = 0

    @property
    def n_records(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def enabled(self) -> bool:
        return self.challenger is not None

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def score(self, request: TransactionRequest) -> FraudPrediction:
        """Score with the champion (synchronous) and shadow-score with the
        challenger (asynchronous, best-effort)."""
        t0 = time.perf_counter()
        result = self.champion.predict_one(request)
        champion_latency_ms = (time.perf_counter() - t0) * 1000

        if self.challenger is None:
            return result

        # Best-effort challenger run. We deliberately do NOT block on the
        # future -- it must never delay the hot path.
        try:
            self._executor.submit(
                self._run_challenger,
                request,
                result.fraud_score,
                champion_latency_ms,
            )
        except RuntimeError:
            # Executor shutting down; skip silently.
            with self._lock:
                self._skipped += 1

        return result

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _run_challenger(
        self,
        request: TransactionRequest,
        champion_score: float,
        champion_latency_ms: float,
    ) -> None:
        if self.challenger is None:
            return
        try:
            t0 = time.perf_counter()
            challenger_result = self.challenger.predict_one(request)
            challenger_latency_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            log.warning("shadow_challenger_failed", error=str(exc))
            with self._lock:
                self._failed += 1
            return

        if challenger_latency_ms > self.challenger_budget_ms:
            log.info(
                "shadow_challenger_over_budget",
                latency_ms=challenger_latency_ms,
                budget_ms=self.challenger_budget_ms,
            )
            with self._lock:
                self._skipped += 1
            return

        decision = ShadowDecision(
            transaction_id=request.transaction_id,
            champion_score=float(champion_score),
            challenger_score=float(challenger_result.fraud_score),
            score_delta=float(challenger_result.fraud_score - champion_score),
            champion_label=champion_score >= self.threshold,
            challenger_label=challenger_result.fraud_score >= self.threshold,
            agreement=(champion_score >= self.threshold)
            == (challenger_result.fraud_score >= self.threshold),
            champion_model=self.champion.model_version,
            challenger_model=self.challenger.model_version,
            champion_latency_ms=champion_latency_ms,
            challenger_latency_ms=challenger_latency_ms,
        )
        with self._lock:
            self._records.append(decision)

        # Hook into Prometheus -- lazy import to keep this module
        # importable without the prometheus extras.
        try:
            from fraud_detection.monitoring.registry import monitoring_metrics

            monitoring_metrics.record_shadow_decision(decision)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def recent_decisions(self, limit: int = 100) -> list[ShadowDecision]:
        with self._lock:
            records = list(self._records)
        return list(reversed(records[-limit:]))

    def summary(self) -> ShadowSummary:
        with self._lock:
            records = list(self._records)
            skipped = self._skipped
            failed = self._failed
        n_total = len(records)
        if n_total == 0:
            return ShadowSummary(
                champion_model=self.champion.model_version,
                challenger_model=self.challenger.model_version
                if self.challenger is not None
                else "",
                challenger_skipped=skipped,
                challenger_failed=failed,
            )
        agree = sum(1 for r in records if r.agreement)
        deltas = [abs(r.score_delta) for r in records]
        return ShadowSummary(
            n_total=n_total,
            n_agreement=agree,
            n_disagreement=n_total - agree,
            agreement_rate=agree / n_total,
            mean_score_delta=sum(deltas) / n_total,
            max_score_delta=max(deltas),
            challenger_skipped=skipped,
            challenger_failed=failed,
            champion_model=self.champion.model_version,
            challenger_model=self.challenger.model_version if self.challenger is not None else "",
            window_start=records[0].timestamp,
            window_end=records[-1].timestamp,
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, *, wait: bool = False) -> None:
        with contextlib.suppress(Exception):  # pragma: no cover -- defensive
            self._executor.shutdown(wait=wait)


__all__ = [
    "ShadowDecision",
    "ShadowDeployment",
    "ShadowSummary",
]
