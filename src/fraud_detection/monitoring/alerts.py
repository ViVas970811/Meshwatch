"""Alert-rule evaluation (Phase 7).

This module gives the *application* a way to evaluate the same conditions
that Prometheus is alerting on -- handy for surfacing them in the React
dashboard or pushing them through the Kafka alert topic without waiting
for Prometheus to fire.

Two layers:

* :class:`AlertRule` -- a tiny declarative spec (name, severity, predicate,
  reason builder) you can stack into an :class:`AlertEvaluator`.
* :func:`default_rules` -- the canonical set of rules that mirrors
  ``configs/prometheus_rules.yml`` so application-fired alerts and
  Prometheus-fired alerts stay aligned.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

AlertSeverity = Literal["info", "warning", "critical"]


@dataclass(slots=True)
class AlertRule:
    """A single declarative rule.

    The ``predicate`` is a pure function over a context dictionary. The
    context typically holds:

    * ``drift_overall``    -- ``float`` overall PSI
    * ``drift_severe``     -- ``int`` count of severe-drift features
    * ``performance``      -- a :class:`PerformanceSnapshot`
    * ``latency_p95``      -- ``float`` recent P95 in seconds
    * ``error_rate``       -- ``float`` 0..1 over the last window
    * ``model_loaded``     -- ``bool``

    ``reason`` is a builder so messages embed live values from the context.
    """

    name: str
    severity: AlertSeverity
    description: str
    predicate: Callable[[dict[str, Any]], bool]
    reason: Callable[[dict[str, Any]], str] = field(default=lambda _ctx: "Triggered")


@dataclass(slots=True)
class Alert:
    """An active alert at a point in time."""

    name: str
    severity: AlertSeverity
    description: str
    reason: str
    fired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if isinstance(d["fired_at"], datetime):
            d["fired_at"] = d["fired_at"].isoformat()
        return d


class AlertEvaluator:
    """Evaluate a list of :class:`AlertRule` against a context."""

    def __init__(self, rules: Iterable[AlertRule]) -> None:
        self.rules: list[AlertRule] = list(rules)

    def evaluate(self, ctx: dict[str, Any]) -> list[Alert]:
        out: list[Alert] = []
        for rule in self.rules:
            try:
                if rule.predicate(ctx):
                    out.append(
                        Alert(
                            name=rule.name,
                            severity=rule.severity,
                            description=rule.description,
                            reason=rule.reason(ctx),
                        )
                    )
            except Exception as exc:  # pragma: no cover -- defensive only
                out.append(
                    Alert(
                        name=rule.name,
                        severity="warning",
                        description=rule.description,
                        reason=f"Rule evaluation failed: {exc!s}",
                    )
                )
        return out


# ---------------------------------------------------------------------------
# Canonical rule set -- mirrors configs/prometheus_rules.yml
# ---------------------------------------------------------------------------


def _get(ctx: dict[str, Any], key: str, default: Any) -> Any:
    val = ctx.get(key, default)
    return default if val is None else val


def default_rules(
    *,
    psi_alert: float = 0.25,
    psi_severe_features: int = 10,
    latency_p95_warn: float = 0.05,  # 50ms -- plan budget
    error_rate_warn: float = 0.05,
    performance_recall_warn: float = 0.55,
    performance_auprc_warn: float = 0.60,
    fraud_rate_spike: float = 0.10,  # 10% production fraud rate
    shadow_agreement_warn: float = 0.90,
) -> list[AlertRule]:
    """Return the canonical set of rules.

    Defaults mirror ``configs/prometheus_rules.yml`` so that the application
    surface (``/api/v1/monitoring/alerts``) reports the same conditions as
    the Prometheus alertmanager. Override any threshold via kwargs if site
    policy differs.
    """
    return [
        AlertRule(
            name="ModelNotLoaded",
            severity="critical",
            description="The serving process is up but the ensemble artifact failed to load.",
            predicate=lambda ctx: not bool(ctx.get("model_loaded", True)),
            reason=lambda _ctx: (
                "FraudPredictor returned None at startup; /api/v1/predict will 503."
            ),
        ),
        AlertRule(
            name="DataDriftSevere",
            severity="critical",
            description=f"Overall PSI breached the alert threshold ({psi_alert}).",
            predicate=lambda ctx: float(_get(ctx, "drift_overall", 0.0)) >= psi_alert,
            reason=lambda ctx: (
                f"Overall PSI = {float(_get(ctx, 'drift_overall', 0.0)):.3f} "
                f">= {psi_alert}; "
                f"{int(_get(ctx, 'drift_severe', 0))} feature(s) above threshold."
            ),
        ),
        AlertRule(
            name="DataDrift",
            severity="warning",
            description=(
                f"More than {psi_severe_features} features show severe drift simultaneously."
            ),
            predicate=lambda ctx: int(_get(ctx, "drift_severe", 0)) > psi_severe_features,
            reason=lambda ctx: (
                f"{int(_get(ctx, 'drift_severe', 0))} features above PSI alert threshold."
            ),
        ),
        AlertRule(
            name="ModelDegradation",
            severity="critical",
            description=f"Production AUPRC fell below {performance_auprc_warn:.2f}.",
            predicate=lambda ctx: _auprc(ctx) is not None and _auprc(ctx) < performance_auprc_warn,
            reason=lambda ctx: (
                f"AUPRC {_auprc(ctx):.3f} on {_n_labelled(ctx)} labelled records "
                f"(threshold {performance_auprc_warn:.2f})."
            ),
        ),
        AlertRule(
            name="ModelPerformanceDegraded",
            severity="warning",
            description=(
                f"Production recall fell below {performance_recall_warn:.2f} "
                "over the latest labelled window."
            ),
            predicate=lambda ctx: (
                _recall(ctx) is not None and _recall(ctx) < performance_recall_warn
            ),
            reason=lambda ctx: f"Recall {_recall(ctx):.3f} on {_n_labelled(ctx)} labelled records.",
        ),
        AlertRule(
            name="FraudRateSpike",
            severity="critical",
            description=(
                f"Production fraud rate exceeded {fraud_rate_spike * 100:.0f}% "
                "(training baseline ~3.5%)."
            ),
            predicate=lambda ctx: float(_get(ctx, "production_fraud_rate", 0.0)) > fraud_rate_spike,
            reason=lambda ctx: (
                f"Production fraud rate = "
                f"{float(_get(ctx, 'production_fraud_rate', 0.0)) * 100:.2f}%, "
                f"reference = {float(_get(ctx, 'reference_fraud_rate', 0.035)) * 100:.2f}%."
            ),
        ),
        AlertRule(
            name="HighLatency",
            severity="warning",
            description=(
                f"Recent P95 inference latency exceeded the {latency_p95_warn * 1000:.0f}ms budget."
            ),
            predicate=lambda ctx: float(_get(ctx, "latency_p95", 0.0)) > latency_p95_warn,
            reason=lambda ctx: f"P95 latency = {float(_get(ctx, 'latency_p95', 0.0)) * 1000:.1f}ms",
        ),
        AlertRule(
            name="HighErrorRate",
            severity="critical",
            description=f"HTTP 5xx rate exceeded {error_rate_warn * 100:.1f}% over the last window.",
            predicate=lambda ctx: float(_get(ctx, "error_rate", 0.0)) > error_rate_warn,
            reason=lambda ctx: f"5xx rate = {float(_get(ctx, 'error_rate', 0.0)) * 100:.2f}%",
        ),
        AlertRule(
            name="ShadowChallengerDisagreement",
            severity="warning",
            description=(
                f"Shadow challenger agreement rate fell below {shadow_agreement_warn:.0%}."
            ),
            predicate=lambda ctx: (
                _shadow_agreement(ctx) is not None
                and _shadow_agreement(ctx) < shadow_agreement_warn
            ),
            reason=lambda ctx: (
                f"Agreement rate = {_shadow_agreement(ctx):.3f} over the recent window."
            ),
        ),
    ]


def _recall(ctx: dict[str, Any]) -> float | None:
    snap = ctx.get("performance")
    if snap is None:
        return None
    if hasattr(snap, "recall") and getattr(snap, "n_labelled", 0):
        return float(snap.recall)
    if isinstance(snap, dict):
        return float(snap.get("recall", 0.0)) if snap.get("n_labelled", 0) else None
    return None


def _n_labelled(ctx: dict[str, Any]) -> int:
    snap = ctx.get("performance")
    if snap is None:
        return 0
    if hasattr(snap, "n_labelled"):
        return int(snap.n_labelled)
    if isinstance(snap, dict):
        return int(snap.get("n_labelled", 0))
    return 0


def _auprc(ctx: dict[str, Any]) -> float | None:
    snap = ctx.get("performance")
    if snap is None:
        return None
    if hasattr(snap, "auprc") and getattr(snap, "n_labelled", 0):
        return None if snap.auprc is None else float(snap.auprc)
    if isinstance(snap, dict):
        auprc = snap.get("auprc")
        n = snap.get("n_labelled", 0)
        if not n or auprc is None:
            return None
        return float(auprc)
    return None


def _shadow_agreement(ctx: dict[str, Any]) -> float | None:
    summary = ctx.get("shadow")
    if summary is None:
        return None
    if hasattr(summary, "agreement_rate") and getattr(summary, "n_total", 0):
        return float(summary.agreement_rate)
    if isinstance(summary, dict):
        n = summary.get("n_total", 0)
        if not n:
            return None
        return float(summary.get("agreement_rate", 1.0))
    return None


__all__ = [
    "Alert",
    "AlertEvaluator",
    "AlertRule",
    "AlertSeverity",
    "default_rules",
]
