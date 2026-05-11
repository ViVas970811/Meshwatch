"""Extended Prometheus metrics for Phase 7.

The serving middleware already exposes per-request metrics under the
``meshwatch_http_*`` and ``meshwatch_predictions_*`` namespaces. Phase 7
adds three more groups:

* **drift** -- the latest overall + per-feature PSI; gauges are convenient
  because Prometheus alerts on absolute values, not deltas.
* **performance** -- the latest precision/recall/F1/AUROC snapshot.
* **agent** -- per-tool invocation counts + latency histograms; the
  Phase 5 investigator publishes here via :meth:`MonitoringMetrics.record_agent_run`.

All groups degrade to no-ops when ``prometheus-client`` is missing, mirroring
the contract of the serving middleware. The single :class:`MonitoringMetrics`
instance attaches itself to the registry that already backs ``/api/v1/metrics``,
so a single scrape gives operators the full picture.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from fraud_detection.serving.middleware import metrics as serving_metrics


class _MetricLike(Protocol):
    def labels(self, *args: Any, **kwargs: Any) -> _MetricLike: ...
    def inc(self, *args: Any, **kwargs: Any) -> None: ...
    def dec(self, *args: Any, **kwargs: Any) -> None: ...
    def observe(self, *args: Any, **kwargs: Any) -> None: ...
    def set(self, *args: Any, **kwargs: Any) -> None: ...


class _NoopMetric:
    """Shared no-op used when prometheus_client is missing."""

    def labels(self, *args: Any, **kwargs: Any) -> _NoopMetric:
        return self

    def inc(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def dec(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def observe(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class MonitoringMetrics:
    """Lazily-constructed Prometheus collectors for the monitoring layer.

    Constructing this class twice on the same process is safe -- it reuses
    the shared registry from :mod:`fraud_detection.serving.middleware`,
    so the second call simply re-fetches the existing collectors.
    """

    def __init__(self) -> None:
        self._enabled = bool(getattr(serving_metrics, "_enabled", False))
        registry = getattr(serving_metrics, "registry", None)

        if not self._enabled or registry is None:
            self.drift_overall = _NoopMetric()
            self.drift_feature = _NoopMetric()
            self.drift_features_severe = _NoopMetric()
            self.drift_features_moderate = _NoopMetric()
            self.label_drift = _NoopMetric()
            self.prediction_distribution_drift = _NoopMetric()
            self.production_fraud_rate = _NoopMetric()
            self.performance_precision = _NoopMetric()
            self.performance_recall = _NoopMetric()
            self.performance_f1 = _NoopMetric()
            self.performance_auroc = _NoopMetric()
            self.performance_auprc = _NoopMetric()
            self.performance_n_labelled = _NoopMetric()
            self.agent_invocations = _NoopMetric()
            self.agent_tool_invocations = _NoopMetric()
            self.agent_latency = _NoopMetric()
            self.shadow_decisions = _NoopMetric()
            self.shadow_agreements = _NoopMetric()
            self.shadow_score_delta = _NoopMetric()
            self.shadow_challenger_latency = _NoopMetric()
            return

        from prometheus_client import (
            Counter,
            Gauge,
            Histogram,
        )

        def _get_or_create(factory_cls: Any, name: str, *args: Any, **kwargs: Any) -> Any:
            """Return an existing collector if already in the registry."""
            existing = self._existing_collector(name)
            if existing is not None:
                return existing
            return factory_cls(name, *args, registry=registry, **kwargs)

        self.drift_overall = _get_or_create(
            Gauge,
            "meshwatch_drift_psi_overall",
            "Overall PSI between the reference (training) and current production windows",
        )
        self.drift_feature = _get_or_create(
            Gauge,
            "meshwatch_drift_psi_feature",
            "Per-feature PSI",
            ["feature", "kind"],
        )
        self.drift_features_severe = _get_or_create(
            Gauge,
            "meshwatch_drift_features_severe",
            "Number of features with PSI >= alert threshold",
        )
        self.drift_features_moderate = _get_or_create(
            Gauge,
            "meshwatch_drift_features_moderate",
            "Number of features with PSI in [warn, alert)",
        )
        self.performance_precision = _get_or_create(
            Gauge, "meshwatch_performance_precision", "Production precision at the alert threshold"
        )
        self.performance_recall = _get_or_create(
            Gauge, "meshwatch_performance_recall", "Production recall at the alert threshold"
        )
        self.performance_f1 = _get_or_create(
            Gauge, "meshwatch_performance_f1", "Production F1 at the alert threshold"
        )
        self.performance_auroc = _get_or_create(
            Gauge, "meshwatch_performance_auroc", "Production AUROC over the labelled window"
        )
        self.performance_auprc = _get_or_create(
            Gauge,
            "meshwatch_performance_auprc",
            "Production AUPRC (average precision) over the labelled window",
        )
        self.performance_n_labelled = _get_or_create(
            Gauge,
            "meshwatch_performance_n_labelled",
            "Number of labelled predictions in the current window",
        )
        self.agent_invocations = _get_or_create(
            Counter,
            "meshwatch_agent_invocations_total",
            "Agent investigations completed",
            ["risk_level", "status"],
        )
        self.agent_tool_invocations = _get_or_create(
            Counter,
            "meshwatch_agent_tool_invocations_total",
            "Tool invocations across all agent runs",
            ["tool", "status"],
        )
        self.agent_latency = _get_or_create(
            Histogram,
            "meshwatch_agent_run_latency_seconds",
            "Agent end-to-end run latency",
            ["risk_level"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0),
        )

        # Label / prediction drift -- the plan calls out three drift types:
        # data drift (handled by drift_overall + drift_feature), label drift
        # (production fraud rate vs. training fraud rate), and prediction
        # distribution drift (score-distribution PSI champion-vs-baseline).
        self.label_drift = _get_or_create(
            Gauge,
            "meshwatch_label_drift",
            "Absolute difference between production and training fraud rates",
        )
        self.prediction_distribution_drift = _get_or_create(
            Gauge,
            "meshwatch_prediction_distribution_drift",
            "PSI between production and reference score distributions",
        )
        self.production_fraud_rate = _get_or_create(
            Gauge,
            "meshwatch_production_fraud_rate",
            "Rolling production fraud rate (predictions above threshold / total)",
        )

        # Shadow deployment metrics.
        self.shadow_decisions = _get_or_create(
            Counter,
            "meshwatch_shadow_decisions_total",
            "Shadow champion/challenger decisions",
            ["champion_model", "challenger_model", "outcome"],
        )
        self.shadow_agreements = _get_or_create(
            Gauge,
            "meshwatch_shadow_agreement_rate",
            "Rolling champion/challenger agreement rate (last window)",
        )
        self.shadow_score_delta = _get_or_create(
            Histogram,
            "meshwatch_shadow_score_delta",
            "Absolute score delta between champion and challenger",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
        )
        self.shadow_challenger_latency = _get_or_create(
            Histogram,
            "meshwatch_shadow_challenger_latency_seconds",
            "Challenger inference latency (off the request hot path)",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _existing_collector(name: str) -> Any | None:
        registry = getattr(serving_metrics, "registry", None)
        if registry is None:
            return None
        # CollectorRegistry exposes ``_names_to_collectors`` (private but stable
        # across the 0.* line of prometheus-client). We poke it so that
        # re-instantiating MonitoringMetrics inside tests doesn't blow up
        # with a "Duplicated timeseries" error.
        names_to_collectors = getattr(registry, "_names_to_collectors", {})
        return names_to_collectors.get(name)

    # ------------------------------------------------------------------
    # high-level writers
    # ------------------------------------------------------------------

    def update_drift(self, report: Any) -> None:
        """Update the drift gauges from a :class:`DriftReport`."""
        from fraud_detection.monitoring.drift import DriftReport

        if not isinstance(report, DriftReport):
            return
        self.drift_overall.set(float(report.overall_psi))
        self.drift_features_severe.set(int(report.n_severe))
        self.drift_features_moderate.set(int(report.n_moderate))
        for f in report.features:
            self.drift_feature.labels(f.feature, f.kind).set(float(f.psi))

    def update_performance(self, snapshot: Any) -> None:
        """Update the performance gauges from a :class:`PerformanceSnapshot`."""
        from fraud_detection.monitoring.performance import PerformanceSnapshot

        if not isinstance(snapshot, PerformanceSnapshot):
            return
        self.performance_precision.set(float(snapshot.precision))
        self.performance_recall.set(float(snapshot.recall))
        self.performance_f1.set(float(snapshot.f1))
        self.performance_n_labelled.set(int(snapshot.n_labelled))
        if snapshot.auroc is not None:
            self.performance_auroc.set(float(snapshot.auroc))
        if snapshot.auprc is not None:
            self.performance_auprc.set(float(snapshot.auprc))

    def record_shadow_decision(self, decision: Any) -> None:
        """Update shadow-deployment counters + histograms.

        Accepts a :class:`ShadowDecision` from
        :mod:`fraud_detection.monitoring.shadow`. We import lazily so this
        registry can be loaded without the shadow module having been imported.
        """
        from fraud_detection.monitoring.shadow import ShadowDecision

        if not isinstance(decision, ShadowDecision):
            return
        outcome = "agree" if decision.agreement else "disagree"
        self.shadow_decisions.labels(
            decision.champion_model, decision.challenger_model, outcome
        ).inc()
        self.shadow_score_delta.observe(abs(float(decision.score_delta)))
        self.shadow_challenger_latency.observe(float(decision.challenger_latency_ms) / 1000.0)

    def update_shadow_summary(self, summary: Any) -> None:
        """Push the rolling shadow summary onto the agreement-rate gauge."""
        from fraud_detection.monitoring.shadow import ShadowSummary

        if not isinstance(summary, ShadowSummary):
            return
        self.shadow_agreements.set(float(summary.agreement_rate))

    def update_label_drift(self, *, production_rate: float, reference_rate: float) -> None:
        """Update label-drift gauges (used by drift_report + monitor.py)."""
        self.production_fraud_rate.set(float(production_rate))
        self.label_drift.set(float(abs(production_rate - reference_rate)))

    def update_prediction_distribution_drift(self, psi: float) -> None:
        """Update the prediction-distribution PSI gauge."""
        self.prediction_distribution_drift.set(float(psi))

    def record_agent_run(
        self,
        *,
        risk_level: str,
        latency_seconds: float,
        status: str = "ok",
        tool_calls: Iterable[tuple[str, str]] | None = None,
    ) -> None:
        """Record a single agent investigation.

        Parameters
        ----------
        risk_level
            LOW / MEDIUM / HIGH / CRITICAL.
        latency_seconds
            Wall-clock seconds for the full LangGraph run.
        status
            ``"ok"`` or ``"error"``.
        tool_calls
            Iterable of ``(tool_name, "ok"|"error")`` pairs to count.
        """
        self.agent_invocations.labels(risk_level, status).inc()
        self.agent_latency.labels(risk_level).observe(max(0.0, float(latency_seconds)))
        if tool_calls is not None:
            for tool, tool_status in tool_calls:
                self.agent_tool_invocations.labels(tool, tool_status).inc()


# Module-level singleton -- safe to import everywhere.
monitoring_metrics = MonitoringMetrics()


__all__ = ["MonitoringMetrics", "monitoring_metrics"]
