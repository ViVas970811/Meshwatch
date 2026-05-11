"""Unit tests for the monitoring Prometheus registry (Phase 7)."""

from __future__ import annotations

import numpy as np

from fraud_detection.monitoring import (
    DriftDetector,
    MonitoringMetrics,
    PerformanceSnapshot,
    monitoring_metrics,
)
from fraud_detection.serving.middleware import metrics as serving_metrics


def _drift_report():
    rng = np.random.default_rng(11)
    ref = {
        "amount": rng.normal(loc=0, scale=1, size=400).tolist(),
        "country": rng.choice(["US", "GB"], size=400).tolist(),
    }
    cur = {
        "amount": rng.normal(loc=3, scale=1, size=400).tolist(),
        "country": rng.choice(["US", "GB"], size=400).tolist(),
    }
    return DriftDetector(ref).detect(cur)


class TestSingleton:
    def test_module_level_singleton_is_a_monitoring_metrics(self):
        assert isinstance(monitoring_metrics, MonitoringMetrics)


class TestUpdaters:
    def test_update_drift_runs_without_error(self):
        report = _drift_report()
        # Should not raise regardless of whether prometheus_client is installed.
        monitoring_metrics.update_drift(report)

    def test_update_drift_writes_overall_gauge_when_enabled(self):
        if not serving_metrics.enabled:
            return
        report = _drift_report()
        monitoring_metrics.update_drift(report)
        rendered = serving_metrics.render().decode("utf-8")
        assert "meshwatch_drift_psi_overall" in rendered

    def test_update_drift_records_per_feature_gauges(self):
        if not serving_metrics.enabled:
            return
        report = _drift_report()
        monitoring_metrics.update_drift(report)
        rendered = serving_metrics.render().decode("utf-8")
        # At least one of the features should appear with a positive gauge.
        assert "meshwatch_drift_psi_feature" in rendered

    def test_update_performance_runs_without_error(self):
        snap = PerformanceSnapshot(
            n_total=10,
            n_labelled=5,
            precision=0.8,
            recall=0.6,
            f1=0.7,
            auroc=0.9,
            auprc=0.85,
            threshold=0.7,
        )
        monitoring_metrics.update_performance(snap)

    def test_update_performance_writes_gauges_when_enabled(self):
        if not serving_metrics.enabled:
            return
        snap = PerformanceSnapshot(
            n_total=10,
            n_labelled=5,
            precision=0.8,
            recall=0.6,
            f1=0.7,
            auroc=0.9,
            auprc=0.85,
        )
        monitoring_metrics.update_performance(snap)
        rendered = serving_metrics.render().decode("utf-8")
        assert "meshwatch_performance_precision" in rendered
        assert "meshwatch_performance_recall" in rendered
        assert "meshwatch_performance_f1" in rendered


class TestAgentRecorder:
    def test_record_agent_run_increments_counters(self):
        # The no-op path should also not raise.
        monitoring_metrics.record_agent_run(
            risk_level="HIGH",
            latency_seconds=0.42,
            status="ok",
            tool_calls=[("analyze_card_history", "ok"), ("retrieve_similar_cases", "ok")],
        )

    def test_record_agent_run_emits_metrics_when_enabled(self):
        if not serving_metrics.enabled:
            return
        monitoring_metrics.record_agent_run(
            risk_level="CRITICAL",
            latency_seconds=1.23,
            tool_calls=[("graph_walk", "ok")],
        )
        rendered = serving_metrics.render().decode("utf-8")
        assert "meshwatch_agent_invocations_total" in rendered
        assert "meshwatch_agent_run_latency_seconds" in rendered

    def test_double_instantiation_is_safe(self):
        # Constructing a second instance must not raise even though it
        # would normally hit "Duplicated timeseries in CollectorRegistry".
        another = MonitoringMetrics()
        another.record_agent_run(risk_level="LOW", latency_seconds=0.01, tool_calls=[])

    def test_update_drift_rejects_non_report(self):
        # Passing a non-DriftReport should silently no-op rather than raise.
        monitoring_metrics.update_drift("not a report")  # type: ignore[arg-type]

    def test_update_performance_rejects_non_snapshot(self):
        monitoring_metrics.update_performance({"precision": 0.9})  # type: ignore[arg-type]
