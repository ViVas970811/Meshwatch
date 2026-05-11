"""Drift detection, performance tracking, alerting (Phase 7).

Public API:

* :class:`DriftDetector`, :class:`DriftReport`, :class:`FeatureDrift` --
  numeric / categorical drift detection with PSI, KS, chi-square and JSD.
* :class:`PerformanceTracker`, :class:`PerformanceSnapshot` -- rolling
  classification metrics keyed by labelled arrivals.
* :class:`MonitoringMetrics`, :data:`monitoring_metrics` -- Prometheus
  collectors for drift, performance, and the agent layer.
* :class:`AlertEvaluator`, :class:`AlertRule`, :func:`default_rules` --
  application-side mirror of the Prometheus alert rules.
* :func:`report_to_json`, :func:`report_to_html` -- drift report
  serialisation.
* :class:`MonitoringState`, :func:`get_state` -- the in-process state
  holder that the FastAPI app, the CLI and the tests all share.
"""

from __future__ import annotations

from fraud_detection.monitoring.alerts import (
    Alert,
    AlertEvaluator,
    AlertRule,
    AlertSeverity,
    default_rules,
)
from fraud_detection.monitoring.drift import (
    DriftDetector,
    DriftDetectorConfig,
    DriftReport,
    DriftSeverity,
    FeatureDrift,
    categorical_psi,
    chi_square,
    evidently_html_report,
    js_divergence,
    ks_test,
    label_drift,
    population_stability_index,
    prediction_distribution_drift,
)
from fraud_detection.monitoring.performance import (
    PerformanceSnapshot,
    PerformanceTracker,
    auprc,
    auroc,
)
from fraud_detection.monitoring.registry import MonitoringMetrics, monitoring_metrics
from fraud_detection.monitoring.reports import (
    report_to_html,
    report_to_json,
    write_html,
    write_json,
)
from fraud_detection.monitoring.shadow import (
    ShadowDecision,
    ShadowDeployment,
    ShadowSummary,
)
from fraud_detection.monitoring.state import MonitoringState, get_state, reset_state

__all__ = [
    "Alert",
    "AlertEvaluator",
    "AlertRule",
    "AlertSeverity",
    "DriftDetector",
    "DriftDetectorConfig",
    "DriftReport",
    "DriftSeverity",
    "FeatureDrift",
    "MonitoringMetrics",
    "MonitoringState",
    "PerformanceSnapshot",
    "PerformanceTracker",
    "ShadowDecision",
    "ShadowDeployment",
    "ShadowSummary",
    "auprc",
    "auroc",
    "categorical_psi",
    "chi_square",
    "default_rules",
    "evidently_html_report",
    "get_state",
    "js_divergence",
    "ks_test",
    "label_drift",
    "monitoring_metrics",
    "population_stability_index",
    "prediction_distribution_drift",
    "report_to_html",
    "report_to_json",
    "reset_state",
    "write_html",
    "write_json",
]
