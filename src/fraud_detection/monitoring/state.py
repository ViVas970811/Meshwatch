"""Module-level state for the monitoring layer.

Both the FastAPI app and the offline CLI need somewhere to stash the
*latest* drift report, the production performance tracker, and the alert
evaluator. Splitting that into a tiny holder keeps the serving module's
``AppState`` clean and lets the test-suite isolate state by importing the
``reset_state`` helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fraud_detection.monitoring.alerts import AlertEvaluator, default_rules
from fraud_detection.monitoring.performance import PerformanceTracker

if TYPE_CHECKING:
    from fraud_detection.monitoring.drift import DriftDetector, DriftReport
    from fraud_detection.monitoring.shadow import ShadowDeployment


@dataclass
class MonitoringState:
    """Live monitoring state shared across processes.

    * ``performance`` -- the rolling labelled-prediction tracker.
    * ``alerts`` -- the application-side alert evaluator.
    * ``drift_detector`` -- the configured detector, ``None`` until a
      reference distribution is registered.
    * ``last_drift_report`` -- the most recent :class:`DriftReport`
      computed via :meth:`recompute_drift` or the CLI.
    """

    performance: PerformanceTracker = field(default_factory=PerformanceTracker)
    alerts: AlertEvaluator = field(default_factory=lambda: AlertEvaluator(default_rules()))
    drift_detector: DriftDetector | None = None
    last_drift_report: DriftReport | None = None
    shadow: ShadowDeployment | None = None
    reference_fraud_rate: float = 0.035  # IEEE-CIS training-time fraud rate


_state: MonitoringState = MonitoringState()


def get_state() -> MonitoringState:
    """Return the process-wide monitoring state."""
    return _state


def reset_state() -> MonitoringState:
    """Replace the process-wide state with a fresh instance (test helper)."""
    global _state
    _state = MonitoringState()
    return _state


__all__ = ["MonitoringState", "get_state", "reset_state"]
