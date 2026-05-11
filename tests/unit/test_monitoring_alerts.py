"""Unit tests for the application-side alert evaluator (Phase 7)."""

from __future__ import annotations

import pytest

from fraud_detection.monitoring.alerts import (
    Alert,
    AlertEvaluator,
    AlertRule,
    default_rules,
)
from fraud_detection.monitoring.performance import PerformanceSnapshot


class TestAlertRule:
    def test_rule_predicate_is_called(self):
        rule = AlertRule(
            name="t",
            severity="info",
            description="",
            predicate=lambda ctx: ctx["x"] > 1,
            reason=lambda ctx: f"x={ctx['x']}",
        )
        evaluator = AlertEvaluator([rule])
        active = evaluator.evaluate({"x": 2})
        assert len(active) == 1
        assert isinstance(active[0], Alert)
        assert active[0].name == "t"
        assert active[0].reason == "x=2"

    def test_rule_does_not_fire_when_predicate_false(self):
        rule = AlertRule(
            name="t",
            severity="info",
            description="",
            predicate=lambda ctx: ctx["x"] > 10,
        )
        evaluator = AlertEvaluator([rule])
        assert evaluator.evaluate({"x": 1}) == []

    def test_predicate_exception_yields_warning_alert(self):
        def _bad_predicate(_ctx: dict) -> bool:
            raise RuntimeError("boom")

        rule = AlertRule(
            name="t",
            severity="critical",
            description="",
            predicate=_bad_predicate,
        )
        evaluator = AlertEvaluator([rule])
        active = evaluator.evaluate({})
        assert len(active) == 1
        assert active[0].severity == "warning"
        assert "Rule evaluation failed" in active[0].reason


class TestDefaultRules:
    def test_default_rules_fire_in_isolation(self):
        rules = default_rules(
            psi_alert=0.25,
            latency_p95_warn=0.05,  # 50ms plan budget
            error_rate_warn=0.05,
            performance_recall_warn=0.55,
            psi_severe_features=10,
            performance_auprc_warn=0.60,
            fraud_rate_spike=0.10,
        )
        evaluator = AlertEvaluator(rules)

        # 1) Model not loaded
        fired = {a.name for a in evaluator.evaluate({"model_loaded": False})}
        assert "ModelNotLoaded" in fired

        # 2) Drift severe (overall PSI >= threshold)
        fired = {
            a.name
            for a in evaluator.evaluate(
                {"model_loaded": True, "drift_overall": 0.30, "drift_severe": 3}
            )
        }
        assert "DataDriftSevere" in fired

        # 3) Many features drifting (plan threshold = >10)
        fired = {a.name for a in evaluator.evaluate({"model_loaded": True, "drift_severe": 15})}
        assert "DataDrift" in fired

        # 4) Performance degraded (recall)
        snap = PerformanceSnapshot(n_labelled=100, recall=0.40, precision=0.9, f1=0.5)
        fired = {a.name for a in evaluator.evaluate({"model_loaded": True, "performance": snap})}
        assert "ModelPerformanceDegraded" in fired

        # 5) Model degraded (AUPRC < 0.60 -- plan threshold)
        snap = PerformanceSnapshot(
            n_labelled=100, recall=0.9, precision=0.9, f1=0.9, auprc=0.45, auroc=0.88
        )
        fired = {a.name for a in evaluator.evaluate({"model_loaded": True, "performance": snap})}
        assert "ModelDegradation" in fired

        # 6) Fraud rate spike (plan threshold = 10%)
        fired = {
            a.name
            for a in evaluator.evaluate(
                {
                    "model_loaded": True,
                    "production_fraud_rate": 0.18,
                    "reference_fraud_rate": 0.035,
                }
            )
        }
        assert "FraudRateSpike" in fired

        # 7) Latency (50ms budget)
        fired = {a.name for a in evaluator.evaluate({"model_loaded": True, "latency_p95": 0.1})}
        assert "HighLatency" in fired

        # 8) Error rate
        fired = {a.name for a in evaluator.evaluate({"model_loaded": True, "error_rate": 0.1})}
        assert "HighErrorRate" in fired

        # 9) Shadow agreement
        shadow_summary = {"n_total": 100, "agreement_rate": 0.75}
        fired = {
            a.name for a in evaluator.evaluate({"model_loaded": True, "shadow": shadow_summary})
        }
        assert "ShadowChallengerDisagreement" in fired

    def test_drift_severe_includes_overall_psi_in_reason(self):
        rules = default_rules()
        evaluator = AlertEvaluator(rules)
        active = evaluator.evaluate(
            {"model_loaded": True, "drift_overall": 0.33, "drift_severe": 2}
        )
        rule = next(a for a in active if a.name == "DataDriftSevere")
        assert "0.330" in rule.reason
        assert "2" in rule.reason

    def test_no_rules_fire_in_a_healthy_context(self):
        rules = default_rules()
        evaluator = AlertEvaluator(rules)
        snap = PerformanceSnapshot(
            n_labelled=100,
            recall=0.95,
            precision=0.95,
            f1=0.95,
            auprc=0.85,
            auroc=0.95,
        )
        ctx = {
            "model_loaded": True,
            "drift_overall": 0.05,
            "drift_severe": 0,
            "performance": snap,
            "production_fraud_rate": 0.03,
            "reference_fraud_rate": 0.035,
            "shadow": {"n_total": 100, "agreement_rate": 0.99},
            "latency_p95": 0.02,
            "error_rate": 0.001,
        }
        assert evaluator.evaluate(ctx) == []

    def test_fraud_rate_spike_reason_includes_baseline(self):
        rules = default_rules(fraud_rate_spike=0.10)
        evaluator = AlertEvaluator(rules)
        active = evaluator.evaluate(
            {
                "model_loaded": True,
                "production_fraud_rate": 0.22,
                "reference_fraud_rate": 0.035,
            }
        )
        rule = next(a for a in active if a.name == "FraudRateSpike")
        assert "22.00%" in rule.reason
        assert "3.50%" in rule.reason

    def test_alert_to_dict_serialises_datetime(self):
        alert = Alert(name="t", severity="info", description="", reason="r")
        d = alert.to_dict()
        assert isinstance(d["fired_at"], str)
        # ISO format check
        assert "T" in d["fired_at"]


class TestRecallRuleFallbacks:
    def test_recall_rule_skips_unlabelled_snapshot(self):
        rules = default_rules(performance_recall_warn=0.99)  # very strict
        evaluator = AlertEvaluator(rules)
        # Zero labels means recall is unknown -- the rule should not fire.
        snap = PerformanceSnapshot(n_labelled=0, recall=0.0)
        active = {a.name for a in evaluator.evaluate({"model_loaded": True, "performance": snap})}
        assert "ModelPerformanceDegraded" not in active

    def test_recall_rule_handles_dict_snapshot(self):
        # The evaluator should accept either the dataclass or a dict so that
        # serialised payloads from the API can flow through unchanged.
        rules = default_rules(performance_recall_warn=0.6)
        evaluator = AlertEvaluator(rules)
        ctx = {"model_loaded": True, "performance": {"n_labelled": 50, "recall": 0.3}}
        fired = {a.name for a in evaluator.evaluate(ctx)}
        assert "ModelPerformanceDegraded" in fired

    @pytest.mark.parametrize("missing_key", ["drift_overall", "latency_p95", "error_rate"])
    def test_default_rules_handle_missing_context_keys(self, missing_key):
        rules = default_rules()
        evaluator = AlertEvaluator(rules)
        ctx = {"model_loaded": True}
        # Should not raise even though most context keys are missing.
        assert evaluator.evaluate(ctx) == []
