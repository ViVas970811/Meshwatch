"""Unit tests for the shadow deployment (Phase 7)."""

from __future__ import annotations

import time

import pytest

from fraud_detection.monitoring.shadow import (
    ShadowDecision,
    ShadowDeployment,
    ShadowSummary,
)
from fraud_detection.serving.schemas import FraudPrediction, TransactionRequest, risk_level


class _FakePredictor:
    """A deterministic predictor whose score depends only on amount.

    Used by both champion and challenger -- by tweaking the slope we can
    deterministically produce agreement / disagreement.
    """

    def __init__(self, *, slope: float, model_version: str, latency_ms: float = 0.0):
        self.slope = float(slope)
        self.model_version = model_version
        self.latency_ms = float(latency_ms)
        self.threshold = 0.7

    def predict_one(self, req: TransactionRequest) -> FraudPrediction:
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
        score = min(0.99, max(0.0, req.transaction_amt * self.slope))
        return FraudPrediction(
            transaction_id=req.transaction_id,
            fraud_probability=score,
            fraud_score=score,
            risk_level=risk_level(score),
            is_fraud_predicted=score >= self.threshold,
            threshold=self.threshold,
            top_features=[],
            latency_ms={"total_ms": self.latency_ms},
            model_version=self.model_version,
        )


def _req(tx_id: str, amt: float) -> TransactionRequest:
    return TransactionRequest(
        transaction_id=tx_id,
        transaction_dt=1_000_000,
        transaction_amt=amt,
        product_cd="W",
        card1=12345,
    )


@pytest.fixture
def deployment():
    champion = _FakePredictor(slope=0.005, model_version="champion-v1")
    challenger = _FakePredictor(slope=0.005, model_version="challenger-v2")
    d = ShadowDeployment(champion=champion, challenger=challenger, max_records=20)
    yield d
    d.shutdown(wait=True)


class TestScoring:
    def test_score_returns_champion_result(self, deployment):
        result = deployment.score(_req("tx_1", 100))
        # Champion slope 0.005 * 100 = 0.5
        assert result.model_version == "champion-v1"
        assert pytest.approx(result.fraud_score, abs=1e-6) == 0.5

    def test_score_logs_challenger_decisions(self, deployment):
        for i in range(5):
            deployment.score(_req(f"tx_{i}", 100 + i))
        # Wait briefly for the challenger to drain.
        for _ in range(50):
            if deployment.n_records >= 5:
                break
            time.sleep(0.02)
        assert deployment.n_records == 5

    def test_disabled_challenger_is_a_noop(self):
        champion = _FakePredictor(slope=0.005, model_version="champion-v1")
        d = ShadowDeployment(champion=champion, challenger=None)
        try:
            result = d.score(_req("tx_1", 100))
            assert result.model_version == "champion-v1"
            assert d.n_records == 0
            assert d.enabled is False
        finally:
            d.shutdown(wait=True)


class TestSummary:
    def test_empty_summary_reports_zero_total(self, deployment):
        summary = deployment.summary()
        assert isinstance(summary, ShadowSummary)
        assert summary.n_total == 0
        assert summary.champion_model == "champion-v1"
        assert summary.challenger_model == "challenger-v2"

    def test_full_agreement_when_slopes_match(self, deployment):
        for i in range(10):
            deployment.score(_req(f"tx_{i}", 200 + i * 50))
        # Drain background work.
        for _ in range(100):
            if deployment.n_records >= 10:
                break
            time.sleep(0.02)
        summary = deployment.summary()
        assert summary.n_total == 10
        assert summary.agreement_rate == pytest.approx(1.0, abs=1e-9)
        assert summary.mean_score_delta == pytest.approx(0.0, abs=1e-9)

    def test_disagreement_when_slopes_diverge(self):
        champion = _FakePredictor(slope=0.001, model_version="champion-v1")
        challenger = _FakePredictor(slope=0.01, model_version="challenger-v2")
        d = ShadowDeployment(champion=champion, challenger=challenger)
        try:
            for i in range(20):
                d.score(_req(f"tx_{i}", 150 + i))  # straddles the 0.7 threshold
            for _ in range(100):
                if d.n_records >= 20:
                    break
                time.sleep(0.02)
            summary = d.summary()
            assert summary.n_disagreement > 0
            assert summary.agreement_rate < 1.0
            assert summary.max_score_delta > 0
        finally:
            d.shutdown(wait=True)


class TestChallengerSafety:
    def test_challenger_over_budget_is_skipped(self):
        champion = _FakePredictor(slope=0.005, model_version="champion-v1")
        slow = _FakePredictor(slope=0.005, model_version="slow-v2", latency_ms=150.0)
        d = ShadowDeployment(
            champion=champion, challenger=slow, challenger_budget_ms=50.0, max_records=10
        )
        try:
            d.score(_req("tx_slow", 100))
            # Wait long enough for the slow challenger to finish (150ms)
            # plus our skip-logging path.
            time.sleep(0.3)
            summary = d.summary()
            # The challenger finished but its latency exceeded the budget,
            # so no record was kept.
            assert summary.n_total == 0
            assert summary.challenger_skipped >= 1
        finally:
            d.shutdown(wait=True)

    def test_failing_challenger_is_counted(self):
        class _Broken:
            model_version = "broken-v2"
            threshold = 0.7

            def predict_one(self, _req):
                raise RuntimeError("intentional failure")

        champion = _FakePredictor(slope=0.005, model_version="champion-v1")
        d = ShadowDeployment(champion=champion, challenger=_Broken())  # type: ignore[arg-type]
        try:
            d.score(_req("tx_1", 100))
            time.sleep(0.1)
            summary = d.summary()
            assert summary.challenger_failed >= 1
            assert summary.n_total == 0
        finally:
            d.shutdown(wait=True)

    def test_attach_detach_challenger(self):
        champion = _FakePredictor(slope=0.005, model_version="champion-v1")
        d = ShadowDeployment(champion=champion, challenger=None)
        try:
            assert d.enabled is False
            challenger = _FakePredictor(slope=0.005, model_version="challenger-v2")
            d.attach_challenger(challenger)
            assert d.enabled is True
            d.detach_challenger()
            assert d.enabled is False
        finally:
            d.shutdown(wait=True)


class TestShadowDecisionRecord:
    def test_decision_to_dict_serialises_timestamp(self):
        decision = ShadowDecision(
            transaction_id="tx_1",
            champion_score=0.5,
            challenger_score=0.75,
            score_delta=0.25,
            champion_label=False,
            challenger_label=True,
            agreement=False,
            champion_model="champion-v1",
            challenger_model="challenger-v2",
            champion_latency_ms=1.0,
            challenger_latency_ms=2.0,
        )
        d = decision.to_dict()
        assert isinstance(d["timestamp"], str)
        assert d["agreement"] is False
