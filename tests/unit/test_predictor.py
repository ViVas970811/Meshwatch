"""Tests for ``fraud_detection.serving.predictor.FraudPredictor``."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from fraud_detection.serving.predictor import FraudPredictor
from fraud_detection.serving.redis_cache import EmbeddingCache
from fraud_detection.serving.schemas import TransactionRequest


def _stub_xgb(score: float = 0.5) -> MagicMock:
    """Return a mock xgb that yields the given score for any input."""
    xgb = MagicMock()
    xgb.predict_proba.side_effect = lambda X: np.full(X.shape[0], score, dtype=np.float32)
    return xgb


def _stub_ensemble(score: float = 0.5):
    ens = MagicMock()
    ens.xgb = _stub_xgb(score)
    ens.gnn = MagicMock()
    ens.gnn.embedding_dim = 4
    ens.feature_columns = [f"gnn_emb_{i:03d}" for i in range(4)] + ["TransactionAmt", "addr1"]
    return ens


@pytest.fixture
def cache() -> EmbeddingCache:
    c = EmbeddingCache(url=None, embedding_dim=4)
    c.connect()
    return c


def test_predict_one_returns_expected_shape(cache):
    ens = _stub_ensemble(score=0.85)
    p = FraudPredictor(
        ensemble=ens,
        embedding_cache=cache,
        feature_columns=ens.feature_columns,
        enable_shap=False,
    )
    req = TransactionRequest(transaction_id=1, transaction_dt=10, transaction_amt=100.0, card1=42)
    out = p.predict_one(req)
    assert out.transaction_id == 1
    assert out.fraud_score == pytest.approx(0.85)
    assert out.is_fraud_predicted  # 0.85 > default 0.7
    assert out.risk_level == "HIGH"


def test_predict_one_below_threshold(cache):
    ens = _stub_ensemble(score=0.1)
    p = FraudPredictor(
        ensemble=ens,
        embedding_cache=cache,
        feature_columns=ens.feature_columns,
        enable_shap=False,
    )
    req = TransactionRequest(transaction_id=1, transaction_dt=10, transaction_amt=20.0)
    out = p.predict_one(req)
    assert not out.is_fraud_predicted
    assert out.risk_level == "LOW"


def test_predict_uses_cached_embedding(cache):
    """If the cache has an embedding for the card, it should be used (not zero)."""
    ens = _stub_ensemble(score=0.5)
    p = FraudPredictor(
        ensemble=ens,
        embedding_cache=cache,
        feature_columns=ens.feature_columns,
        enable_shap=False,
    )
    # Pre-warm cache with a known vector for card1=42.
    cache.set(42, np.arange(4, dtype=np.float32) * 10)
    req = TransactionRequest(transaction_id=1, transaction_dt=10, transaction_amt=50.0, card1=42)
    p.predict_one(req)
    # The mock xgb received the cache vector concatenated with tabular features.
    # Inspect the call args (last call):
    last_X = ens.xgb.predict_proba.call_args_list[-1][0][0]
    np.testing.assert_array_equal(last_X[0, :4], np.arange(4, dtype=np.float32) * 10)


def test_predict_zero_embedding_for_cold_card(cache):
    ens = _stub_ensemble(score=0.5)
    p = FraudPredictor(
        ensemble=ens,
        embedding_cache=cache,
        feature_columns=ens.feature_columns,
        enable_shap=False,
    )
    req = TransactionRequest(transaction_id=1, transaction_dt=10, transaction_amt=50.0, card1=99999)
    p.predict_one(req)
    last_X = ens.xgb.predict_proba.call_args_list[-1][0][0]
    np.testing.assert_array_equal(last_X[0, :4], np.zeros(4, dtype=np.float32))


def test_predict_batch_keeps_order(cache):
    ens = _stub_ensemble(score=0.5)
    # Make the score depend on input so we can verify ordering.
    ens.xgb.predict_proba.side_effect = lambda X: np.linspace(0.0, 1.0, X.shape[0]).astype(
        np.float32
    )
    p = FraudPredictor(
        ensemble=ens,
        embedding_cache=cache,
        feature_columns=ens.feature_columns,
        enable_shap=False,
    )
    reqs = [
        TransactionRequest(transaction_id=i, transaction_dt=i, transaction_amt=10.0 + i)
        for i in range(5)
    ]
    out = p.predict_batch(reqs)
    assert [r.transaction_id for r in out] == [0, 1, 2, 3, 4]
    # Scores should be linspace(0, 1, 5)
    assert out[0].fraud_score == pytest.approx(0.0)
    assert out[-1].fraud_score == pytest.approx(1.0, abs=1e-5)


def test_predict_latency_breakdown_included(cache):
    ens = _stub_ensemble(score=0.5)
    p = FraudPredictor(
        ensemble=ens,
        embedding_cache=cache,
        feature_columns=ens.feature_columns,
        enable_shap=False,
    )
    req = TransactionRequest(transaction_id=1, transaction_dt=10, transaction_amt=10.0)
    out = p.predict_one(req)
    for k in ("embedding_ms", "tabular_ms", "xgboost_ms", "total_ms"):
        assert k in out.latency_ms
        assert out.latency_ms[k] >= 0


def test_predict_info(cache):
    ens = _stub_ensemble(score=0.5)
    p = FraudPredictor(
        ensemble=ens,
        embedding_cache=cache,
        feature_columns=ens.feature_columns,
        enable_shap=False,
    )
    info = p.info()
    assert info["embedding_dim"] == 4
    assert "cache_stats" in info
    assert info["shap_enabled"] is False
