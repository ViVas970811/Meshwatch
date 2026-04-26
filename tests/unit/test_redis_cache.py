"""Tests for ``fraud_detection.serving.redis_cache.EmbeddingCache``."""

from __future__ import annotations

import time

import numpy as np
import pytest

from fraud_detection.serving.redis_cache import EmbeddingCache


@pytest.fixture
def cache() -> EmbeddingCache:
    c = EmbeddingCache(url=None, ttl_seconds=60, embedding_dim=8)
    c.connect()
    return c


def test_in_memory_fallback_connects_without_url():
    c = EmbeddingCache(url=None)
    assert c.connect() is True
    assert c.connected
    assert not c.is_redis()


def test_set_and_get_roundtrip(cache):
    vec = np.arange(8, dtype=np.float32)
    cache.set("card-1", vec)
    out = cache.get("card-1")
    assert out is not None
    np.testing.assert_array_equal(out, vec)


def test_get_missing_returns_none(cache):
    assert cache.get("missing") is None


def test_ttl_expiry(cache):
    vec = np.zeros(8, dtype=np.float32)
    cache.set("ephemeral", vec, ttl_seconds=0)
    # Brief sleep so the in-memory TTL definitely expires.
    time.sleep(0.01)
    assert cache.get("ephemeral") is None


def test_size_and_clear(cache):
    for i in range(5):
        cache.set(f"k{i}", np.zeros(8, dtype=np.float32))
    assert cache.size() == 5
    cache.clear()
    assert cache.size() == 0


def test_delete(cache):
    cache.set("kkk", np.ones(8, dtype=np.float32))
    cache.delete("kkk")
    assert cache.get("kkk") is None


def test_dim_mismatch_logs_but_stores(cache):
    """A wrong-dim embedding should log a warning but not crash."""
    cache.set("weird", np.zeros(16, dtype=np.float32))
    out = cache.get("weird")
    assert out is not None
    assert out.shape[0] == 16


def test_warm_up_bulk_load(cache):
    ids = ["a", "b", "c"]
    embs = np.arange(24, dtype=np.float32).reshape(3, 8)
    n = cache.warm_up(ids, embs)
    assert n == 3
    assert cache.size() == 3
    np.testing.assert_array_equal(cache.get("b"), embs[1])


def test_warm_up_shape_mismatch_raises(cache):
    with pytest.raises(ValueError):
        cache.warm_up(["a", "b"], np.zeros((3, 8), dtype=np.float32))


def test_stats_in_memory(cache):
    cache.set("k", np.zeros(8, dtype=np.float32))
    stats = cache.stats()
    assert stats["backend"] == "in_memory"
    assert stats["size"] == 1
    assert stats["embedding_dim"] == 8
    assert stats["connected"] is True


def test_in_memory_dtype_normalisation(cache):
    """Setting a float64 array should still produce float32 on read."""
    cache.set("dtype-test", np.ones(8, dtype=np.float64))
    out = cache.get("dtype-test")
    assert out.dtype == np.float32


def test_redis_failure_falls_back(monkeypatch):
    """When the URL points at a dead Redis, ``connect`` still succeeds in-memory."""
    cache = EmbeddingCache(url="redis://127.0.0.1:1/0", embedding_dim=8)
    assert cache.connect() is True
    assert cache.connected
    assert not cache.is_redis()  # fell back to in-memory
    cache.set("k", np.zeros(8, dtype=np.float32))
    assert cache.get("k") is not None
