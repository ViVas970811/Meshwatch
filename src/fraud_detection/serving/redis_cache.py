"""Embedding cache with a graceful in-memory fallback.

Phase 4's hot path (~50ms budget) needs sub-millisecond reads of the GNN
embedding for the requested card. In production we hit Redis; in tests
and on local CPU runs Redis may not be available, so we transparently
fall back to a thread-safe in-memory dict.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import numpy as np

from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


class EmbeddingCache:
    """Stores per-card 64-d GNN embeddings with TTL support.

    ``connect()`` tries Redis first, falls back to a local dict. Callers
    don't need to know which backend is active -- ``get`` and ``set``
    have the same semantics either way.
    """

    DEFAULT_TTL_SECONDS = 3600  # 1 hour
    KEY_PREFIX = "meshwatch:emb:"

    def __init__(
        self,
        url: str | None = None,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        embedding_dim: int = 64,
    ) -> None:
        self.url = url
        self.ttl_seconds = ttl_seconds
        self.embedding_dim = embedding_dim
        self._redis: Any | None = None
        self._lock = threading.Lock()
        self._mem: dict[str, tuple[float, np.ndarray]] = {}
        self._connected = False

    # ------------------------------------------------------------------
    # connect / status
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Try Redis; on failure fall back to in-memory store. Always
        returns the *operational* status (whether the cache is usable),
        not whether Redis itself was reached."""
        if not self.url:
            log.info("embedding_cache_in_memory_no_url")
            self._connected = True
            return True
        try:
            import redis

            client = redis.Redis.from_url(self.url, socket_timeout=2.0)
            client.ping()
            self._redis = client
            self._connected = True
            log.info("redis_connected", url=self.url)
        except Exception as exc:
            log.warning("redis_unavailable_using_in_memory", error=str(exc))
            self._redis = None
            self._connected = True
        return self._connected

    def is_redis(self) -> bool:
        return self._redis is not None

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------

    def _key(self, card_id: int | str) -> str:
        return f"{self.KEY_PREFIX}{card_id}"

    def get(self, card_id: int | str) -> np.ndarray | None:
        """Return the cached embedding, or ``None`` if missing/expired."""
        key = self._key(card_id)
        if self._redis is not None:
            try:
                blob = self._redis.get(key)
            except Exception as exc:
                log.warning("redis_get_failed", error=str(exc))
                return None
            if blob is None:
                return None
            return np.frombuffer(blob, dtype=np.float32)

        # In-memory path with manual TTL.
        with self._lock:
            entry = self._mem.get(key)
            if entry is None:
                return None
            expires_at, vec = entry
            if expires_at < time.time():
                self._mem.pop(key, None)
                return None
            return vec

    def set(
        self,
        card_id: int | str,
        embedding: np.ndarray,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store ``embedding`` under ``card_id`` with the configured TTL."""
        if embedding.ndim != 1:
            embedding = embedding.reshape(-1)
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        if embedding.shape[0] != self.embedding_dim:
            log.warning(
                "embedding_dim_mismatch",
                expected=self.embedding_dim,
                got=int(embedding.shape[0]),
            )
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        key = self._key(card_id)

        if self._redis is not None:
            try:
                self._redis.setex(key, ttl, embedding.tobytes())
            except Exception as exc:
                log.warning("redis_set_failed", error=str(exc))
            return

        with self._lock:
            self._mem[key] = (time.time() + ttl, embedding)

    def delete(self, card_id: int | str) -> None:
        key = self._key(card_id)
        if self._redis is not None:
            try:
                self._redis.delete(key)
            except Exception as exc:
                log.warning("redis_delete_failed", error=str(exc))
            return
        with self._lock:
            self._mem.pop(key, None)

    def clear(self) -> None:
        if self._redis is not None:
            try:
                # Only clear our own keys, never the whole DB.
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor=cursor, match=f"{self.KEY_PREFIX}*")
                    if keys:
                        self._redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as exc:
                log.warning("redis_clear_failed", error=str(exc))
            return
        with self._lock:
            self._mem.clear()

    def size(self) -> int:
        if self._redis is not None:
            try:
                return sum(1 for _ in self._redis.scan_iter(match=f"{self.KEY_PREFIX}*"))
            except Exception:
                return 0
        with self._lock:
            return len(self._mem)

    # ------------------------------------------------------------------
    # bulk warm-up (pre-load embeddings extracted from the GNN)
    # ------------------------------------------------------------------

    def warm_up(
        self,
        card_ids: list[int | str],
        embeddings: np.ndarray,
        *,
        ttl_seconds: int | None = None,
    ) -> int:
        """Bulk-load cached embeddings (e.g. at server startup).

        Returns the number of cache entries written.
        """
        if embeddings.ndim != 2 or embeddings.shape[0] != len(card_ids):
            msg = (
                f"warm_up: card_ids ({len(card_ids)}) must match "
                f"embeddings.shape[0] ({embeddings.shape[0]})"
            )
            raise ValueError(msg)
        n_written = 0
        for cid, vec in zip(card_ids, embeddings, strict=True):
            self.set(cid, vec, ttl_seconds=ttl_seconds)
            n_written += 1
        log.info("embedding_cache_warmup_done", n=n_written)
        return n_written

    # ------------------------------------------------------------------
    # serialisation helpers (used by tests + diagnostics)
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "redis" if self.is_redis() else "in_memory",
            "size": self.size(),
            "ttl_seconds": self.ttl_seconds,
            "embedding_dim": self.embedding_dim,
            "connected": self._connected,
        }

    def __repr__(self) -> str:
        return f"EmbeddingCache({json.dumps(self.stats())})"


__all__ = ["EmbeddingCache"]
