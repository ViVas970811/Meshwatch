"""Optional Langfuse tracing for the fraud-investigation agent (Phase 5).

Langfuse is only imported when ``FRAUD_LANGFUSE_ENABLED=true`` and the
package is installed. Otherwise we hand callers a no-op span so the
investigation code path doesn't have to care whether tracing is on.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator
from typing import Any

from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


class _NoopSpan:
    """A drop-in replacement for ``langfuse`` spans when tracing is off."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.t0 = time.perf_counter()
        self.metadata: dict[str, Any] = {}

    def update(self, **kwargs: Any) -> None:
        self.metadata.update(kwargs)

    def end(self, **kwargs: Any) -> None:
        self.metadata.update(kwargs)
        elapsed_ms = (time.perf_counter() - self.t0) * 1000
        log.debug("agent_span", name=self.name, elapsed_ms=round(elapsed_ms, 2), **self.metadata)


class AgentTracer:
    """Minimal facade so node code can call ``with tracer.span(...)`` regardless."""

    def __init__(self) -> None:
        self._client: Any | None = None
        self._enabled = False
        self._try_connect()

    def _try_connect(self) -> None:
        if os.environ.get("FRAUD_LANGFUSE_ENABLED", "false").lower() != "true":
            return
        try:
            from langfuse import Langfuse  # type: ignore[import-untyped]

            self._client = Langfuse(
                public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
                host=os.environ.get("LANGFUSE_HOST"),
            )
            self._enabled = True
            log.info("langfuse_connected")
        except Exception as exc:
            log.warning("langfuse_unavailable", error=str(exc))
            self._client = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextlib.contextmanager
    def span(self, name: str, **metadata: Any) -> Iterator[Any]:
        """Context manager yielding a span object; no-op if tracing is off."""
        span: Any
        if self._client is not None:
            try:
                span = self._client.span(name=name, metadata=metadata or None)
            except Exception:
                span = _NoopSpan(name)
                span.update(**metadata)
        else:
            span = _NoopSpan(name)
            span.update(**metadata)
        try:
            yield span
        finally:
            with contextlib.suppress(Exception):
                span.end()


__all__ = ["AgentTracer"]
