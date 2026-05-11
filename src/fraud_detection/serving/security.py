"""Security hardening middleware (Phase 8).

Two pieces, both pure Starlette so they slot into the FastAPI app with no
external dependencies:

* :class:`ApiKeyAuthMiddleware` -- API-key authentication. Accepts a key
  via the ``X-API-Key`` header (or ``Authorization: Bearer <key>``). When
  no keys are configured the middleware is a no-op so local dev keeps
  working without an env var dance; in production set
  ``FRAUD_API_KEYS="key1,key2"`` to lock the API down.
* :class:`RateLimitMiddleware` -- per-client token bucket rate limiter.
  Default budget is 100 requests/second per identity, matching the
  plan's spec. Identity = API key if present, else remote IP. A small
  set of endpoints (``/api/v1/health``, ``/api/v1/metrics``, ``/docs``,
  ``/openapi.json``) are exempt so monitoring scrapers don't get
  throttled.

Both middlewares are deliberately self-contained -- no Redis dependency
on the hot path. For a multi-replica deployment swap the in-memory bucket
for a Redis-backed one (Phase 4 already ships a Redis client we could
reuse); for the project's local-dev + small-scale production use, the
in-memory implementation is sufficient and adds zero latency.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterable

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# API key auth
# ---------------------------------------------------------------------------


def _load_api_keys_from_env(var: str = "FRAUD_API_KEYS") -> set[str]:
    raw = os.environ.get(var, "")
    return {k.strip() for k in raw.split(",") if k.strip()}


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid API key when keys are configured.

    Parameters
    ----------
    api_keys
        Iterable of accepted API keys. When empty or ``None`` the
        middleware is a no-op (warns once at startup).
    header_name
        Header to read. Defaults to ``X-API-Key``.
    exempt_paths
        Paths that bypass auth entirely -- health, metrics, OpenAPI,
        docs, dashboard's WebSocket. Match is by ``startswith`` so a
        single entry like ``/api/v1/metrics`` covers exposed scrape
        endpoints; ``/api/v1/health`` covers liveness probes.
    """

    DEFAULT_EXEMPT: tuple[str, ...] = (
        "/",
        "/api/v1/health",
        "/api/v1/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    )

    def __init__(
        self,
        app,
        *,
        api_keys: Iterable[str] | None = None,
        header_name: str = "X-API-Key",
        exempt_paths: Iterable[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._keys: set[str] = set(api_keys or [])
        self._header_name = header_name.lower()
        self._exempt: tuple[str, ...] = tuple(exempt_paths or self.DEFAULT_EXEMPT)
        if not self._keys:
            log.warning("api_key_auth_disabled_no_keys_configured")

    @property
    def enabled(self) -> bool:
        return bool(self._keys)

    def _is_exempt(self, path: str) -> bool:
        return any(path == p or path.startswith(p + "/") for p in self._exempt)

    @staticmethod
    def _extract_key(request: Request, header_name: str) -> str | None:
        # Primary header (case-insensitive).
        for name, value in request.headers.items():
            if name.lower() == header_name:
                return value
        # Fallback: Authorization: Bearer <key>.
        auth = request.headers.get("authorization")
        if auth and auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return None

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self._keys or self._is_exempt(request.url.path):
            return await call_next(request)
        key = self._extract_key(request, self._header_name)
        if key is None:
            return JSONResponse({"detail": "Missing API key"}, status_code=401)
        if key not in self._keys:
            log.info("api_key_invalid", path=request.url.path)
            return JSONResponse({"detail": "Invalid API key"}, status_code=403)
        # Annotate the request scope so the rate limiter (which runs after
        # this middleware) can key on the API key instead of just the IP.
        request.scope["api_key"] = key
        return await call_next(request)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Lock-protected token bucket.

    A bucket holds at most ``capacity`` tokens; each successful request
    spends one token. Tokens regenerate at ``rate`` per second.
    """

    __slots__ = ("capacity", "lock", "rate", "tokens", "updated")

    def __init__(self, capacity: int, rate: float) -> None:
        self.capacity = float(capacity)
        self.rate = float(rate)
        self.tokens = float(capacity)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def take(self, now: float | None = None) -> tuple[bool, float]:
        """Spend a token. Returns ``(allowed, retry_after_seconds)``."""
        with self.lock:
            current = now if now is not None else time.monotonic()
            elapsed = max(0.0, current - self.updated)
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.updated = current
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, 0.0
            retry = max(0.0, (1.0 - self.tokens) / self.rate) if self.rate else 1.0
            return False, retry


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-identity token-bucket rate limiter.

    Identity = API key when present in the request scope (set by
    :class:`ApiKeyAuthMiddleware`), else the client IP from
    ``request.client.host``.

    Parameters
    ----------
    rate
        Steady-state requests / second. Default ``100`` matches the plan.
    burst
        Bucket capacity. Defaults to ``rate`` (1-second burst).
    exempt_paths
        Paths bypassed entirely.
    max_buckets
        Cap on the in-memory bucket cache so a flood of unique IPs
        cannot exhaust memory. Oldest buckets are evicted on overflow.
    """

    DEFAULT_EXEMPT: tuple[str, ...] = (
        "/api/v1/health",
        "/api/v1/metrics",
    )

    def __init__(
        self,
        app,
        *,
        rate: float = 100.0,
        burst: int | None = None,
        exempt_paths: Iterable[str] | None = None,
        max_buckets: int = 1024,
    ) -> None:
        super().__init__(app)
        self._rate = float(rate)
        self._capacity = int(burst if burst is not None else max(1, round(rate)))
        self._exempt: tuple[str, ...] = tuple(exempt_paths or self.DEFAULT_EXEMPT)
        self._buckets: dict[str, _TokenBucket] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._max_buckets = int(max_buckets)

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def capacity(self) -> int:
        return self._capacity

    def _is_exempt(self, path: str) -> bool:
        return any(path == p or path.startswith(p + "/") for p in self._exempt)

    def _identity(self, request: Request) -> str:
        key = request.scope.get("api_key")
        if key:
            return f"key:{key}"
        host = request.client.host if request.client else "unknown"
        return f"ip:{host}"

    def _bucket_for(self, identity: str) -> _TokenBucket:
        with self._lock:
            bucket = self._buckets.get(identity)
            if bucket is None:
                bucket = _TokenBucket(self._capacity, self._rate)
                self._buckets[identity] = bucket
                self._order.append(identity)
                while len(self._order) > self._max_buckets:
                    oldest = self._order.pop(0)
                    self._buckets.pop(oldest, None)
            return bucket

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if self._is_exempt(request.url.path):
            return await call_next(request)
        identity = self._identity(request)
        bucket = self._bucket_for(identity)
        allowed, retry_after = bucket.take()
        if not allowed:
            log.info(
                "rate_limit_exceeded",
                identity=identity,
                path=request.url.path,
                retry_after=retry_after,
            )
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": f"{retry_after:.2f}"},
            )
        return await call_next(request)


# ---------------------------------------------------------------------------
# Helper -- assemble both middlewares from environment variables
# ---------------------------------------------------------------------------


def configure_security(app, *, env_prefix: str = "FRAUD") -> None:
    """Install :class:`ApiKeyAuthMiddleware` + :class:`RateLimitMiddleware`.

    Reads from environment variables (all optional):

    * ``{prefix}_API_KEYS``           comma-separated list of accepted keys
    * ``{prefix}_RATE_LIMIT_RPS``     steady-state req/s (default 100)
    * ``{prefix}_RATE_LIMIT_BURST``   bucket capacity (default = rps)
    * ``{prefix}_RATE_LIMIT_DISABLED`` set to ``true`` to skip the limiter
    """
    api_keys = _load_api_keys_from_env(f"{env_prefix}_API_KEYS")
    rate = float(os.environ.get(f"{env_prefix}_RATE_LIMIT_RPS", "100"))
    burst_raw = os.environ.get(f"{env_prefix}_RATE_LIMIT_BURST")
    burst = int(burst_raw) if burst_raw else None
    disabled = os.environ.get(f"{env_prefix}_RATE_LIMIT_DISABLED", "false").lower() == "true"

    if not disabled:
        app.add_middleware(RateLimitMiddleware, rate=rate, burst=burst)
    app.add_middleware(ApiKeyAuthMiddleware, api_keys=api_keys)


__all__ = [
    "ApiKeyAuthMiddleware",
    "RateLimitMiddleware",
    "configure_security",
]
