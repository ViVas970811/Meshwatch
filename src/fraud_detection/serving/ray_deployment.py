"""Optional Ray Serve deployment of the fraud-prediction service.

Use cases:

* Horizontal scaling -- spin up multiple replicas of the predictor
  behind one HTTP endpoint.
* Resource isolation -- pin GPU vs CPU replicas separately.
* Production-grade autoscaling and rolling deploys.

Phase 4's plan calls for FastAPI + Ray Serve at <50ms P95. For local dev
on a 16GB Ryzen Ray adds more overhead than benefit, so we keep it
**opt-in** via a CLI flag. The plain FastAPI app in :mod:`app` is the
default.

Run with::

    python scripts/serve.py --ray --num-replicas 2

or programmatically::

    deployment = build_deployment()
    serve.run(deployment, name="fraud_detection")
"""

from __future__ import annotations

from typing import Any

from fraud_detection.serving.app import app as fastapi_app
from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


def _check_ray_available() -> tuple[Any, Any] | None:
    """Lazy-import ray + ray.serve, returning ``None`` if unavailable."""
    try:
        import ray
        from ray import serve

        return ray, serve
    except ImportError as exc:
        log.warning("ray_serve_unavailable", error=str(exc))
        return None


def build_deployment(
    *,
    num_replicas: int = 1,
    num_cpus_per_replica: float = 0.5,
    max_ongoing_requests: int = 64,
    health_check_period_s: float = 10.0,
):
    """Wrap the FastAPI app in a :func:`@serve.deployment`.

    Returns the configured deployment binding (call ``serve.run(binding)``
    to launch).

    Raises
    ------
    RuntimeError
        If ``ray`` / ``ray.serve`` aren't installed.
    """
    bundle = _check_ray_available()
    if bundle is None:
        msg = "Ray Serve is not installed; install with: pip install -e '.[serve]'"
        raise RuntimeError(msg)
    _, serve = bundle

    @serve.deployment(
        num_replicas=num_replicas,
        ray_actor_options={"num_cpus": num_cpus_per_replica},
        max_ongoing_requests=max_ongoing_requests,
        health_check_period_s=health_check_period_s,
    )
    @serve.ingress(fastapi_app)
    class FraudDetectionDeployment:
        """Ray Serve replica wrapping the FastAPI app.

        Each replica owns its own :class:`FraudPredictor`, Redis client,
        and Kafka producer (set up by the FastAPI lifespan when the
        wrapped app boots).
        """

        def __init__(self) -> None:
            log.info("ray_serve_replica_started")

    return FraudDetectionDeployment.bind()


def run_deployment(
    *,
    num_replicas: int = 1,
    host: str = "0.0.0.0",
    port: int = 8000,
    blocking: bool = True,
) -> None:
    """One-shot helper for ``scripts/serve.py --ray``.

    Starts a local Ray cluster (or attaches to an existing one), launches
    the deployment, and blocks until interrupted.
    """
    bundle = _check_ray_available()
    if bundle is None:
        msg = "Ray Serve is not installed; install with: pip install -e '.[serve]'"
        raise RuntimeError(msg)
    ray, serve = bundle

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)
        log.info("ray_initialized")

    serve.start(
        http_options={"host": host, "port": port},
        detached=False,
    )
    deployment = build_deployment(num_replicas=num_replicas)
    handle = serve.run(deployment, name="fraud_detection")
    log.info("ray_serve_running", host=host, port=port, num_replicas=num_replicas)

    if not blocking:
        return handle

    import time

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        log.info("ray_serve_shutdown_signal")
    finally:
        serve.shutdown()
        ray.shutdown()
        log.info("ray_serve_stopped")


__all__ = ["build_deployment", "run_deployment"]
