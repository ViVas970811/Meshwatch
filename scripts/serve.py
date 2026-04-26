#!/usr/bin/env python
"""Launch the Meshwatch fraud-detection API.

Plain FastAPI / uvicorn (default)::

    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8000 --reload

Ray Serve (multi-replica, opt-in)::

    python scripts/serve.py --ray --num-replicas 2

Common env vars (all optional -- the app falls back gracefully):
    FRAUD_ENSEMBLE_DIR        path to data/models/ensemble
    REDIS_URL                 e.g. redis://localhost:6379/0
    KAFKA_BOOTSTRAP_SERVERS   e.g. localhost:9092
    KAFKA_FRAUD_TOPIC         default "fraud_alerts"
    FRAUD_ALERT_THRESHOLD     default 0.7
    FRAUD_ENABLE_SHAP         "true" / "false" (default true)
    FRAUD_LOG_LEVEL           default "INFO"
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import click  # noqa: E402

from fraud_detection.utils.config import load_config  # noqa: E402
from fraud_detection.utils.logging import configure_logging, get_logger  # noqa: E402


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev only)")
@click.option(
    "--workers",
    default=1,
    show_default=True,
    type=int,
    help="Uvicorn worker count (ignored with --ray or --reload)",
)
@click.option("--ray", is_flag=True, help="Run via Ray Serve instead of plain uvicorn")
@click.option("--num-replicas", default=1, show_default=True, type=int)
@click.option("--log-level", default="info", show_default=True)
def main(
    host: str,
    port: int,
    reload: bool,
    workers: int,
    ray: bool,
    num_replicas: int,
    log_level: str,
) -> None:
    cfg = load_config()
    configure_logging(level=cfg.logging.level, json=cfg.logging.use_json)
    log = get_logger("serve")

    if ray:
        log.info(
            "starting_ray_serve",
            host=host,
            port=port,
            num_replicas=num_replicas,
        )
        from fraud_detection.serving.ray_deployment import run_deployment

        run_deployment(num_replicas=num_replicas, host=host, port=port, blocking=True)
        return

    log.info("starting_uvicorn", host=host, port=port, workers=workers, reload=reload)
    import uvicorn

    uvicorn.run(
        "fraud_detection.serving.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level=log_level,
        access_log=False,  # we have our own structured access log
    )


if __name__ == "__main__":
    main()
