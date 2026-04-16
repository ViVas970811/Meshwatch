"""Structured logging for the fraud-detection project.

Wraps ``structlog`` on top of the stdlib logging module so that third-party
libraries (pandas, sklearn, kaggle, torch) funnel into the same JSON stream
as our own code.

Usage
-----

>>> from fraud_detection.utils.logging import configure_logging, get_logger
>>> configure_logging(level="INFO", json=True)
>>> log = get_logger(__name__)
>>> log.info("preprocessing_complete", rows=590540, drop_cols=23)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.typing import EventDict, Processor

# Reused across calls so repeated ``configure_logging`` calls are idempotent.
_CONFIGURED = False


def _add_service_name(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
    """Attach the service name on every record for easy log filtering."""
    event_dict.setdefault("service", "fraud-detection-gnn")
    return event_dict


def configure_logging(
    level: str = "INFO",
    *,
    json: bool = True,
    static_context: dict[str, Any] | None = None,
) -> None:
    """Configure structlog + stdlib logging.

    Parameters
    ----------
    level
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    json
        If ``True`` emit one JSON object per line (good for docker/k8s).
        If ``False`` pretty-print with color (good for local dev).
    static_context
        Extra key/value pairs to attach to every record.
    """
    global _CONFIGURED

    log_level = getattr(logging, level.upper(), logging.INFO)

    # --- stdlib root logger: a single stderr StreamHandler ---------------------
    root = logging.getLogger()
    # Remove handlers we may have previously installed so re-configuration works
    # cleanly in tests / notebook reloads.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(log_level)
    root.addHandler(handler)
    root.setLevel(log_level)

    # --- structlog processor chain ---------------------------------------------
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _add_service_name,
    ]

    if static_context:
        ctx_copy = dict(static_context)

        def _inject_static(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
            for k, v in ctx_copy.items():
                event_dict.setdefault(k, v)
            return event_dict

        shared_processors.append(_inject_static)

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    # Lazy factory: grab sys.stderr fresh on each logger instantiation so
    # pytest's stderr capture swap between tests doesn't leave us holding a
    # closed file descriptor.
    def _stderr_factory(*_args: Any, **_kwargs: Any) -> structlog.PrintLogger:
        return structlog.PrintLogger(file=sys.stderr)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=_stderr_factory,
        cache_logger_on_first_use=False,
    )

    # Tame a few notoriously noisy stdlib loggers during data pipeline runs
    for noisy in ("urllib3", "botocore", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(max(log_level, logging.WARNING))

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    If :func:`configure_logging` has not been called yet, apply sane defaults
    first so that importing and using the logger immediately from a script
    still produces output.
    """
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)


__all__ = ["configure_logging", "get_logger"]
