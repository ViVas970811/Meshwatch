"""Tests for ``fraud_detection.utils.logging``."""

from __future__ import annotations

import json
import logging

import pytest

from fraud_detection.utils import logging as fd_logging


def test_configure_logging_is_idempotent():
    fd_logging.configure_logging(level="INFO", json=True)
    fd_logging.configure_logging(level="DEBUG", json=False)
    # Should not raise on repeated calls.
    assert logging.getLogger().level == logging.DEBUG


def test_get_logger_returns_bound_logger():
    log = fd_logging.get_logger("test")
    # structlog bound loggers expose info/debug/warning/error.
    for method in ("info", "debug", "warning", "error", "exception"):
        assert callable(getattr(log, method))


def test_json_output_is_parseable(capsys: pytest.CaptureFixture[str]):
    fd_logging.configure_logging(level="INFO", json=True)
    log = fd_logging.get_logger("test")
    log.info("hello_world", foo="bar", n=42)

    captured = capsys.readouterr().err.strip().splitlines()
    assert captured, "expected at least one log line on stderr"
    last = captured[-1]
    data = json.loads(last)
    assert data["event"] == "hello_world"
    assert data["foo"] == "bar"
    assert data["n"] == 42
    assert data["service"] == "fraud-detection-gnn"
    assert data["level"] == "info"
    assert "timestamp" in data


def test_static_context_is_merged(capsys: pytest.CaptureFixture[str]):
    fd_logging.configure_logging(
        level="INFO", json=True, static_context={"env": "test", "run_id": "abc"}
    )
    log = fd_logging.get_logger("test")
    log.info("event")

    last = capsys.readouterr().err.strip().splitlines()[-1]
    data = json.loads(last)
    assert data["env"] == "test"
    assert data["run_id"] == "abc"


def test_log_level_filters(capsys: pytest.CaptureFixture[str]):
    fd_logging.configure_logging(level="WARNING", json=True)
    log = fd_logging.get_logger("test")
    log.debug("should_be_dropped")
    log.info("also_dropped")
    log.warning("kept")

    lines = [ln for ln in capsys.readouterr().err.strip().splitlines() if ln.strip()]
    events = [json.loads(ln)["event"] for ln in lines]
    assert "should_be_dropped" not in events
    assert "also_dropped" not in events
    assert "kept" in events
