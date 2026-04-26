"""Tests for the Kafka producer + consumer in-memory fallback."""

from __future__ import annotations

import asyncio
import contextlib

from fraud_detection.serving.schemas import FraudAlert
from fraud_detection.streaming import FraudAlertConsumer, FraudAlertProducer


def _alert(i: int = 1, score: float = 0.85) -> FraudAlert:
    return FraudAlert(
        transaction_id=i,
        fraud_score=score,
        risk_level="HIGH",
        transaction_amt=42.0,
        card_id="card-1",
    )


# ---------------------------------------------------------------------------
# Producer in-memory fallback
# ---------------------------------------------------------------------------


def test_producer_connects_without_broker():
    p = FraudAlertProducer(bootstrap_servers=None)
    assert p.connect()
    assert not p.is_kafka
    assert p.connected


def test_producer_publish_in_memory():
    p = FraudAlertProducer(bootstrap_servers=None)
    p.connect()
    assert p.publish(_alert(1))
    assert p.publish(_alert(2))
    drained = p.drain_in_memory()
    assert len(drained) == 2
    assert drained[0].transaction_id == 1


def test_producer_drain_clears_buffer():
    p = FraudAlertProducer(bootstrap_servers=None)
    p.connect()
    p.publish(_alert(1))
    p.drain_in_memory()
    assert p.drain_in_memory() == []


def test_producer_stats_track_publishes():
    p = FraudAlertProducer(bootstrap_servers=None)
    p.connect()
    for i in range(3):
        p.publish(_alert(i))
    stats = p.stats()
    assert stats["backend"] == "in_memory"
    assert stats["n_published"] == 3
    assert stats["topic"] == "fraud_alerts"


def test_producer_dead_broker_falls_back(monkeypatch):
    """Unreachable broker should not crash; we end up in in-memory mode."""
    p = FraudAlertProducer(bootstrap_servers="127.0.0.1:1")
    assert p.connect()
    assert p.connected
    # Whether it ends up in kafka or in-memory mode depends on the local
    # confluent_kafka behavior; what matters is that publish() doesn't
    # raise for the in-memory fallback path.
    p.publish(_alert(1))


def test_producer_close_is_safe_when_not_connected():
    p = FraudAlertProducer(bootstrap_servers=None)
    # close before connect should be a no-op
    p.close()
    assert not p.connected


# ---------------------------------------------------------------------------
# Consumer in-memory fallback
# ---------------------------------------------------------------------------


def test_consumer_connects_without_broker():
    c = FraudAlertConsumer(bootstrap_servers=None)
    assert c.connect()
    assert not c.is_kafka


def test_consumer_in_memory_push_drain():
    c = FraudAlertConsumer(bootstrap_servers=None)
    c.connect()
    c.push_in_memory(_alert(1))
    c.push_in_memory(_alert(2))
    drained = c._drain_in_memory()
    assert len(drained) == 2


def test_consumer_async_loop_dispatches_to_handler():
    c = FraudAlertConsumer(bootstrap_servers=None, poll_timeout_seconds=0.01)
    c.connect()
    c.push_in_memory(_alert(1))
    c.push_in_memory(_alert(2))

    received: list[FraudAlert] = []

    async def handler(alert: FraudAlert) -> None:
        received.append(alert)
        if len(received) >= 2:
            c.stop()  # signal the loop to exit

    async def runner():
        # Wrap the async consume loop in a timeout so the test can never hang.
        await asyncio.wait_for(c.consume_async(handler), timeout=2.0)

    with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
        asyncio.run(runner())

    assert len(received) == 2
    assert received[0].transaction_id == 1
    assert received[1].transaction_id == 2


def test_consumer_decode_handles_bad_json():
    c = FraudAlertConsumer(bootstrap_servers=None)
    assert c._decode(b"not json {") is None
    assert c._decode(b"") is None
    assert c._decode(None) is None


def test_consumer_decode_validates_schema():
    c = FraudAlertConsumer(bootstrap_servers=None)
    # Valid alert
    a = _alert(7).model_dump_json().encode()
    decoded = c._decode(a)
    assert decoded is not None
    assert decoded.transaction_id == 7
    # Wrong shape -> rejected
    assert c._decode(b'{"foo": "bar"}') is None


def test_consumer_stats():
    c = FraudAlertConsumer(bootstrap_servers=None, group_id="test-group")
    c.connect()
    c.push_in_memory(_alert(1))
    s = c.stats()
    assert s["backend"] == "in_memory"
    assert s["group_id"] == "test-group"
    assert s["in_memory_buffer_size"] == 1
