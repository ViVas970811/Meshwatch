"""Kafka producer for fraud alerts.

Wraps ``confluent_kafka.Producer`` and falls back to an in-process queue
when no broker is reachable -- the FastAPI app stays operational on
laptops without Kafka, and tests don't need a docker-compose stack.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Any

from fraud_detection.serving.schemas import FraudAlert
from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


class FraudAlertProducer:
    """Publish :class:`FraudAlert` events to the ``fraud_alerts`` topic.

    Parameters
    ----------
    bootstrap_servers
        Kafka bootstrap string (e.g. ``"localhost:9092"``). If ``None``
        or unreachable, falls back to an in-memory queue.
    topic
        Topic to publish to (default ``fraud_alerts``).
    flush_timeout_seconds
        Default flush timeout used by :meth:`flush` and :meth:`close`.
    on_delivery
        Optional delivery-report callback. Receives ``(err, msg)`` per
        ``confluent_kafka`` convention; called on the producer's poll
        thread.
    """

    DEFAULT_TOPIC = "fraud_alerts"

    def __init__(
        self,
        *,
        bootstrap_servers: str | None = None,
        topic: str = DEFAULT_TOPIC,
        client_id: str = "meshwatch-api",
        flush_timeout_seconds: float = 5.0,
        on_delivery: Callable[[Any, Any], None] | None = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.client_id = client_id
        self.flush_timeout_seconds = flush_timeout_seconds
        self._on_delivery = on_delivery
        self._producer: Any | None = None
        self._mem_queue: deque[FraudAlert] = deque(maxlen=10_000)
        self._lock = threading.Lock()
        self._connected = False
        self._n_published = 0
        self._n_failures = 0

    # ------------------------------------------------------------------
    # connect / status
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Try to construct a confluent_kafka.Producer; fall back on failure."""
        if not self.bootstrap_servers:
            log.info("kafka_producer_in_memory_no_bootstrap")
            self._connected = True
            return True
        try:
            from confluent_kafka import Producer

            conf = {
                "bootstrap.servers": self.bootstrap_servers,
                "client.id": self.client_id,
                "linger.ms": 5,
                "compression.type": "snappy",
                "enable.idempotence": True,
                "acks": "all",
                "message.timeout.ms": 5000,
            }
            self._producer = Producer(conf)
            # Test connectivity with a metadata fetch (cheap, no message).
            self._producer.list_topics(timeout=2.0)
            self._connected = True
            log.info("kafka_producer_connected", bootstrap=self.bootstrap_servers)
        except Exception as exc:
            log.warning("kafka_producer_unavailable_using_in_memory", error=str(exc))
            self._producer = None
            self._connected = True  # in-memory mode is "operational"
        return self._connected

    @property
    def is_kafka(self) -> bool:
        return self._producer is not None

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # publish
    # ------------------------------------------------------------------

    def publish(self, alert: FraudAlert) -> bool:
        """Send one alert. Returns True if accepted (not necessarily delivered)."""
        if self._producer is not None:
            payload = alert.model_dump_json().encode("utf-8")
            key = str(alert.transaction_id).encode("utf-8")
            try:
                self._producer.produce(
                    topic=self.topic,
                    value=payload,
                    key=key,
                    on_delivery=self._on_delivery_wrapper,
                )
                # poll(0) lets the librdkafka background thread fire the
                # delivery callbacks without blocking.
                self._producer.poll(0)
                with self._lock:
                    self._n_published += 1
                return True
            except BufferError as exc:
                log.warning("kafka_buffer_full", error=str(exc))
                with self._lock:
                    self._n_failures += 1
                return False
            except Exception as exc:
                log.warning("kafka_produce_failed", error=str(exc))
                with self._lock:
                    self._n_failures += 1
                return False

        # In-memory path (testing / no broker).
        with self._lock:
            self._mem_queue.append(alert)
            self._n_published += 1
        return True

    def _on_delivery_wrapper(self, err: Any, msg: Any) -> None:
        if err is not None:
            log.warning("kafka_delivery_failed", error=str(err))
            with self._lock:
                self._n_failures += 1
        if self._on_delivery is not None:
            try:
                self._on_delivery(err, msg)
            except Exception:
                log.exception("kafka_on_delivery_callback_error")

    # ------------------------------------------------------------------
    # housekeeping
    # ------------------------------------------------------------------

    def flush(self, timeout: float | None = None) -> int:
        """Force-flush pending messages. Returns # of messages still in the queue."""
        if self._producer is None:
            return 0
        return int(self._producer.flush(timeout or self.flush_timeout_seconds))

    def close(self) -> None:
        """Flush + drop the underlying producer."""
        if self._producer is not None:
            try:
                self._producer.flush(self.flush_timeout_seconds)
            except Exception as exc:
                log.warning("kafka_close_flush_failed", error=str(exc))
            self._producer = None
        self._connected = False

    # Test helpers ------------------------------------------------------

    def drain_in_memory(self) -> list[FraudAlert]:
        """Return + clear the in-memory queue (no-op in Kafka mode)."""
        if self._producer is not None:
            return []
        with self._lock:
            out = list(self._mem_queue)
            self._mem_queue.clear()
            return out

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "kafka" if self.is_kafka else "in_memory",
            "topic": self.topic,
            "n_published": self._n_published,
            "n_failures": self._n_failures,
            "connected": self._connected,
            "in_memory_queue_size": len(self._mem_queue),
        }


__all__ = ["FraudAlertProducer"]
