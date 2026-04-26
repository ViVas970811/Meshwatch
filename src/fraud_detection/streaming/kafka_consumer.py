"""Kafka consumer for fraud alerts -- poll loop with WebSocket fan-out.

The Phase 4 architecture uses Kafka as the durable buffer between the
scoring service and any downstream consumer (the dashboard's
``/ws/alerts`` endpoint, the agent's investigation queue, etc.). This
module exposes :class:`FraudAlertConsumer` which:

* Subscribes to one or more alert topics (default ``fraud_alerts``).
* Calls a user-supplied callback for each decoded :class:`FraudAlert`.
* Falls back to an in-process broker stub when Kafka isn't reachable so
  tests + local dev work without docker-compose.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fraud_detection.serving.schemas import FraudAlert
from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)

AlertHandler = Callable[[FraudAlert], None] | Callable[[FraudAlert], Awaitable[None]]


class FraudAlertConsumer:
    """Background poll loop that feeds alerts into a callback.

    Run with :meth:`start` (returns immediately, dispatch happens on a
    daemon thread) and stop with :meth:`stop`. For async use cases, see
    :meth:`consume_async`.
    """

    def __init__(
        self,
        *,
        bootstrap_servers: str | None = None,
        topic: str = "fraud_alerts",
        group_id: str = "meshwatch-api",
        handler: AlertHandler | None = None,
        poll_timeout_seconds: float = 1.0,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.handler = handler
        self.poll_timeout_seconds = poll_timeout_seconds

        self._consumer: Any | None = None
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False
        self._n_consumed = 0
        self._n_errors = 0

        # in-memory fallback: peer producers register here and we drain.
        self._mem_buffer: list[FraudAlert] = []
        self._mem_lock = threading.Lock()

        # Pending fire-and-forget handler tasks; tracked so the loop
        # doesn't GC them mid-execution (cf. RUF006 / asyncio docs).
        self._pending_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # connect
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if not self.bootstrap_servers:
            log.info("kafka_consumer_in_memory_no_bootstrap")
            self._connected = True
            return True
        try:
            from confluent_kafka import Consumer

            self._consumer = Consumer(
                {
                    "bootstrap.servers": self.bootstrap_servers,
                    "group.id": self.group_id,
                    "auto.offset.reset": "latest",
                    "enable.auto.commit": True,
                    "session.timeout.ms": 6000,
                }
            )
            self._consumer.subscribe([self.topic])
            self._connected = True
            log.info("kafka_consumer_connected", topic=self.topic, group=self.group_id)
        except Exception as exc:
            log.warning("kafka_consumer_unavailable_using_in_memory", error=str(exc))
            self._consumer = None
            self._connected = True
        return self._connected

    @property
    def is_kafka(self) -> bool:
        return self._consumer is not None

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # in-memory bridge (used when there's no Kafka)
    # ------------------------------------------------------------------

    def push_in_memory(self, alert: FraudAlert) -> None:
        """Inject an alert into the in-memory buffer (used by tests and the
        FastAPI app's local fan-out when Kafka isn't reachable)."""
        with self._mem_lock:
            self._mem_buffer.append(alert)

    def _drain_in_memory(self) -> list[FraudAlert]:
        with self._mem_lock:
            out = list(self._mem_buffer)
            self._mem_buffer.clear()
            return out

    # ------------------------------------------------------------------
    # blocking poll (sync)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn a daemon thread that runs the poll loop."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run_blocking_loop,
            name="meshwatch-kafka-consumer",
            daemon=True,
        )
        self._thread.start()
        log.info("kafka_consumer_started")

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._consumer is not None:
            with contextlib.suppress(Exception):
                self._consumer.close()
            self._consumer = None
        log.info("kafka_consumer_stopped", n_consumed=self._n_consumed)

    def _run_blocking_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                if self._consumer is not None:
                    msg = self._consumer.poll(self.poll_timeout_seconds)
                    if msg is None:
                        continue
                    if msg.error():
                        log.warning("kafka_consumer_msg_error", error=str(msg.error()))
                        self._n_errors += 1
                        continue
                    payload = msg.value()
                    self._dispatch_payload(payload)
                else:
                    # in-memory fallback: drain buffer
                    for a in self._drain_in_memory():
                        self._dispatch_alert(a)
                    time.sleep(self.poll_timeout_seconds)
            except Exception as exc:
                log.exception("kafka_consumer_loop_error", error=str(exc))
                self._n_errors += 1
                time.sleep(self.poll_timeout_seconds)

    # ------------------------------------------------------------------
    # async variant (used by FastAPI WebSocket handler)
    # ------------------------------------------------------------------

    async def consume_async(self, handler: AlertHandler) -> None:
        """Yield alerts via an async-compatible handler.

        Runs until :meth:`stop` is called. Designed to live in an
        ``asyncio.Task`` started from a FastAPI lifespan / dependency.
        """
        loop = asyncio.get_running_loop()
        while not self._stop_evt.is_set():
            alerts: list[FraudAlert] = []
            if self._consumer is not None:
                # Run blocking poll on a worker thread to avoid blocking the loop.
                msg = await loop.run_in_executor(
                    None, self._consumer.poll, self.poll_timeout_seconds
                )
                if msg is None:
                    continue
                if msg.error():
                    log.warning("kafka_consumer_msg_error", error=str(msg.error()))
                    self._n_errors += 1
                    continue
                payload = msg.value()
                a = self._decode(payload)
                if a is not None:
                    alerts.append(a)
            else:
                alerts = self._drain_in_memory()
                if not alerts:
                    await asyncio.sleep(self.poll_timeout_seconds)
                    continue

            for alert in alerts:
                self._n_consumed += 1
                result = handler(alert)
                if asyncio.iscoroutine(result):
                    await result

    # ------------------------------------------------------------------
    # decode + dispatch helpers
    # ------------------------------------------------------------------

    def _decode(self, payload: bytes | str | None) -> FraudAlert | None:
        if not payload:
            return None
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8", errors="replace")
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            log.warning("kafka_consumer_decode_failed", error=str(exc))
            return None
        try:
            return FraudAlert.model_validate(data)
        except Exception as exc:
            log.warning("kafka_consumer_schema_mismatch", error=str(exc))
            return None

    def _dispatch_payload(self, payload: bytes | str | None) -> None:
        a = self._decode(payload)
        if a is not None:
            self._dispatch_alert(a)

    def _dispatch_alert(self, alert: FraudAlert) -> None:
        self._n_consumed += 1
        if self.handler is None:
            return
        try:
            result = self.handler(alert)
            if asyncio.iscoroutine(result):
                # Fire-and-forget on the running loop if any. We track the
                # task so the running loop doesn't GC it mid-execution; we
                # don't need to await it (the handler is one-shot per alert).
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(result)
                    self._pending_tasks.add(task)
                    task.add_done_callback(self._pending_tasks.discard)
                except RuntimeError:
                    # No running loop on this thread -- run synchronously.
                    asyncio.run(result)
        except Exception:
            log.exception("kafka_consumer_handler_error")
            self._n_errors += 1

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "kafka" if self.is_kafka else "in_memory",
            "topic": self.topic,
            "group_id": self.group_id,
            "n_consumed": self._n_consumed,
            "n_errors": self._n_errors,
            "connected": self._connected,
            "in_memory_buffer_size": len(self._mem_buffer),
        }


__all__ = ["AlertHandler", "FraudAlertConsumer"]
