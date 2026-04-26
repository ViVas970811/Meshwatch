"""Kafka producer + consumer for fraud-alert streaming (Phase 4).

Both classes degrade gracefully to an in-memory queue when no broker is
reachable, so unit tests + local laptop dev work without docker-compose.
"""

from fraud_detection.streaming.kafka_consumer import AlertHandler, FraudAlertConsumer
from fraud_detection.streaming.kafka_producer import FraudAlertProducer

__all__ = ["AlertHandler", "FraudAlertConsumer", "FraudAlertProducer"]
