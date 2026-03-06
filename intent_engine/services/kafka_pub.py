from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


class RiskEventPublisher:
    def publish(self, payload: Dict[str, Any]) -> None:
        raise NotImplementedError()


class NoopPublisher(RiskEventPublisher):
    def publish(self, payload: Dict[str, Any]) -> None:
        return


@dataclass
class KafkaPublisher(RiskEventPublisher):
    producer: Any
    topic: str

    def publish(self, payload: Dict[str, Any]) -> None:
        self.producer.send(self.topic, payload)


def build_publisher(bootstrap_servers: str, topic: str) -> RiskEventPublisher:
    if not bootstrap_servers:
        return NoopPublisher()
    if importlib.util.find_spec("kafka") is None:
        return NoopPublisher()

    try:
        from kafka import KafkaProducer  # type: ignore
    except Exception:
        return NoopPublisher()

    producer = KafkaProducer(
        bootstrap_servers=str(bootstrap_servers),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=0,
        linger_ms=0,
        retries=0,
    )

    return KafkaPublisher(producer=producer, topic=str(topic))

