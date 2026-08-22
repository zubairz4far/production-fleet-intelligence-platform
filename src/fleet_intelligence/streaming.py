from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

from .serving import EtaModelService, EtaPredictionRequest, EtaPredictionResponse


class ConsumerMessage(Protocol):
    def value(self) -> bytes | None: ...

    def error(self) -> Any: ...


class ConsumerLike(Protocol):
    def poll(self, timeout: float) -> ConsumerMessage | None: ...

    def commit(self, message: ConsumerMessage, asynchronous: bool = False) -> Any: ...

    def close(self) -> None: ...


def decode_eta_event(payload: bytes | str) -> EtaPredictionRequest:
    raw = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Kafka ETA event must be a JSON object")
    return EtaPredictionRequest.model_validate(data)


def process_one_message(
    consumer: ConsumerLike,
    service: EtaModelService,
    *,
    timeout_seconds: float = 1.0,
) -> EtaPredictionResponse | None:
    message = consumer.poll(timeout_seconds)
    if message is None:
        return None
    if message.error():
        raise RuntimeError(f"Kafka consumer error: {message.error()}")
    payload = message.value()
    if payload is None:
        raise ValueError("Kafka ETA event has no payload")
    request = decode_eta_event(payload)
    response = service.predict(request)
    consumer.commit(message=message, asynchronous=False)
    return response


def create_kafka_consumer(
    *,
    bootstrap_servers: str,
    group_id: str,
    topic: str,
    extra_config: dict[str, Any] | None = None,
) -> ConsumerLike:
    try:
        from confluent_kafka import Consumer
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError('Install the "mlops" extra to use Kafka integration') from exc

    config: dict[str, Any] = {
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    }
    if extra_config:
        config.update(extra_config)
    consumer = Consumer(config)
    consumer.subscribe([topic])
    return consumer


def consume_forever(
    consumer: ConsumerLike,
    service: EtaModelService,
    *,
    on_prediction: Callable[[EtaPredictionResponse], None] | None = None,
) -> None:
    try:
        while True:
            response = process_one_message(consumer, service)
            if response is not None and on_prediction is not None:
                on_prediction(response)
    finally:
        consumer.close()
