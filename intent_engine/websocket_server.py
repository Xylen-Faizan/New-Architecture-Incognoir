from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Set


def _get_env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None or val == "" else str(val)


WS_HOST = _get_env("INTENT_WS_HOST", "0.0.0.0")
WS_PORT = int(_get_env("INTENT_WS_PORT", "8765"))
KAFKA_BOOTSTRAP = _get_env("INTENT_ENGINE_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = _get_env("INTENT_ENGINE_KAFKA_TOPIC_RISK_EVENTS", "risk_events")
KAFKA_GROUP_ID = _get_env("INTENT_WS_KAFKA_GROUP_ID", "intent_ws_broadcaster")


async def _broadcast_loop(clients: Set[Any]) -> None:
    try:
        from kafka import KafkaConsumer  # type: ignore
    except Exception as e:
        raise RuntimeError("kafka-python is required for websocket_server") from e

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )

    loop = asyncio.get_running_loop()

    while True:
        msg = await loop.run_in_executor(None, consumer.poll, 1000)
        if not msg:
            await asyncio.sleep(0.01)
            continue

        for _, messages in msg.items():
            for m in messages:
                payload: Dict[str, Any] = m.value
                if not clients:
                    continue
                data = json.dumps(payload)
                await asyncio.gather(*(c.send(data) for c in list(clients)), return_exceptions=True)


async def _run_server() -> None:
    try:
        import websockets  # type: ignore
    except Exception as e:
        raise RuntimeError("websockets is required for websocket_server") from e

    clients: Set[Any] = set()

    async def handler(websocket):
        clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            clients.discard(websocket)

    async with websockets.serve(handler, WS_HOST, WS_PORT):
        await _broadcast_loop(clients)


def main() -> None:
    asyncio.run(_run_server())


if __name__ == "__main__":
    main()

