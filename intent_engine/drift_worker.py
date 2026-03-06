from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

from .config import settings
from .services.cache import build_cache
from .services.math_utils import kl_divergence
from .services.markov import load_markov_artifacts


def _get_env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None or val == "" else str(val)


KAFKA_BOOTSTRAP = _get_env("INTENT_ENGINE_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = _get_env("INTENT_ENGINE_KAFKA_TOPIC_RISK_EVENTS", "risk_events")
KAFKA_GROUP_ID = _get_env("INTENT_DRIFT_KAFKA_GROUP_ID", "intent_drift_worker")
PUBLISH_EVERY_SECONDS = float(_get_env("INTENT_DRIFT_PUBLISH_EVERY_SECONDS", "5"))
MAX_TRANSITIONS = int(_get_env("INTENT_DRIFT_MAX_TRANSITIONS", "20000"))


@dataclass
class DriftState:
    prev_top_state_by_session: Dict[str, int]
    transitions: Deque[Tuple[int, int]]


def _transition_matrix_from_transitions(num_states: int, transitions: Deque[Tuple[int, int]], alpha: float) -> np.ndarray:
    counts = np.ones((num_states, num_states), dtype=np.float64) * float(alpha)
    for i, j in transitions:
        if 0 <= i < num_states and 0 <= j < num_states:
            counts[i, j] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    return counts / row_sums


def main() -> None:
    try:
        from kafka import KafkaConsumer  # type: ignore
    except Exception as e:
        raise RuntimeError("kafka-python is required for drift_worker") from e

    cache = build_cache(settings.redis_url)

    model_dir = Path(settings.model_dir)
    markov_path = model_dir / "markov.pkl"
    markov = load_markov_artifacts(markov_path)

    a_baseline = getattr(markov.transition_model, "a_counts")
    if isinstance(a_baseline, np.ndarray):
        a0_counts = a_baseline
    else:
        a0_counts = np.asarray(a_baseline, dtype=np.float64)
    a0 = (a0_counts / np.where(a0_counts.sum(axis=1, keepdims=True) > 0.0, a0_counts.sum(axis=1, keepdims=True), 1.0)).astype(
        np.float64
    )

    h = int(a0.shape[0])
    alpha = 0.5

    state = DriftState(prev_top_state_by_session={}, transitions=deque(maxlen=int(MAX_TRANSITIONS)))

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )

    last_publish = 0.0

    for msg in consumer:
        payload: Dict[str, Any] = msg.value
        session_id = str(payload.get("session_id") or "")
        if not session_id:
            continue
        top_state_val = payload.get("top_intent_state")
        if top_state_val is None:
            continue
        top_state = int(top_state_val)

        prev = state.prev_top_state_by_session.get(session_id)
        state.prev_top_state_by_session[session_id] = top_state
        if prev is not None:
            state.transitions.append((int(prev), int(top_state)))

        now = time.time()
        if now - last_publish < float(PUBLISH_EVERY_SECONDS):
            continue
        last_publish = now

        a_recent = _transition_matrix_from_transitions(h, state.transitions, alpha=alpha)
        kl = float(kl_divergence(a_recent, a0))

        out = {
            "ok": True,
            "timestamp_unix_ms": int(now * 1000),
            "hmm_transition_kl": float(kl),
            "recent_transition_count": int(len(state.transitions)),
        }

        cache.set_json("metrics:drift", out, ttl_seconds=int(settings.session_ttl_seconds))


if __name__ == "__main__":
    main()

