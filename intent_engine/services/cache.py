from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import importlib.util


class SessionCache:
    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError()

    def set_json(self, key: str, value: Dict[str, Any], ttl_seconds: int) -> None:
        raise NotImplementedError()

    def delete(self, key: str) -> None:
        raise NotImplementedError()


@dataclass
class _Entry:
    expires_at: float
    value: Dict[str, Any]


class InMemoryTTLCache(SessionCache):
    def __init__(self) -> None:
        self._store: Dict[str, _Entry] = {}

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            self._store.pop(key, None)
            return None
        return dict(entry.value)

    def set_json(self, key: str, value: Dict[str, Any], ttl_seconds: int) -> None:
        expires_at = time.time() + float(ttl_seconds)
        self._store[key] = _Entry(expires_at=expires_at, value=dict(value))

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


class RedisCache(SessionCache):
    def __init__(self, redis_url: str) -> None:
        if importlib.util.find_spec("redis") is None:
            raise RuntimeError("redis package is not installed")
        import redis  # type: ignore

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        raw = self._client.get(key)
        if not raw:
            return None
        return json.loads(raw)

    def set_json(self, key: str, value: Dict[str, Any], ttl_seconds: int) -> None:
        self._client.set(key, json.dumps(value), ex=int(ttl_seconds))

    def delete(self, key: str) -> None:
        self._client.delete(key)


def build_cache(redis_url: str) -> SessionCache:
    if redis_url:
        try:
            return RedisCache(redis_url)
        except Exception:
            return InMemoryTTLCache()
    return InMemoryTTLCache()

