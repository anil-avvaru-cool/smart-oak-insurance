from __future__ import annotations

import json
from typing import Any

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None


class OnlineFeatureStore:
    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_client = None
        self._in_memory_store: dict[str, str] = {}

        if redis_url and redis is not None:
            self._redis_client = redis.from_url(redis_url)

    def set_features(self, key: str, features: dict[str, Any], ttl_seconds: int | None = 3600) -> None:
        payload = json.dumps(features)
        if self._redis_client is not None:
            self._redis_client.set(key, payload, ex=ttl_seconds)
            return
        self._in_memory_store[key] = payload

    def get_features(self, key: str) -> dict[str, Any] | None:
        if self._redis_client is not None:
            raw = self._redis_client.get(key)
            return json.loads(raw) if raw else None
        raw = self._in_memory_store.get(key)
        return json.loads(raw) if raw else None

    def delete_features(self, key: str) -> None:
        if self._redis_client is not None:
            self._redis_client.delete(key)
            return
        self._in_memory_store.pop(key, None)
