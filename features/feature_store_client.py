from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .offline_pipeline import build_claim_snapshot, build_quote_snapshot, write_snapshot
from .online_serving import OnlineFeatureStore


class FeatureStoreClient:
    def __init__(self, offline_dir: Path, online_store: OnlineFeatureStore | None = None) -> None:
        self.offline_dir = offline_dir
        self.online_store = online_store or OnlineFeatureStore()
        self.offline_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, snapshot: dict[str, Any]) -> Path:
        return write_snapshot(snapshot, self.offline_dir)

    def load_snapshot(self, record_id: str, record_type: str) -> dict[str, Any] | None:
        path = self.offline_dir / f"{record_type}s" / f"{record_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def upsert_online_features(self, key: str, features: dict[str, Any], ttl_seconds: int | None = 3600) -> None:
        self.online_store.set_features(key, features, ttl_seconds)

    def get_online_features(self, key: str) -> dict[str, Any] | None:
        return self.online_store.get_features(key)

    def save_quote(self, quote_payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = build_quote_snapshot(quote_payload)
        self.save_snapshot(snapshot)
        self.upsert_online_features(snapshot["record_id"], snapshot["features"])
        return snapshot

    def save_claim(self, claim_payload: dict[str, Any], underwriting_features: dict[str, Any] | None = None) -> dict[str, Any]:
        snapshot = build_claim_snapshot(claim_payload, underwriting_features)
        self.save_snapshot(snapshot)
        self.upsert_online_features(snapshot["record_id"], snapshot["features"])
        return snapshot
