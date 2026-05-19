from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .feature_definitions import FEATURE_STORE_VERSION, build_claim_feature_vector, build_quote_feature_vector
from .online_serving import OnlineFeatureStore


def current_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_feature_snapshot(
    record_id: str,
    record_type: str,
    features: dict,
    state: str,
    regulatory_mask_applied: bool = True,
) -> dict:
    return {
        "record_id": record_id,
        "record_type": record_type,
        "timestamp": current_utc_timestamp(),
        "feature_store_version": FEATURE_STORE_VERSION,
        "state": state,
        "regulatory_mask_applied": regulatory_mask_applied,
        "features": features,
    }


def build_quote_snapshot(quote_payload: dict) -> dict:
    quote_id = quote_payload.get("quote_id", "quote-unknown")
    state = quote_payload.get("state", "UNKNOWN")
    features = build_quote_feature_vector(quote_payload)
    return generate_feature_snapshot(
        record_id=quote_id,
        record_type="quote",
        features=features,
        state=state,
        regulatory_mask_applied=True,
    )


def build_claim_snapshot(claim_payload: dict, underwriting_features: dict | None = None) -> dict:
    claim_id = claim_payload.get("claim_id", "claim-unknown")
    state = claim_payload.get("state", "UNKNOWN")
    features = build_claim_feature_vector(claim_payload, underwriting_features)
    return generate_feature_snapshot(
        record_id=claim_id,
        record_type="claim",
        features=features,
        state=state,
        regulatory_mask_applied=False,
    )


def _sanitize_nans(obj: Any) -> Any:
    """Recursively replace float NaN with None so json.dumps produces valid JSON."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nans(v) for v in obj]
    return obj


def write_snapshot(snapshot: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{snapshot['record_id']}.json"
    file_path.write_text(json.dumps(_sanitize_nans(snapshot), indent=2))
    return file_path


def _to_native(val: Any) -> Any:
    """Convert numpy scalars to native Python types for JSON serialization."""
    if hasattr(val, "item"):
        native = val.item()
        if isinstance(native, float) and math.isnan(native):
            return None
        return native
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def run_offline_pipeline(
    quotes_path: Path,
    claims_path: Path,
    output_dir: Path,
    online_store: OnlineFeatureStore | None = None,
) -> tuple[int, int]:
    """Batch-process all quotes and claims into feature snapshots.

    Pass 1 — quotes: builds feature vectors, writes JSON snapshots, pushes to online
    store, and indexes underwriting features by quote_id for cross-platform linking.

    Pass 2 — claims: joins each claim to its underwriting context via quote_id
    (the shared data spine), builds claim feature vectors, writes snapshots.

    Returns (quotes_written, claims_written).
    """
    quotes_df = pd.read_parquet(quotes_path)
    claims_df = pd.read_parquet(claims_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: quotes → build underwriting feature index keyed by quote_id
    underwriting_index: dict[str, dict] = {}
    quotes_written = 0

    for _, row in quotes_df.iterrows():
        payload = {k: _to_native(v) for k, v in row.items()}
        snapshot = build_quote_snapshot(payload)
        write_snapshot(snapshot, output_dir)
        if online_store is not None:
            online_store.set_features(snapshot["record_id"], snapshot["features"])
        underwriting_index[str(row["quote_id"])] = snapshot["features"]
        quotes_written += 1

    # Pass 2: claims → join underwriting context via quote_id (shared data spine)
    claims_written = 0

    for _, row in claims_df.iterrows():
        payload = {k: _to_native(v) for k, v in row.items()}
        underwriting = underwriting_index.get(str(row.get("quote_id", "")))
        snapshot = build_claim_snapshot(payload, underwriting)
        write_snapshot(snapshot, output_dir)
        if online_store is not None:
            online_store.set_features(snapshot["record_id"], snapshot["features"])
        claims_written += 1

    return quotes_written, claims_written
