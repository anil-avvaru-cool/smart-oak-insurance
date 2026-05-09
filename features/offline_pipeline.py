from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .feature_definitions import FEATURE_STORE_VERSION, build_claim_feature_vector, build_quote_feature_vector


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


def write_snapshot(snapshot: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{snapshot['record_id']}.json"
    file_path.write_text(json.dumps(snapshot, indent=2))
    return file_path
