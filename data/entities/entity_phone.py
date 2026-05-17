from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from data.config import CLAIMS_OUTPUT, QUOTES_OUTPUT, RAW_DATA_DIR


def resolve_phones() -> pd.DataFrame:
    quotes_df = pd.read_parquet(QUOTES_OUTPUT)

    phones: list[dict] = []
    seen_phone_keys: set[str] = set()

    for _, row in quotes_df.iterrows():
        phone_data = _generate_phone(row["quote_id"])
        phone_normalized = _normalize_phone(phone_data)
        phone_key = _make_phone_key(phone_normalized)

        if phone_key not in seen_phone_keys:
            seen_phone_keys.add(phone_key)
            is_valid = len(phone_normalized) == 10 and phone_normalized.isdigit()
            phones.append({
                "phone_id": str(uuid.uuid4()),
                "phone_normalized": phone_normalized if is_valid else None,
                "phone_hash": hashlib.sha256(phone_normalized.encode()).hexdigest()[:16] if is_valid else None,
                "is_valid": is_valid,
                "phone_type": "mobile",
            })

    phones_df = pd.DataFrame(phones)

    entities_dir = RAW_DATA_DIR / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    output_path = entities_dir / "phones.parquet"
    phones_df.to_parquet(output_path, index=False)

    assert phones_df[phones_df["is_valid"]]["phone_hash"].isnull().sum() == 0, "Valid phones must have hash"
    print(f"Resolved {len(phones_df)} unique phones to {output_path}")
    return phones_df


def _generate_phone(entity_id: str) -> str:
    seed_hash = hashlib.sha256(f"{entity_id}:phone".encode()).hexdigest()
    seed_int = int(seed_hash[:16], 16)
    np.random.seed(seed_int % (2**31))

    area_code = 200 + (seed_int % 800)
    exchange = 100 + ((seed_int // 800) % 900)
    number = (seed_int // 720000) % 10000

    return f"{area_code}{exchange}{number:04d}"


def _normalize_phone(phone: str) -> str:
    phone_clean = "".join(c for c in str(phone) if c.isdigit())
    if len(phone_clean) == 11 and phone_clean[0] == "1":
        phone_clean = phone_clean[1:]
    if len(phone_clean) == 10:
        return phone_clean
    return phone_clean


def _make_phone_key(phone_normalized: str) -> str:
    key = phone_normalized.encode()
    return hashlib.sha256(key).hexdigest()
