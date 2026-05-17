from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from data.config import CLAIMS_OUTPUT, QUOTES_OUTPUT, RAW_DATA_DIR


def resolve_addresses() -> pd.DataFrame:
    quotes_df = pd.read_parquet(QUOTES_OUTPUT)

    addresses: list[dict] = []
    seen_address_keys: set[str] = set()

    for _, row in quotes_df.iterrows():
        address_data = _generate_address(row["quote_id"], row["state"])
        address_normalized = _normalize_address(address_data)
        address_key = _make_address_key(address_normalized)

        if address_key not in seen_address_keys:
            seen_address_keys.add(address_key)
            addresses.append({
                "address_id": str(uuid.uuid4()),
                "address_normalized": address_normalized,
                "city": address_data["city"],
                "state": row["state"],
                "zip": address_data["zip"],
                "address_hash": hashlib.sha256(address_normalized.encode()).hexdigest()[:16],
            })

    addresses_df = pd.DataFrame(addresses)

    entities_dir = RAW_DATA_DIR / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    output_path = entities_dir / "addresses.parquet"
    addresses_df.to_parquet(output_path, index=False)

    assert addresses_df["address_hash"].isnull().sum() == 0, "No null address_hash allowed"
    print(f"Resolved {len(addresses_df)} unique addresses to {output_path}")
    return addresses_df


def _generate_address(entity_id: str, state: str) -> dict:
    seed_hash = hashlib.sha256(f"{entity_id}:{state}".encode()).hexdigest()
    seed_int = int(seed_hash[:16], 16)
    np.random.seed(seed_int % (2**31))

    cities_by_state = {
        "CA": ["Los Angeles", "San Francisco", "San Diego", "Sacramento", "Oakland"],
        "TX": ["Houston", "Dallas", "Austin", "San Antonio", "Fort Worth"],
        "FL": ["Miami", "Tampa", "Jacksonville", "Orlando", "St. Petersburg"],
        "NY": ["New York", "Buffalo", "Rochester", "Yonkers", "Syracuse"],
        "PA": ["Philadelphia", "Pittsburgh", "Allentown", "Erie", "Reading"],
    }

    cities = cities_by_state.get(state, ["Main City", "Downtown", "Suburbs", "North", "South"])
    street_nums = [100 + (seed_int % 9900), 200 + (seed_int // 100 % 9800), 1000 + (seed_int // 10000 % 8000)]
    street_names = ["Main St", "Oak Ave", "Elm St", "Pine Ave", "Maple Dr", "Oak Dr"]
    city = cities[seed_int % len(cities)]
    street_num = street_nums[0]
    street = street_names[(seed_int // len(cities)) % len(street_names)]
    zip_code = f"{10000 + (seed_int % 90000)}"

    return {
        "address_line1": f"{street_num} {street}",
        "city": city,
        "state": state,
        "zip": zip_code,
    }


def _normalize_address(address_data: dict) -> str:
    line1 = (address_data.get("address_line1") or "").strip()
    line1 = line1.replace("St.", "Street").replace("Ave.", "Avenue").replace("Blvd.", "Boulevard")
    line1 = line1.replace("Dr.", "Drive").replace("Rd.", "Road").replace("Ln.", "Lane")

    city = (address_data.get("city") or "").strip().title()
    state = (address_data.get("state") or "").strip().upper()
    zip_code = (address_data.get("zip") or "").strip()

    return f"{line1}, {city}, {state} {zip_code}".replace("  ", " ")


def _make_address_key(address_normalized: str) -> str:
    key = address_normalized.lower().encode()
    return hashlib.sha256(key).hexdigest()
