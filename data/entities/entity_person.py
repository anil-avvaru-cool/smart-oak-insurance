from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from data.config import CLAIMS_OUTPUT, QUOTES_OUTPUT, RAW_DATA_DIR


def resolve_persons() -> pd.DataFrame:
    quotes_df = pd.read_parquet(QUOTES_OUTPUT)
    claims_df = pd.read_parquet(CLAIMS_OUTPUT)

    persons: list[dict] = []
    seen_person_keys: set[str] = set()
    person_id_map: dict[str, str] = {}

    for _, row in quotes_df.iterrows():
        person_data = _generate_person(row["quote_id"], row["driver_age"], row["state"])
        person_key = _make_person_key(person_data["name_normalized"], person_data["dob"], row["state"])

        if person_key not in seen_person_keys:
            seen_person_keys.add(person_key)
            person_id = str(uuid.uuid4())
            person_id_map[person_key] = person_id
            persons.append({
                "person_id": person_id,
                "name_normalized": person_data["name_normalized"],
                "dob": person_data["dob"],
                "state": row["state"],
                "role": "policyholder",
            })

    for _, row in claims_df.iterrows():
        person_data = _generate_person(row["claim_id"], 35, row["state"])
        person_key = _make_person_key(person_data["name_normalized"], person_data["dob"], row["state"])

        if person_key not in seen_person_keys:
            seen_person_keys.add(person_key)
            person_id = str(uuid.uuid4())
            person_id_map[person_key] = person_id
            persons.append({
                "person_id": person_id,
                "name_normalized": person_data["name_normalized"],
                "dob": person_data["dob"],
                "state": row["state"],
                "role": "claimant",
            })

    persons_df = pd.DataFrame(persons)

    entities_dir = RAW_DATA_DIR / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    output_path = entities_dir / "persons.parquet"
    persons_df.to_parquet(output_path, index=False)

    assert persons_df["person_id"].duplicated().sum() == 0, "No duplicate person_ids allowed"
    print(f"Resolved {len(persons_df)} unique persons to {output_path}")
    return persons_df


def _generate_person(entity_id: str, age: int, state: str) -> dict:
    seed_hash = hashlib.sha256(f"{entity_id}:{age}:{state}".encode()).hexdigest()
    seed_int = int(seed_hash[:16], 16)
    np.random.seed(seed_int % (2**31))

    first_names = ["James", "Mary", "Robert", "Patricia", "Michael", "Jennifer",
                   "William", "Linda", "David", "Barbara", "Richard", "Susan"]
    last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Garcia",
                  "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez"]

    first = first_names[seed_int % len(first_names)]
    last = last_names[(seed_int // len(first_names)) % len(last_names)]

    dob = datetime.now() - timedelta(days=age * 365 + 100 + (seed_int % 200))

    return {
        "name_normalized": f"{first} {last}".strip().title(),
        "dob": dob.strftime("%Y-%m-%d"),
    }


def _make_person_key(name: str, dob: str, state: str) -> str:
    key = f"{name.lower().replace(' ', '')}:{dob}:{state}".encode()
    return hashlib.sha256(key).hexdigest()
