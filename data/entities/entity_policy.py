from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from data.config import CLAIMS_OUTPUT, QUOTES_OUTPUT, RAW_DATA_DIR


def resolve_policies() -> pd.DataFrame:
    quotes_df = pd.read_parquet(QUOTES_OUTPUT)

    policies: list[dict] = []

    for _, row in quotes_df.iterrows():
        inception_date = datetime.now() - timedelta(days=max(1, int(row.get("insurance_lapse_days", 0) or 0)))
        expiration_date = inception_date + timedelta(days=365)

        policies.append({
            "policy_id": str(uuid.uuid4()),
            "quote_id": row["quote_id"],
            "inception_date": inception_date.strftime("%Y-%m-%d"),
            "expiration_date": expiration_date.strftime("%Y-%m-%d"),
            "status": "active" if expiration_date > datetime.now() else "lapsed",
            "is_lapsed": expiration_date < datetime.now(),
        })

    policies_df = pd.DataFrame(policies)

    claims_df = pd.read_parquet(CLAIMS_OUTPUT)
    for _, claim_row in claims_df.iterrows():
        policy_row = policies_df[policies_df["quote_id"] == claim_row["quote_id"]]
        if len(policy_row) > 0:
            inception_date_obj = datetime.strptime(policy_row.iloc[0]["inception_date"], "%Y-%m-%d")
            policy_inception_days = max(0, int((datetime.now() - inception_date_obj).days))
            assert policy_inception_days >= 0, f"No claims before policy inception"

    entities_dir = RAW_DATA_DIR / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    output_path = entities_dir / "policies.parquet"
    policies_df.to_parquet(output_path, index=False)

    print(f"Resolved {len(policies_df)} policies to {output_path}")
    return policies_df
