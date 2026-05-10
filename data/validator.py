from __future__ import annotations

import pandas as pd

from data.config import CLAIMS_OUTPUT, QUOTES_OUTPUT
from data.states import US_STATE_ABBREVIATIONS


def validate_quote_dataset(quotes_df: pd.DataFrame) -> None:
    print("Validating quote dataset...")
    print(f"  records: {len(quotes_df)}")
    print(f"  quote state coverage: {quotes_df['state'].nunique()} states")
    print(f"  risk score range: {quotes_df['risk_score_at_issuance'].min():.3f} - {quotes_df['risk_score_at_issuance'].max():.3f}")
    invalid_states = quotes_df.loc[~quotes_df["state"].isin(US_STATE_ABBREVIATIONS), "state"]
    if len(invalid_states):
        print(f"  ❌ invalid state values found: {sorted(invalid_states.unique())}")
    else:
        print("  ✅ state values are valid")
    if (quotes_df["credit_score"] < 500).any() or (quotes_df["credit_score"] > 850).any():
        print("  ❌ credit score out of range")
    else:
        print("  ✅ credit score range looks healthy")


def validate_claim_dataset(claims_df: pd.DataFrame) -> None:
    print("Validating claim dataset...")
    print(f"  records: {len(claims_df)}")
    fraud_rate = claims_df[claims_df["is_fraud"] == True].shape[0] / max(1, len(claims_df))
    print(f"  fraud rate: {fraud_rate:.2%}")
    if fraud_rate < 0.15 or fraud_rate > 0.45:
        print("  ⚠️ fraud rate is outside expected training range")
    else:
        print("  ✅ fraud rate is within expected training range")
    if (claims_df["policy_inception_days"] < 0).any():
        print("  ❌ negative policy_inception_days found")
    else:
        print("  ✅ policy inception values are valid")
    if (claims_df["reporting_delay_days"] < 0).any():
        print("  ❌ negative reporting_delay_days found")
    else:
        print("  ✅ reporting delay values are valid")


def validate_data(quotes_df: pd.DataFrame | None = None, claims_df: pd.DataFrame | None = None) -> None:
    if quotes_df is None:
        quotes_df = pd.read_parquet(QUOTES_OUTPUT)
    if claims_df is None:
        claims_df = pd.read_parquet(CLAIMS_OUTPUT)

    validate_quote_dataset(quotes_df)
    validate_claim_dataset(claims_df)
    print("Validation complete.")
