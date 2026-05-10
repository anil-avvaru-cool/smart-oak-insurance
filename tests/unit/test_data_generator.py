from __future__ import annotations

import tempfile
from pathlib import Path

from data.generator import generate_quotes, generate_claims


def test_generate_quotes_creates_expected_columns() -> None:
    quotes_df = generate_quotes(seed=123)
    assert "quote_id" in quotes_df.columns
    assert "risk_score_at_issuance" in quotes_df.columns
    assert len(quotes_df) > 0


def test_generate_claims_links_to_quotes() -> None:
    quotes_df = generate_quotes(seed=123)
    claims_df = generate_claims(quotes_df, seed=456)
    assert "quote_id" in claims_df.columns
    assert "is_fraud" in claims_df.columns
    assert len(claims_df) == len(quotes_df)
    assert claims_df["quote_id"].isin(quotes_df["quote_id"]).all()


def test_generate_data_writes_parquet_files() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        quotes_df = generate_quotes(seed=123)
        claims_df = generate_claims(quotes_df, seed=456)
        quotes_path = output_dir / "quotes.parquet"
        claims_path = output_dir / "claims.parquet"
        quotes_df.to_parquet(quotes_path, index=False)
        claims_df.to_parquet(claims_path, index=False)
        assert quotes_path.exists()
        assert claims_path.exists()


def test_state_abbreviation_list_includes_dc_and_count_is_51() -> None:
    from data.states import US_STATE_ABBREVIATIONS

    assert len(US_STATE_ABBREVIATIONS) == 51
    assert "DC" in US_STATE_ABBREVIATIONS
