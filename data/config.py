from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
RANDOM_SEED = 42
QUOTE_FILE_NAME = "quotes.parquet"
CLAIM_FILE_NAME = "claims.parquet"
QUOTES_OUTPUT = RAW_DATA_DIR / QUOTE_FILE_NAME
CLAIMS_OUTPUT = RAW_DATA_DIR / CLAIM_FILE_NAME
