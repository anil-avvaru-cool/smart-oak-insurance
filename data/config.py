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
OFFLINE_FEATURES_DIR = PROCESSED_DATA_DIR / "features"
RISK_MODELS_DIR = PROCESSED_DATA_DIR / "risk_models"
FRAUD_MODELS_DIR = PROCESSED_DATA_DIR / "fraud_models"

# DEC-013: generation date spread — quotes and claims are spread over this many days
QUOTE_DATE_RANGE_DAYS = 365
# DEC-013: fraction of claims that remain open (claim_closed_at is null)
OPEN_CLAIM_RATE = 0.15
# DEC-013: fraction of fraud claims where fraud_confirmed_at is null (label window not yet closed)
UNCONFIRMED_FRAUD_RATE = 0.20

# PSI drift monitoring constants (monitoring/psi_drift.py)
PSI_REFERENCE_VERSION = "v1.0.0"
PSI_CURRENT_WINDOW_DAYS = 14
PSI_MIN_RECORDS = 500

# Champion-Challenger rollout config (monitoring/champion_challenger.py — CC-3)
# 0 = shadow-only (no live traffic to challenger); set to 10 / 25 / 50 / 100 during phased rollout
CC_TRAFFIC_SPLIT_PCT: int = 0
CC_ROLLOUT_STAGES: list[int] = [10, 25, 50, 100]
