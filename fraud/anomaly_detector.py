"""Stage 4b — Isolation Forest anomaly detector (unsupervised fraud signal)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .train_xgb_fraud import CLAIM_FRAUD_FEATURE_COLS, _ensure_risk_score, prepare_fraud_features

# IF decision_function returns higher values for normal, lower for anomalous.
# Multiplying by this scale before sigmoid sharpens the mapping; 3.0 empirically
# places ~95th-percentile anomalies near 0.8 and typical records near 0.3.
_SIGMOID_SCALE = 3.0


def _compute_medians(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    return {c: float(df[c].median()) for c in cols if c in df.columns}


def _impute(X: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = X.copy()
    for col, med in medians.items():
        if col in out.columns:
            out[col] = out[col].fillna(med)
    return out


def decision_to_anomaly_score(raw: np.ndarray) -> np.ndarray:
    """Convert IF decision_function output to [0,1] anomaly score (1 = most anomalous)."""
    return 1.0 / (1.0 + np.exp(raw * _SIGMOID_SCALE))


def train_anomaly_detector(claims_path: Path, quotes_path: Path, output_dir: Path) -> dict:
    """Train Isolation Forest on claim features (unsupervised — is_fraud label unused).

    Saves:
      fraud_anomaly_model.joblib   — fitted IsolationForest
      fraud_anomaly_medians.json   — per-feature median imputation values
      fraud_anomaly_metrics.json   — training diagnostics
    """
    df = pd.read_parquet(claims_path)
    df = _ensure_risk_score(df, quotes_path)
    df = prepare_fraud_features(df)

    available_cols = [c for c in CLAIM_FRAUD_FEATURE_COLS if c in df.columns]
    medians = _compute_medians(df, available_cols)
    X = _impute(df[available_cols], medians)

    # Contamination set to actual fraud rate; clamped to sklearn's valid range [0.05, 0.5].
    # Architecture note: synthetic data has ~33% fraud vs ~15% production.
    # scale_pos_weight in XGBoost handles the imbalance; IF contamination reflects
    # the realistic expected proportion of anomalous records.
    fraud_rate = float(df["is_fraud"].mean()) if "is_fraud" in df.columns else 0.33
    contamination = float(np.clip(fraud_rate, 0.05, 0.5))

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X.values)

    raw_scores = iso.decision_function(X.values)
    anomaly_scores = decision_to_anomaly_score(raw_scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, output_dir / "fraud_anomaly_model.joblib")
    (output_dir / "fraud_anomaly_medians.json").write_text(json.dumps(medians, indent=2))

    metrics = {
        "contamination": round(contamination, 4),
        "n_estimators": iso.n_estimators,
        "n_training_records": len(X),
        "anomaly_score_mean": round(float(anomaly_scores.mean()), 4),
        "anomaly_score_p50": round(float(np.percentile(anomaly_scores, 50)), 4),
        "anomaly_score_p95": round(float(np.percentile(anomaly_scores, 95)), 4),
        "available_feature_cols": available_cols,
    }
    (output_dir / "fraud_anomaly_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
