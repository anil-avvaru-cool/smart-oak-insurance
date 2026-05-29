"""Stage 4a — XGBoost Fraud Classifier: P(fraud | structured claim features)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Claim-era features for fraud scoring.
# Nulls are intentional — XGBoost learns the null path:
#   graph_hop_distance: null = no fraud path within 6 hops (GRAPH_HOP_SENTINEL filtered in feature_definitions)
#   telematics_*: null = non-enrolled (trio convention in feature_definitions)
CLAIM_FRAUD_FEATURE_COLS: list[str] = [
    "policy_inception_days",           # short = high-risk (cross-platform spine signal)
    "prior_claims_count",
    "reported_injury_count",
    "reporting_delay_days",
    "attorney_present",
    "claimant_count",
    "graph_hop_distance",              # null = no fraud path found
    "shared_attribute_count",
    "attorney_centrality_score",
    "narrative_inconsistency_score",
    "narrative_complexity_score",
    "risk_score_at_issuance",          # underwriting → claims spine
    "ip_geolocation_delta_miles",
    "device_fingerprint_match",
    "submission_hour",
    "telematics_distraction_score",    # null for non-enrolled
    "telematics_hard_brake_rate",
    "telematics_crash_match",
    "telematics_commute_entropy",
    "telematics_enrolled",
]

TARGET = "is_fraud"


def prepare_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cast bool columns to float so XGBoost sees numeric inputs."""
    out = df.copy()
    for col in ("attorney_present", "device_fingerprint_match", "telematics_enrolled"):
        if col in out.columns:
            out[col] = out[col].astype(float)
    return out


def _ensure_risk_score(claims_df: pd.DataFrame, quotes_path: Path) -> pd.DataFrame:
    """Join risk_score_at_issuance from quotes if absent or all-null in claims."""
    col = "risk_score_at_issuance"
    if col in claims_df.columns and claims_df[col].notna().any():
        return claims_df
    quotes_df = pd.read_parquet(quotes_path, columns=["quote_id", col])
    return claims_df.merge(quotes_df, on="quote_id", how="left", suffixes=("", "_q"))


def train(claims_path: Path, quotes_path: Path, output_dir: Path) -> dict:
    """Train XGBoost fraud model and write fraud_model.json + fraud_metrics.json.

    Returns metrics dict: auc, gini, log_loss, average_precision, feature_importance_pct.
    """
    df = pd.read_parquet(claims_path)
    df = _ensure_risk_score(df, quotes_path)
    df = prepare_fraud_features(df).dropna(subset=[TARGET])

    available_cols = [c for c in CLAIM_FRAUD_FEATURE_COLS if c in df.columns]
    X = df[available_cols]
    y = df[TARGET].astype(int)

    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    scale_pos_weight = neg / pos

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, preds))
    gini = 2 * auc - 1
    ll = float(log_loss(y_test, preds))
    ap = float(average_precision_score(y_test, preds))

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "fraud_model.json")

    raw_importance = model.get_booster().get_score(importance_type="gain")
    total_gain = sum(raw_importance.values()) or 1.0
    feature_importance_pct = dict(
        sorted(
            {f: round(raw_importance.get(f, 0.0) / total_gain * 100, 2) for f in available_cols}.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    metrics = {
        "auc": round(auc, 4),
        "gini": round(gini, 4),
        "log_loss": round(ll, 4),
        "average_precision": round(ap, 4),
        "scale_pos_weight": round(scale_pos_weight, 2),
        "train_positives": pos,
        "train_negatives": neg,
        "feature_importance_pct": feature_importance_pct,
        "available_feature_cols": available_cols,
    }
    (output_dir / "fraud_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "fraud_features.json").write_text(json.dumps(available_cols, indent=2))
    return metrics


def train_with_pu_labels(
    claims_path: Path,
    quotes_path: Path,
    pu_labels_path: Path,
    output_dir: Path,
) -> dict:
    """Stage 5b — Train XGBoost using PU-adjusted sample weights.

    Loads pu_adjusted_labels.parquet (written by label_quality.build_pu_adjusted_dataset)
    and trains an XGBoost model with per-record sample weights:
      confirmed fraud  → weight 2.0  (high-confidence positive)
      suspected mislabel → weight 0.5  (retain but penalise)
      all others       → weight 1.0

    Saves fraud_model_pu.json + fraud_metrics_pu.json.
    Returns metrics dict (same schema as train(), plus pu_label_adjusted=True).
    """
    df = pd.read_parquet(claims_path)
    df = _ensure_risk_score(df, quotes_path)
    df = prepare_fraud_features(df).dropna(subset=[TARGET]).reset_index(drop=True)

    pu_df = pd.read_parquet(pu_labels_path, columns=["claim_id", "sample_weight"])
    df = df.merge(pu_df, on="claim_id", how="inner")

    available_cols = [c for c in CLAIM_FRAUD_FEATURE_COLS if c in df.columns]
    X = df[available_cols]
    y = df[TARGET].astype(int).values
    weights = df["sample_weight"].values

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    w_train = weights[train_idx]

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, preds))
    gini = 2 * auc - 1
    ll = float(log_loss(y_test, preds))
    ap = float(average_precision_score(y_test, preds))

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "fraud_model_pu.json")

    raw_importance = model.get_booster().get_score(importance_type="gain")
    total_gain = sum(raw_importance.values()) or 1.0
    feature_importance_pct = dict(
        sorted(
            {f: round(raw_importance.get(f, 0.0) / total_gain * 100, 2) for f in available_cols}.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    metrics = {
        "auc": round(auc, 4),
        "gini": round(gini, 4),
        "log_loss": round(ll, 4),
        "average_precision": round(ap, 4),
        "scale_pos_weight": round(spw, 2),
        "train_positives": pos,
        "train_negatives": neg,
        "feature_importance_pct": feature_importance_pct,
        "available_feature_cols": available_cols,
        "pu_label_adjusted": True,
    }
    (output_dir / "fraud_metrics_pu.json").write_text(json.dumps(metrics, indent=2))
    return metrics
