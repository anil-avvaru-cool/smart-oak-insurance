"""Stage 2a — Frequency model: P(claim > 0) via XGBoost binary logistic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Quote-era features consumed by both sub-models so the hurdle product
# P(claim) × E[cost | claim] is computable at underwriting time from a
# single feature vector.
QUOTE_FEATURE_COLS: list[str] = [
    "credit_score",               # null in CA/MA/MI/HI — XGBoost native null path
    "prior_loss_frequency",
    "prior_loss_severity_avg",
    "insurance_lapse_days",
    "violation_severity_index",
    "household_driver_density",
    "driver_age",
    "years_licensed",
    "vehicle_msrp_power_ratio",   # derived below; matches feature_definitions.py
    "vehicle_adas_score",
    "vehicle_age_years",
    "geohash_risk_score",
    "annual_mileage_estimate",
    "telematics_distraction_score",    # null for non-enrolled
    "telematics_hard_brake_rate",      # null for non-enrolled
    "telematics_crash_match",          # null for non-enrolled
    "telematics_commute_entropy",      # null for non-enrolled
    "telematics_enrolled",
]

TARGET = "claim_occurred"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive computed columns and cast types to match the feature store contract."""
    out = df.copy()
    vehicle_power = out["vehicle_power"].clip(lower=1.0)
    out["vehicle_msrp_power_ratio"] = out["vehicle_msrp"] / vehicle_power
    out["telematics_enrolled"] = out["telematics_enrolled"].astype(float)
    return out


def train(quotes_path: Path, output_dir: Path) -> dict:
    df = pd.read_parquet(quotes_path)
    df = prepare_features(df)

    X = df[QUOTE_FEATURE_COLS]
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

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "frequency_model.json")

    raw_importance = model.get_booster().get_score(importance_type="gain")
    total_gain = sum(raw_importance.values()) or 1.0
    feature_importance_pct = dict(
        sorted(
            {f: round(raw_importance.get(f, 0.0) / total_gain * 100, 2) for f in QUOTE_FEATURE_COLS}.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    metrics = {
        "auc": round(auc, 4),
        "gini": round(gini, 4),
        "log_loss": round(ll, 4),
        "scale_pos_weight": round(scale_pos_weight, 2),
        "train_positives": pos,
        "train_negatives": neg,
        "feature_importance_pct": feature_importance_pct,
    }
    (output_dir / "frequency_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "frequency_features.json").write_text(
        json.dumps(QUOTE_FEATURE_COLS, indent=2)
    )
    return metrics
