"""Stage 2b — Severity model: E[cost | claim] via XGBoost Gamma objective."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from .train_frequency import QUOTE_FEATURE_COLS, prepare_features

TARGET = "incurred_loss_usd"


def train(quotes_path: Path, claims_path: Path, output_dir: Path) -> dict:
    quotes = pd.read_parquet(quotes_path)
    claims = pd.read_parquet(claims_path)

    # Join to quote-era features — severity at underwriting uses only pre-claim
    # information so the hurdle product is computable at quote time.
    df = claims[["quote_id", TARGET]].merge(
        quotes[["quote_id", "vehicle_msrp", "vehicle_power"] + [
            c for c in QUOTE_FEATURE_COLS
            if c not in ("vehicle_msrp_power_ratio", "telematics_enrolled")
        ] + ["telematics_enrolled"]],
        on="quote_id",
        how="inner",
    )
    df = prepare_features(df)

    X = df[QUOTE_FEATURE_COLS]
    y = df[TARGET]

    # Severity model watch-out: ~1,600 claim records at 20K quotes / 8% claim rate.
    # Use 60/20/20 (not 80/20) so the val set catches Gamma divergence early.
    # If train_rows < 960, bump quotes volume to 50K before tuning hyperparameters.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42  # 0.25 × 0.8 = 0.2
    )

    if len(X_train) < 960:
        import warnings
        warnings.warn(
            f"Severity model has only {len(X_train)} training rows "
            f"(expected ≥960). Bump quotes volume to 50K before adjusting "
            f"hyperparameters — the bottleneck is data, not the model.",
            stacklevel=2,
        )

    model = xgb.XGBRegressor(
        objective="reg:gamma",
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        eval_metric="gamma-deviance",
        early_stopping_rounds=30,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds = model.predict(X_test)
    mae = float(np.mean(np.abs(y_test.values - preds)))
    rmse = float(np.sqrt(np.mean((y_test.values - preds) ** 2)))
    mape = float(np.mean(np.abs((y_test.values - preds) / y_test.values)))

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "severity_model.json")

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
        "mae_usd": round(mae, 2),
        "rmse_usd": round(rmse, 2),
        "mape": round(mape, 4),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "best_iteration": int(model.best_iteration),
        "low_data_warning": len(X_train) < 960,
        "target_mean_usd": round(float(y.mean()), 2),
        "feature_importance_pct": feature_importance_pct,
    }
    (output_dir / "severity_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
