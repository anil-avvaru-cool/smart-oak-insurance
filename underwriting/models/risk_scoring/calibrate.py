"""Stage 3 — Calibration: Platt-scale frequency model + bias-correct severity model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from .train_frequency import QUOTE_FEATURE_COLS, TARGET as FREQ_TARGET, prepare_features
from .train_severity import TARGET as SEV_TARGET


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


def _ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Expected Calibration Error + reliability diagram data (equal-width bins)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    diagram: list[dict] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() == 0:
            continue
        frac_pos = float(y_true[mask].mean())
        mean_pred = float(y_pred[mask].mean())
        count = int(mask.sum())
        ece += count * abs(frac_pos - mean_pred)
        diagram.append(
            {
                "bin_lower": round(float(lo), 2),
                "bin_upper": round(float(hi), 2),
                "predicted_freq": round(mean_pred, 4),
                "actual_freq": round(frac_pos, 4),
                "count": count,
            }
        )
    return ece / n, diagram


def calibrate_frequency(quotes_path: Path, model_dir: Path, output_dir: Path) -> dict:
    """Fit Platt scaling on the frequency model's held-out test set.

    Reproduces the same 80/20 train/test split (random_state=42) used in
    train_frequency so the calibration set is guaranteed clean.
    Writes frequency_calibration.json and frequency_calibration_metrics.json.
    """
    df = pd.read_parquet(quotes_path)
    df = prepare_features(df)

    X = df[QUOTE_FEATURE_COLS]
    y = df[FREQ_TARGET].astype(int)

    _, X_cal, _, y_cal = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if y_cal.sum() == 0 or y_cal.sum() == len(y_cal):
        raise ValueError(
            "Calibration set contains only one class — increase dataset size before calibrating."
        )

    freq_model = xgb.XGBClassifier()
    freq_model.load_model(model_dir / "frequency_model.json")

    p_raw = freq_model.predict_proba(X_cal)[:, 1]

    # Platt scaling: logistic regression on logit(p_raw) → true labels
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(p_raw).reshape(-1, 1), y_cal.values)
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])

    p_calibrated = 1.0 / (1.0 + np.exp(-(a * _logit(p_raw) + b)))

    brier_before = float(brier_score_loss(y_cal, p_raw))
    brier_after = float(brier_score_loss(y_cal, p_calibrated))
    ece_before, _ = _ece(y_cal.values, p_raw)
    ece_after, reliability_diagram = _ece(y_cal.values, p_calibrated)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frequency_calibration.json").write_text(
        json.dumps({"method": "platt", "a": round(a, 6), "b": round(b, 6)}, indent=2)
    )

    metrics = {
        "method": "platt",
        "platt_a": round(a, 6),
        "platt_b": round(b, 6),
        "brier_before": round(brier_before, 6),
        "brier_after": round(brier_after, 6),
        "brier_improvement_pct": round((brier_before - brier_after) / brier_before * 100, 2),
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
        "p_claim_mean_before": round(float(p_raw.mean()), 4),
        "p_claim_mean_after": round(float(p_calibrated.mean()), 4),
        "actual_claim_rate": round(float(y_cal.mean()), 4),
        "reliability_diagram": reliability_diagram,
    }
    (output_dir / "frequency_calibration_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    return metrics


def calibrate_severity(
    quotes_path: Path, claims_path: Path, model_dir: Path, output_dir: Path
) -> dict:
    """Fit multiplicative bias correction on the severity model's held-out test set.

    Reproduces the same 80/20 split (random_state=42) used in train_severity.
    Writes severity_calibration.json and severity_calibration_metrics.json.
    """
    quotes = pd.read_parquet(quotes_path)
    claims = pd.read_parquet(claims_path)

    df = claims[["quote_id", SEV_TARGET]].merge(
        quotes[
            ["quote_id", "vehicle_msrp", "vehicle_power"]
            + [
                c
                for c in QUOTE_FEATURE_COLS
                if c not in ("vehicle_msrp_power_ratio", "telematics_enrolled")
            ]
            + ["telematics_enrolled"]
        ],
        on="quote_id",
        how="inner",
    )
    df = prepare_features(df)

    X = df[QUOTE_FEATURE_COLS]
    y = df[SEV_TARGET]

    _, X_cal, _, y_cal = train_test_split(X, y, test_size=0.2, random_state=42)

    sev_model = xgb.XGBRegressor()
    sev_model.load_model(model_dir / "severity_model.json")

    p_raw = sev_model.predict(X_cal)
    mean_predicted = float(p_raw.mean())
    mean_actual = float(y_cal.mean())
    bias_correction = mean_actual / mean_predicted if mean_predicted > 0 else 1.0

    p_calibrated = p_raw * bias_correction

    mae_before = float(np.mean(np.abs(y_cal.values - p_raw)))
    mae_after = float(np.mean(np.abs(y_cal.values - p_calibrated)))
    rmse_before = float(np.sqrt(np.mean((y_cal.values - p_raw) ** 2)))
    rmse_after = float(np.sqrt(np.mean((y_cal.values - p_calibrated) ** 2)))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "severity_calibration.json").write_text(
        json.dumps(
            {"method": "multiplicative", "bias_correction": round(bias_correction, 6)},
            indent=2,
        )
    )

    metrics = {
        "method": "multiplicative",
        "bias_correction": round(bias_correction, 6),
        "mae_before_usd": round(mae_before, 2),
        "mae_after_usd": round(mae_after, 2),
        "rmse_before_usd": round(rmse_before, 2),
        "rmse_after_usd": round(rmse_after, 2),
        "mean_actual_usd": round(mean_actual, 2),
        "mean_predicted_before_usd": round(mean_predicted, 2),
        "mean_predicted_after_usd": round(float(p_calibrated.mean()), 2),
    }
    (output_dir / "severity_calibration_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    return metrics
