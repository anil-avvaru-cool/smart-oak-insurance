"""Stage 5 — Label Quality & PU Learning.

Architecture (Section 5):
  PU Learning        → treats negatives as "unlabeled," not confirmed clean
                       (Elkan-Noto two-step: de-bias P(s=1|x) → P(y=1|x))
  Confident Learning → identifies likely mislabeled examples via OOF calibration
                       (cleanlab-style: off-diagonal in per-class threshold table)
  Label Smoothing    → soft targets reduce overconfidence on noisy negatives
  Delayed labels     → confirmed fraud keyed on fraud_confirmed_at (90/180d windows)

Outputs (all written to FRAUD_MODELS_DIR):
  pu_adjusted_labels.parquet  — per-claim: pu_score, is_fraud_soft, sample_weight,
                                 suspected_mislabel, confirmed_fraud flags
  label_quality_report.json   — noise rate estimates, c_estimate, counts
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .train_xgb_fraud import (
    CLAIM_FRAUD_FEATURE_COLS,
    TARGET,
    _ensure_risk_score,
    prepare_fraud_features,
)

# Default confirmed-fraud window: SIU closures within 90 days of FNOL submission.
# Claims with fraud_confirmed_at > 90d post-FNOL are treated as "unlabeled"
# for training purposes — the label may arrive but is outside the training window.
CONFIRMED_WINDOW_DAYS = 90

# Label smoothing strength: 0.05 → positives become 0.95, negatives become 0.05.
# Suspected mislabels are assigned 0.5 (maximum uncertainty).
LABEL_SMOOTH_ALPHA = 0.05

# PU Learning XGBoost hyperparameters (lighter than the production model;
# purpose is to estimate the label frequency c, not to be the final scorer).
_PU_XGB_PARAMS: dict = dict(
    objective="binary:logistic",
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    verbosity=0,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def build_confirmed_label_window(
    df: pd.DataFrame,
    window_days: int = CONFIRMED_WINDOW_DAYS,
) -> pd.Series:
    """Boolean mask: is_fraud=1 AND fraud_confirmed_at within window_days of fnol.

    Claims outside this window may still be fraud — they are simply not yet
    confirmed within the training cohort cut. They retain is_fraud=1 labels but
    receive normal (not boosted) sample weights.
    """
    if "fraud_confirmed_at" not in df.columns or "fnol_submitted_at" not in df.columns:
        return pd.Series(False, index=df.index)

    fca = pd.to_datetime(df["fraud_confirmed_at"], errors="coerce")
    fnol = pd.to_datetime(df["fnol_submitted_at"], errors="coerce")
    days_to_confirm = (fca - fnol).dt.days
    return (df[TARGET] == 1) & fca.notna() & (days_to_confirm <= window_days)


def _train_oof_xgb(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Train XGBoost via stratified K-fold and return OOF predicted probabilities."""
    oof = np.zeros(len(y))
    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    spw = neg / max(pos, 1)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in kfold.split(X, y):
        m = xgb.XGBClassifier(scale_pos_weight=spw, **_PU_XGB_PARAMS)
        m.fit(X.iloc[train_idx], y[train_idx], verbose=False)
        oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
    return oof


# ── Core algorithms ───────────────────────────────────────────────────────────


def run_pu_learning(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, float]:
    """Elkan-Noto PU Learning: estimate true P(y=1|x) when negatives are unlabeled.

    SCAR assumption: labeled positives are selected completely at random from
    all true positives (reasonable for SIU investigation selection in insurance).

    Algorithm:
      Step 1 — train f(x) = P(s=1|x) on biased dataset (s = observed label)
               via OOF cross-validation to avoid overfitting the de-bias estimate.
      Step 2 — estimate c = E[f(x) | x in P] (label frequency among true positives).
      Step 3 — de-bias: P(y=1|x) = min(1, P(s=1|x) / c).

    Returns:
        pu_scores:   de-biased fraud probability [0,1] for every record
        c_estimate:  estimated P(s=1|y=1) — proportion of true positives labeled
    """
    oof = _train_oof_xgb(X, y, n_splits=n_splits)
    # c = average predicted probability among confirmed positives (the P set)
    c_estimate = float(oof[y == 1].mean())
    c_estimate = max(c_estimate, 0.01)  # guard against degenerate c near 0
    pu_scores = np.clip(oof / c_estimate, 0.0, 1.0)
    return pu_scores, c_estimate


def run_confident_learning(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Confident Learning (cleanlab-style) via OOF probability calibration.

    For each class, the per-class average OOF probability is used as a
    threshold. Records whose predicted class disagrees with their given label
    above threshold are flagged as suspected mislabels.

    Off-diagonal logic (2-class case):
      Label=0, P(fraud) > avg_P(fraud | labeled=1)  → suspected mislabeled negative
      Label=1, P(fraud) < avg_P(fraud | labeled=0)  → suspected mislabeled positive

    This is more conservative than full cleanlab (no count-matrix pruning) but
    requires only sklearn + xgboost and no external dependency.

    Returns:
        oof_proba:         OOF predicted P(fraud) for every record
        suspected_mislabel: boolean mask of likely label errors
    """
    oof_proba = _train_oof_xgb(X, y, n_splits=n_splits)

    # Per-class average OOF probability (used as decision thresholds)
    threshold_pos = float(oof_proba[y == 1].mean())  # what the model expects of positives
    threshold_neg = float(oof_proba[y == 0].mean())  # what the model expects of negatives

    mislabeled_neg = (y == 0) & (oof_proba > threshold_pos)
    mislabeled_pos = (y == 1) & (oof_proba < threshold_neg)

    return oof_proba, mislabeled_neg | mislabeled_pos


def apply_label_smoothing(
    y: np.ndarray,
    suspected_mislabel: np.ndarray,
    alpha: float = LABEL_SMOOTH_ALPHA,
) -> np.ndarray:
    """Produce soft label targets in [0, 1].

    Confirmed labels:      1 - alpha  or  alpha
    Suspected mislabels:   0.5  (maximum uncertainty — let the model decide)
    """
    y_smooth = np.where(y == 1, 1.0 - alpha, alpha).astype(float)
    y_smooth[suspected_mislabel] = 0.5
    return y_smooth


# ── Orchestrator ──────────────────────────────────────────────────────────────


def build_pu_adjusted_dataset(
    claims_path: Path,
    quotes_path: Path,
    output_dir: Path,
    confirmed_window_days: int = CONFIRMED_WINDOW_DAYS,
    label_smooth_alpha: float = LABEL_SMOOTH_ALPHA,
    n_cv_splits: int = 5,
) -> dict:
    """Run the full Label Quality pipeline and write artifacts.

    Pipeline:
      1. Mark confirmed fraud via fraud_confirmed_at within confirmed_window_days.
      2. PU Learning → de-biased P(y=1|x) for each record.
      3. Confident Learning → suspected mislabels via OOF calibration.
      4. Label smoothing → soft targets in [0, 1].
      5. Sample weights: confirmed=2.0, suspected_mislabel=0.5, others=1.0.

    Artifacts written to output_dir:
      pu_adjusted_labels.parquet
      label_quality_report.json

    Returns the label quality report dict.
    """
    df = pd.read_parquet(claims_path)
    df = _ensure_risk_score(df, quotes_path)
    df = prepare_fraud_features(df).dropna(subset=[TARGET]).reset_index(drop=True)

    available_cols = [c for c in CLAIM_FRAUD_FEATURE_COLS if c in df.columns]
    X = df[available_cols]
    y = df[TARGET].astype(int).values

    # Step 1: confirmed fraud window
    confirmed_mask = build_confirmed_label_window(df, confirmed_window_days)
    n_confirmed = int(confirmed_mask.sum())

    print(f"  [5a] Confirmed fraud within {confirmed_window_days}d window: {n_confirmed} / {int((y==1).sum())} positives")

    # Step 2: PU Learning
    print("  [5b] PU Learning: estimating de-biased P(y=1|x) via OOF…")
    pu_scores, c_estimate = run_pu_learning(X, y, n_splits=n_cv_splits)
    n_high_pu_neg = int((pu_scores[y == 0] > 0.5).sum())
    print(f"       c_estimate = {c_estimate:.4f}  (label frequency among true positives)")
    print(f"       negatives with PU score > 0.5: {n_high_pu_neg} (likely undetected fraud)")

    # Step 3: Confident Learning
    print("  [5c] Confident Learning: identifying suspected mislabels via OOF calibration…")
    oof_proba, suspected_mislabel = run_confident_learning(X, y, n_splits=n_cv_splits)
    n_mislabel_neg = int((suspected_mislabel & (y == 0)).sum())
    n_mislabel_pos = int((suspected_mislabel & (y == 1)).sum())
    noise_rate_neg = n_mislabel_neg / max(int((y == 0).sum()), 1)
    print(f"       suspected mislabeled negatives: {n_mislabel_neg}  (noise rate: {noise_rate_neg:.2%})")
    print(f"       suspected mislabeled positives: {n_mislabel_pos}")

    # OOF AUC diagnostic
    oof_auc = float(roc_auc_score(y, oof_proba))
    print(f"       OOF AUC on hard labels: {oof_auc:.4f}")

    # Step 4: Label smoothing
    is_fraud_soft = apply_label_smoothing(y, suspected_mislabel, alpha=label_smooth_alpha)

    # Step 5: Sample weights
    sample_weight = np.ones(len(y), dtype=float)
    sample_weight[confirmed_mask.values] = 2.0      # confirmed fraud: higher confidence
    sample_weight[suspected_mislabel] = 0.5          # suspected mislabels: penalize but retain

    # Write label artifact
    output_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(
        {
            "claim_id": df["claim_id"],
            "is_fraud_orig": y,
            "pu_score": pu_scores.round(4),
            "oof_proba": oof_proba.round(4),
            "is_fraud_soft": is_fraud_soft.round(4),
            "sample_weight": sample_weight,
            "suspected_mislabel": suspected_mislabel,
            "confirmed_fraud": confirmed_mask.values,
        }
    )
    out_df.to_parquet(output_dir / "pu_adjusted_labels.parquet", index=False)

    report = {
        "n_records": int(len(y)),
        "n_positives": int((y == 1).sum()),
        "n_negatives": int((y == 0).sum()),
        "n_confirmed_fraud": n_confirmed,
        "confirmed_window_days": confirmed_window_days,
        "c_estimate": round(c_estimate, 4),
        "estimated_noise_rate_negatives": round(noise_rate_neg, 4),
        "n_suspected_mislabel_negatives": n_mislabel_neg,
        "n_suspected_mislabel_positives": n_mislabel_pos,
        "n_total_suspected_mislabels": int(suspected_mislabel.sum()),
        "n_high_pu_score_negatives": n_high_pu_neg,
        "oof_auc": round(oof_auc, 4),
        "label_smooth_alpha": label_smooth_alpha,
        "avg_pu_score_positives": round(float(pu_scores[y == 1].mean()), 4),
        "avg_pu_score_negatives": round(float(pu_scores[y == 0].mean()), 4),
        "avg_soft_label_positives": round(float(is_fraud_soft[y == 1].mean()), 4),
        "avg_soft_label_negatives": round(float(is_fraud_soft[y == 0].mean()), 4),
    }
    (output_dir / "label_quality_report.json").write_text(json.dumps(report, indent=2))
    return report
