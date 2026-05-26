"""Concept Drift Monitoring — CD-1 through CD-3 (DEC-013).

PSI catches population shift; concept drift monitors the *relationship*
between features and outcomes. Two signals are tracked:
  - Gini trend: champion model AUC/Gini on a rolling holdout cohort
  - Loss ratio by tier: incurred / risk_score_at_issuance per policy tier

A label-lag guard (CD-3) warns when too few fraud labels have confirmed
within the 90-day window — in that case, drift signals may be unreliable.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from data.config import (
    CLAIMS_OUTPUT,
    PSI_CURRENT_WINDOW_DAYS,
    PSI_MIN_RECORDS,
    QUOTES_OUTPUT,
    RISK_MODELS_DIR,
)
from monitoring.psi_drift import InsufficientDataError, LabelWindowReport
from underwriting.models.risk_scoring.train_frequency import (
    QUOTE_FEATURE_COLS,
    prepare_features,
)

logger = logging.getLogger(__name__)

# Thresholds from architecture doc
LOSS_RATIO_DRIFT_THRESHOLD: float = 0.10  # flag tiers where ratio drifts >10% from baseline
LABEL_LAG_WARNING_THRESHOLD: float = 0.50  # confirmation_rate_90d below this = unreliable

_GINI_HISTORY_FILE = "gini_history.json"


# ── CD-1: Gini trend tracker ──────────────────────────────────────────────────


def append_gini_entry(
    quotes_path: Path = QUOTES_OUTPUT,
    models_dir: Path = RISK_MODELS_DIR,
    output_dir: Path = RISK_MODELS_DIR,
    window_days: int = PSI_CURRENT_WINDOW_DAYS,
    min_records: int = PSI_MIN_RECORDS,
    as_of: pd.Timestamp | None = None,
) -> dict:
    """Compute Gini on the rolling holdout cohort and append to gini_history.json.

    Scores the current window of quotes with the champion frequency model and
    computes AUC/Gini against actual ``claim_occurred`` labels. Appends the
    result to ``output_dir/gini_history.json``, creating the file if absent.

    Returns the new entry dict.
    Raises FileNotFoundError if the champion model is absent.
    Raises InsufficientDataError if current window has fewer than min_records.
    """
    if as_of is None:
        as_of = pd.Timestamp.now()

    model_path = models_dir / "frequency_model.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Champion frequency model not found: {model_path}. Run --train-risk-model first."
        )

    df = pd.read_parquet(quotes_path)
    df = prepare_features(df)

    window_start = as_of - pd.Timedelta(days=window_days)
    ts = pd.to_datetime(df["quote_requested_at"])
    current = df[(ts >= window_start) & (ts <= as_of)]

    n_current = len(current)
    if n_current < min_records:
        raise InsufficientDataError(
            f"Gini holdout window {window_start.date()} → {as_of.date()} "
            f"has {n_current} records (minimum {min_records})."
        )

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    X = current[QUOTE_FEATURE_COLS]
    y = current["claim_occurred"].astype(int)

    preds = model.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, preds))
    gini = round(2 * auc - 1, 4)

    entry = {
        "computed_at": as_of.isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": as_of.isoformat(),
        "n_records": n_current,
        "auc": round(auc, 4),
        "gini": gini,
        "model_path": str(model_path),
    }

    history_path = output_dir / _GINI_HISTORY_FILE
    if history_path.exists():
        history = json.loads(history_path.read_text())
    else:
        history = {"schema_version": "1.0", "entries": []}

    history["entries"].append(entry)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))

    logger.info(
        "Gini history updated: window=%s→%s n=%d gini=%.4f",
        window_start.date(),
        as_of.date(),
        n_current,
        gini,
    )
    return entry


# ── CD-2: Loss ratio by tier ──────────────────────────────────────────────────


def compute_loss_ratio_by_tier(
    claims_path: Path = CLAIMS_OUTPUT,
    window_days: int = PSI_CURRENT_WINDOW_DAYS,
    as_of: pd.Timestamp | None = None,
) -> dict:
    """Compute actual loss ratio per policy_tier_at_issuance on closed claims.

    Reference cohort: closed claims whose fnol_submitted_at precedes the window.
    Current cohort: closed claims within the rolling window ending at as_of.
    Loss ratio = sum(incurred_loss_usd) / sum(risk_score_at_issuance) per tier.

    Flags tiers where the current ratio drifts more than LOSS_RATIO_DRIFT_THRESHOLD
    from the reference ratio as a fraction of the reference ratio.
    """
    if as_of is None:
        as_of = pd.Timestamp.now()

    df = pd.read_parquet(claims_path)
    closed = df[df["claim_closed_at"].notna()].copy()
    closed_fnol = pd.to_datetime(closed["fnol_submitted_at"])

    window_start = as_of - pd.Timedelta(days=window_days)
    ref = closed[closed_fnol < window_start]
    curr = closed[(closed_fnol >= window_start) & (closed_fnol <= as_of)]

    def _ratio_by_tier(subset: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        for tier, grp in subset.groupby("policy_tier_at_issuance"):
            denom = float(grp["risk_score_at_issuance"].sum())
            if denom > 0:
                out[str(tier)] = float(grp["incurred_loss_usd"].sum() / denom)
        return out

    ref_ratios = _ratio_by_tier(ref)
    curr_ratios = _ratio_by_tier(curr)

    tiers = sorted(set(ref_ratios) | set(curr_ratios))
    tier_details: list[dict] = []
    drifted_tiers: list[str] = []

    for tier in tiers:
        ref_r = ref_ratios.get(tier)
        curr_r = curr_ratios.get(tier)

        if ref_r is None:
            status = "insufficient_reference"
        elif curr_r is None:
            status = "no_current_data"
        else:
            relative_change = abs(curr_r - ref_r) / max(ref_r, 1e-9)
            if relative_change > LOSS_RATIO_DRIFT_THRESHOLD:
                status = "drifted"
                drifted_tiers.append(tier)
            else:
                status = "stable"

        tier_details.append(
            {
                "tier": tier,
                "reference_loss_ratio": round(ref_r, 4) if ref_r is not None else None,
                "current_loss_ratio": round(curr_r, 4) if curr_r is not None else None,
                "status": status,
            }
        )

    if drifted_tiers:
        logger.warning(
            "Loss ratio drift in %d tier(s): %s (threshold %.0f%%)",
            len(drifted_tiers),
            drifted_tiers,
            LOSS_RATIO_DRIFT_THRESHOLD * 100,
        )

    return {
        "as_of": as_of.isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": as_of.isoformat(),
        "n_reference_closed": len(ref),
        "n_current_closed": len(curr),
        "drift_threshold": LOSS_RATIO_DRIFT_THRESHOLD,
        "drifted_tiers": drifted_tiers,
        "tier_details": tier_details,
    }


# ── CD-3: Label lag warning ───────────────────────────────────────────────────


def check_label_lag(
    label_window: LabelWindowReport,
    threshold: float = LABEL_LAG_WARNING_THRESHOLD,
) -> dict:
    """Warn when the 90-day fraud confirmation rate is below threshold.

    A low rate means many recent fraud labels have not yet resolved —
    the Gini and loss ratio signals computed on this window may be noisy.
    """
    below = label_window.confirmation_rate_90d < threshold
    warning = (
        f"Label lag: confirmation_rate_90d={label_window.confirmation_rate_90d:.2%} "
        f"< {threshold:.0%} — concept drift signals may be unreliable."
        if below
        else None
    )
    if below:
        logger.warning(warning)

    return {
        "confirmation_rate_90d": label_window.confirmation_rate_90d,
        "confirmation_rate_180d": label_window.confirmation_rate_180d,
        "threshold": threshold,
        "below_threshold": below,
        "warning": warning,
    }


# ── CD-4: Public entry point ──────────────────────────────────────────────────


def run_concept_drift_check(
    quotes_path: Path = QUOTES_OUTPUT,
    claims_path: Path = CLAIMS_OUTPUT,
    models_dir: Path = RISK_MODELS_DIR,
    output_dir: Path = RISK_MODELS_DIR,
    window_days: int = PSI_CURRENT_WINDOW_DAYS,
    min_records: int = PSI_MIN_RECORDS,
    as_of: pd.Timestamp | None = None,
    label_window: LabelWindowReport | None = None,
) -> dict:
    """Run all concept drift checks and return a summary dict for the drift report.

    CD-1: Gini trend on rolling holdout cohort (appends to gini_history.json).
    CD-2: Loss ratio by tier on closed-claims window.
    CD-3: Label lag warning when confirmation_rate_90d is below threshold.

    Pass ``label_window`` from an already-computed PSI claims report to avoid
    re-reading claims.parquet. Gini errors are non-fatal and recorded in
    ``gini_error`` so loss ratio and label lag are always emitted.
    """
    if as_of is None:
        as_of = pd.Timestamp.now()

    gini_entry: dict | None = None
    gini_error: str | None = None
    try:
        gini_entry = append_gini_entry(
            quotes_path=quotes_path,
            models_dir=models_dir,
            output_dir=output_dir,
            window_days=window_days,
            min_records=min_records,
            as_of=as_of,
        )
    except Exception as exc:
        gini_error = str(exc)
        logger.warning("CD-1 Gini trend skipped: %s", exc)

    loss_ratio = compute_loss_ratio_by_tier(
        claims_path=claims_path,
        window_days=window_days,
        as_of=as_of,
    )

    label_lag: dict | None = None
    if label_window is not None:
        label_lag = check_label_lag(label_window)

    return {
        "as_of": as_of.isoformat(),
        "gini_trend": gini_entry,
        "gini_error": gini_error,
        "loss_ratio_by_tier": loss_ratio,
        "label_lag": label_lag,
    }
