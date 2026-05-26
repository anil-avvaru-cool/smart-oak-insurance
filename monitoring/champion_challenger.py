"""Champion-Challenger framework — four-stage loop: Shadow → Gini check → Phased rollout → Promote.

Challenger model artifact naming convention:
  frequency_model_challenger.json            frequency_calibration_challenger.json
  severity_model_challenger.json             severity_calibration_challenger.json
  (plus matching _metrics and _features variants)

Champion artifacts (always current production):
  frequency_model.json                       frequency_calibration.json
  severity_model.json                        severity_calibration.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from data.config import QUOTES_OUTPUT, RISK_MODELS_DIR
from underwriting.models.risk_scoring.train_frequency import (
    QUOTE_FEATURE_COLS,
    prepare_features,
)

logger = logging.getLogger(__name__)

ModelSlot = Literal["champion", "challenger"]

# Minimum accumulated shadow records before Gini comparison is meaningful
_MIN_SHADOW_RECORDS = 1_000
# Challenger Gini must be at most this many points below champion Gini to pass
_GINI_PASS_DELTA = -0.02

# Challenger → champion name mapping (same index positions)
_CHAMPION_ARTIFACTS = [
    "frequency_model.json",
    "frequency_calibration.json",
    "frequency_features.json",
    "frequency_metrics.json",
    "frequency_calibration_metrics.json",
    "severity_model.json",
    "severity_calibration.json",
    "severity_metrics.json",
    "severity_calibration_metrics.json",
    "hurdle_metrics.json",
]
_CHALLENGER_ARTIFACTS = [
    "frequency_model_challenger.json",
    "frequency_calibration_challenger.json",
    "frequency_features_challenger.json",
    "frequency_metrics_challenger.json",
    "frequency_calibration_metrics_challenger.json",
    "severity_model_challenger.json",
    "severity_calibration_challenger.json",
    "severity_metrics_challenger.json",
    "severity_calibration_metrics_challenger.json",
    "hurdle_metrics_challenger.json",
]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _load_freq_model(models_dir: Path, slot: ModelSlot) -> xgb.XGBClassifier:
    suffix = "" if slot == "champion" else "_challenger"
    path = models_dir / f"frequency_model{suffix}.json"
    if not path.exists():
        raise FileNotFoundError(f"{slot} frequency model not found: {path}")
    m = xgb.XGBClassifier()
    m.load_model(path)
    return m


def _apply_platt(raw_probs: np.ndarray, cal_path: Path) -> np.ndarray:
    cal = json.loads(cal_path.read_text())
    a, b = cal["a"], cal["b"]
    clipped = np.clip(raw_probs, 1e-9, 1.0 - 1e-9)
    logit = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-(a * logit + b)))


def _calibrated_p_claim(X: pd.DataFrame, models_dir: Path, slot: ModelSlot) -> np.ndarray:
    model = _load_freq_model(models_dir, slot)
    raw = model.predict_proba(X)[:, 1]
    suffix = "" if slot == "champion" else "_challenger"
    cal_path = models_dir / f"frequency_calibration{suffix}.json"
    if cal_path.exists():
        return _apply_platt(raw, cal_path)
    return raw


def _gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return 2.0 * float(roc_auc_score(y_true, y_score)) - 1.0


def _append_event(output_dir: Path, event: dict) -> None:
    """Append a structured event to challenger_events.jsonl."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "challenger_events.jsonl"
    with log_path.open("a") as fh:
        fh.write(json.dumps(event) + "\n")


# ── CC-1: Shadow mode scoring ────────────────────────────────────────────────


def score_shadow(
    quotes_path: Path = QUOTES_OUTPUT,
    models_dir: Path = RISK_MODELS_DIR,
    output_dir: Path = RISK_MODELS_DIR,
    triggered_by: str = "drift_check",
) -> Path:
    """Run challenger model on all quotes; write observation-only predictions parquet.

    Does NOT route live traffic. Outputs accumulate for later Gini comparison (CC-2).
    Returns the path of the written parquet file.

    Raises FileNotFoundError if frequency_model_challenger.json is absent.
    """
    chal_model_path = models_dir / "frequency_model_challenger.json"
    if not chal_model_path.exists():
        raise FileNotFoundError(
            f"Challenger model not found: {chal_model_path}. "
            "Train a challenger model and save it as frequency_model_challenger.json."
        )

    quotes_df = pd.read_parquet(quotes_path)
    df = prepare_features(quotes_df)
    X = df[QUOTE_FEATURE_COLS]

    p_claim = _calibrated_p_claim(X, models_dir, "challenger")

    ts = pd.Timestamp.now()
    out = pd.DataFrame(
        {
            "quote_id": quotes_df["quote_id"].values,
            "challenger_p_claim": p_claim,
            "scored_at": ts,
            "triggered_by": triggered_by,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    archived_dir = output_dir / "archived"
    archived_dir.mkdir(exist_ok=True)
    for old in output_dir.glob("challenger_predictions_*.parquet"):
        shutil.move(str(old), archived_dir / old.name)

    ts_str = ts.strftime("%Y%m%dT%H%M%S%f")[:17]
    out_path = output_dir / f"challenger_predictions_{ts_str}.parquet"
    out.to_parquet(out_path, index=False)

    logger.info("Shadow scoring complete: %d quotes → %s", len(out), out_path)
    _append_event(
        output_dir,
        {
            "event": "shadow_scoring",
            "triggered_by": triggered_by,
            "n_quotes": len(out),
            "output": str(out_path),
            "ts": ts.isoformat(),
        },
    )
    return out_path


# ── CC-2: Gini comparison harness ───────────────────────────────────────────


def compare_gini(
    quotes_path: Path = QUOTES_OUTPUT,
    models_dir: Path = RISK_MODELS_DIR,
    output_dir: Path = RISK_MODELS_DIR,
    min_shadow_records: int = _MIN_SHADOW_RECORDS,
    gini_pass_delta: float = _GINI_PASS_DELTA,
) -> dict:
    """Compare challenger Gini against champion Gini on the accumulated shadow cohort.

    Loads all challenger_predictions_*.parquet files, joins to actual outcomes from
    quotes.parquet, and scores the same cohort with the champion model.

    Returns a dict with: passed, champion_gini, challenger_gini, delta, n_records.
    Raises FileNotFoundError if no shadow files exist.
    Raises ValueError if too few records or single-class cohort.
    """
    archived_dir = output_dir / "archived"
    shadow_files = sorted(
        list(output_dir.glob("challenger_predictions_*.parquet")) +
        list(archived_dir.glob("challenger_predictions_*.parquet") if archived_dir.exists() else [])
    )
    if not shadow_files:
        raise FileNotFoundError(
            "No shadow prediction files found. Run --drift-check to accumulate shadow data."
        )

    shadow_df = pd.concat(
        [pd.read_parquet(f) for f in shadow_files], ignore_index=True
    )
    shadow_df = shadow_df.drop_duplicates(subset=["quote_id"], keep="last")

    if len(shadow_df) < min_shadow_records:
        raise ValueError(
            f"Only {len(shadow_df)} shadow records (minimum {min_shadow_records}). "
            "Accumulate more shadow data before comparing Gini."
        )

    quotes_df = pd.read_parquet(quotes_path)
    cohort_quotes = quotes_df[quotes_df["quote_id"].isin(shadow_df["quote_id"])].copy()
    df_features = prepare_features(cohort_quotes)

    p_champ = _calibrated_p_claim(df_features[QUOTE_FEATURE_COLS], models_dir, "champion")
    df_features = df_features.copy()
    df_features["champion_p_claim"] = p_champ

    merged = shadow_df.merge(
        df_features[["quote_id", "champion_p_claim", "claim_occurred"]],
        on="quote_id",
        how="inner",
    )

    y_true = merged["claim_occurred"].astype(int).values
    y_champ = merged["champion_p_claim"].values
    y_chal = merged["challenger_p_claim"].values

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        raise ValueError("Shadow cohort has only one class — cannot compute Gini.")

    champion_gini = _gini(y_true, y_champ)
    challenger_gini = _gini(y_true, y_chal)
    delta = challenger_gini - champion_gini
    passed = delta >= gini_pass_delta

    result = {
        "passed": passed,
        "champion_gini": round(champion_gini, 4),
        "challenger_gini": round(challenger_gini, 4),
        "delta": round(delta, 4),
        "n_records": int(len(merged)),
        "n_shadow_files": len(shadow_files),
        "gini_pass_delta_threshold": gini_pass_delta,
        "ts": pd.Timestamp.now().isoformat(),
    }

    _append_event(output_dir, {"event": "gini_comparison", **result})
    logger.info(
        "Gini comparison: champion=%.4f challenger=%.4f delta=%.4f → %s",
        champion_gini,
        challenger_gini,
        delta,
        "PASS" if passed else "FAIL",
    )
    return result


# ── CC-3: Traffic routing ────────────────────────────────────────────────────


def route_quote(quote_id: str, traffic_split_pct: int | None = None) -> ModelSlot:
    """Deterministically route a quote to champion or challenger via SHA-256 hash.

    traffic_split_pct=0 (shadow mode) always returns "champion" for actual serving.
    Reads CC_TRAFFIC_SPLIT_PCT from data.config when traffic_split_pct is None.
    """
    if traffic_split_pct is None:
        from data.config import CC_TRAFFIC_SPLIT_PCT
        traffic_split_pct = CC_TRAFFIC_SPLIT_PCT
    if traffic_split_pct == 0:
        return "champion"
    digest = int(hashlib.sha256(str(quote_id).encode()).hexdigest(), 16)
    return "challenger" if (digest % 100) < traffic_split_pct else "champion"


def log_routing_event(
    quote_id: str,
    model_used: ModelSlot,
    output_dir: Path = RISK_MODELS_DIR,
) -> None:
    """Append a routing decision row to routing_log.parquet."""
    log_path = output_dir / "routing_log.parquet"
    row = pd.DataFrame(
        [{"quote_id": quote_id, "model_used": model_used, "routed_at": pd.Timestamp.now()}]
    )
    if log_path.exists():
        existing = pd.read_parquet(log_path)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_parquet(log_path, index=False)


# ── CC-4: Promote workflow ───────────────────────────────────────────────────


def promote_challenger(
    models_dir: Path = RISK_MODELS_DIR,
    output_dir: Path = RISK_MODELS_DIR,
) -> dict:
    """Copy challenger artifacts to champion slot; archive previous champion with timestamp suffix.

    For each pair in (_CHALLENGER_ARTIFACTS, _CHAMPION_ARTIFACTS):
      1. Archive existing champion as <stem>_archived_<ts><ext>
      2. Copy challenger → champion name

    Returns a summary dict: ts, promoted, archived, missing_challenger.
    Missing challenger artifacts are skipped with a warning — does not raise.
    """
    ts_str = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    archived: list[str] = []
    promoted: list[str] = []
    missing_challenger: list[str] = []

    for champ_name, chal_name in zip(_CHAMPION_ARTIFACTS, _CHALLENGER_ARTIFACTS):
        champ_path = models_dir / champ_name
        chal_path = models_dir / chal_name

        if not chal_path.exists():
            missing_challenger.append(chal_name)
            continue

        if champ_path.exists():
            stem, ext = champ_path.stem, champ_path.suffix
            archive_path = models_dir / f"{stem}_archived_{ts_str}{ext}"
            shutil.copy2(champ_path, archive_path)
            archived.append(archive_path.name)

        shutil.copy2(chal_path, champ_path)
        promoted.append(champ_name)

    if missing_challenger:
        logger.warning(
            "Promotion skipped for %d missing challenger artifacts: %s",
            len(missing_challenger),
            missing_challenger,
        )

    result = {
        "ts": ts_str,
        "promoted": promoted,
        "archived": archived,
        "missing_challenger": missing_challenger,
    }
    _append_event(output_dir, {"event": "promotion", **result})
    logger.info(
        "Promotion complete: %d artifacts promoted, %d archived", len(promoted), len(archived)
    )
    return result
