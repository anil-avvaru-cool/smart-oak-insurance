"""SHAP Stability Monitoring — SH-1 through SH-2 (DEC-013).

Tracks global mean absolute SHAP values for the frequency model across
retraining cycles to detect feature-importance brittleness and drift.

Brittleness threshold (from architecture doc): any single feature whose
SHAP share exceeds 40% of total importance triggers a warning. A model
dominated by one signal is fragile — regulatory review or data change
targeting that feature can collapse its Gini overnight.

Snapshot schema (shap_snapshot_<ts>.json):
  schema_version, generated_at, model, n_samples,
  features, mean_abs_shap, shap_share, ranked_features,
  max_shap_share_feature, max_shap_share, brittleness_warning
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from data.config import QUOTES_OUTPUT, RISK_MODELS_DIR
from underwriting.models.risk_scoring.train_frequency import (
    QUOTE_FEATURE_COLS,
    prepare_features,
)

logger = logging.getLogger(__name__)

SHAP_BRITTLENESS_THRESHOLD: float = 0.40
_SHAP_SAMPLE_SIZE: int = 5_000


# ── SH-1: Snapshot writer ────────────────────────────────────────────────────


def write_shap_snapshot(
    quotes_path: Path = QUOTES_OUTPUT,
    models_dir: Path = RISK_MODELS_DIR,
    output_dir: Path = RISK_MODELS_DIR,
    sample_size: int = _SHAP_SAMPLE_SIZE,
) -> Path:
    """Compute global mean |SHAP| values for the champion frequency model.

    Uses TreeExplainer (exact, no approximation) on a stratified sample of
    quotes to bound runtime. Writes ``shap_snapshot_<ts>.json`` to
    ``output_dir``.

    Returns the path of the written snapshot.
    Raises FileNotFoundError if frequency_model.json is absent.
    """
    model_path = models_dir / "frequency_model.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Frequency model not found: {model_path}. Run --train-risk-model first."
        )

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    df = pd.read_parquet(quotes_path)
    df = prepare_features(df)
    X = df[QUOTE_FEATURE_COLS]

    if len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=42).reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_matrix = explainer.shap_values(X)  # (n_samples, n_features)

    mean_abs = np.abs(shap_matrix).mean(axis=0)
    total = float(mean_abs.sum()) or 1.0
    shares = (mean_abs / total).tolist()

    ranked = sorted(
        zip(QUOTE_FEATURE_COLS, mean_abs.tolist(), shares),
        key=lambda t: t[2],
        reverse=True,
    )
    ranked_features = [
        {"feature": f, "mean_abs_shap": round(m, 8), "shap_share": round(s, 6)}
        for f, m, s in ranked
    ]

    max_share = float(max(shares))
    max_feature = QUOTE_FEATURE_COLS[int(np.argmax(shares))]
    brittleness = max_share > SHAP_BRITTLENESS_THRESHOLD

    if brittleness:
        logger.warning(
            "SHAP brittleness: %s has %.1f%% share (threshold %.0f%%)",
            max_feature, max_share * 100, SHAP_BRITTLENESS_THRESHOLD * 100,
        )

    ts = pd.Timestamp.now()
    snapshot = {
        "schema_version": "1.0",
        "generated_at": ts.isoformat(),
        "model": "frequency_model",
        "model_path": str(model_path),
        "n_samples": int(len(X)),
        "features": QUOTE_FEATURE_COLS,
        "mean_abs_shap": [round(v, 8) for v in mean_abs.tolist()],
        "shap_share": [round(v, 6) for v in shares],
        "ranked_features": ranked_features,
        "max_shap_share_feature": max_feature,
        "max_shap_share": round(max_share, 6),
        "brittleness_warning": brittleness,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ts_str = ts.strftime("%Y%m%dT%H%M%S%f")[:17]
    out_path = output_dir / f"shap_snapshot_{ts_str}.json"
    out_path.write_text(json.dumps(snapshot, indent=2))

    logger.info(
        "SHAP snapshot written: n_samples=%d max_feature=%s max_share=%.1f%% → %s",
        len(X), max_feature, max_share * 100, out_path,
    )
    return out_path


# ── SH-2: Drift check ────────────────────────────────────────────────────────


def compare_shap_snapshots(
    output_dir: Path = RISK_MODELS_DIR,
    brittleness_threshold: float = SHAP_BRITTLENESS_THRESHOLD,
) -> dict:
    """Compare the two most recent SHAP snapshots.

    Returns a comparison dict with per-feature share deltas, rank shifts,
    and brittleness warnings on the current snapshot.

    Raises FileNotFoundError if fewer than two snapshots exist.
    """
    snapshot_files = sorted(output_dir.glob("shap_snapshot_*.json"))
    if len(snapshot_files) < 2:
        raise FileNotFoundError(
            f"Need at least 2 SHAP snapshots for comparison; found {len(snapshot_files)}. "
            "Run --train-risk-model to generate a new snapshot."
        )

    prev = json.loads(snapshot_files[-2].read_text())
    curr = json.loads(snapshot_files[-1].read_text())

    prev_shares: dict[str, float] = dict(zip(prev["features"], prev["shap_share"]))
    curr_shares: dict[str, float] = dict(zip(curr["features"], curr["shap_share"]))
    prev_rank: dict[str, int] = {
        e["feature"]: i + 1 for i, e in enumerate(prev["ranked_features"])
    }
    curr_rank: dict[str, int] = {
        e["feature"]: i + 1 for i, e in enumerate(curr["ranked_features"])
    }

    all_features = sorted(set(prev_shares) | set(curr_shares))

    feature_comparison = sorted(
        [
            {
                "feature": f,
                "prev_share": round(prev_shares.get(f, 0.0), 6),
                "curr_share": round(curr_shares.get(f, 0.0), 6),
                "delta": round(curr_shares.get(f, 0.0) - prev_shares.get(f, 0.0), 6),
                "prev_rank": prev_rank.get(f),
                "curr_rank": curr_rank.get(f),
                "rank_shift": (
                    (prev_rank.get(f) or 0) - (curr_rank.get(f) or 0)
                ),
            }
            for f in all_features
        ],
        key=lambda x: x["curr_share"],
        reverse=True,
    )

    warnings: list[str] = []
    for f, share in curr_shares.items():
        if share > brittleness_threshold:
            warnings.append(
                f"BRITTLENESS: {f} has SHAP share {share:.1%} "
                f"(threshold {brittleness_threshold:.0%})"
            )

    result = {
        "previous_snapshot": snapshot_files[-2].name,
        "current_snapshot": snapshot_files[-1].name,
        "previous_generated_at": prev["generated_at"],
        "current_generated_at": curr["generated_at"],
        "previous_n_samples": prev["n_samples"],
        "current_n_samples": curr["n_samples"],
        "max_shap_share_feature": curr["max_shap_share_feature"],
        "max_shap_share": curr["max_shap_share"],
        "brittleness_warning": curr["brittleness_warning"],
        "warnings": warnings,
        "feature_comparison": feature_comparison,
        "ts": pd.Timestamp.now().isoformat(),
    }

    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info(
            "SHAP check: no brittleness. max_share=%s=%.1f%%",
            curr["max_shap_share_feature"], curr["max_shap_share"] * 100,
        )

    return result
