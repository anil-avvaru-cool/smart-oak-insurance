"""Stage 2c — Hurdle Model: Risk Score = P(claim) × E[cost | claim]."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from .train_frequency import QUOTE_FEATURE_COLS, prepare_features


class HurdleModel:
    """Combined frequency × severity risk scorer.

    Both sub-models share QUOTE_FEATURE_COLS so the combined risk score is
    computable at underwriting time from a single feature vector.
    """

    def __init__(self, model_dir: Path) -> None:
        self._freq = xgb.XGBClassifier()
        self._freq.load_model(model_dir / "frequency_model.json")
        self._sev = xgb.XGBRegressor()
        self._sev.load_model(model_dir / "severity_model.json")
        self._model_dir = model_dir
        self._freq_explainer: shap.TreeExplainer | None = None

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Return risk_score = P(claim) × E[cost | claim] for each row."""
        X = prepare_features(df)[QUOTE_FEATURE_COLS]
        p_claim = self._freq.predict_proba(X)[:, 1]
        e_cost = self._sev.predict(X)
        return p_claim * e_cost

    def score_with_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with p_claim, e_cost_usd, and risk_score columns."""
        X = prepare_features(df)[QUOTE_FEATURE_COLS]
        p_claim = self._freq.predict_proba(X)[:, 1]
        e_cost = self._sev.predict(X)
        return pd.DataFrame(
            {"p_claim": p_claim, "e_cost_usd": e_cost, "risk_score": p_claim * e_cost},
            index=df.index,
        )

    def explain(self, df: pd.DataFrame, top_n: int = 5) -> list[list[dict[str, Any]]]:
        """Return per-row SHAP reason codes from the frequency sub-model.

        Each entry is a list of up to *top_n* dicts sorted by absolute SHAP
        contribution descending::

            [{"feature": "prior_loss_frequency", "shap_pct": 34.1}, ...]

        SHAP values are in log-odds space (frequency model output); percentages
        are computed from absolute contributions so direction is preserved in the
        sign of shap_pct (positive = raises P(claim), negative = lowers it).
        """
        if self._freq_explainer is None:
            self._freq_explainer = shap.TreeExplainer(self._freq)

        X = prepare_features(df)[QUOTE_FEATURE_COLS]
        shap_values = self._freq_explainer.shap_values(X)  # shape (n, n_features)

        results: list[list[dict[str, Any]]] = []
        for row_shap in shap_values:
            abs_total = float(np.abs(row_shap).sum()) or 1.0
            ranked = sorted(
                enumerate(row_shap), key=lambda x: abs(x[1]), reverse=True
            )[:top_n]
            results.append(
                [
                    {
                        "feature": QUOTE_FEATURE_COLS[i],
                        "shap_pct": round(float(v) / abs_total * 100, 1),
                    }
                    for i, v in ranked
                ]
            )
        return results


def evaluate(quotes_path: Path, model_dir: Path) -> dict:
    """Score all quotes and report combined Gini + risk score distribution."""
    from sklearn.metrics import roc_auc_score

    df = pd.read_parquet(quotes_path)
    model = HurdleModel(model_dir)
    result = model.score_with_components(df)

    auc = float(roc_auc_score(df["claim_occurred"].astype(int), result["risk_score"]))
    gini = 2 * auc - 1

    metrics = {
        "hurdle_gini": round(gini, 4),
        "risk_score_mean": round(float(result["risk_score"].mean()), 2),
        "risk_score_p50": round(float(result["risk_score"].median()), 2),
        "risk_score_p95": round(float(result["risk_score"].quantile(0.95)), 2),
        "p_claim_mean": round(float(result["p_claim"].mean()), 4),
        "e_cost_mean_usd": round(float(result["e_cost_usd"].mean()), 2),
    }
    (model_dir / "hurdle_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
