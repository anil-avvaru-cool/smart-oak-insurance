"""Unit tests for Section 4 Ensemble Fraud Scoring."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fraud.train_xgb_fraud import (
    CLAIM_FRAUD_FEATURE_COLS,
    prepare_fraud_features,
    train as train_xgb_fraud,
)
from fraud.anomaly_detector import train_anomaly_detector
from fraud.ensemble_fraud_scorer import (
    ACTIVE_SIGNALS,
    EnsembleFraudScorer,
    train_ensemble,
    train_stacking,
)


# ── Test data factory ─────────────────────────────────────────────────────────


def _make_claims(n: int = 300, seed: int = 0, fraud_rate: float = 0.33) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    is_fraud = (rng.random(n) < fraud_rate).astype(int)
    return pd.DataFrame(
        {
            "claim_id": [f"C-{i:05d}" for i in range(n)],
            "quote_id": [f"Q-{i:05d}" for i in range(n)],
            "is_fraud": is_fraud,
            "policy_inception_days": np.where(
                is_fraud, rng.integers(0, 30, n), rng.integers(30, 730, n)
            ).astype(float),
            "prior_claims_count": rng.integers(0, 5, n).astype(float),
            "reported_injury_count": rng.integers(0, 4, n).astype(float),
            "reporting_delay_days": rng.integers(0, 90, n).astype(float),
            "attorney_present": is_fraud.astype(bool),
            "claimant_count": rng.integers(1, 6, n).astype(float),
            "graph_hop_distance": np.where(
                is_fraud, rng.integers(1, 4, n).astype(float), np.nan
            ),
            "shared_attribute_count": np.where(
                is_fraud, rng.integers(1, 5, n), 0
            ).astype(float),
            "attorney_centrality_score": np.where(
                is_fraud, rng.uniform(0.3, 1.0, n), rng.uniform(0.0, 0.2, n)
            ),
            "narrative_inconsistency_score": rng.uniform(0, 1, n),
            "narrative_complexity_score": rng.uniform(0, 1, n),
            "risk_score_at_issuance": rng.uniform(500, 15_000, n),
            "ip_geolocation_delta_miles": np.where(
                is_fraud, rng.uniform(50, 500, n), rng.uniform(0, 20, n)
            ),
            "device_fingerprint_match": (~is_fraud.astype(bool)),
            "submission_hour": rng.integers(0, 24, n).astype(float),
            "telematics_distraction_score": np.where(
                rng.random(n) > 0.3, rng.uniform(0, 1, n), np.nan
            ),
            "telematics_hard_brake_rate": np.where(
                rng.random(n) > 0.3, rng.uniform(0, 1, n), np.nan
            ),
            "telematics_crash_match": np.where(
                rng.random(n) > 0.3, rng.uniform(0, 1, n), np.nan
            ),
            "telematics_commute_entropy": np.where(
                rng.random(n) > 0.3, rng.uniform(0, 1, n), np.nan
            ),
            "telematics_enrolled": rng.random(n) > 0.4,
        }
    )


def _make_quotes(n: int = 300, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "quote_id": [f"Q-{i:05d}" for i in range(n)],
            "risk_score_at_issuance": rng.uniform(500, 15_000, n),
        }
    )


def _write_data(tmp_path, n: int = 300):
    claims = _make_claims(n)
    quotes = _make_quotes(n)
    claims.to_parquet(tmp_path / "claims.parquet")
    quotes.to_parquet(tmp_path / "quotes.parquet")
    return claims, quotes


# ── Stage 4a: XGBoost fraud model ────────────────────────────────────────────


def test_xgb_fraud_trains_and_saves_artifacts(tmp_path):
    _write_data(tmp_path)
    metrics = train_xgb_fraud(tmp_path / "claims.parquet", tmp_path / "quotes.parquet", tmp_path)

    assert (tmp_path / "fraud_model.json").exists()
    assert (tmp_path / "fraud_metrics.json").exists()
    assert (tmp_path / "fraud_features.json").exists()
    assert 0.0 <= metrics["auc"] <= 1.0
    assert -1.0 <= metrics["gini"] <= 1.0
    assert metrics["log_loss"] > 0
    assert 0.0 <= metrics["average_precision"] <= 1.0


def test_xgb_fraud_feature_importance_sums_to_100(tmp_path):
    _write_data(tmp_path)
    metrics = train_xgb_fraud(tmp_path / "claims.parquet", tmp_path / "quotes.parquet", tmp_path)
    imp = metrics["feature_importance_pct"]
    assert abs(sum(imp.values()) - 100.0) < 0.5
    vals = list(imp.values())
    assert vals == sorted(vals, reverse=True)


def test_xgb_fraud_scale_pos_weight_reflects_imbalance(tmp_path):
    _write_data(tmp_path, n=300)
    metrics = train_xgb_fraud(tmp_path / "claims.parquet", tmp_path / "quotes.parquet", tmp_path)
    neg, pos = metrics["train_negatives"], metrics["train_positives"]
    expected_spw = round(neg / pos, 2)
    assert metrics["scale_pos_weight"] == pytest.approx(expected_spw, rel=0.01)


def test_prepare_fraud_features_casts_bools(tmp_path):
    claims, _ = _write_data(tmp_path)
    out = prepare_fraud_features(claims)
    assert out["attorney_present"].dtype in (float, np.float64)
    assert out["device_fingerprint_match"].dtype in (float, np.float64)
    assert out["telematics_enrolled"].dtype in (float, np.float64)


# ── Stage 4b: Anomaly detector ───────────────────────────────────────────────


def test_anomaly_detector_saves_artifacts(tmp_path):
    _write_data(tmp_path)
    metrics = train_anomaly_detector(
        tmp_path / "claims.parquet", tmp_path / "quotes.parquet", tmp_path
    )
    assert (tmp_path / "fraud_anomaly_model.joblib").exists()
    assert (tmp_path / "fraud_anomaly_medians.json").exists()
    assert (tmp_path / "fraud_anomaly_metrics.json").exists()
    assert 0.0 <= metrics["anomaly_score_mean"] <= 1.0
    assert 0.0 <= metrics["anomaly_score_p95"] <= 1.0


def test_anomaly_detector_contamination_clamped(tmp_path):
    claims, quotes = _make_claims(300, fraud_rate=0.99), _make_quotes(300)
    claims.to_parquet(tmp_path / "claims.parquet")
    quotes.to_parquet(tmp_path / "quotes.parquet")
    metrics = train_anomaly_detector(
        tmp_path / "claims.parquet", tmp_path / "quotes.parquet", tmp_path
    )
    assert metrics["contamination"] <= 0.5


# ── Stage 4c: Stacking meta-learner ──────────────────────────────────────────


def test_stacking_saves_meta_learner(tmp_path):
    _write_data(tmp_path)
    # Need base models first
    train_xgb_fraud(tmp_path / "claims.parquet", tmp_path / "quotes.parquet", tmp_path)
    train_anomaly_detector(tmp_path / "claims.parquet", tmp_path / "quotes.parquet", tmp_path)
    metrics = train_stacking(tmp_path / "claims.parquet", tmp_path / "quotes.parquet", tmp_path)

    assert (tmp_path / "fraud_meta_learner.json").exists()
    assert (tmp_path / "fraud_meta_metrics.json").exists()

    meta = json.loads((tmp_path / "fraud_meta_learner.json").read_text())
    assert meta["method"] == "logistic_regression"
    assert len(meta["coef"]) == 2
    assert meta["input_features"] == ACTIVE_SIGNALS
    assert meta["cv_folds"] == 5

    assert 0.0 <= metrics["oof_auc"] <= 1.0
    assert -1.0 <= metrics["oof_gini"] <= 1.0


# ── train_ensemble orchestrator ───────────────────────────────────────────────


def test_train_ensemble_returns_all_metrics(tmp_path):
    _write_data(tmp_path)
    all_metrics = train_ensemble(
        tmp_path / "claims.parquet", tmp_path / "quotes.parquet", tmp_path
    )
    assert "xgb_fraud" in all_metrics
    assert "anomaly_detector" in all_metrics
    assert "stacking_meta_learner" in all_metrics


# ── EnsembleFraudScorer inference ─────────────────────────────────────────────


def _train_all(tmp_path, n: int = 300):
    _write_data(tmp_path, n)
    train_ensemble(tmp_path / "claims.parquet", tmp_path / "quotes.parquet", tmp_path)


def test_scorer_loads_and_scores(tmp_path):
    _train_all(tmp_path)
    claims = _make_claims(50)
    scorer = EnsembleFraudScorer(tmp_path)
    result = scorer.score(claims)

    assert set(result.columns) == {
        "fraud_score", "ci_lower", "ci_upper",
        "p_xgb", "p_anomaly", "p_gnn", "p_nlp", "p_vision",
    }
    assert result.shape == (50, 8)


def test_scorer_fraud_score_bounded(tmp_path):
    _train_all(tmp_path)
    claims = _make_claims(100)
    scorer = EnsembleFraudScorer(tmp_path)
    result = scorer.score(claims)

    assert (result["fraud_score"] >= 0.0).all()
    assert (result["fraud_score"] <= 1.0).all()
    assert (result["ci_lower"] >= 0.0).all()
    assert (result["ci_upper"] <= 1.0).all()


def test_scorer_ci_ordering(tmp_path):
    _train_all(tmp_path)
    claims = _make_claims(100)
    scorer = EnsembleFraudScorer(tmp_path)
    result = scorer.score(claims)

    assert (result["ci_lower"] <= result["fraud_score"]).all()
    assert (result["fraud_score"] <= result["ci_upper"]).all()


def test_scorer_pending_signals_are_null(tmp_path):
    _train_all(tmp_path)
    claims = _make_claims(50)
    scorer = EnsembleFraudScorer(tmp_path)
    result = scorer.score(claims)

    for col in ["p_gnn", "p_nlp", "p_vision"]:
        assert result[col].isna().all(), f"{col} should be None for pending signals"


def test_scorer_explain_shape_and_structure(tmp_path):
    _train_all(tmp_path)
    claims = _make_claims(30)
    scorer = EnsembleFraudScorer(tmp_path)
    reason_codes = scorer.explain(claims, top_n=5)

    assert len(reason_codes) == 30
    for row in reason_codes:
        assert len(row) <= 5
        for entry in row:
            assert "feature" in entry and "shap_pct" in entry
            assert entry["feature"] in scorer._feature_cols
            assert isinstance(entry["shap_pct"], float)


def test_scorer_explain_top_n_respected(tmp_path):
    _train_all(tmp_path)
    claims = _make_claims(20)
    scorer = EnsembleFraudScorer(tmp_path)
    for top_n in (1, 3, 5):
        codes = scorer.explain(claims, top_n=top_n)
        assert all(len(row) <= top_n for row in codes)


def test_scorer_explainer_cached(tmp_path):
    _train_all(tmp_path)
    claims = _make_claims(20)
    scorer = EnsembleFraudScorer(tmp_path)
    scorer.explain(claims)
    ref = scorer._xgb_explainer
    scorer.explain(claims)
    assert scorer._xgb_explainer is ref


def test_scorer_index_preserved(tmp_path):
    _train_all(tmp_path)
    claims = _make_claims(50)
    claims.index = range(100, 150)
    scorer = EnsembleFraudScorer(tmp_path)
    result = scorer.score(claims)
    assert list(result.index) == list(range(100, 150))


def test_scorer_high_fraud_signal_scores_higher(tmp_path):
    """High-risk claims (attorney, short inception, shared attrs) should outscore clean claims."""
    _train_all(tmp_path)
    scorer = EnsembleFraudScorer(tmp_path)

    base = {c: 0.0 for c in CLAIM_FRAUD_FEATURE_COLS}

    high_risk = pd.DataFrame([{
        **base,
        "policy_inception_days": 5.0,
        "attorney_present": 1.0,
        "shared_attribute_count": 4.0,
        "graph_hop_distance": 1.0,
        "attorney_centrality_score": 0.9,
        "risk_score_at_issuance": 12_000.0,
        "ip_geolocation_delta_miles": 300.0,
        "device_fingerprint_match": 0.0,
    }])
    low_risk = pd.DataFrame([{
        **base,
        "policy_inception_days": 400.0,
        "attorney_present": 0.0,
        "shared_attribute_count": 0.0,
        "graph_hop_distance": np.nan,
        "attorney_centrality_score": 0.01,
        "risk_score_at_issuance": 1_000.0,
        "ip_geolocation_delta_miles": 2.0,
        "device_fingerprint_match": 1.0,
    }])

    high_score = scorer.score(high_risk)["fraud_score"].iloc[0]
    low_score = scorer.score(low_risk)["fraud_score"].iloc[0]
    assert high_score > low_score
