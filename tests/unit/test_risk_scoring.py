from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from underwriting.models.risk_scoring.train_frequency import (
    QUOTE_FEATURE_COLS,
    train as train_frequency,
)
from underwriting.models.risk_scoring.train_severity import train as train_severity
from underwriting.models.risk_scoring.hurdle_model import HurdleModel, evaluate


def _make_quotes(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "quote_id": [f"Q-{i:05d}" for i in range(n)],
            "credit_score": np.where(
                rng.random(n) > 0.1, rng.integers(550, 800, n).astype(float), np.nan
            ),
            "prior_loss_frequency": rng.uniform(0, 2, n),
            "prior_loss_severity_avg": rng.uniform(0, 10_000, n),
            "insurance_lapse_days": rng.integers(0, 365, n).astype(float),
            "violation_severity_index": rng.uniform(0, 5, n),
            "household_driver_density": rng.uniform(0.5, 3.0, n),
            "driver_age": rng.integers(18, 75, n).astype(float),
            "years_licensed": rng.integers(0, 50, n).astype(float),
            "vehicle_msrp": rng.uniform(15_000, 80_000, n),
            "vehicle_power": rng.uniform(100, 400, n),
            "vehicle_adas_score": rng.uniform(0, 1, n),
            "vehicle_age_years": rng.integers(0, 20, n).astype(float),
            "geohash_risk_score": rng.uniform(0, 1, n),
            "annual_mileage_estimate": rng.uniform(5_000, 25_000, n),
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
            "claim_occurred": rng.random(n) > 0.9,
        }
    )


def _make_claims(quotes: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "quote_id": quotes["quote_id"].values,
            "incurred_loss_usd": rng.gamma(shape=2.0, scale=5_000.0, size=len(quotes)),
        }
    )


def test_frequency_trains_and_saves_model(tmp_path):
    quotes = _make_quotes()
    quotes.to_parquet(tmp_path / "quotes.parquet")

    metrics = train_frequency(tmp_path / "quotes.parquet", tmp_path)

    assert (tmp_path / "frequency_model.json").exists()
    assert (tmp_path / "frequency_features.json").exists()
    assert 0.0 <= metrics["auc"] <= 1.0
    assert -1.0 <= metrics["gini"] <= 1.0
    assert metrics["log_loss"] > 0


def test_severity_trains_and_saves_model(tmp_path):
    quotes = _make_quotes()
    claims = _make_claims(quotes)
    quotes.to_parquet(tmp_path / "quotes.parquet")
    claims.to_parquet(tmp_path / "claims.parquet")

    metrics = train_severity(
        tmp_path / "quotes.parquet", tmp_path / "claims.parquet", tmp_path
    )

    assert (tmp_path / "severity_model.json").exists()
    assert metrics["mae_usd"] > 0
    assert metrics["rmse_usd"] >= metrics["mae_usd"]


def test_hurdle_model_score_shape_and_sign(tmp_path):
    quotes = _make_quotes()
    claims = _make_claims(quotes)
    quotes.to_parquet(tmp_path / "quotes.parquet")
    claims.to_parquet(tmp_path / "claims.parquet")

    train_frequency(tmp_path / "quotes.parquet", tmp_path)
    train_severity(tmp_path / "quotes.parquet", tmp_path / "claims.parquet", tmp_path)

    model = HurdleModel(tmp_path)
    scores = model.score(quotes)

    assert scores.shape == (len(quotes),)
    assert (scores >= 0).all()


def test_hurdle_model_components_multiply_to_risk_score(tmp_path):
    quotes = _make_quotes()
    claims = _make_claims(quotes)
    quotes.to_parquet(tmp_path / "quotes.parquet")
    claims.to_parquet(tmp_path / "claims.parquet")

    train_frequency(tmp_path / "quotes.parquet", tmp_path)
    train_severity(tmp_path / "quotes.parquet", tmp_path / "claims.parquet", tmp_path)

    model = HurdleModel(tmp_path)
    result = model.score_with_components(quotes)

    assert set(result.columns) == {"p_claim", "e_cost_usd", "risk_score"}
    np.testing.assert_allclose(
        result["risk_score"].values,
        (result["p_claim"] * result["e_cost_usd"]).values,
        rtol=1e-6,
    )


def test_evaluate_returns_gini_and_distribution(tmp_path):
    quotes = _make_quotes()
    claims = _make_claims(quotes)
    quotes.to_parquet(tmp_path / "quotes.parquet")
    claims.to_parquet(tmp_path / "claims.parquet")

    train_frequency(tmp_path / "quotes.parquet", tmp_path)
    train_severity(tmp_path / "quotes.parquet", tmp_path / "claims.parquet", tmp_path)

    metrics = evaluate(tmp_path / "quotes.parquet", tmp_path)

    assert "hurdle_gini" in metrics
    assert "risk_score_mean" in metrics
    assert metrics["risk_score_p95"] >= metrics["risk_score_p50"]


import json as _json


def test_frequency_metrics_include_feature_importance(tmp_path):
    quotes = _make_quotes()
    quotes.to_parquet(tmp_path / "quotes.parquet")

    train_frequency(tmp_path / "quotes.parquet", tmp_path)

    metrics = _json.loads((tmp_path / "frequency_metrics.json").read_text())
    assert "feature_importance_pct" in metrics
    imp = metrics["feature_importance_pct"]
    assert set(imp.keys()) == set(QUOTE_FEATURE_COLS)
    assert abs(sum(imp.values()) - 100.0) < 0.5  # sums to ~100%
    assert all(v >= 0 for v in imp.values())
    vals = list(imp.values())
    assert vals == sorted(vals, reverse=True)


def test_severity_metrics_include_feature_importance(tmp_path):
    quotes = _make_quotes()
    claims = _make_claims(quotes)
    quotes.to_parquet(tmp_path / "quotes.parquet")
    claims.to_parquet(tmp_path / "claims.parquet")

    from underwriting.models.risk_scoring.train_severity import train as train_severity

    train_severity(tmp_path / "quotes.parquet", tmp_path / "claims.parquet", tmp_path)

    metrics = _json.loads((tmp_path / "severity_metrics.json").read_text())
    assert "feature_importance_pct" in metrics
    imp = metrics["feature_importance_pct"]
    assert set(imp.keys()) == set(QUOTE_FEATURE_COLS)
    assert abs(sum(imp.values()) - 100.0) < 0.5
    vals = list(imp.values())
    assert vals == sorted(vals, reverse=True)


def test_hurdle_model_explain_shape_and_structure(tmp_path):
    quotes = _make_quotes()
    claims = _make_claims(quotes)
    quotes.to_parquet(tmp_path / "quotes.parquet")
    claims.to_parquet(tmp_path / "claims.parquet")

    train_frequency(tmp_path / "quotes.parquet", tmp_path)
    train_severity(tmp_path / "quotes.parquet", tmp_path / "claims.parquet", tmp_path)

    model = HurdleModel(tmp_path)
    reason_codes = model.explain(quotes, top_n=5)

    assert len(reason_codes) == len(quotes)
    for row in reason_codes:
        assert len(row) <= 5
        for entry in row:
            assert "feature" in entry and "shap_pct" in entry
            assert entry["feature"] in QUOTE_FEATURE_COLS
            assert isinstance(entry["shap_pct"], float)


def test_hurdle_model_explain_top_n_respected(tmp_path):
    quotes = _make_quotes()
    claims = _make_claims(quotes)
    quotes.to_parquet(tmp_path / "quotes.parquet")
    claims.to_parquet(tmp_path / "claims.parquet")

    train_frequency(tmp_path / "quotes.parquet", tmp_path)
    train_severity(tmp_path / "quotes.parquet", tmp_path / "claims.parquet", tmp_path)

    model = HurdleModel(tmp_path)
    for top_n in (1, 3, 5):
        codes = model.explain(quotes, top_n=top_n)
        assert all(len(row) <= top_n for row in codes)


def test_hurdle_model_explain_explainer_cached(tmp_path):
    quotes = _make_quotes()
    claims = _make_claims(quotes)
    quotes.to_parquet(tmp_path / "quotes.parquet")
    claims.to_parquet(tmp_path / "claims.parquet")

    train_frequency(tmp_path / "quotes.parquet", tmp_path)
    train_severity(tmp_path / "quotes.parquet", tmp_path / "claims.parquet", tmp_path)

    model = HurdleModel(tmp_path)
    model.explain(quotes)
    explainer_ref = model._freq_explainer
    model.explain(quotes)
    assert model._freq_explainer is explainer_ref  # same object, not re-created


# ── Stage 3 Calibration ───────────────────────────────────────────────────────

from underwriting.models.risk_scoring.calibrate import (
    calibrate_frequency,
    calibrate_severity,
)


def _train_both(tmp_path, quotes, claims):
    train_frequency(tmp_path / "quotes.parquet", tmp_path)
    train_severity(tmp_path / "quotes.parquet", tmp_path / "claims.parquet", tmp_path)


def test_calibrate_frequency_writes_artifacts(tmp_path):
    quotes = _make_quotes(n=500)
    claims = _make_claims(quotes)
    quotes.to_parquet(tmp_path / "quotes.parquet")
    claims.to_parquet(tmp_path / "claims.parquet")
    _train_both(tmp_path, quotes, claims)

    metrics = calibrate_frequency(tmp_path / "quotes.parquet", tmp_path, tmp_path)

    assert (tmp_path / "frequency_calibration.json").exists()
    assert (tmp_path / "frequency_calibration_metrics.json").exists()
    cal = _json.loads((tmp_path / "frequency_calibration.json").read_text())
    assert cal["method"] == "platt"
    assert "a" in cal and "b" in cal
    assert "brier_before" in metrics
    assert "brier_after" in metrics
    assert "ece_before" in metrics
    assert "ece_after" in metrics
    assert "reliability_diagram" in metrics
    assert isinstance(metrics["reliability_diagram"], list)


def test_calibrate_severity_writes_artifacts(tmp_path):
    quotes = _make_quotes(n=500)
    claims = _make_claims(quotes)
    quotes.to_parquet(tmp_path / "quotes.parquet")
    claims.to_parquet(tmp_path / "claims.parquet")
    _train_both(tmp_path, quotes, claims)

    metrics = calibrate_severity(
        tmp_path / "quotes.parquet", tmp_path / "claims.parquet", tmp_path, tmp_path
    )

    assert (tmp_path / "severity_calibration.json").exists()
    assert (tmp_path / "severity_calibration_metrics.json").exists()
    cal = _json.loads((tmp_path / "severity_calibration.json").read_text())
    assert cal["method"] == "multiplicative"
    assert cal["bias_correction"] > 0
    assert "mae_before_usd" in metrics
    assert "mae_after_usd" in metrics
    assert metrics["mean_predicted_after_usd"] == pytest.approx(
        metrics["mean_actual_usd"], rel=0.01
    )


def test_hurdle_model_loads_calibration_on_init(tmp_path):
    quotes = _make_quotes(n=500)
    claims = _make_claims(quotes)
    quotes.to_parquet(tmp_path / "quotes.parquet")
    claims.to_parquet(tmp_path / "claims.parquet")
    _train_both(tmp_path, quotes, claims)

    model_no_cal = HurdleModel(tmp_path)
    assert not model_no_cal.calibration_applied
    assert model_no_cal._platt_a is None
    assert model_no_cal._sev_bias == 1.0

    calibrate_frequency(tmp_path / "quotes.parquet", tmp_path, tmp_path)
    calibrate_severity(
        tmp_path / "quotes.parquet", tmp_path / "claims.parquet", tmp_path, tmp_path
    )

    model_cal = HurdleModel(tmp_path)
    assert model_cal.calibration_applied
    assert model_cal._platt_a is not None
    assert model_cal._sev_bias != 1.0


def test_hurdle_calibrated_probability_is_bounded(tmp_path):
    quotes = _make_quotes(n=500)
    claims = _make_claims(quotes)
    quotes.to_parquet(tmp_path / "quotes.parquet")
    claims.to_parquet(tmp_path / "claims.parquet")
    _train_both(tmp_path, quotes, claims)
    calibrate_frequency(tmp_path / "quotes.parquet", tmp_path, tmp_path)
    calibrate_severity(
        tmp_path / "quotes.parquet", tmp_path / "claims.parquet", tmp_path, tmp_path
    )

    model = HurdleModel(tmp_path)
    result = model.score_with_components(quotes)

    assert (result["p_claim"] >= 0).all() and (result["p_claim"] <= 1).all()
    assert (result["e_cost_usd"] > 0).all()
    assert (result["risk_score"] >= 0).all()


def test_hurdle_calibration_risk_score_invariant(tmp_path):
    """risk_score == p_claim * e_cost_usd must hold even with calibration applied."""
    quotes = _make_quotes(n=500)
    claims = _make_claims(quotes)
    quotes.to_parquet(tmp_path / "quotes.parquet")
    claims.to_parquet(tmp_path / "claims.parquet")
    _train_both(tmp_path, quotes, claims)
    calibrate_frequency(tmp_path / "quotes.parquet", tmp_path, tmp_path)
    calibrate_severity(
        tmp_path / "quotes.parquet", tmp_path / "claims.parquet", tmp_path, tmp_path
    )

    model = HurdleModel(tmp_path)
    result = model.score_with_components(quotes)

    np.testing.assert_allclose(
        result["risk_score"].values,
        (result["p_claim"] * result["e_cost_usd"]).values,
        rtol=1e-6,
    )


def test_calibration_metrics_structure(tmp_path):
    quotes = _make_quotes(n=500)
    claims = _make_claims(quotes)
    quotes.to_parquet(tmp_path / "quotes.parquet")
    claims.to_parquet(tmp_path / "claims.parquet")
    _train_both(tmp_path, quotes, claims)

    freq_metrics = calibrate_frequency(tmp_path / "quotes.parquet", tmp_path, tmp_path)
    sev_metrics = calibrate_severity(
        tmp_path / "quotes.parquet", tmp_path / "claims.parquet", tmp_path, tmp_path
    )

    assert 0.0 <= freq_metrics["ece_before"] <= 1.0
    assert 0.0 <= freq_metrics["ece_after"] <= 1.0
    assert freq_metrics["brier_before"] > 0
    assert freq_metrics["brier_after"] > 0
    assert 0.0 < freq_metrics["actual_claim_rate"] < 1.0

    assert sev_metrics["bias_correction"] > 0
    assert sev_metrics["mae_before_usd"] > 0
    assert sev_metrics["mae_after_usd"] > 0
