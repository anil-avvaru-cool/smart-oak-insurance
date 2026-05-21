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
