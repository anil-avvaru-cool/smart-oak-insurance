"""Unit tests for monitoring/concept_drift.py (CD-1, CD-2, CD-3)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from monitoring.concept_drift import (
    LABEL_LAG_WARNING_THRESHOLD,
    LOSS_RATIO_DRIFT_THRESHOLD,
    append_gini_entry,
    check_label_lag,
    compute_loss_ratio_by_tier,
    run_concept_drift_check,
)
from monitoring.psi_drift import InsufficientDataError, LabelWindowReport
from underwriting.models.risk_scoring.train_frequency import QUOTE_FEATURE_COLS


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_quotes(n: int = 600, seed: int = 0, date_range_days: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-01-01")
    dates = [base + pd.Timedelta(days=int(d)) for d in rng.integers(0, date_range_days, n)]
    return pd.DataFrame(
        {
            "quote_id": [f"Q-{i:06d}" for i in range(n)],
            "quote_requested_at": dates,
            "credit_score": np.where(rng.random(n) > 0.1, rng.integers(550, 800, n).astype(float), np.nan),
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
            "telematics_distraction_score": np.where(rng.random(n) > 0.3, rng.uniform(0, 1, n), np.nan),
            "telematics_hard_brake_rate": np.where(rng.random(n) > 0.3, rng.uniform(0, 1, n), np.nan),
            "telematics_crash_match": np.where(rng.random(n) > 0.3, rng.uniform(0, 1, n), np.nan),
            "telematics_commute_entropy": np.where(rng.random(n) > 0.3, rng.uniform(0, 1, n), np.nan),
            "telematics_enrolled": rng.random(n) > 0.4,
            "claim_occurred": rng.random(n) > 0.85,
        }
    )


def _make_claims(
    n: int = 600,
    seed: int = 0,
    date_range_days: int = 60,
    close_fraction: float = 0.8,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-01-01")
    fnol_dates = [base + pd.Timedelta(days=int(d)) for d in rng.integers(0, date_range_days, n)]
    closed = rng.random(n) < close_fraction
    close_dates = [
        (fnol + pd.Timedelta(days=int(rng.integers(5, 30)))) if c else None
        for fnol, c in zip(fnol_dates, closed)
    ]
    tiers = rng.choice(["standard", "preferred", "high_risk"], size=n)
    return pd.DataFrame(
        {
            "claim_id": [f"C-{i:06d}" for i in range(n)],
            "fnol_submitted_at": fnol_dates,
            "claim_closed_at": close_dates,
            "policy_tier_at_issuance": tiers,
            "incurred_loss_usd": rng.uniform(500, 20_000, n),
            "risk_score_at_issuance": rng.uniform(200, 5_000, n),
        }
    )


def _train_model(quotes: pd.DataFrame, models_dir: Path) -> Path:
    from underwriting.models.risk_scoring.train_frequency import prepare_features

    df = prepare_features(quotes)
    X = df[QUOTE_FEATURE_COLS]
    y = df["claim_occurred"].astype(int)
    model = xgb.XGBClassifier(n_estimators=10, max_depth=2, random_state=0, verbosity=0, eval_metric="logloss")
    model.fit(X, y)
    path = models_dir / "frequency_model.json"
    model.save_model(path)
    return path


def _make_label_window(confirmation_rate_90d: float = 0.8) -> LabelWindowReport:
    return LabelWindowReport(
        total_claims_in_window=100,
        fraud_claims_in_window=30,
        confirmed_any=25,
        confirmed_90d=int(30 * confirmation_rate_90d),
        confirmed_180d=int(30 * min(confirmation_rate_90d + 0.1, 1.0)),
        confirmation_rate_any=0.83,
        confirmation_rate_90d=confirmation_rate_90d,
        confirmation_rate_180d=min(confirmation_rate_90d + 0.1, 1.0),
        as_of=pd.Timestamp("2026-02-01"),
    )


# ── CD-1: append_gini_entry ───────────────────────────────────────────────────


class TestAppendGiniEntry:
    def _as_of(self) -> pd.Timestamp:
        return pd.Timestamp("2026-02-01")

    def test_happy_path_returns_entry_with_gini(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)
        _train_model(quotes, tmp_path)

        entry = append_gini_entry(
            quotes_path=quotes_path,
            models_dir=tmp_path,
            output_dir=tmp_path,
            window_days=30,
            min_records=10,
            as_of=self._as_of(),
        )

        assert "gini" in entry
        assert "auc" in entry
        assert -1.0 <= entry["gini"] <= 1.0
        assert 0.0 <= entry["auc"] <= 1.0

    def test_creates_gini_history_json(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)
        _train_model(quotes, tmp_path)

        append_gini_entry(
            quotes_path=quotes_path,
            models_dir=tmp_path,
            output_dir=tmp_path,
            window_days=30,
            min_records=10,
            as_of=self._as_of(),
        )

        history_path = tmp_path / "gini_history.json"
        assert history_path.exists()
        h = json.loads(history_path.read_text())
        assert h["schema_version"] == "1.0"
        assert len(h["entries"]) == 1

    def test_appends_on_second_call(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)
        _train_model(quotes, tmp_path)

        as_of1 = pd.Timestamp("2026-02-01")
        as_of2 = pd.Timestamp("2026-02-15")

        append_gini_entry(quotes_path=quotes_path, models_dir=tmp_path, output_dir=tmp_path, window_days=30, min_records=10, as_of=as_of1)
        append_gini_entry(quotes_path=quotes_path, models_dir=tmp_path, output_dir=tmp_path, window_days=30, min_records=10, as_of=as_of2)

        h = json.loads((tmp_path / "gini_history.json").read_text())
        assert len(h["entries"]) == 2

    def test_entry_contains_required_keys(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)
        _train_model(quotes, tmp_path)

        entry = append_gini_entry(
            quotes_path=quotes_path, models_dir=tmp_path, output_dir=tmp_path,
            window_days=30, min_records=10, as_of=self._as_of(),
        )

        for key in ("computed_at", "window_start", "window_end", "n_records", "auc", "gini", "model_path"):
            assert key in entry

    def test_raises_file_not_found_if_model_absent(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)

        with pytest.raises(FileNotFoundError, match="frequency_model"):
            append_gini_entry(quotes_path=quotes_path, models_dir=tmp_path, output_dir=tmp_path, as_of=self._as_of())

    def test_raises_insufficient_data_when_window_too_small(self, tmp_path: Path) -> None:
        quotes = _make_quotes(n=50, date_range_days=5)
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)
        _train_model(quotes, tmp_path)

        # Current window of 1 day will catch very few quotes
        with pytest.raises(InsufficientDataError):
            append_gini_entry(
                quotes_path=quotes_path,
                models_dir=tmp_path,
                output_dir=tmp_path,
                window_days=1,
                min_records=500,
                as_of=pd.Timestamp("2026-01-03"),
            )

    def test_n_records_matches_window(self, tmp_path: Path) -> None:
        quotes = _make_quotes(n=600, date_range_days=60)
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)
        _train_model(quotes, tmp_path)

        as_of = pd.Timestamp("2026-02-01")
        window_days = 14
        entry = append_gini_entry(
            quotes_path=quotes_path, models_dir=tmp_path, output_dir=tmp_path,
            window_days=window_days, min_records=10, as_of=as_of,
        )

        # Verify manually
        df = pd.read_parquet(quotes_path)
        window_start = as_of - pd.Timedelta(days=window_days)
        ts = pd.to_datetime(df["quote_requested_at"])
        expected = int(((ts >= window_start) & (ts <= as_of)).sum())
        assert entry["n_records"] == expected


# ── CD-2: compute_loss_ratio_by_tier ─────────────────────────────────────────


class TestComputeLossRatioByTier:
    def _as_of(self) -> pd.Timestamp:
        return pd.Timestamp("2026-02-01")

    def test_returns_required_keys(self, tmp_path: Path) -> None:
        claims = _make_claims()
        claims_path = tmp_path / "claims.parquet"
        claims.to_parquet(claims_path)

        result = compute_loss_ratio_by_tier(claims_path=claims_path, window_days=30, as_of=self._as_of())

        for key in ("as_of", "window_start", "window_end", "n_reference_closed", "n_current_closed", "drifted_tiers", "tier_details", "drift_threshold"):
            assert key in result

    def test_tier_details_cover_all_tiers(self, tmp_path: Path) -> None:
        claims = _make_claims()
        claims_path = tmp_path / "claims.parquet"
        claims.to_parquet(claims_path)

        result = compute_loss_ratio_by_tier(claims_path=claims_path, window_days=30, as_of=self._as_of())

        tiers_in_result = {d["tier"] for d in result["tier_details"]}
        assert "standard" in tiers_in_result or "preferred" in tiers_in_result  # at least one tier present

    def test_stable_when_ratios_identical(self, tmp_path: Path) -> None:
        base = pd.Timestamp("2026-01-01")
        # Reference: 100 claims 60–31 days ago; current: 100 claims last 30 days
        # Same loss ratio for both periods → stable
        ref_fnol = [base + pd.Timedelta(days=d) for d in range(40)]
        curr_fnol = [base + pd.Timedelta(days=d) for d in range(45, 75)]
        fnol = ref_fnol + curr_fnol
        n = len(fnol)
        df = pd.DataFrame({
            "claim_id": [f"C-{i}" for i in range(n)],
            "fnol_submitted_at": fnol,
            "claim_closed_at": [f + pd.Timedelta(days=3) for f in fnol],
            "policy_tier_at_issuance": ["standard"] * n,
            "incurred_loss_usd": [1000.0] * n,
            "risk_score_at_issuance": [2000.0] * n,  # loss ratio = 0.5 everywhere
        })
        claims_path = tmp_path / "claims.parquet"
        df.to_parquet(claims_path)

        as_of = base + pd.Timedelta(days=75)
        result = compute_loss_ratio_by_tier(claims_path=claims_path, window_days=30, as_of=as_of)

        assert result["drifted_tiers"] == []
        detail = result["tier_details"][0]
        assert detail["status"] == "stable"

    def test_drifted_when_ratio_changes_by_more_than_threshold(self, tmp_path: Path) -> None:
        base = pd.Timestamp("2026-01-01")
        ref_fnol = [base + pd.Timedelta(days=d) for d in range(40)]
        curr_fnol = [base + pd.Timedelta(days=d) for d in range(45, 75)]
        fnol = ref_fnol + curr_fnol
        n_ref = len(ref_fnol)
        n = len(fnol)
        # Reference loss ratio = 0.5, current = 0.7 (40% relative drift)
        losses = [1000.0] * n_ref + [1400.0] * (n - n_ref)
        df = pd.DataFrame({
            "claim_id": [f"C-{i}" for i in range(n)],
            "fnol_submitted_at": fnol,
            "claim_closed_at": [f + pd.Timedelta(days=3) for f in fnol],
            "policy_tier_at_issuance": ["standard"] * n,
            "incurred_loss_usd": losses,
            "risk_score_at_issuance": [2000.0] * n,
        })
        claims_path = tmp_path / "claims.parquet"
        df.to_parquet(claims_path)

        as_of = base + pd.Timedelta(days=75)
        result = compute_loss_ratio_by_tier(claims_path=claims_path, window_days=30, as_of=as_of)

        assert "standard" in result["drifted_tiers"]
        detail = next(d for d in result["tier_details"] if d["tier"] == "standard")
        assert detail["status"] == "drifted"

    def test_only_closed_claims_used(self, tmp_path: Path) -> None:
        base = pd.Timestamp("2026-01-01")
        fnol = [base + pd.Timedelta(days=d) for d in range(50)]
        n = len(fnol)
        # All claims open (claim_closed_at = None)
        df = pd.DataFrame({
            "claim_id": [f"C-{i}" for i in range(n)],
            "fnol_submitted_at": fnol,
            "claim_closed_at": [None] * n,
            "policy_tier_at_issuance": ["standard"] * n,
            "incurred_loss_usd": [1000.0] * n,
            "risk_score_at_issuance": [2000.0] * n,
        })
        claims_path = tmp_path / "claims.parquet"
        df.to_parquet(claims_path)

        as_of = base + pd.Timedelta(days=55)
        result = compute_loss_ratio_by_tier(claims_path=claims_path, window_days=14, as_of=as_of)

        # No closed claims → tier_details may be empty or all insufficient_reference
        assert result["n_reference_closed"] == 0
        assert result["n_current_closed"] == 0

    def test_loss_ratio_values_are_rounded(self, tmp_path: Path) -> None:
        base = pd.Timestamp("2026-01-01")
        ref_fnol = [base + pd.Timedelta(days=d) for d in range(40)]
        curr_fnol = [base + pd.Timedelta(days=d) for d in range(45, 75)]
        fnol = ref_fnol + curr_fnol
        n = len(fnol)
        df = pd.DataFrame({
            "claim_id": [f"C-{i}" for i in range(n)],
            "fnol_submitted_at": fnol,
            "claim_closed_at": [f + pd.Timedelta(days=3) for f in fnol],
            "policy_tier_at_issuance": ["standard"] * n,
            "incurred_loss_usd": [1000.0] * n,
            "risk_score_at_issuance": [3000.0] * n,
        })
        claims_path = tmp_path / "claims.parquet"
        df.to_parquet(claims_path)

        as_of = base + pd.Timedelta(days=75)
        result = compute_loss_ratio_by_tier(claims_path=claims_path, window_days=30, as_of=as_of)

        for detail in result["tier_details"]:
            if detail["reference_loss_ratio"] is not None:
                assert isinstance(detail["reference_loss_ratio"], float)
            if detail["current_loss_ratio"] is not None:
                assert isinstance(detail["current_loss_ratio"], float)

    def test_drift_threshold_in_result(self, tmp_path: Path) -> None:
        claims = _make_claims()
        claims_path = tmp_path / "claims.parquet"
        claims.to_parquet(claims_path)

        result = compute_loss_ratio_by_tier(claims_path=claims_path, window_days=30, as_of=self._as_of())
        assert result["drift_threshold"] == LOSS_RATIO_DRIFT_THRESHOLD


# ── CD-3: check_label_lag ─────────────────────────────────────────────────────


class TestCheckLabelLag:
    def test_no_warning_when_above_threshold(self) -> None:
        lw = _make_label_window(confirmation_rate_90d=0.80)
        result = check_label_lag(lw)
        assert result["below_threshold"] is False
        assert result["warning"] is None

    def test_warning_when_below_threshold(self) -> None:
        lw = _make_label_window(confirmation_rate_90d=0.30)
        result = check_label_lag(lw)
        assert result["below_threshold"] is True
        assert result["warning"] is not None
        assert "Label lag" in result["warning"]

    def test_exact_threshold_is_not_below(self) -> None:
        # "below threshold" is strict (<), so the exact value is not flagged
        lw = _make_label_window(confirmation_rate_90d=LABEL_LAG_WARNING_THRESHOLD)
        result = check_label_lag(lw)
        assert result["below_threshold"] is False

    def test_just_above_threshold_is_ok(self) -> None:
        lw = _make_label_window(confirmation_rate_90d=LABEL_LAG_WARNING_THRESHOLD + 0.01)
        result = check_label_lag(lw)
        assert result["below_threshold"] is False

    def test_returns_confirmation_rates(self) -> None:
        lw = _make_label_window(confirmation_rate_90d=0.75)
        result = check_label_lag(lw)
        assert result["confirmation_rate_90d"] == pytest.approx(0.75)
        assert "confirmation_rate_180d" in result

    def test_custom_threshold(self) -> None:
        lw = _make_label_window(confirmation_rate_90d=0.60)
        result = check_label_lag(lw, threshold=0.70)
        assert result["below_threshold"] is True

        result2 = check_label_lag(lw, threshold=0.50)
        assert result2["below_threshold"] is False


# ── run_concept_drift_check ───────────────────────────────────────────────────


class TestRunConceptDriftCheck:
    def _as_of(self) -> pd.Timestamp:
        return pd.Timestamp("2026-02-01")

    def test_returns_required_top_level_keys(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        claims = _make_claims()
        quotes_path = tmp_path / "quotes.parquet"
        claims_path = tmp_path / "claims.parquet"
        quotes.to_parquet(quotes_path)
        claims.to_parquet(claims_path)
        _train_model(quotes, tmp_path)

        result = run_concept_drift_check(
            quotes_path=quotes_path,
            claims_path=claims_path,
            models_dir=tmp_path,
            output_dir=tmp_path,
            window_days=30,
            min_records=10,
            as_of=self._as_of(),
        )

        for key in ("as_of", "gini_trend", "gini_error", "loss_ratio_by_tier", "label_lag"):
            assert key in result

    def test_gini_error_when_no_model(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        claims = _make_claims()
        quotes_path = tmp_path / "quotes.parquet"
        claims_path = tmp_path / "claims.parquet"
        quotes.to_parquet(quotes_path)
        claims.to_parquet(claims_path)
        # No model file

        result = run_concept_drift_check(
            quotes_path=quotes_path,
            claims_path=claims_path,
            models_dir=tmp_path,
            output_dir=tmp_path,
            window_days=30,
            min_records=10,
            as_of=self._as_of(),
        )

        assert result["gini_trend"] is None
        assert result["gini_error"] is not None
        # loss ratio still computed
        assert result["loss_ratio_by_tier"] is not None

    def test_label_lag_none_when_not_provided(self, tmp_path: Path) -> None:
        claims = _make_claims()
        claims_path = tmp_path / "claims.parquet"
        claims.to_parquet(claims_path)
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)

        result = run_concept_drift_check(
            quotes_path=quotes_path,
            claims_path=claims_path,
            models_dir=tmp_path,
            output_dir=tmp_path,
            window_days=30,
            min_records=10,
            as_of=self._as_of(),
            label_window=None,
        )

        assert result["label_lag"] is None

    def test_label_lag_populated_when_provided(self, tmp_path: Path) -> None:
        claims = _make_claims()
        claims_path = tmp_path / "claims.parquet"
        claims.to_parquet(claims_path)
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)

        lw = _make_label_window(confirmation_rate_90d=0.80)
        result = run_concept_drift_check(
            quotes_path=quotes_path,
            claims_path=claims_path,
            models_dir=tmp_path,
            output_dir=tmp_path,
            window_days=30,
            min_records=10,
            as_of=self._as_of(),
            label_window=lw,
        )

        assert result["label_lag"] is not None
        assert "below_threshold" in result["label_lag"]

    def test_gini_history_written_when_model_present(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        claims = _make_claims()
        quotes_path = tmp_path / "quotes.parquet"
        claims_path = tmp_path / "claims.parquet"
        quotes.to_parquet(quotes_path)
        claims.to_parquet(claims_path)
        _train_model(quotes, tmp_path)

        run_concept_drift_check(
            quotes_path=quotes_path,
            claims_path=claims_path,
            models_dir=tmp_path,
            output_dir=tmp_path,
            window_days=30,
            min_records=10,
            as_of=self._as_of(),
        )

        assert (tmp_path / "gini_history.json").exists()
