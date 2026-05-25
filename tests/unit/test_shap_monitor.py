"""Unit tests for monitoring/shap_monitor.py (SH-1, SH-2)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from monitoring.shap_monitor import (
    SHAP_BRITTLENESS_THRESHOLD,
    compare_shap_snapshots,
    write_shap_snapshot,
)
from underwriting.models.risk_scoring.train_frequency import QUOTE_FEATURE_COLS


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_quotes(n: int = 400, seed: int = 0) -> pd.DataFrame:
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
            "claim_occurred": rng.random(n) > 0.85,
        }
    )


def _train_and_save_model(quotes: pd.DataFrame, models_dir: Path) -> Path:
    """Train a minimal XGBoost model and save to models_dir/frequency_model.json."""
    from underwriting.models.risk_scoring.train_frequency import prepare_features

    df = prepare_features(quotes)
    X = df[QUOTE_FEATURE_COLS]
    y = df["claim_occurred"].astype(int)

    model = xgb.XGBClassifier(
        n_estimators=10, max_depth=2, random_state=0, verbosity=0, eval_metric="logloss"
    )
    model.fit(X, y)
    model_path = models_dir / "frequency_model.json"
    model.save_model(model_path)
    return model_path


def _make_snapshot_dict(
    features: list[str],
    shares: list[float],
    generated_at: str = "2026-01-01T00:00:00",
    n_samples: int = 100,
) -> dict:
    mean_abs = [s * 0.5 for s in shares]
    ranked = sorted(
        zip(features, mean_abs, shares),
        key=lambda t: t[2],
        reverse=True,
    )
    max_share = max(shares)
    return {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "model": "frequency_model",
        "model_path": "/tmp/frequency_model.json",
        "n_samples": n_samples,
        "features": features,
        "mean_abs_shap": mean_abs,
        "shap_share": shares,
        "ranked_features": [
            {"feature": f, "mean_abs_shap": round(m, 8), "shap_share": round(s, 6)}
            for f, m, s in ranked
        ],
        "max_shap_share_feature": features[shares.index(max_share)],
        "max_shap_share": max_share,
        "brittleness_warning": max_share > SHAP_BRITTLENESS_THRESHOLD,
    }


# ── SH-1: write_shap_snapshot ─────────────────────────────────────────────────


class TestWriteShapSnapshot:
    def test_writes_valid_json_snapshot(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)
        _train_and_save_model(quotes, tmp_path)

        out_path = write_shap_snapshot(
            quotes_path=quotes_path,
            models_dir=tmp_path,
            output_dir=tmp_path,
        )

        assert out_path.exists()
        snap = json.loads(out_path.read_text())
        assert snap["schema_version"] == "1.0"
        assert snap["model"] == "frequency_model"
        assert snap["n_samples"] > 0

    def test_snapshot_filename_matches_pattern(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)
        _train_and_save_model(quotes, tmp_path)

        out_path = write_shap_snapshot(
            quotes_path=quotes_path, models_dir=tmp_path, output_dir=tmp_path
        )
        assert out_path.name.startswith("shap_snapshot_")
        assert out_path.suffix == ".json"

    def test_features_match_quote_feature_cols(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)
        _train_and_save_model(quotes, tmp_path)

        out_path = write_shap_snapshot(
            quotes_path=quotes_path, models_dir=tmp_path, output_dir=tmp_path
        )
        snap = json.loads(out_path.read_text())
        assert snap["features"] == QUOTE_FEATURE_COLS

    def test_shap_shares_sum_to_one(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)
        _train_and_save_model(quotes, tmp_path)

        out_path = write_shap_snapshot(
            quotes_path=quotes_path, models_dir=tmp_path, output_dir=tmp_path
        )
        snap = json.loads(out_path.read_text())
        total = sum(snap["shap_share"])
        assert abs(total - 1.0) < 1e-4

    def test_ranked_features_descending_by_share(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)
        _train_and_save_model(quotes, tmp_path)

        out_path = write_shap_snapshot(
            quotes_path=quotes_path, models_dir=tmp_path, output_dir=tmp_path
        )
        snap = json.loads(out_path.read_text())
        shares = [e["shap_share"] for e in snap["ranked_features"]]
        assert shares == sorted(shares, reverse=True)

    def test_max_shap_share_consistent_with_ranked(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)
        _train_and_save_model(quotes, tmp_path)

        out_path = write_shap_snapshot(
            quotes_path=quotes_path, models_dir=tmp_path, output_dir=tmp_path
        )
        snap = json.loads(out_path.read_text())
        assert snap["max_shap_share_feature"] == snap["ranked_features"][0]["feature"]
        assert abs(snap["max_shap_share"] - snap["ranked_features"][0]["shap_share"]) < 1e-9

    def test_sample_size_respected(self, tmp_path: Path) -> None:
        quotes = _make_quotes(n=600)
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)
        _train_and_save_model(quotes, tmp_path)

        out_path = write_shap_snapshot(
            quotes_path=quotes_path,
            models_dir=tmp_path,
            output_dir=tmp_path,
            sample_size=100,
        )
        snap = json.loads(out_path.read_text())
        assert snap["n_samples"] == 100

    def test_missing_model_raises_file_not_found(self, tmp_path: Path) -> None:
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)

        with pytest.raises(FileNotFoundError, match="frequency_model"):
            write_shap_snapshot(
                quotes_path=quotes_path, models_dir=tmp_path, output_dir=tmp_path
            )

    def test_brittleness_warning_flag_set(self, tmp_path: Path) -> None:
        """When a single feature dominates, brittleness_warning must be True."""
        quotes = _make_quotes()
        quotes_path = tmp_path / "quotes.parquet"
        quotes.to_parquet(quotes_path)

        # Build a model that uses only one feature (all others zeroed)
        from underwriting.models.risk_scoring.train_frequency import prepare_features

        df = prepare_features(quotes)
        X = df[QUOTE_FEATURE_COLS].copy()
        # Null all but first feature so XGBoost gains concentrate on it
        for col in QUOTE_FEATURE_COLS[1:]:
            X[col] = np.nan
        y = df["claim_occurred"].astype(int)

        model = xgb.XGBClassifier(
            n_estimators=20, max_depth=3, random_state=0, verbosity=0, eval_metric="logloss"
        )
        model.fit(X, y)
        model.save_model(tmp_path / "frequency_model.json")

        out_path = write_shap_snapshot(
            quotes_path=quotes_path, models_dir=tmp_path, output_dir=tmp_path
        )
        snap = json.loads(out_path.read_text())
        assert snap["brittleness_warning"] is True
        assert snap["max_shap_share"] > SHAP_BRITTLENESS_THRESHOLD


# ── SH-2: compare_shap_snapshots ─────────────────────────────────────────────


class TestCompareShapSnapshots:
    def _write_snapshot(self, output_dir: Path, ts_str: str, shares: list[float]) -> None:
        snap = _make_snapshot_dict(QUOTE_FEATURE_COLS, shares, generated_at=f"{ts_str}T00:00:00")
        (output_dir / f"shap_snapshot_{ts_str}.json").write_text(json.dumps(snap))

    def test_raises_when_fewer_than_two_snapshots(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="2 SHAP snapshots"):
            compare_shap_snapshots(output_dir=tmp_path)

    def test_raises_when_only_one_snapshot(self, tmp_path: Path) -> None:
        uniform = [1.0 / len(QUOTE_FEATURE_COLS)] * len(QUOTE_FEATURE_COLS)
        self._write_snapshot(tmp_path, "20260101T000000", uniform)
        with pytest.raises(FileNotFoundError):
            compare_shap_snapshots(output_dir=tmp_path)

    def test_happy_path_returns_required_keys(self, tmp_path: Path) -> None:
        n = len(QUOTE_FEATURE_COLS)
        uniform = [1.0 / n] * n
        self._write_snapshot(tmp_path, "20260101T000000", uniform)
        self._write_snapshot(tmp_path, "20260201T000000", uniform)

        result = compare_shap_snapshots(output_dir=tmp_path)
        for key in (
            "previous_snapshot",
            "current_snapshot",
            "feature_comparison",
            "brittleness_warning",
            "max_shap_share_feature",
            "max_shap_share",
            "warnings",
            "ts",
        ):
            assert key in result

    def test_identifies_correct_previous_and_current(self, tmp_path: Path) -> None:
        n = len(QUOTE_FEATURE_COLS)
        uniform = [1.0 / n] * n
        self._write_snapshot(tmp_path, "20260101T000000", uniform)
        self._write_snapshot(tmp_path, "20260201T000000", uniform)

        result = compare_shap_snapshots(output_dir=tmp_path)
        assert "20260101" in result["previous_snapshot"]
        assert "20260201" in result["current_snapshot"]

    def test_feature_comparison_covers_all_features(self, tmp_path: Path) -> None:
        n = len(QUOTE_FEATURE_COLS)
        uniform = [1.0 / n] * n
        self._write_snapshot(tmp_path, "20260101T000000", uniform)
        self._write_snapshot(tmp_path, "20260201T000000", uniform)

        result = compare_shap_snapshots(output_dir=tmp_path)
        returned_features = {row["feature"] for row in result["feature_comparison"]}
        assert returned_features == set(QUOTE_FEATURE_COLS)

    def test_delta_computed_correctly(self, tmp_path: Path) -> None:
        n = len(QUOTE_FEATURE_COLS)
        uniform = [1.0 / n] * n
        shifted = uniform.copy()
        shifted[0] = shifted[0] + 0.10
        # Normalize to sum=1
        total = sum(shifted)
        shifted = [s / total for s in shifted]

        self._write_snapshot(tmp_path, "20260101T000000", uniform)
        self._write_snapshot(tmp_path, "20260201T000000", shifted)

        result = compare_shap_snapshots(output_dir=tmp_path)
        feat0 = QUOTE_FEATURE_COLS[0]
        row = next(r for r in result["feature_comparison"] if r["feature"] == feat0)
        assert row["delta"] == pytest.approx(row["curr_share"] - row["prev_share"], abs=1e-5)

    def test_no_warnings_when_below_brittleness_threshold(self, tmp_path: Path) -> None:
        n = len(QUOTE_FEATURE_COLS)
        # Each feature well below 40%
        uniform = [1.0 / n] * n
        assert max(uniform) < SHAP_BRITTLENESS_THRESHOLD

        self._write_snapshot(tmp_path, "20260101T000000", uniform)
        self._write_snapshot(tmp_path, "20260201T000000", uniform)

        result = compare_shap_snapshots(output_dir=tmp_path)
        assert result["brittleness_warning"] is False
        assert result["warnings"] == []

    def test_warning_emitted_when_above_brittleness_threshold(self, tmp_path: Path) -> None:
        n = len(QUOTE_FEATURE_COLS)
        # First feature gets 50% share (above 40% threshold)
        shares = [0.50] + [0.50 / (n - 1)] * (n - 1)

        self._write_snapshot(tmp_path, "20260101T000000", shares)
        self._write_snapshot(tmp_path, "20260201T000000", shares)

        result = compare_shap_snapshots(output_dir=tmp_path)
        assert result["brittleness_warning"] is True
        assert len(result["warnings"]) >= 1
        assert "BRITTLENESS" in result["warnings"][0]

    def test_rank_shift_zero_for_identical_snapshots(self, tmp_path: Path) -> None:
        n = len(QUOTE_FEATURE_COLS)
        uniform = [1.0 / n] * n
        self._write_snapshot(tmp_path, "20260101T000000", uniform)
        self._write_snapshot(tmp_path, "20260201T000000", uniform)

        result = compare_shap_snapshots(output_dir=tmp_path)
        for row in result["feature_comparison"]:
            assert row["rank_shift"] == 0

    def test_feature_comparison_sorted_descending_by_current_share(
        self, tmp_path: Path
    ) -> None:
        n = len(QUOTE_FEATURE_COLS)
        rng = np.random.default_rng(7)
        shares_raw = rng.dirichlet(np.ones(n))
        shares = shares_raw.tolist()

        self._write_snapshot(tmp_path, "20260101T000000", [1.0 / n] * n)
        self._write_snapshot(tmp_path, "20260201T000000", shares)

        result = compare_shap_snapshots(output_dir=tmp_path)
        curr_shares = [r["curr_share"] for r in result["feature_comparison"]]
        assert curr_shares == sorted(curr_shares, reverse=True)
