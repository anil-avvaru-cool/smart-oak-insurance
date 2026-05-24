"""Unit tests for monitoring/psi_drift.py.

All tests operate on synthetic in-memory Series/DataFrames — no parquet I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring.psi_drift import (
    LABEL_WINDOW_90D,
    LABEL_WINDOW_180D,
    PSI_MAJOR_DRIFT_THRESHOLD,
    PSI_STABLE_THRESHOLD,
    InsufficientDataError,
    LabelWindowReport,
    PSIDatasetReport,
    PSIFeatureResult,
    _drift_status,
    _run_dataset_psi,
    compute_label_window,
    compute_psi_series,
)


# ── _drift_status ─────────────────────────────────────────────────────────────


class TestDriftStatus:
    def test_stable(self) -> None:
        assert _drift_status(0.05) == "stable"

    def test_minor_shift(self) -> None:
        assert _drift_status(0.15) == "minor_shift"

    def test_major_shift(self) -> None:
        assert _drift_status(0.30) == "major_shift"

    def test_boundary_stable_upper(self) -> None:
        assert _drift_status(PSI_STABLE_THRESHOLD - 1e-12) == "stable"

    def test_boundary_minor_at_stable(self) -> None:
        assert _drift_status(PSI_STABLE_THRESHOLD) == "minor_shift"

    def test_boundary_major(self) -> None:
        assert _drift_status(PSI_MAJOR_DRIFT_THRESHOLD) == "major_shift"


# ── compute_psi_series — numeric ──────────────────────────────────────────────


class TestComputePsiSeriesNumeric:
    def test_identical_series_psi_near_zero(self) -> None:
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(50, 10, 2000))
        psi, _ = compute_psi_series(data, data.copy())
        assert psi < PSI_STABLE_THRESHOLD

    def test_heavily_shifted_distribution_triggers_major_drift(self) -> None:
        rng = np.random.default_rng(42)
        reference = pd.Series(rng.normal(0.0, 1.0, 2000))
        current = pd.Series(rng.normal(8.0, 1.0, 2000))
        psi, _ = compute_psi_series(reference, current)
        assert psi >= PSI_MAJOR_DRIFT_THRESHOLD

    def test_null_bin_always_present(self) -> None:
        reference = pd.Series([1.0, 2.0, 3.0, None, None])
        current = pd.Series([1.0, 2.0, 3.0, 4.0])
        _, bins = compute_psi_series(reference, current)
        assert any(b.bin_label == "(null)" for b in bins)

    def test_shifting_null_rate_increases_psi(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.normal(5.0, 1.0, 800).tolist()
        ref = pd.Series(base + [None] * 200)         # 20 % null
        curr_shifted = pd.Series(base[:500] + [None] * 500)  # 50 % null
        psi_shifted, _ = compute_psi_series(ref, curr_shifted)
        psi_same, _ = compute_psi_series(ref, ref.copy())
        assert psi_shifted > psi_same

    def test_all_nulls_reference_returns_one_null_bin(self) -> None:
        ref = pd.Series([None, None, None], dtype=float)
        curr = pd.Series([None, None, None], dtype=float)
        psi, bins = compute_psi_series(ref, curr)
        assert len(bins) == 1
        assert bins[0].bin_label == "(null)"
        assert psi < PSI_STABLE_THRESHOLD

    def test_psi_contributions_sum_to_total(self) -> None:
        rng = np.random.default_rng(7)
        ref = pd.Series(rng.normal(0.0, 1.0, 500))
        curr = pd.Series(rng.normal(2.0, 1.0, 500))
        psi, bins = compute_psi_series(ref, curr)
        assert abs(sum(b.psi_contribution for b in bins) - psi) < 1e-6

    def test_bin_ref_counts_sum_to_n_reference(self) -> None:
        rng = np.random.default_rng(3)
        ref = pd.Series(rng.uniform(0, 10, 300).tolist() + [None] * 50)
        curr = pd.Series(rng.uniform(0, 10, 200).tolist() + [None] * 30)
        _, bins = compute_psi_series(ref, curr)
        assert sum(b.ref_count for b in bins) == len(ref)

    def test_bin_curr_counts_sum_to_n_current(self) -> None:
        rng = np.random.default_rng(3)
        ref = pd.Series(rng.uniform(0, 10, 300).tolist() + [None] * 50)
        curr = pd.Series(rng.uniform(0, 10, 200).tolist() + [None] * 30)
        _, bins = compute_psi_series(ref, curr)
        assert sum(b.curr_count for b in bins) == len(curr)

    def test_current_values_outside_reference_range_land_in_first_last_bin(self) -> None:
        # Reference: [0, 1]; Current: includes values well outside that range
        ref = pd.Series(np.linspace(0.0, 1.0, 200))
        curr = pd.Series(np.linspace(-5.0, 10.0, 200))
        _, bins = compute_psi_series(ref, curr)
        # All current counts must be accounted for (no values lost)
        assert sum(b.curr_count for b in bins) == len(curr)


# ── compute_psi_series — categorical ─────────────────────────────────────────


class TestComputePsiSeriesCategorical:
    def test_identical_series_psi_near_zero(self) -> None:
        cats = pd.Series(["A", "B", "C", "A", "B"] * 400)
        psi, _ = compute_psi_series(cats, cats.copy(), is_categorical=True)
        assert psi < PSI_STABLE_THRESHOLD

    def test_new_category_in_current_captured_in_unseen_bin(self) -> None:
        reference = pd.Series(["A", "B"] * 500)
        current = pd.Series(["A", "B", "C", "C", "C"] * 200)
        psi, bins = compute_psi_series(reference, current, is_categorical=True)
        unseen = next(b for b in bins if b.bin_label == "(unseen)")
        assert unseen.curr_count > 0
        assert unseen.ref_count == 0

    def test_null_bin_present(self) -> None:
        reference = pd.Series(["A", "B", None, "A"])
        current = pd.Series(["A", "B", "A", "B"])
        _, bins = compute_psi_series(reference, current, is_categorical=True)
        assert any(b.bin_label == "(null)" for b in bins)

    def test_fully_shifted_categories_trigger_major_drift(self) -> None:
        reference = pd.Series(["A"] * 500)
        current = pd.Series(["B"] * 500)
        psi, _ = compute_psi_series(reference, current, is_categorical=True)
        assert psi >= PSI_MAJOR_DRIFT_THRESHOLD

    def test_bin_counts_complete_for_categorical(self) -> None:
        reference = pd.Series(["X", "Y", None] * 100)
        current = pd.Series(["X", "Z", None] * 80)
        _, bins = compute_psi_series(reference, current, is_categorical=True)
        assert sum(b.ref_count for b in bins) == len(reference)
        assert sum(b.curr_count for b in bins) == len(current)


# ── compute_label_window ──────────────────────────────────────────────────────


class TestComputeLabelWindow:
    _AS_OF = pd.Timestamp("2025-06-01")

    def _make_claims(
        self,
        n_window: int = 20,
        n_outside: int = 100,
        fraud_rate: float = 0.5,
        confirm_lag_days: int = 60,
    ) -> pd.DataFrame:
        """Build a minimal claims DataFrame with the required columns."""
        rows: list[dict] = []
        as_of = self._AS_OF

        # Claims outside the window (reference cohort)
        for i in range(n_outside):
            fnol = as_of - pd.Timedelta(days=30 + i)
            rows.append({"fnol_submitted_at": fnol, "is_fraud": False, "fraud_confirmed_at": None})

        # Claims inside the 14-day window
        for i in range(n_window):
            fnol = as_of - pd.Timedelta(days=i)
            is_fraud = i % int(1 / fraud_rate) == 0 if fraud_rate > 0 else False
            if is_fraud:
                confirmed = fnol + pd.Timedelta(days=confirm_lag_days)
            else:
                confirmed = None
            rows.append({"fnol_submitted_at": fnol, "is_fraud": is_fraud, "fraud_confirmed_at": confirmed})

        return pd.DataFrame(rows)

    def test_window_filters_correct_records(self) -> None:
        df = self._make_claims(n_window=14, n_outside=100)
        report = compute_label_window(df, as_of=self._AS_OF, window_days=14)
        assert report.total_claims_in_window == 14

    def test_confirmation_rates_bounded_0_to_1(self) -> None:
        df = self._make_claims(n_window=20, confirm_lag_days=60)
        report = compute_label_window(df, as_of=self._AS_OF, window_days=14)
        assert 0.0 <= report.confirmation_rate_any <= 1.0
        assert 0.0 <= report.confirmation_rate_90d <= 1.0
        assert 0.0 <= report.confirmation_rate_180d <= 1.0

    def test_90d_leq_180d_confirmation_rate(self) -> None:
        df = self._make_claims(n_window=40, confirm_lag_days=100)
        report = compute_label_window(df, as_of=self._AS_OF, window_days=14)
        assert report.confirmed_90d <= report.confirmed_180d
        assert report.confirmation_rate_90d <= report.confirmation_rate_180d

    def test_fast_confirm_lag_all_within_90d(self) -> None:
        df = self._make_claims(n_window=20, fraud_rate=1.0, confirm_lag_days=30)
        report = compute_label_window(df, as_of=self._AS_OF, window_days=14)
        if report.fraud_claims_in_window > 0:
            assert report.confirmed_90d == report.confirmed_any

    def test_slow_confirm_lag_none_within_90d(self) -> None:
        df = self._make_claims(n_window=20, fraud_rate=1.0, confirm_lag_days=150)
        report = compute_label_window(df, as_of=self._AS_OF, window_days=14)
        assert report.confirmed_90d == 0

    def test_no_fraud_returns_zero_rates(self) -> None:
        df = pd.DataFrame({
            "fnol_submitted_at": [self._AS_OF - pd.Timedelta(days=i) for i in range(5)],
            "is_fraud": [False] * 5,
            "fraud_confirmed_at": [None] * 5,
        })
        report = compute_label_window(df, as_of=self._AS_OF, window_days=14)
        assert report.fraud_claims_in_window == 0
        assert report.confirmation_rate_any == 0.0

    def test_as_of_matches_report(self) -> None:
        df = self._make_claims()
        report = compute_label_window(df, as_of=self._AS_OF, window_days=14)
        assert report.as_of == self._AS_OF


# ── _run_dataset_psi ──────────────────────────────────────────────────────────


class TestRunDatasetPsi:
    _AS_OF = pd.Timestamp("2025-06-01")

    def _make_df(self, n: int = 800, spread_days: int = 60) -> pd.DataFrame:
        rng = np.random.default_rng(10)
        dates = pd.date_range(end=self._AS_OF - pd.Timedelta(days=1), periods=n, freq="h")
        return pd.DataFrame({
            "quote_requested_at": dates,
            "score": rng.normal(0.5, 0.15, n),
            "category": rng.choice(["A", "B", "C"], n),
            "flag": rng.random(n) > 0.5,
        })

    def test_raises_insufficient_data_error(self) -> None:
        df = self._make_df(n=10)
        with pytest.raises(InsufficientDataError):
            _run_dataset_psi(
                df=df,
                time_col="quote_requested_at",
                numeric_features=["score"],
                categorical_features=["category"],
                boolean_features=["flag"],
                dataset_name="quotes",
                reference_version="v1.0.0",
                current_window_days=14,
                min_records=500,
                as_of=self._AS_OF,
            )

    def test_returns_correct_report_structure(self) -> None:
        df = self._make_df(n=800)
        report = _run_dataset_psi(
            df=df,
            time_col="quote_requested_at",
            numeric_features=["score"],
            categorical_features=["category"],
            boolean_features=["flag"],
            dataset_name="quotes",
            reference_version="v1.0.0",
            current_window_days=14,
            min_records=5,
            as_of=self._AS_OF,
        )
        assert isinstance(report, PSIDatasetReport)
        assert report.dataset == "quotes"
        assert report.reference_version == "v1.0.0"
        assert report.n_current > 0
        assert report.n_reference > 0
        assert len(report.feature_results) == 3  # score + category + flag
        assert report.max_psi == max(r.psi for r in report.feature_results)
        assert report.label_window is None

    def test_major_drift_triggers_champion_challenger(self) -> None:
        rng = np.random.default_rng(99)
        as_of = self._AS_OF
        window_start = as_of - pd.Timedelta(days=14)

        # Reference: stable distribution
        ref_dates = pd.date_range(end=window_start - pd.Timedelta(hours=1), periods=500, freq="h")
        ref_scores = rng.normal(0.2, 0.05, 500)

        # Current: shifted by ~8 standard deviations — guaranteed major drift
        curr_dates = pd.date_range(start=window_start, periods=50, freq="h")
        curr_scores = rng.normal(0.9, 0.05, 50)

        df = pd.DataFrame({
            "quote_requested_at": list(ref_dates) + list(curr_dates),
            "score": list(ref_scores) + list(curr_scores),
        })

        report = _run_dataset_psi(
            df=df,
            time_col="quote_requested_at",
            numeric_features=["score"],
            categorical_features=[],
            boolean_features=[],
            dataset_name="quotes",
            reference_version="v1.0.0",
            current_window_days=14,
            min_records=1,
            as_of=as_of,
        )
        assert report.champion_challenger_triggered is True
        assert report.max_psi >= PSI_MAJOR_DRIFT_THRESHOLD

    def test_stable_distribution_does_not_trigger_champion_challenger(self) -> None:
        rng = np.random.default_rng(5)
        as_of = self._AS_OF

        dates = pd.date_range(end=as_of - pd.Timedelta(hours=1), periods=1000, freq="h")
        scores = rng.normal(0.5, 0.1, 1000)

        df = pd.DataFrame({"quote_requested_at": dates, "score": scores})

        report = _run_dataset_psi(
            df=df,
            time_col="quote_requested_at",
            numeric_features=["score"],
            categorical_features=[],
            boolean_features=[],
            dataset_name="quotes",
            reference_version="v1.0.0",
            current_window_days=14,
            min_records=5,
            as_of=as_of,
        )
        assert report.champion_challenger_triggered is False

    def test_missing_feature_column_returns_insufficient_data_status(self) -> None:
        df = self._make_df(n=800)
        report = _run_dataset_psi(
            df=df,
            time_col="quote_requested_at",
            numeric_features=["nonexistent_feature"],
            categorical_features=[],
            boolean_features=[],
            dataset_name="quotes",
            reference_version="v1.0.0",
            current_window_days=14,
            min_records=5,
            as_of=self._AS_OF,
        )
        result = report.feature_results[0]
        assert result.feature == "nonexistent_feature"
        assert result.status == "insufficient_data"
        assert result.psi == 0.0

    def test_label_window_passed_through_to_claims_report(self) -> None:
        df = self._make_df(n=800)
        label_window = LabelWindowReport(
            total_claims_in_window=10,
            fraud_claims_in_window=5,
            confirmed_any=3,
            confirmed_90d=2,
            confirmed_180d=3,
            confirmation_rate_any=0.6,
            confirmation_rate_90d=0.4,
            confirmation_rate_180d=0.6,
            as_of=self._AS_OF,
        )
        report = _run_dataset_psi(
            df=df,
            time_col="quote_requested_at",
            numeric_features=["score"],
            categorical_features=[],
            boolean_features=[],
            dataset_name="claims",
            reference_version="v1.0.0",
            current_window_days=14,
            min_records=5,
            as_of=self._AS_OF,
            label_window=label_window,
        )
        assert report.label_window is label_window
