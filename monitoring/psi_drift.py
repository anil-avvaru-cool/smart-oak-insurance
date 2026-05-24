"""PSI (Population Stability Index) drift monitoring — Phase 6 (DEC-013).

Time axis:
  - Quotes keyed on ``quote_requested_at``
  - Claims keyed on ``fnol_submitted_at``

Reference cohort: all rows whose timestamp precedes the current window.
Current cohort:   rolling ``PSI_CURRENT_WINDOW_DAYS`` window ending at ``as_of``.

Null treatment: nulls are **never** filtered before PSI computation.
They form their own bin so that a shift in null rate is visible as drift.

Champion-challenger trigger: any feature with PSI >= 0.25 sets
``champion_challenger_triggered = True`` on the dataset report.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from data.config import (
    CLAIMS_OUTPUT,
    PSI_CURRENT_WINDOW_DAYS,
    PSI_MIN_RECORDS,
    PSI_REFERENCE_VERSION,
    QUOTES_OUTPUT,
)

logger = logging.getLogger(__name__)

# Industry-standard PSI thresholds
PSI_STABLE_THRESHOLD = 0.10
PSI_MAJOR_DRIFT_THRESHOLD = 0.25
_EPSILON = 1e-9
_N_BINS_DEFAULT = 10

# Concept drift label confirmation windows (days from fnol_submitted_at)
LABEL_WINDOW_90D = 90
LABEL_WINDOW_180D = 180

# ── Feature lists ────────────────────────────────────────────────────────────

QUOTE_NUMERIC_FEATURES: list[str] = [
    "credit_score",
    "prior_loss_frequency",
    "prior_loss_severity_avg",
    "insurance_lapse_days",
    "violation_severity_index",
    "driver_age",
    "vehicle_msrp",
    "vehicle_age_years",
    "annual_mileage_estimate",
    "geohash_risk_score",
    "risk_score_at_issuance",
    "vehicle_adas_score",
]
QUOTE_CATEGORICAL_FEATURES: list[str] = ["state", "policy_tier_at_issuance"]
QUOTE_BOOLEAN_FEATURES: list[str] = ["telematics_enrolled"]

CLAIM_NUMERIC_FEATURES: list[str] = [
    "reporting_delay_days",
    "policy_inception_days",
    "claimant_count",
    "incurred_loss_usd",
    "narrative_inconsistency_score",
    "narrative_complexity_score",
    "risk_score_at_issuance",
    "ip_geolocation_delta_miles",
]
CLAIM_CATEGORICAL_FEATURES: list[str] = ["submission_channel"]
CLAIM_BOOLEAN_FEATURES: list[str] = ["attorney_present", "is_fraud"]

# ── Types ────────────────────────────────────────────────────────────────────

DriftStatus = Literal["stable", "minor_shift", "major_shift", "insufficient_data"]


def _drift_status(psi: float) -> DriftStatus:
    if psi < PSI_STABLE_THRESHOLD:
        return "stable"
    if psi < PSI_MAJOR_DRIFT_THRESHOLD:
        return "minor_shift"
    return "major_shift"


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class BinDetail:
    bin_label: str
    ref_count: int
    curr_count: int
    ref_fraction: float
    curr_fraction: float
    psi_contribution: float


@dataclass
class PSIFeatureResult:
    feature: str
    psi: float
    n_reference: int
    n_current: int
    null_fraction_reference: float
    null_fraction_current: float
    status: DriftStatus
    bins: list[BinDetail] = field(default_factory=list)


@dataclass
class LabelWindowReport:
    """Fraud label confirmation rates for the 90- and 180-day concept drift windows."""

    total_claims_in_window: int
    fraud_claims_in_window: int
    confirmed_any: int
    confirmed_90d: int
    confirmed_180d: int
    confirmation_rate_any: float
    confirmation_rate_90d: float
    confirmation_rate_180d: float
    as_of: pd.Timestamp


@dataclass
class PSIDatasetReport:
    dataset: str  # "quotes" or "claims"
    reference_version: str
    current_window_days: int
    current_window_start: pd.Timestamp
    current_window_end: pd.Timestamp
    n_reference: int
    n_current: int
    feature_results: list[PSIFeatureResult]
    max_psi: float
    champion_challenger_triggered: bool
    label_window: LabelWindowReport | None  # populated for claims only
    generated_at: pd.Timestamp


class InsufficientDataError(ValueError):
    """Current window has fewer records than PSI_MIN_RECORDS."""


# ── Core PSI computation ─────────────────────────────────────────────────────


def _psi_from_fractions(
    ref_fractions: np.ndarray,
    curr_fractions: np.ndarray,
    bin_labels: list[str],
    ref_counts: np.ndarray,
    curr_counts: np.ndarray,
) -> tuple[float, list[BinDetail]]:
    ref_c = np.clip(ref_fractions, _EPSILON, None)
    curr_c = np.clip(curr_fractions, _EPSILON, None)
    contributions = (curr_c - ref_c) * np.log(curr_c / ref_c)
    psi = float(np.sum(contributions))
    bins = [
        BinDetail(
            bin_label=lbl,
            ref_count=int(rc),
            curr_count=int(cc),
            ref_fraction=float(rf),
            curr_fraction=float(cf),
            psi_contribution=float(contrib),
        )
        for lbl, rc, cc, rf, cf, contrib in zip(
            bin_labels, ref_counts, curr_counts,
            ref_fractions, curr_fractions, contributions,
        )
    ]
    return psi, bins


def _bin_numeric(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = _N_BINS_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Bin numeric series with a dedicated null bin.

    Bin edges are derived from reference quantiles and extended to ±inf so
    current values outside the reference range fall into the first / last bin
    rather than being lost.
    """
    ref_null = reference.isna()
    curr_null = current.isna()
    ref_nonnull = reference.dropna().astype(float)
    curr_nonnull = current.dropna().astype(float)
    n_ref = len(reference)
    n_curr = len(current)

    if len(ref_nonnull) == 0:
        # All reference values are null — single null bin only
        ref_counts = np.array([float(ref_null.sum())])
        curr_counts = np.array([float(curr_null.sum())])
        return ref_counts, curr_counts, ref_counts / max(n_ref, 1), curr_counts / max(n_curr, 1), ["(null)"]

    # Compute quantile cut-points on reference non-null values
    quantile_probs = np.linspace(0.0, 1.0, n_bins + 1)
    inner_edges = np.unique(np.quantile(ref_nonnull, quantile_probs))

    # Extend to ±inf so all finite current values land in a bin
    if len(inner_edges) >= 2:
        bin_edges: np.ndarray = np.concatenate([[-np.inf], inner_edges[1:-1], [np.inf]])
    else:
        bin_edges = np.array([-np.inf, np.inf])

    ref_cut = pd.cut(ref_nonnull, bins=bin_edges, include_lowest=True, right=True)
    curr_cut = pd.cut(curr_nonnull, bins=bin_edges, include_lowest=True, right=True)

    cats = ref_cut.cat.categories
    ref_bin = ref_cut.value_counts(sort=False).reindex(cats, fill_value=0).values.astype(float)
    curr_bin = curr_cut.value_counts(sort=False).reindex(cats, fill_value=0).values.astype(float)

    # Append null bin
    ref_counts = np.append(ref_bin, float(ref_null.sum()))
    curr_counts = np.append(curr_bin, float(curr_null.sum()))

    bin_labels = [str(c) for c in cats] + ["(null)"]
    return (
        ref_counts,
        curr_counts,
        ref_counts / max(n_ref, 1),
        curr_counts / max(n_curr, 1),
        bin_labels,
    )


def _bin_categorical(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Bin categorical / boolean series.

    Bins: one per reference category + ``(unseen)`` + ``(null)``.
    """
    ref_null = reference.isna()
    curr_null = current.isna()
    ref_nonnull = reference.dropna().astype(str)
    curr_nonnull = current.dropna().astype(str)
    n_ref = len(reference)
    n_curr = len(current)

    ref_cats = sorted(ref_nonnull.unique().tolist())
    ref_vc = ref_nonnull.value_counts()
    curr_vc = curr_nonnull.value_counts()

    ref_counts = np.array([float(ref_vc.get(c, 0)) for c in ref_cats] + [0.0, float(ref_null.sum())])
    curr_counts = np.array(
        [float(curr_vc.get(c, 0)) for c in ref_cats]
        + [float((~curr_nonnull.isin(ref_cats)).sum()), float(curr_null.sum())]
    )
    bin_labels = ref_cats + ["(unseen)", "(null)"]
    return (
        ref_counts,
        curr_counts,
        ref_counts / max(n_ref, 1),
        curr_counts / max(n_curr, 1),
        bin_labels,
    )


def compute_psi_series(
    reference: pd.Series,
    current: pd.Series,
    is_categorical: bool = False,
    n_bins: int = _N_BINS_DEFAULT,
) -> tuple[float, list[BinDetail]]:
    """Compute PSI between ``reference`` and ``current`` series.

    Nulls are treated as a dedicated bin — never filtered before PSI.
    Returns ``(psi_value, per_bin_details)``.
    """
    if is_categorical:
        ref_counts, curr_counts, ref_fracs, curr_fracs, labels = _bin_categorical(reference, current)
    else:
        ref_counts, curr_counts, ref_fracs, curr_fracs, labels = _bin_numeric(reference, current, n_bins)
    return _psi_from_fractions(ref_fracs, curr_fracs, labels, ref_counts, curr_counts)


# ── Feature-level PSI ────────────────────────────────────────────────────────


def _feature_psi(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    feature: str,
    is_categorical: bool,
) -> PSIFeatureResult:
    ref_s = ref_df[feature] if feature in ref_df.columns else pd.Series(dtype=float, name=feature)
    curr_s = curr_df[feature] if feature in curr_df.columns else pd.Series(dtype=float, name=feature)

    n_ref = len(ref_s)
    n_curr = len(curr_s)

    if n_ref == 0 or n_curr == 0:
        return PSIFeatureResult(
            feature=feature,
            psi=0.0,
            n_reference=n_ref,
            n_current=n_curr,
            null_fraction_reference=0.0,
            null_fraction_current=0.0,
            status="insufficient_data",
            bins=[],
        )

    ref_null_frac = float(ref_s.isna().mean())
    curr_null_frac = float(curr_s.isna().mean())
    psi, bins = compute_psi_series(ref_s, curr_s, is_categorical=is_categorical)

    return PSIFeatureResult(
        feature=feature,
        psi=round(psi, 6),
        n_reference=n_ref,
        n_current=n_curr,
        null_fraction_reference=ref_null_frac,
        null_fraction_current=curr_null_frac,
        status=_drift_status(psi),
        bins=bins,
    )


# ── Label window query ───────────────────────────────────────────────────────


def compute_label_window(
    claims_df: pd.DataFrame,
    as_of: pd.Timestamp,
    window_days: int = PSI_CURRENT_WINDOW_DAYS,
) -> LabelWindowReport:
    """Query 90/180-day fraud label confirmation rates in the current window.

    Filters claims whose ``fnol_submitted_at`` falls within the rolling window.
    A drop in confirmation rate signals that the label window has not yet closed —
    this is expected but should be tracked to detect true concept drift.
    """
    window_start = as_of - pd.Timedelta(days=window_days)
    fnol = pd.to_datetime(claims_df["fnol_submitted_at"])
    in_window = claims_df[(fnol >= window_start) & (fnol <= as_of)]

    fraud = in_window[in_window["is_fraud"].fillna(False).astype(bool)]
    total = len(in_window)
    fraud_total = len(fraud)

    if fraud_total == 0:
        return LabelWindowReport(
            total_claims_in_window=total,
            fraud_claims_in_window=0,
            confirmed_any=0,
            confirmed_90d=0,
            confirmed_180d=0,
            confirmation_rate_any=0.0,
            confirmation_rate_90d=0.0,
            confirmation_rate_180d=0.0,
            as_of=as_of,
        )

    confirmed = fraud[fraud["fraud_confirmed_at"].notna()].copy()
    confirmed_any = len(confirmed)

    if confirmed_any > 0:
        lag = (
            pd.to_datetime(confirmed["fraud_confirmed_at"])
            - pd.to_datetime(confirmed["fnol_submitted_at"])
        ).dt.days
        confirmed_90d = int((lag <= LABEL_WINDOW_90D).sum())
        confirmed_180d = int((lag <= LABEL_WINDOW_180D).sum())
    else:
        confirmed_90d = 0
        confirmed_180d = 0

    return LabelWindowReport(
        total_claims_in_window=total,
        fraud_claims_in_window=fraud_total,
        confirmed_any=confirmed_any,
        confirmed_90d=confirmed_90d,
        confirmed_180d=confirmed_180d,
        confirmation_rate_any=confirmed_any / fraud_total,
        confirmation_rate_90d=confirmed_90d / fraud_total,
        confirmation_rate_180d=confirmed_180d / fraud_total,
        as_of=as_of,
    )


# ── Dataset-level PSI ────────────────────────────────────────────────────────


def _run_dataset_psi(
    df: pd.DataFrame,
    time_col: str,
    numeric_features: list[str],
    categorical_features: list[str],
    boolean_features: list[str],
    dataset_name: str,
    reference_version: str,
    current_window_days: int,
    min_records: int,
    as_of: pd.Timestamp,
    label_window: LabelWindowReport | None = None,
) -> PSIDatasetReport:
    window_end = as_of
    window_start = as_of - pd.Timedelta(days=current_window_days)

    ts = pd.to_datetime(df[time_col])
    curr_df = df[(ts >= window_start) & (ts <= window_end)]
    ref_df = df[ts < window_start]

    n_curr = len(curr_df)
    if n_curr < min_records:
        raise InsufficientDataError(
            f"{dataset_name}: current window {window_start.date()} → {window_end.date()} "
            f"has {n_curr} records (minimum {min_records}). "
            f"Run --generate-data to produce more records."
        )

    results: list[PSIFeatureResult] = []
    for feat in numeric_features:
        results.append(_feature_psi(ref_df, curr_df, feat, is_categorical=False))
    for feat in categorical_features + boolean_features:
        results.append(_feature_psi(ref_df, curr_df, feat, is_categorical=True))

    max_psi = max((r.psi for r in results), default=0.0)
    cc_triggered = any(r.psi >= PSI_MAJOR_DRIFT_THRESHOLD for r in results)

    if cc_triggered:
        drifted = [r.feature for r in results if r.psi >= PSI_MAJOR_DRIFT_THRESHOLD]
        logger.warning(
            "%s drift: champion-challenger loop triggered. max_psi=%.4f features=%s",
            dataset_name, max_psi, drifted,
        )

    return PSIDatasetReport(
        dataset=dataset_name,
        reference_version=reference_version,
        current_window_days=current_window_days,
        current_window_start=window_start,
        current_window_end=window_end,
        n_reference=len(ref_df),
        n_current=n_curr,
        feature_results=results,
        max_psi=max_psi,
        champion_challenger_triggered=cc_triggered,
        label_window=label_window,
        generated_at=as_of,
    )


# ── Public entry point ───────────────────────────────────────────────────────


def run_drift_check(
    quotes_path: Path = QUOTES_OUTPUT,
    claims_path: Path = CLAIMS_OUTPUT,
    current_window_days: int = PSI_CURRENT_WINDOW_DAYS,
    min_records: int = PSI_MIN_RECORDS,
    reference_version: str = PSI_REFERENCE_VERSION,
    as_of: pd.Timestamp | None = None,
) -> dict[str, PSIDatasetReport]:
    """Compute PSI drift for quotes and claims.

    Reads directly from ``data/raw/`` parquet files — not from the feature store.
    Returns ``{"quotes": PSIDatasetReport, "claims": PSIDatasetReport}``.
    Raises ``InsufficientDataError`` if either current window is below ``min_records``.
    """
    if as_of is None:
        as_of = pd.Timestamp.now()

    quotes_df = pd.read_parquet(quotes_path)
    claims_df = pd.read_parquet(claims_path)

    label_window = compute_label_window(claims_df, as_of=as_of, window_days=current_window_days)

    quotes_report = _run_dataset_psi(
        df=quotes_df,
        time_col="quote_requested_at",
        numeric_features=QUOTE_NUMERIC_FEATURES,
        categorical_features=QUOTE_CATEGORICAL_FEATURES,
        boolean_features=QUOTE_BOOLEAN_FEATURES,
        dataset_name="quotes",
        reference_version=reference_version,
        current_window_days=current_window_days,
        min_records=min_records,
        as_of=as_of,
    )

    claims_report = _run_dataset_psi(
        df=claims_df,
        time_col="fnol_submitted_at",
        numeric_features=CLAIM_NUMERIC_FEATURES,
        categorical_features=CLAIM_CATEGORICAL_FEATURES,
        boolean_features=CLAIM_BOOLEAN_FEATURES,
        dataset_name="claims",
        reference_version=reference_version,
        current_window_days=current_window_days,
        min_records=min_records,
        as_of=as_of,
        label_window=label_window,
    )

    return {"quotes": quotes_report, "claims": claims_report}
