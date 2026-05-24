"""
Phase 8 tests — DEC-013 temporal integrity (8.1), snapshot audit datetime fields (8.3),
and telematics_enrolled_but_missing signal path (8.4).
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.generator import generate_claims, generate_quotes
from features.feature_definitions import build_claim_feature_vector, build_telematics_features
from features.offline_pipeline import build_claim_snapshot, build_quote_snapshot
from data.validator import _check_snapshot_envelope


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generated_data():
    quotes_df = generate_quotes(seed=42)
    claims_df = generate_claims(quotes_df, seed=43)
    return quotes_df, claims_df


# ---------------------------------------------------------------------------
# 8.1 — DEC-013 temporal ordering and datetime arithmetic consistency
# ---------------------------------------------------------------------------

class TestTemporalOrdering:
    def test_quote_requested_at_populated_for_all_quotes(self, generated_data):
        quotes_df, _ = generated_data
        assert "quote_requested_at" in quotes_df.columns
        assert quotes_df["quote_requested_at"].notna().all(), "quote_requested_at must be populated for every quote"

    def test_quote_completed_at_after_requested_at(self, generated_data):
        quotes_df, _ = generated_data
        assert (quotes_df["quote_completed_at"] >= quotes_df["quote_requested_at"]).all()

    def test_fnol_submitted_at_populated_for_all_claims(self, generated_data):
        _, claims_df = generated_data
        assert "fnol_submitted_at" in claims_df.columns
        assert claims_df["fnol_submitted_at"].notna().all(), "fnol_submitted_at must be populated for every claim"

    def test_fnol_after_loss_event(self, generated_data):
        _, claims_df = generated_data
        fnol = pd.to_datetime(claims_df["fnol_submitted_at"])
        loss = pd.to_datetime(claims_df["loss_event_datetime"])
        bad = (fnol < loss).sum()
        assert bad == 0, f"{bad} claims have fnol_submitted_at before loss_event_datetime"

    def test_claim_closed_at_after_fnol(self, generated_data):
        _, claims_df = generated_data
        fnol = pd.to_datetime(claims_df["fnol_submitted_at"])
        closed = pd.to_datetime(claims_df["claim_closed_at"])
        mask = closed.notna()
        assert mask.any(), "Expected some closed claims"
        bad = (closed[mask] < fnol[mask]).sum()
        assert bad == 0, f"{bad} closed claims have claim_closed_at before fnol_submitted_at"

    def test_fraud_confirmed_at_after_fnol(self, generated_data):
        _, claims_df = generated_data
        fnol = pd.to_datetime(claims_df["fnol_submitted_at"])
        confirmed = pd.to_datetime(claims_df["fraud_confirmed_at"])
        mask = confirmed.notna()
        assert mask.any(), "Expected some fraud-confirmed claims"
        bad = (confirmed[mask] < fnol[mask]).sum()
        assert bad == 0, f"{bad} claims have fraud_confirmed_at before fnol_submitted_at"

    def test_reporting_delay_days_consistent_with_datetime_diff(self, generated_data):
        _, claims_df = generated_data
        fnol = pd.to_datetime(claims_df["fnol_submitted_at"])
        loss = pd.to_datetime(claims_df["loss_event_datetime"])
        derived = (fnol - loss).dt.days
        delta = (claims_df["reporting_delay_days"] - derived).abs()
        # Allow ≤1 day tolerance for intra-day rounding
        bad = (delta > 1).sum()
        assert bad == 0, f"{bad} claims have reporting_delay_days inconsistent with fnol−loss diff"

    def test_fraud_confirmed_at_null_for_legitimate_claims(self, generated_data):
        _, claims_df = generated_data
        legit = claims_df[claims_df["is_fraud"] == False]
        bad = legit["fraud_confirmed_at"].notna().sum()
        assert bad == 0, f"{bad} legitimate claims have non-null fraud_confirmed_at"

    def test_policy_inception_days_non_negative(self, generated_data):
        _, claims_df = generated_data
        assert (claims_df["policy_inception_days"] >= 0).all(), "policy_inception_days must be non-negative"

    def test_reporting_delay_days_non_negative(self, generated_data):
        _, claims_df = generated_data
        assert (claims_df["reporting_delay_days"] >= 0).all(), "reporting_delay_days must be non-negative"


# ---------------------------------------------------------------------------
# 8.1 — DEC-005: graph feature stubs set correctly by generator
# ---------------------------------------------------------------------------

class TestGraphFeatureStubs:
    def test_graph_hop_distance_is_sentinel_before_neo4j(self, generated_data):
        _, claims_df = generated_data
        assert (claims_df["graph_hop_distance"] == 999).all(), (
            "graph_hop_distance must be 999 sentinel until --compute-graph-features runs"
        )

    def test_shared_attribute_count_is_zero_stub(self, generated_data):
        _, claims_df = generated_data
        assert (claims_df["shared_attribute_count"] == 0).all(), (
            "shared_attribute_count must be 0 stub until --compute-graph-features runs"
        )

    def test_attorney_centrality_score_is_zero_stub(self, generated_data):
        _, claims_df = generated_data
        assert (claims_df["attorney_centrality_score"] == 0.0).all(), (
            "attorney_centrality_score must be 0.0 stub until --compute-graph-features runs"
        )

    def test_graph_hop_sentinel_becomes_null_in_feature_vector(self):
        payload = {"graph_hop_distance": 999, "shared_attribute_count": 0, "attorney_centrality_score": 0.0}
        features = build_claim_feature_vector(payload)
        assert features["graph_hop_distance"] is None, (
            "feature builder must convert sentinel 999 → null so XGBoost uses its null path"
        )

    def test_non_sentinel_hop_distance_passes_through(self):
        payload = {"graph_hop_distance": 2}
        features = build_claim_feature_vector(payload)
        assert features["graph_hop_distance"] == 2


# ---------------------------------------------------------------------------
# 8.3 — Snapshot audit datetime fields and envelope validation
# ---------------------------------------------------------------------------

class TestSnapshotAuditDatetimeFields:
    def _minimal_quote_payload(self):
        return {
            "quote_id": "q-001",
            "state": "TX",
            "quote_requested_at": pd.Timestamp("2025-01-10 09:00:00"),
            "quote_completed_at": pd.Timestamp("2025-01-10 09:02:30"),
            "vehicle_msrp": 30_000.0,
            "vehicle_power": 180.0,
            "vehicle_adas_score": 0.5,
            "vehicle_age_years": 3,
            "risk_score_at_issuance": 0.35,
            "policy_tier_at_issuance": "gold",
        }

    def _minimal_claim_payload(self):
        return {
            "claim_id": "c-001",
            "state": "TX",
            "fnol_submitted_at": pd.Timestamp("2025-02-15 14:00:00"),
            "loss_event_datetime": pd.Timestamp("2025-02-12 18:30:00"),
            "policy_inception_days": 32,
            "reporting_delay_days": 3,
        }

    def test_quote_snapshot_has_quote_requested_at(self):
        snap = build_quote_snapshot(self._minimal_quote_payload())
        assert "quote_requested_at" in snap
        assert snap["quote_requested_at"] is not None

    def test_quote_snapshot_has_quote_completed_at(self):
        snap = build_quote_snapshot(self._minimal_quote_payload())
        assert "quote_completed_at" in snap
        assert snap["quote_completed_at"] is not None

    def test_quote_snapshot_has_datetime_note(self):
        snap = build_quote_snapshot(self._minimal_quote_payload())
        assert "_datetime_note" in snap
        assert isinstance(snap["_datetime_note"], str)

    def test_claim_snapshot_has_fnol_submitted_at(self):
        snap = build_claim_snapshot(self._minimal_claim_payload())
        assert "fnol_submitted_at" in snap
        assert snap["fnol_submitted_at"] is not None

    def test_claim_snapshot_has_loss_event_datetime(self):
        snap = build_claim_snapshot(self._minimal_claim_payload())
        assert "loss_event_datetime" in snap
        assert snap["loss_event_datetime"] is not None

    def test_claim_snapshot_has_datetime_note(self):
        snap = build_claim_snapshot(self._minimal_claim_payload())
        assert "_datetime_note" in snap

    def test_quote_snapshot_datetime_fields_are_iso_strings(self):
        snap = build_quote_snapshot(self._minimal_quote_payload())
        # Must be parseable ISO-8601 strings, not raw Timestamp objects
        pd.Timestamp(snap["quote_requested_at"])
        pd.Timestamp(snap["quote_completed_at"])

    def test_claim_snapshot_datetime_fields_are_iso_strings(self):
        snap = build_claim_snapshot(self._minimal_claim_payload())
        pd.Timestamp(snap["fnol_submitted_at"])
        pd.Timestamp(snap["loss_event_datetime"])

    def test_envelope_validator_accepts_valid_quote_snapshot(self):
        snap = build_quote_snapshot(self._minimal_quote_payload())
        errs = _check_snapshot_envelope(snap, "test-quote")
        assert errs == [], f"Unexpected envelope errors: {errs}"

    def test_envelope_validator_accepts_valid_claim_snapshot(self):
        snap = build_claim_snapshot(self._minimal_claim_payload())
        errs = _check_snapshot_envelope(snap, "test-claim")
        assert errs == [], f"Unexpected envelope errors: {errs}"

    def test_envelope_validator_rejects_quote_missing_quote_requested_at(self):
        snap = build_quote_snapshot(self._minimal_quote_payload())
        del snap["quote_requested_at"]
        errs = _check_snapshot_envelope(snap, "test-quote")
        assert any("quote_requested_at" in e for e in errs)

    def test_envelope_validator_rejects_claim_missing_fnol_submitted_at(self):
        snap = build_claim_snapshot(self._minimal_claim_payload())
        del snap["fnol_submitted_at"]
        errs = _check_snapshot_envelope(snap, "test-claim")
        assert any("fnol_submitted_at" in e for e in errs)

    def test_quote_snapshot_null_datetimes_when_payload_missing(self):
        payload = {**self._minimal_quote_payload()}
        del payload["quote_requested_at"]
        snap = build_quote_snapshot(payload)
        assert snap["quote_requested_at"] is None


# ---------------------------------------------------------------------------
# 8.4 — telematics_enrolled_rate → telematics_enrolled_but_missing signal path
# ---------------------------------------------------------------------------

class TestTelematicsEnrolledButMissingSignal:
    def test_enrolled_with_no_data_sets_enrolled_but_missing_true(self):
        payload = {
            "telematics_enrolled": True,
            "telematics_distraction_score": None,
            "telematics_hard_brake_rate": None,
            "telematics_crash_match": None,
            "telematics_commute_entropy": None,
        }
        features = build_telematics_features(payload)
        assert features["telematics_enrolled_but_missing"] is True
        assert features["telematics_available"] is False

    def test_enrolled_with_data_sets_enrolled_but_missing_false(self):
        payload = {
            "telematics_enrolled": True,
            "telematics_distraction_score": 0.35,
            "telematics_hard_brake_rate": 0.04,
            "telematics_crash_match": 0.7,
            "telematics_commute_entropy": 0.5,
        }
        features = build_telematics_features(payload)
        assert features["telematics_enrolled_but_missing"] is False
        assert features["telematics_available"] is True

    def test_not_enrolled_sets_enrolled_but_missing_false(self):
        payload = {
            "telematics_enrolled": False,
            "telematics_distraction_score": None,
            "telematics_hard_brake_rate": None,
            "telematics_crash_match": None,
            "telematics_commute_entropy": None,
        }
        features = build_telematics_features(payload)
        assert features["telematics_enrolled_but_missing"] is False
        assert features["telematics_available"] is False

    def test_enrolled_but_missing_mutually_exclusive_with_available(self):
        # Invariant: can never have both enrolled_but_missing=True and available=True
        payloads = [
            {"telematics_enrolled": True, "telematics_distraction_score": None},
            {"telematics_enrolled": True, "telematics_distraction_score": 0.5},
            {"telematics_enrolled": False, "telematics_distraction_score": None},
        ]
        for p in payloads:
            f = build_telematics_features(p)
            assert not (f["telematics_enrolled_but_missing"] and f["telematics_available"]), (
                f"Invariant broken for payload {p}: enrolled_but_missing={f['telematics_enrolled_but_missing']}, "
                f"available={f['telematics_available']}"
            )

    def test_claim_feature_vector_pulls_enrolled_from_underwriting_context(self):
        claim_payload = {
            "telematics_distraction_score": None,
            "telematics_hard_brake_rate": None,
            "telematics_crash_match": None,
            "telematics_commute_entropy": None,
        }
        underwriting = {"telematics_enrolled": True}
        features = build_claim_feature_vector(claim_payload, underwriting_features=underwriting)
        assert features["telematics_enrolled_but_missing"] is True, (
            "Claim feature vector must pull telematics_enrolled from underwriting context "
            "so enrolled_but_missing is detectable even when claim payload lacks the field"
        )

    def test_claim_feature_vector_enrolled_false_without_underwriting_context(self):
        claim_payload = {
            "telematics_distraction_score": None,
            "telematics_hard_brake_rate": None,
            "telematics_crash_match": None,
            "telematics_commute_entropy": None,
        }
        features = build_claim_feature_vector(claim_payload, underwriting_features=None)
        assert features["telematics_enrolled_but_missing"] is False

    def test_generator_produces_some_enrolled_but_missing_claims(self, generated_data):
        quotes_df, claims_df = generated_data
        enrolled = claims_df["telematics_enrolled"]
        has_data = claims_df["telematics_distraction_score"].notna()
        enrolled_missing = enrolled & ~has_data
        assert enrolled_missing.any(), (
            "Expected at least one claim where telematics_enrolled=True but no data "
            "(fraud archetypes with telematics_enrolled_rate > telematics_opt_in_rate should produce this)"
        )

    def test_generator_enrolled_but_missing_rate_is_nonzero_and_partial(self, generated_data):
        """The telematics_enrolled_rate archetype field must propagate to a non-trivial
        fraction of enrolled_but_missing claims — verifies the full signal path."""
        _, claims_df = generated_data
        enrolled = claims_df["telematics_enrolled"]
        has_data = claims_df["telematics_distraction_score"].notna()
        enrolled_missing_rate = (enrolled & ~has_data).mean()
        assert enrolled_missing_rate > 0.0, "Expected a non-zero enrolled_but_missing rate"
        assert enrolled_missing_rate < 1.0, "Expected enrolled_but_missing rate < 1 (some claims have data)"
