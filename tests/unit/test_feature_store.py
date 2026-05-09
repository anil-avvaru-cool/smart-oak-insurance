from __future__ import annotations

import tempfile
from pathlib import Path

from features.feature_definitions import (
    FEATURE_STORE_VERSION,
    apply_state_regulatory_mask,
    build_claim_feature_vector,
    build_quote_feature_vector,
)
from features.feature_store_client import FeatureStoreClient
from features.online_serving import OnlineFeatureStore


def test_apply_state_regulatory_mask_restricted_state() -> None:
    features = {"credit_score": 720, "credit_eligible": True}
    validated = apply_state_regulatory_mask(features, "CA")
    assert validated["credit_score"] is None
    assert validated["credit_eligible"] is False


def test_build_quote_feature_vector_applies_state_mask() -> None:
    payload = {
        "quote_id": "Q-0001",
        "state": "CA",
        "credit_score": 700,
        "vehicle_msrp": 45000,
        "vehicle_power": 200,
        "vehicle_age_years": 3,
        "annual_mileage_estimate": 12000,
        "telematics": None,
        "telematics_enrolled": True,
    }
    features = build_quote_feature_vector(payload)
    assert features["state"] == "CA"
    assert features["credit_score"] is None
    assert features["credit_eligible"] is False
    assert features["vehicle_msrp_power_ratio"] == 225.0
    assert features["telematics_available"] is False
    assert features["telematics_enrolled_but_missing"] is True


def test_build_claim_feature_vector_combines_underwriting_features() -> None:
    payload = {
        "claim_id": "C-0001",
        "state": "TX",
        "policy_inception_days": 20,
        "prior_claims_count": 1,
        "reported_injury_count": 0,
        "reporting_delay_days": 2,
        "attorney_present": False,
        "submission_hour": 14,
        "claimant_count": 1,
        "device_fingerprint_match": True,
        "submission_channel": "mobile",
        "telematics": {"crash_match": 0.8},
    }
    underwriting_features = {"risk_score_at_issuance": 0.54, "policy_tier_at_issuance": "gold"}
    features = build_claim_feature_vector(payload, underwriting_features)
    assert features["risk_score_at_issuance"] == 0.54
    assert features["policy_tier_at_issuance"] == "gold"
    assert features["telematics_available"] is True
    assert features["telematics_crash_match"] == 0.8


def test_feature_store_client_persists_offline_and_online() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        offline_dir = Path(temp_dir) / "feature_store"
        client = FeatureStoreClient(offline_dir=offline_dir, online_store=OnlineFeatureStore())

        quote_payload = {
            "quote_id": "Q-123",
            "state": "NV",
            "credit_score": 680,
            "vehicle_msrp": 30000,
            "vehicle_power": 150,
            "vehicle_age_years": 2,
            "annual_mileage_estimate": 15000,
            "telematics": None,
            "telematics_enrolled": False,
        }

        snapshot = client.save_quote(quote_payload)
        assert snapshot["feature_store_version"] == FEATURE_STORE_VERSION
        assert (offline_dir / "Q-123.json").exists()

        stored = client.get_online_features("Q-123")
        assert stored is not None
        assert stored["state"] == "NV"
        assert stored["telematics_available"] is False

        loaded = client.load_snapshot("Q-123")
        assert loaded is not None
        assert loaded["record_id"] == "Q-123"
        assert loaded["record_type"] == "quote"
