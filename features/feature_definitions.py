from __future__ import annotations

import math
from typing import Any

FeatureVector = dict[str, Any]

FEATURE_STORE_VERSION = "v1.0.0"
CREDIT_RESTRICTED_STATES = {"CA", "MA", "MI", "HI"}
GRAPH_HOP_SENTINEL = 999  # written by graph_features.py when no fraud path found within 6 hops


def _nan_to_none(v: Any) -> Any:
    return None if (isinstance(v, float) and math.isnan(v)) else v


def build_telematics_features(payload: dict) -> FeatureVector:
    distraction_score = _nan_to_none(payload.get("telematics_distraction_score"))
    hard_brake_rate = _nan_to_none(payload.get("telematics_hard_brake_rate"))
    crash_match = _nan_to_none(payload.get("telematics_crash_match"))
    commute_entropy = _nan_to_none(payload.get("telematics_commute_entropy"))
    available = any(v is not None for v in (distraction_score, hard_brake_rate, crash_match, commute_entropy))
    return {
        "telematics_distraction_score": distraction_score,
        "telematics_hard_brake_rate": hard_brake_rate,
        "telematics_crash_match": crash_match,
        "telematics_commute_entropy": commute_entropy,
        "telematics_available": available,
        "telematics_enrolled_but_missing": bool(payload.get("telematics_enrolled", False) and not available),
    }


def apply_state_regulatory_mask(features: FeatureVector, state: str) -> FeatureVector:
    state_code = (state or "").upper()
    if state_code in CREDIT_RESTRICTED_STATES:
        features["credit_score"] = None
        features["credit_eligible"] = False
    else:
        features["credit_eligible"] = True
    return features


def build_quote_feature_vector(quote_payload: dict) -> FeatureVector:
    vehicle_msrp = float(quote_payload.get("vehicle_msrp", 0.0) or 0.0)
    vehicle_power = float(quote_payload.get("vehicle_power", 1.0) or 1.0)
    vehicle_age_years = int(quote_payload.get("vehicle_age_years", 0) or 0)

    features: FeatureVector = {
        "credit_score": quote_payload.get("credit_score"),
        "prior_loss_frequency": float(quote_payload.get("prior_loss_frequency", 0.0) or 0.0),
        "prior_loss_severity_avg": float(quote_payload.get("prior_loss_severity_avg", 0.0) or 0.0),
        "insurance_lapse_days": int(quote_payload.get("insurance_lapse_days", 0) or 0),
        "violation_severity_index": float(quote_payload.get("violation_severity_index", 0.0) or 0.0),
        "household_driver_density": float(quote_payload.get("household_driver_density", 0.0) or 0.0),
        "driver_age": int(quote_payload.get("driver_age", 0) or 0),
        "years_licensed": int(quote_payload.get("years_licensed", 0) or 0),
        "vehicle_msrp_power_ratio": vehicle_msrp / max(vehicle_power, 1.0),
        "vehicle_adas_score": float(quote_payload.get("vehicle_adas_score", 0.0) or 0.0),
        "vehicle_age_years": vehicle_age_years,
        "geohash_risk_score": float(quote_payload.get("geohash_risk_score", 0.0) or 0.0),
        "state": quote_payload.get("state", "UNKNOWN"),
        "annual_mileage_estimate": float(quote_payload.get("annual_mileage_estimate", 0.0) or 0.0),
        "risk_score_at_issuance": quote_payload.get("risk_score_at_issuance"),
        "policy_tier_at_issuance": quote_payload.get("policy_tier_at_issuance"),
    }

    features.update(build_telematics_features(quote_payload))
    return apply_state_regulatory_mask(features, quote_payload.get("state", ""))


def build_claim_feature_vector(claim_payload: dict, underwriting_features: dict | None = None) -> FeatureVector:
    if underwriting_features is None:
        underwriting_features = {}

    features: FeatureVector = {
        "policy_inception_days": int(claim_payload.get("policy_inception_days", 0) or 0),
        "prior_claims_count": int(claim_payload.get("prior_claims_count", 0) or 0),
        "reported_injury_count": int(claim_payload.get("reported_injury_count", 0) or 0),
        "reporting_delay_days": int(claim_payload.get("reporting_delay_days", 0) or 0),
        "attorney_present": bool(claim_payload.get("attorney_present", False)),
        "submission_hour": int(claim_payload.get("submission_hour", 0) or 0),
        "claimant_count": int(claim_payload.get("claimant_count", 1) or 1),
        "graph_hop_distance": None if (_raw_hop := claim_payload.get("graph_hop_distance")) is None or int(_raw_hop) == GRAPH_HOP_SENTINEL else int(_raw_hop),
        "shared_attribute_count": int(claim_payload.get("shared_attribute_count", 0) or 0),
        "attorney_centrality_score": float(claim_payload.get("attorney_centrality_score", 0.0) or 0.0),
        "narrative_inconsistency_score": float(claim_payload.get("narrative_inconsistency_score", 0.0) or 0.0),
        "narrative_complexity_score": float(claim_payload.get("narrative_complexity_score", 0.0) or 0.0),
        "risk_score_at_issuance": underwriting_features.get("risk_score_at_issuance") or claim_payload.get("risk_score_at_issuance"),
        "policy_tier_at_issuance": underwriting_features.get("policy_tier_at_issuance") or claim_payload.get("policy_tier_at_issuance"),
        "ip_geolocation_delta_miles": float(claim_payload.get("ip_geolocation_delta_miles", 0.0) or 0.0),
        "device_fingerprint_match": bool(claim_payload.get("device_fingerprint_match", True)),
        "submission_channel": claim_payload.get("submission_channel", "unknown"),
    }

    # telematics_enrolled lives in quotes, not claims — pull from underwriting context
    telem_enrolled = claim_payload.get("telematics_enrolled", underwriting_features.get("telematics_enrolled", False))
    features.update(build_telematics_features({**claim_payload, "telematics_enrolled": telem_enrolled}))
    return features
