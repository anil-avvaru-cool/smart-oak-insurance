from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker

from data.archetypes_claims import CLAIM_ARCHETYPES
from data.archetypes_underwriting import UNDERWRITING_ARCHETYPES
from data.config import CLAIMS_OUTPUT, QUOTES_OUTPUT, RAW_DATA_DIR, RANDOM_SEED
from data.states import US_STATE_ABBREVIATIONS


def _bounded_normal(rng: np.random.Generator, mean: float, std: float, minimum: float, maximum: float) -> float:
    return float(np.clip(rng.normal(mean, std), minimum, maximum))


def _assign_policy_tier(risk_score: float) -> str:
    if risk_score < 0.25:
        return "platinum"
    if risk_score < 0.45:
        return "gold"
    if risk_score < 0.65:
        return "silver"
    return "bronze"


def _score_quote(quote: dict[str, Any]) -> float:
    score = 0.0
    score += max(0.0, (760.0 - float(quote["credit_score"] or 760.0))) / 300.0
    score += min(1.0, float(quote["prior_loss_frequency"]) * 0.35)
    score += min(1.0, float(quote["prior_loss_severity_avg"]) / 20000.0 * 0.35)
    score += min(1.0, float(quote["insurance_lapse_days"]) / 120.0 * 0.20)
    score += min(1.0, float(quote["violation_severity_index"]) / 5.0 * 0.18)
    if quote.get("telematics_enrolled", False) and not quote.get("telematics"):
        score += 0.12
    if quote["vehicle_power"] > 220:
        score += 0.08
    return float(np.clip(score, 0.0, 1.0))


def _choose_claim_archetype(is_fraud: bool, rng: np.random.Generator) -> dict[str, Any]:
    filtered = [archetype for archetype in CLAIM_ARCHETYPES if archetype.is_fraud == is_fraud]
    weights = [1.0 for _ in filtered]
    chosen = rng.choice(filtered, p=np.array(weights) / np.sum(weights))
    return chosen


def _sample_telematics(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "distraction_score": float(np.clip(rng.normal(0.4, 0.18), 0.0, 1.0)),
        "hard_brake_rate": float(np.clip(rng.normal(0.04, 0.02), 0.0, 0.18)),
        "crash_match": float(np.clip(rng.normal(0.7, 0.18), 0.0, 1.0)),
        "commute_entropy": float(np.clip(rng.normal(0.45, 0.13), 0.0, 1.0)),
    }


def generate_quotes(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    faker = Faker()
    records: list[dict[str, Any]] = []

    for archetype in UNDERWRITING_ARCHETYPES:
        for _ in range(archetype.volume):
            state = rng.choice(US_STATE_ABBREVIATIONS)
            has_telematics = rng.random() < archetype.telematics_opt_in_rate
            telematics = _sample_telematics(rng) if has_telematics else None
            credit_score = int(_bounded_normal(rng, archetype.credit_score_mean, archetype.credit_score_std, 500, 850))
            prior_loss_frequency = float(np.clip(_bounded_normal(rng, archetype.prior_loss_frequency_mean, 0.12, 0.0, 1.0), 0.0, 1.0))
            prior_loss_severity_avg = float(np.clip(_bounded_normal(rng, archetype.prior_loss_severity_mean, 1600, 500.0, 20000.0), 0.0, None))
            quote_payload = {
                "quote_id": faker.uuid4(),
                "state": state,
                "credit_score": credit_score,
                "prior_loss_frequency": prior_loss_frequency,
                "prior_loss_severity_avg": prior_loss_severity_avg,
                "insurance_lapse_days": int(_bounded_normal(rng, archetype.insurance_lapse_days_mean, 12, 0, 180)),
                "violation_severity_index": float(_bounded_normal(rng, archetype.violation_severity_index_mean, 0.8, 0.0, 5.0)),
                "household_driver_density": float(_bounded_normal(rng, archetype.household_driver_density_mean, 0.9, 0.5, 6.0)),
                "driver_age": int(_bounded_normal(rng, archetype.driver_age_mean, 6, 16, 80)),
                "years_licensed": int(_bounded_normal(rng, archetype.years_licensed_mean, 4, 0, 60)),
                "vehicle_msrp": float(_bounded_normal(rng, archetype.vehicle_msrp_mean, 11000, 12000, 150000)),
                "vehicle_power": float(_bounded_normal(rng, archetype.vehicle_power_mean, 30, 80, 500)),
                "vehicle_adas_score": float(_bounded_normal(rng, archetype.vehicle_adas_score_mean, 0.12, 0.0, 1.0)),
                "vehicle_age_years": int(_bounded_normal(rng, 3.5, 2.5, 0, 20)),
                "geohash_risk_score": float(np.clip(rng.random() * 0.4 + 0.1, 0.0, 1.0)),
                "annual_mileage_estimate": float(_bounded_normal(rng, archetype.annual_mileage_mean, 4300, 1000, 40000)),
                "telematics": telematics,
                "telematics_enrolled": rng.random() < archetype.telematics_opt_in_rate,
            }
            quote_payload["risk_score_at_issuance"] = _score_quote(quote_payload)
            quote_payload["policy_tier_at_issuance"] = _assign_policy_tier(quote_payload["risk_score_at_issuance"])
            records.append(quote_payload)

    quotes_df = pd.DataFrame(records)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    quotes_df.to_parquet(QUOTES_OUTPUT, index=False)
    return quotes_df


def generate_claims(quotes_df: pd.DataFrame, seed: int = RANDOM_SEED + 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    faker = Faker()
    records: list[dict[str, Any]] = []

    for _, quote in quotes_df.iterrows():
        fraud_probability = float(np.clip(0.08 + quote["risk_score_at_issuance"] * 0.45, 0.02, 0.85))
        is_fraud = rng.random() < fraud_probability
        archetype = _choose_claim_archetype(is_fraud, rng)

        claim_payload = {
            "claim_id": faker.uuid4(),
            "quote_id": quote["quote_id"],
            "state": quote["state"],
            "policy_inception_days": int(max(0.0, rng.normal(quote["risk_score_at_issuance"] * 120.0, 35.0))),
            "prior_claims_count": int(np.clip(rng.poisson(quote["prior_loss_frequency"] * 1.8 + 0.3), 0, 5)),
            "reported_injury_count": int(np.clip(rng.normal(1.0 if not is_fraud else 2.1, 1.1), 0, 5)),
            "reporting_delay_days": int(np.clip(rng.normal(archetype.reporting_delay_mean, archetype.reporting_delay_std), 0, 45)),
            "attorney_present": rng.random() < archetype.attorney_present_prob,
            "submission_hour": int(rng.integers(0, 24)),
            "claimant_count": int(np.clip(rng.poisson(archetype.claimant_count_lambda), 1, 8)),
            "graph_hop_distance": int(np.clip(rng.poisson(archetype.graph_hop_distance_lambda), 0, 6)),
            "shared_attribute_count": int(np.clip(rng.normal(archetype.shared_attribute_count_mean, 1.1), 0, 8)),
            "attorney_centrality_score": float(np.clip(rng.normal(archetype.attorney_centrality_mean, 0.18), 0.0, 1.0)),
            "narrative_inconsistency_score": float(np.clip(rng.normal(archetype.narrative_inconsistency_mean, 0.16), 0.0, 1.0)),
            "narrative_complexity_score": float(np.clip(rng.normal(archetype.narrative_complexity_mean, 0.18), 0.0, 1.0)),
            "device_fingerprint_match": rng.random() < archetype.device_fingerprint_match_prob,
            "submission_channel": rng.choice(["mobile", "agent_portal", "web", "broker"], p=[0.4, 0.2, 0.3, 0.1]),
            "telematics": quote["telematics"] if rng.random() < archetype.telematics_opt_in_rate else None,
            "risk_score_at_issuance": quote["risk_score_at_issuance"],
            "policy_tier_at_issuance": quote["policy_tier_at_issuance"],
            "is_fraud": is_fraud,
            "ip_geolocation_delta_km": float(np.clip(rng.normal(archetype.ip_geolocation_delta_mean, 5.0), 0.0, 100.0)),
        }
        records.append(claim_payload)

    claims_df = pd.DataFrame(records)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    claims_df.to_parquet(CLAIMS_OUTPUT, index=False)
    return claims_df


def generate_data(seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    quotes_df = generate_quotes(seed)
    claims_df = generate_claims(quotes_df, seed + 1)
    return quotes_df, claims_df


def main() -> None:
    quotes_df, claims_df = generate_data()
    print(f"Generated {len(quotes_df)} quotes to {QUOTES_OUTPUT}")
    print(f"Generated {len(claims_df)} claims to {CLAIMS_OUTPUT}")


if __name__ == "__main__":
    main()
