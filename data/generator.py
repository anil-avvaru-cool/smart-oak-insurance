from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker

from data.archetypes_claims import CLAIM_ARCHETYPES
from data.archetypes_underwriting import UNDERWRITING_ARCHETYPES
from data.config import (
    CLAIMS_OUTPUT,
    OPEN_CLAIM_RATE,
    QUOTE_DATE_RANGE_DAYS,
    QUOTES_OUTPUT,
    RAW_DATA_DIR,
    RANDOM_SEED,
    UNCONFIRMED_FRAUD_RATE,
)
from data.states import US_STATE_ABBREVIATIONS


def _bounded_normal(rng: np.random.Generator, mean: float, std: float, minimum: float, maximum: float) -> float:
    return float(np.clip(rng.normal(mean, std), minimum, maximum))


def _sample_gamma_loss(rng: np.random.Generator, mean_usd: float, cv: float) -> float:
    """Sample incurred loss from Gamma(shape=1/cv², scale=mean*cv²). Clamped to [$500, $250k]."""
    shape = 1.0 / (cv ** 2)
    scale = mean_usd * (cv ** 2)
    return float(np.clip(rng.gamma(shape, scale), 500.0, 250_000.0))


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
    if quote.get("telematics_enrolled", False) and quote.get("telematics_distraction_score") is None:
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
    reference_date = pd.Timestamp.now().normalize()

    for archetype in UNDERWRITING_ARCHETYPES:
        for _ in range(archetype.volume):
            state = rng.choice(US_STATE_ABBREVIATIONS)
            has_telematics = rng.random() < archetype.telematics_opt_in_rate
            telematics = _sample_telematics(rng) if has_telematics else {}
            credit_score = int(_bounded_normal(rng, archetype.credit_score_mean, archetype.credit_score_std, 500, 850))
            prior_loss_frequency = float(np.clip(_bounded_normal(rng, archetype.prior_loss_frequency_mean, 0.12, 0.0, 1.0), 0.0, 1.0))
            prior_loss_severity_avg = float(np.clip(_bounded_normal(rng, archetype.prior_loss_severity_mean, 1600, 500.0, 20000.0), 0.0, None))

            # DEC-013: spread quotes uniformly over the past QUOTE_DATE_RANGE_DAYS
            days_back = int(rng.integers(0, QUOTE_DATE_RANGE_DAYS))
            quote_requested_at = reference_date - pd.Timedelta(days=days_back) + pd.Timedelta(
                hours=int(rng.integers(8, 20)), minutes=int(rng.integers(0, 60))
            )
            # processing offset: 1 second to 5 minutes
            quote_completed_at = quote_requested_at + pd.Timedelta(seconds=int(rng.integers(1, 300)))

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
                "telematics_distraction_score": telematics.get("distraction_score"),
                "telematics_hard_brake_rate": telematics.get("hard_brake_rate"),
                "telematics_crash_match": telematics.get("crash_match"),
                "telematics_commute_entropy": telematics.get("commute_entropy"),
                "telematics_enrolled": rng.random() < archetype.telematics_opt_in_rate,
                "quote_requested_at": quote_requested_at,
                "quote_completed_at": quote_completed_at,
            }
            quote_payload["risk_score_at_issuance"] = _score_quote(quote_payload)
            quote_payload["policy_tier_at_issuance"] = _assign_policy_tier(quote_payload["risk_score_at_issuance"])
            quote_payload["archetype_name"] = archetype.name
            quote_payload["claim_occurred"] = bool(rng.random() < archetype.annual_claim_rate)
            records.append(quote_payload)

    quotes_df = pd.DataFrame(records)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    quotes_df.to_parquet(QUOTES_OUTPUT, index=False)
    return quotes_df


def generate_claims(quotes_df: pd.DataFrame, seed: int = RANDOM_SEED + 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    faker = Faker()
    records: list[dict[str, Any]] = []

    # DEC-013: look up quote_completed_at to derive policy_inception_days from datetimes
    quote_completed_at_map: dict[str, pd.Timestamp] = dict(
        zip(quotes_df["quote_id"], quotes_df["quote_completed_at"])
    )

    for _, quote in quotes_df.iterrows():
        fraud_probability = float(np.clip(0.08 + quote["risk_score_at_issuance"] * 0.45, 0.02, 0.85))
        is_fraud = rng.random() < fraud_probability
        archetype = _choose_claim_archetype(is_fraud, rng)

        quote_completed_at = pd.Timestamp(quote_completed_at_map[quote["quote_id"]])

        # DEC-013: loss event occurs N days after policy inception (quote_completed_at proxy)
        days_after_policy = max(0.0, rng.normal(35.0 if is_fraud else 90.0, 30.0))
        loss_hour = int(np.clip(
            rng.normal(archetype.loss_event_hour_dist[0], archetype.loss_event_hour_dist[1]), 0, 23
        ))
        loss_event_datetime = (quote_completed_at + pd.Timedelta(days=int(days_after_policy))).replace(
            hour=loss_hour,
            minute=int(rng.integers(0, 60)),
            second=int(rng.integers(0, 60)),
            microsecond=0,
        )

        # DEC-013: fnol derived from loss event + reporting delay sample
        reporting_delay_raw = float(np.clip(
            rng.normal(archetype.reporting_delay_mean, archetype.reporting_delay_std), 0, 45
        ))
        fnol_submitted_at = loss_event_datetime + pd.Timedelta(days=reporting_delay_raw)

        # DEC-013: derived ints computed from datetimes, not independently sampled
        reporting_delay_days = int((fnol_submitted_at - loss_event_datetime).days)
        policy_inception_days = max(0, (loss_event_datetime.date() - quote_completed_at.date()).days)

        # claim_closed_at: null at OPEN_CLAIM_RATE probability
        if rng.random() >= OPEN_CLAIM_RATE:
            close_days = max(1.0, rng.normal(
                archetype.claim_open_duration_days_dist[0],
                archetype.claim_open_duration_days_dist[1],
            ))
            claim_closed_at: pd.Timestamp | None = fnol_submitted_at + pd.Timedelta(days=close_days)
        else:
            claim_closed_at = None

        # fraud_confirmed_at: null for legitimate claims; null at UNCONFIRMED_FRAUD_RATE for fraud
        if archetype.is_fraud and rng.random() >= UNCONFIRMED_FRAUD_RATE:
            confirm_lag = max(1.0, rng.normal(
                archetype.fraud_confirmation_lag_days_dist[0],
                archetype.fraud_confirmation_lag_days_dist[1],
            ))
            fraud_confirmed_at: pd.Timestamp | None = fnol_submitted_at + pd.Timedelta(days=confirm_lag)
        else:
            fraud_confirmed_at = None

        # telematics_enrolled_rate drives the enrolled_but_missing fraud signal
        _telematics_enrolled = rng.random() < archetype.telematics_enrolled_rate
        _data_rate = (
            archetype.telematics_opt_in_rate / max(archetype.telematics_enrolled_rate, 1e-6)
            if _telematics_enrolled else 0.0
        )
        _has_claim_telematics = _telematics_enrolled and rng.random() < _data_rate

        claim_payload = {
            "claim_id": faker.uuid4(),
            "quote_id": quote["quote_id"],
            "state": quote["state"],
            "policy_inception_days": policy_inception_days,
            "prior_claims_count": int(np.clip(rng.poisson(quote["prior_loss_frequency"] * 1.8 + 0.3), 0, 5)),
            "reported_injury_count": int(np.clip(rng.normal(1.0 if not is_fraud else 2.1, 1.1), 0, 5)),
            "reporting_delay_days": reporting_delay_days,
            "attorney_present": rng.random() < archetype.attorney_present_prob,
            "submission_hour": fnol_submitted_at.hour,
            "claimant_count": int(np.clip(rng.poisson(archetype.claimant_count_lambda), 1, 8)),
            # DEC-005: graph features are stubs until --compute-graph-features overwrites them via Neo4j
            "graph_hop_distance": 999,
            "shared_attribute_count": 0,
            "attorney_centrality_score": 0.0,
            "narrative_inconsistency_score": float(np.clip(rng.normal(archetype.narrative_inconsistency_mean, 0.20), 0.0, 1.0)),
            "narrative_complexity_score": float(np.clip(rng.normal(archetype.narrative_complexity_mean, 0.18), 0.0, 1.0)),
            "device_fingerprint_match": rng.random() < archetype.device_fingerprint_match_prob,
            "submission_channel": rng.choice(["mobile", "agent_portal", "web", "broker"], p=[0.4, 0.2, 0.3, 0.1]),
            "telematics_enrolled": _telematics_enrolled,
            "telematics_distraction_score": quote["telematics_distraction_score"] if _has_claim_telematics else None,
            "telematics_hard_brake_rate": quote["telematics_hard_brake_rate"] if _has_claim_telematics else None,
            "telematics_crash_match": quote["telematics_crash_match"] if _has_claim_telematics else None,
            "telematics_commute_entropy": quote["telematics_commute_entropy"] if _has_claim_telematics else None,
            "risk_score_at_issuance": quote["risk_score_at_issuance"],
            "policy_tier_at_issuance": quote["policy_tier_at_issuance"],
            "is_fraud": is_fraud,
            "ip_geolocation_delta_miles": float(np.clip(rng.normal(archetype.ip_geolocation_delta_mean, 5.0), 0.0, 100.0)),
            "incurred_loss_usd": _sample_gamma_loss(rng, archetype.loss_mean_usd, archetype.loss_cv),
            "archetype_name": archetype.name,
            "loss_event_datetime": loss_event_datetime,
            "fnol_submitted_at": fnol_submitted_at,
            "claim_closed_at": claim_closed_at,
            "fraud_confirmed_at": fraud_confirmed_at,
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
