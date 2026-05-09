# Feature Store Implementation Guide
### Smart Oak Insurance — AI Platform
> Version: 2026-Q2 | Platform: Claims + Underwriting

---

## Overview

The feature store is the **single source of truth** for all model inputs across both platforms. It has three layers with distinct purposes:

| Layer | Technology | Purpose | Latency |
|---|---|---|---|
| Online store | Redis | Live inference serving | <10ms |
| Offline store | S3 + Parquet | Model training, batch enrichment | Minutes |
| Feature snapshots | JSON files (S3) | Audit trail, regulatory replay | N/A |

**Critical rule:** `feature_definitions.py` is the single source of truth for feature names, types, and null policy. Both online and offline pipelines import from this file. Never define feature names in two places.

---

## Null Policy

Two categories of nullable features. The rule is the same for both: **null, never imputed with a default.**

### Category 1 — Regulatory null (credit score)

```python
# feature_definitions.py
CREDIT_RESTRICTED_STATES = {"CA", "MA", "MI", "HI"}

def get_credit_score(raw_credit_score: float, state: str) -> float | None:
    if state in CREDIT_RESTRICTED_STATES:
        return None  # Regulatory — do NOT impute
    return raw_credit_score
```

Imputing a neutral value (e.g. `650`) risks acting as a credit proxy in restricted states, which regulators prohibit.

### Category 2 — Behavioral null (telematics)

~60% of users do not have telematics. This is not missing data — it is a signal. Non-telematics users skew slightly higher risk on average due to adverse selection (safer drivers are more likely to opt in). XGBoost's native missing-value handling learns this pattern from training data. Imputing a neutral value masks it.

---

## Telematics Convention

Every nullable telematics signal follows a **trio pattern**. This is a platform-wide convention — apply it to any future nullable signal group.

```python
# feature_definitions.py

def build_telematics_features(
    telematics: TelematicsRecord | None,
    policy: PolicyRecord
) -> dict:
    return {
        # 1. Raw signal — nullable, XGBoost handles via learned null path
        "telematics_distraction_score":     telematics.distraction_score if telematics else None,
        "telematics_hard_brake_rate":        telematics.hard_brake_rate   if telematics else None,
        "telematics_crash_match":            telematics.crash_match        if telematics else None,
        "commute_entropy":                   telematics.commute_entropy    if telematics else None,

        # 2. Availability flag — derived bool, always populated
        "telematics_available":              telematics is not None,

        # 3. Enrolled-but-missing fraud signal — claims platform only
        # Distinguishes "no device" from "enrolled but feed suspiciously absent at claim time"
        "telematics_enrolled_but_missing":   policy.telematics_enrolled and telematics is None,
    }
```

| Feature | Type | Nullable | Platform |
|---|---|---|---|
| `telematics_distraction_score` | float | ✅ Yes | Both |
| `telematics_hard_brake_rate` | float | ✅ Yes | Both |
| `telematics_crash_match` | float | ✅ Yes | Claims |
| `commute_entropy` | float | ✅ Yes | Quotes |
| `telematics_available` | bool | ❌ No | Both |
| `telematics_enrolled_but_missing` | bool | ❌ No | Claims |

---

## Feature Sets

### Claims Platform — 20 fraud features

| # | Feature | Type | Nullable | Signal Layer |
|---|---|---|---|---|
| 1 | `policy_inception_days` | int | No | Tabular |
| 2 | `prior_claims_count` | int | No | Tabular |
| 3 | `reported_injury_count` | int | No | Tabular |
| 4 | `reporting_delay_days` | int | No | Tabular |
| 5 | `attorney_present` | bool | No | Tabular |
| 6 | `submission_hour` | int | No | Tabular |
| 7 | `claimant_count` | int | No | Tabular |
| 8 | `telematics_available` | bool | No | Telematics (derived) |
| 9 | `telematics_crash_match` | float | ✅ Yes | Telematics |
| 10 | `telematics_enrolled_but_missing` | bool | No | Telematics (derived) |
| 11 | `graph_hop_distance` | int | No | Graph |
| 12 | `shared_attribute_count` | int | No | Graph |
| 13 | `attorney_centrality_score` | float | No | Graph |
| 14 | `narrative_inconsistency_score` | float | No | NLP |
| 15 | `narrative_complexity_score` | float | No | NLP |
| 16 | `risk_score_at_issuance` | float | No | Shared ← underwriting |
| 17 | `policy_tier_at_issuance` | str | No | Shared ← underwriting |
| 18 | `ip_geolocation_delta_km` | float | No | Device/geo |
| 19 | `device_fingerprint_match` | bool | No | Device/geo |
| 20 | `submission_channel` | str | No | Tabular |

### Quotes Platform — 20 risk features

| # | Feature | Type | Nullable | Signal Layer |
|---|---|---|---|---|
| 1 | `credit_score` | float | ✅ Yes | Tabular (state-gated) |
| 2 | `credit_eligible` | bool | No | Derived |
| 3 | `prior_loss_frequency` | float | No | Tabular |
| 4 | `prior_loss_severity_avg` | float | No | Tabular |
| 5 | `insurance_lapse_days` | int | No | Tabular |
| 6 | `violation_severity_index` | float | No | Tabular (MVR) |
| 7 | `household_driver_density` | float | No | Tabular |
| 8 | `driver_age` | int | No | Tabular |
| 9 | `years_licensed` | int | No | Tabular |
| 10 | `telematics_available` | bool | No | Telematics (derived) |
| 11 | `telematics_distraction_score` | float | ✅ Yes | Telematics |
| 12 | `telematics_hard_brake_rate` | float | ✅ Yes | Telematics |
| 13 | `commute_entropy` | float | ✅ Yes | Telematics |
| 14 | `vehicle_msrp_power_ratio` | float | No | Vehicle |
| 15 | `vehicle_adas_score` | float | No | Vehicle |
| 16 | `vehicle_age_years` | int | No | Vehicle |
| 17 | `geohash_risk_score` | float | No | Territorial (derived) |
| 18 | `state` | str | No | Territorial |
| 19 | `annual_mileage_estimate` | float | No | Derived |
| 20 | `risk_score_at_issuance` | float | No | Derived (model output) |

---

## Feature Snapshot Schema (Audit Trail)

One JSON file per claim or quote. Immutable. Stored in S3.

```json
{
  "record_id": "CLM-20260505-00123",
  "record_type": "claim",
  "timestamp": "2026-05-05T14:32:11.004Z",
  "feature_store_version": "v1.0.0",
  "regulatory_mask_applied": true,
  "state": "CA",
  "features": {
    "policy_inception_days": 12,
    "prior_claims_count": 1,
    "telematics_available": false,
    "telematics_crash_match": null,
    "telematics_enrolled_but_missing": true,
    "graph_hop_distance": 2,
    "risk_score_at_issuance": 0.71
  }
}
```

**Required fields on every snapshot:**
- `feature_store_version` — enables snapshot replay if feature logic changes
- `regulatory_mask_applied` — confirms state mask was applied before vector assembly
- `timestamp` — millisecond precision for regulatory reproducibility

---

## State Regulatory Mask

Applied in `feature_definitions.py` **before** the feature vector is assembled — not at model level.

```python
CREDIT_RESTRICTED_STATES = {"CA", "MA", "MI", "HI"}

def apply_state_regulatory_mask(features: dict, state: str) -> dict:
    """
    Apply state-specific regulatory constraints.
    Must be called before passing features to any model.
    Sets restricted features to null — never imputes a default.
    """
    if state in CREDIT_RESTRICTED_STATES:
        features["credit_score"] = None
        features["credit_eligible"] = False
    else:
        features["credit_eligible"] = True
    return features
```

| State | Credit Score | Notes |
|---|---|---|
| California | ❌ null | Prohibited by statute |
| Massachusetts | ❌ null | Prohibited by statute |
| Michigan | ❌ null | Prohibited by statute |
| Hawaii | ❌ null | Prohibited by statute |
| All others | ✅ populated | Subject to CBIS regulations |

---

## Cross-Platform Shared Features

Three features are shared between underwriting and claims via the feature store. They must use **identical function implementations** in `feature_definitions.py`.

| Feature | Direction | Notes |
|---|---|---|
| `telematics_available` | Both platforms | Same derivation logic |
| `telematics_distraction_score` | Both platforms | Same null policy |
| `risk_score_at_issuance` | Underwriting → Claims | Stored at quote time; re-read at FNOL |

**Generation dependency:** Quotes must be generated and risk-scored before claims dataset is assembled. `risk_score_at_issuance` must be populated in the feature store before claims training begins.

---

## Volume Estimates

| Scale | Claims Parquet | Quotes Parquet | Redis | Neo4j | JSON Snapshots |
|---|---|---|---|---|---|
| 20K (experiment) | ~5MB | ~5MB | ~60MB | ~100MB | ~40MB |
| 100K | ~25MB | ~25MB | ~300MB | ~500MB | ~200MB |
| 1M | ~250MB | ~250MB | ~3GB | ~5GB | ~2GB |
| 10M+ | S3 + Spark | S3 + Spark | ElastiCache | Neptune | S3 |

At 20K records the entire stack runs in local Docker (Neo4j + Redis + API).