# Feature Store Implementation Guide
### Smart Oak Insurance — AI Platform
> Version: 2026-Q2 | Platform: Claims + Underwriting

---

## Overview

The feature store is the **single source of truth** for all model inputs across
both platforms. It has three layers with distinct purposes:

| Layer | Technology | Purpose | Latency |
|---|---|---|---|
| Online store | Redis | Live inference serving | <10ms |
| Offline store | S3 + Parquet | Model training, batch enrichment | Minutes |
| Feature snapshots | JSON files (S3) | Audit trail, regulatory replay | N/A |

**Critical rule:** `feature_definitions.py` is the single source of truth for
feature names, types, and null policy. Both online and offline pipelines import
from this file. Never define feature names in two places.

**Pre-computation dependency:** All feature store inputs are resolved entities
from the `entities/` layer (`entity_vehicle.py`, `entity_person.py`,
`entity_address.py`, `entity_phone.py`, `entity_policy.py`). Raw generator
output is not fed directly into the feature store pipeline. Entity resolution
is completed offline before any feature computation or graph construction begins.

**Source datetime columns:** `quote_requested_at`, `fnol_submitted_at`,
`policy_inception_date`, and `quote_completed_at` are raw parquet pipeline
columns added per DEC-013. They are consumed by `monitoring/psi_drift.py`
directly from `data/raw/` — they are **not** forwarded into `data/processed/`
or the feature vector. The feature store serves only the derived int features
(`reporting_delay_days`, `policy_inception_days`) to models.

---

## Null Policy

Two categories of nullable features. The rule is the same for both: **null,
never imputed with a default.**

### Category 1 — Regulatory null (credit score)

```python
# feature_definitions.py
CREDIT_RESTRICTED_STATES = {"CA", "MA", "MI", "HI"}

def get_credit_score(raw_credit_score: float, state: str) -> float | None:
    if state in CREDIT_RESTRICTED_STATES:
        return None  # Regulatory — do NOT impute
    return raw_credit_score
```

Imputing a neutral value (e.g. `650`) risks acting as a credit proxy in
restricted states, which regulators prohibit.

### Category 2 — Behavioral null (telematics)

~60% of users do not have telematics. This is not missing data — it is a signal.
Non-telematics users skew slightly higher risk on average due to adverse selection
(safer drivers are more likely to opt in). XGBoost's native missing-value handling
learns this pattern from training data. Imputing a neutral value masks it.

---

## Telematics Convention

Every nullable telematics signal follows a **trio pattern**. This is a
platform-wide convention — apply it to any future nullable signal group.

```python
# feature_definitions.py

def build_telematics_features(
    telematics: TelematicsRecord | None,   # resolved by entity_vehicle.py —
                                           # OBD-II device → VIN linkage and
                                           # enrollment status resolution
    policy: PolicyRecord                   # resolved by entity_policy.py
) -> dict:
    return {
        # 1. Raw signal — nullable, XGBoost handles via learned null path
        "telematics_distraction_score":   telematics.distraction_score if telematics else None,
        "telematics_hard_brake_rate":     telematics.hard_brake_rate   if telematics else None,
        "telematics_crash_match":         telematics.crash_match        if telematics else None,
        "commute_entropy":                telematics.commute_entropy    if telematics else None,

        # 2. Availability flag — derived bool, always populated
        "telematics_available":           telematics is not None,

        # 3. Enrolled-but-missing fraud signal — claims platform only
        # Distinguishes "no device" from "enrolled but feed suspiciously absent
        # at claim time"
        "telematics_enrolled_but_missing": policy.telematics_enrolled and telematics is None,
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

## Source Datetime Columns (DEC-013)

Source datetimes are **pipeline columns only** — they live in `data/raw/` and
`data/entities/` and are consumed directly by `monitoring/psi_drift.py`. They
do not enter `data/processed/`, the feature vector, or the online store.

The feature store continues to serve the derived int features to models.
The derivation relationships are the canonical definitions in
`feature_definitions.py`:

```python
# feature_definitions.py — derivation documentation (no logic change)

# policy_inception_days: how many days before coverage started relative to
# quote completion. Source columns live in data/raw/quotes.parquet and
# data/entities/policies.parquet.
# policy_inception_days = (quote_completed_at.date - policy_inception_date).days

# reporting_delay_days: how many days elapsed between the loss event and FNOL
# submission. Source columns live in data/raw/claims.parquet.
# reporting_delay_days = (fnol_submitted_at - loss_event_datetime).days
```

### quotes.parquet — source datetime columns

| Column | Type | Nullable | Written By | Consumer |
|---|---|---|---|---|
| `quote_requested_at` | datetime | No | `generator.py` | `psi_drift.py` — PSI current-period window anchor |
| `quote_completed_at` | datetime | No | `hurdle_model.py` | `feature_definitions.py` — source for `policy_inception_days` |
| `policy_inception_date` | date | No | `entity_policy.py` | `feature_definitions.py` — source for `policy_inception_days` |

### claims.parquet — source datetime columns

| Column | Type | Nullable | Written By | Consumer |
|---|---|---|---|---|
| `loss_event_datetime` | datetime | No | `generator.py` | `feature_definitions.py` — source for `reporting_delay_days` |
| `fnol_submitted_at` | datetime | No | `generator.py` | `psi_drift.py` — PSI current-period window anchor; source for `reporting_delay_days` |
| `claim_closed_at` | datetime | Yes — open claims | `generator.py` | Pipeline use only |
| `fraud_confirmed_at` | datetime | Yes — unconfirmed | Label store, post-SIU | `psi_drift.py` — 90/180-day concept drift label window (DEC-007) |

**Propagation rule:** Source datetime columns terminate at `data/raw/`. They are
never forwarded into `data/processed/` or the feature snapshot JSON. If
`psi_drift.py` needs a datetime it reads directly from `data/raw/` — it does not
go through the feature store. This boundary enforces Option A from DEC-013 and
prevents source datetimes from accidentally entering model training.

---

## Feature Sets

### Claims Platform — 20 fraud features

| # | Feature | Type | Nullable | Signal Layer |
|---|---|---|---|---|
| 1 | `policy_inception_days` | int | No | Tabular ² |
| 2 | `prior_claims_count` | int | No | Tabular |
| 3 | `reported_injury_count` | int | No | Tabular |
| 4 | `reporting_delay_days` | int | No | Tabular ³ |
| 5 | `attorney_present` | bool | No | Tabular |
| 6 | `submission_hour` | int | No | Tabular |
| 7 | `claimant_count` | int | No | Tabular |
| 8 | `telematics_available` | bool | No | Telematics (derived) |
| 9 | `telematics_crash_match` | float | ✅ Yes | Telematics |
| 10 | `telematics_enrolled_but_missing` | bool | No | Telematics (derived) |
| 11 | `graph_hop_distance` | int | No | Graph ¹ |
| 12 | `shared_attribute_count` | int | No | Graph ¹ |
| 13 | `attorney_centrality_score` | float | No | Graph ¹ |
| 14 | `narrative_inconsistency_score` | float | No | NLP |
| 15 | `narrative_complexity_score` | float | No | NLP |
| 16 | `risk_score_at_issuance` | float | No | Shared ← underwriting |
| 17 | `policy_tier_at_issuance` | str | No | Shared ← underwriting |
| 18 | `ip_geolocation_delta_miles` | float | No | Device/geo |
| 19 | `device_fingerprint_match` | bool | No | Device/geo |
| 20 | `submission_channel` | str | No | Tabular |

> ¹ **Graph feature reliability note:** `graph_hop_distance`, `shared_attribute_count`,
> and `attorney_centrality_score` depend on normalized Address and Phone entities
> upstream (`entity_address.py`, `entity_phone.py`). If `shares_address` or
> `shares_phone` edges are built from unnormalized strings, Louvain community
> detection degrades silently — two records for "123 Main St" and "123 Main Street"
> will not share an edge and fraud rings will appear disconnected. Verify address
> and phone normalization coverage in `validator.py` before graph construction.

> ² **`policy_inception_days` derivation:** Derived int in the feature vector.
> Source columns are `policy_inception_date` (from `entity_policy.py`) and
> `quote_completed_at` (from `hurdle_model.py`), both in `data/raw/quotes.parquet`
> and `data/entities/policies.parquet`. Formula:
> `(quote_completed_at.date - policy_inception_date).days`.
> Source datetimes do not enter the feature vector — derived int only.

> ³ **`reporting_delay_days` derivation:** Derived int in the feature vector.
> Source columns are `loss_event_datetime` and `fnol_submitted_at`, both in
> `data/raw/claims.parquet`. Formula:
> `(fnol_submitted_at - loss_event_datetime).days`.
> Source datetimes do not enter the feature vector — derived int only.
> `validator.py` enforces consistency between the derived int and its source
> datetimes on every generation run.

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
| 14 | `vehicle_msrp_power_ratio` | float | No | Vehicle ← entity_vehicle.py |
| 15 | `vehicle_adas_score` | float | No | Vehicle ← entity_vehicle.py |
| 16 | `vehicle_age_years` | int | No | Vehicle ← entity_vehicle.py |
| 17 | `geohash_risk_score` | float | No | Territorial (derived) |
| 18 | `state` | str | No | Territorial |
| 19 | `annual_mileage_estimate` | float | No | Derived |
| 20 | `risk_score_at_issuance` | float | No | Derived (model output) |

> **`policy_inception_days` derivation (quotes platform):** Not a direct feature
> in the quotes feature vector — it appears in the claims feature vector (row 1
> above). However, `policy_inception_date` and `quote_completed_at` are written
> to `data/raw/quotes.parquet` by `entity_policy.py` and `hurdle_model.py`
> respectively, and propagated to `data/raw/claims.parquet` at claim generation
> time. The derived int is computed in `feature_definitions.py` for the claims
> feature vector only.

---

## Feature Snapshot Schema (Audit Trail)

One JSON file per claim or quote. Immutable. Stored in S3.

```json
{
  "record_id": "CLM-20260505-00123",
  "record_type": "claim",
  "feature_store_version": "v1.0.0",
  "regulatory_mask_applied": true,
  "state": "CA",

  "_datetime_note": {
    "timestamp": "Feature vector freeze time — when the feature store assembled this snapshot for scoring. Used for audit trail and regulatory replay.",
    "fnol_submitted_at": "Event time — when the claimant submitted the FNOL. PSI current-period window anchor. Sourced from data/raw/claims.parquet, not the feature store.",
    "loss_event_datetime": "Loss event time — when the accident occurred. Source for reporting_delay_days derivation. Not a feature vector field."
  },

  "timestamp": "2026-05-05T14:32:11.004Z",
  "fnol_submitted_at": "2026-05-05T09:14:00.000Z",
  "loss_event_datetime": "2026-05-04T23:45:00.000Z",

  "features": {
    "policy_inception_days": 12,
    "prior_claims_count": 1,
    "reporting_delay_days": 9,
    "telematics_available": false,
    "telematics_crash_match": null,
    "telematics_enrolled_but_missing": true,
    "graph_hop_distance": 2,
    "risk_score_at_issuance": 0.71,
    "vin": "1HGCM82633A123456",
    "vehicle_age_years": 4,
    "msrp_usd": 28000,
    "horsepower": 192,
    "msrp_to_power_ratio": 145.8,
    "adas_aeb_efficacy": 0.74
  }
}
```

**Three distinct timestamps on every claim snapshot — never confuse them:**

| Field | Meaning | Used By |
|---|---|---|
| `timestamp` | Feature vector freeze time — when the feature store assembled this record for scoring | Audit trail, regulatory replay |
| `fnol_submitted_at` | Event time — when the claimant submitted the FNOL | `psi_drift.py` PSI window anchor |
| `loss_event_datetime` | Loss event time — when the accident occurred | Derivation source for `reporting_delay_days`; included for full audit context |

> `fnol_submitted_at` and `loss_event_datetime` are included in the snapshot for
> audit completeness. They are **not** feature vector fields and must not be passed
> to any model. Their presence in the snapshot allows investigators and regulators
> to reconstruct the full claim timeline from a single record without joining back
> to raw parquet.

**For quotes, the equivalent snapshot fields are:**

| Field | Meaning | Used By |
|---|---|---|
| `timestamp` | Feature vector freeze time | Audit trail, regulatory replay |
| `quote_requested_at` | When the customer initiated the quote | `psi_drift.py` PSI window anchor |
| `quote_completed_at` | When risk scoring completed | Derivation source for `policy_inception_days` |

**Required fields on every snapshot:**
- `feature_store_version` — enables snapshot replay if feature logic changes
- `regulatory_mask_applied` — confirms state mask was applied before vector assembly
- `timestamp` — millisecond precision for regulatory reproducibility
- `fnol_submitted_at` (claims) / `quote_requested_at` (quotes) — PSI window anchor

---

## State Regulatory Mask

Applied in `feature_definitions.py` **before** the feature vector is assembled
— not at model level.

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

Six features are shared between underwriting and claims via the feature store.
They must use **identical function implementations** in `feature_definitions.py`.

| Feature | Direction | Source | Notes |
|---|---|---|---|
| `telematics_available` | Both platforms | `entity_vehicle.py` OBD-II linkage | Same derivation logic |
| `telematics_distraction_score` | Both platforms | `entity_vehicle.py` OBD-II linkage | Same null policy |
| `risk_score_at_issuance` | Underwriting → Claims | Model output | Stored at quote time; re-read at FNOL |
| `vehicle_age_years` | Both platforms | `entity_vehicle.py` → Feature Store | Same derivation logic |
| `vehicle_msrp_power_ratio` | Both platforms | `entity_vehicle.py` → Feature Store | Same derivation logic |
| `vehicle_adas_score` | Both platforms | `entity_vehicle.py` → Feature Store | OEM-specific efficacy, not binary flag |

**Source datetime columns are shared raw parquet columns — not feature vector entries.**
The following columns flow from `data/raw/quotes.parquet` into `data/raw/claims.parquet`
at claim generation time but do not enter the feature vector on either platform:

| Column | Direction | Notes |
|---|---|---|
| `quote_requested_at` | Quotes raw parquet | PSI anchor for quotes platform; not propagated to claims |
| `quote_completed_at` | Quotes → Claims raw parquet | Source for `policy_inception_days` derivation on both platforms |
| `policy_inception_date` | Quotes → Claims raw parquet | Source for `policy_inception_days` derivation on both platforms |
| `fnol_submitted_at` | Claims raw parquet | PSI anchor for claims platform only |

These columns must never appear in `feature_definitions.py` feature-building
functions or be passed to any model. If they appear in `data/processed/`, that
is a pipeline bug.

**Generation dependency:** Quotes must be generated and risk-scored before claims
dataset is assembled. `risk_score_at_issuance` must be populated in the feature
store before claims training begins. Entity resolution (`data/entities/`) must be
complete before any shared feature is computed.

---

## Volume Estimates

| Scale | Claims Parquet | Quotes Parquet | Redis | Neo4j | JSON Snapshots |
|---|---|---|---|---|---|
| 20K (experiment) | ~5MB | ~5MB | ~60MB | ~100MB | ~40MB |
| 100K | ~25MB | ~25MB | ~300MB | ~500MB | ~200MB |
| 1M | ~250MB | ~250MB | ~3GB | ~5GB | ~2GB |
| 10M+ | S3 + Spark | S3 + Spark | ElastiCache | Neptune | S3 |

At 20K records the entire stack runs in local Docker (Neo4j + Redis + API).
Entity resolution outputs (`data/entities/`) add ~2MB at 20K scale — negligible.
Source datetime columns add negligible parquet size overhead at all scales —
datetime64 columns are efficiently encoded by parquet compression.