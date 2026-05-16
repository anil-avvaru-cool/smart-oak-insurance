# Insurance Risk Scoring Model — Architecture Guidelines

## Overview

This document outlines the architecture of a production-grade insurance risk
scoring pipeline built on a two-stage Hurdle model. The system separates the
**risk score** (a model output) from the **premium** (a business decision),
enabling regulatory auditability, jurisdictional compliance, and continuous model
improvement through a Champion-Challenger framework.

---

## Core Principle: Risk Score vs. Premium

The model produces a **risk score**, not a premium. Premium is a downstream
business calculation:

```
Premium = Risk Score × Base Rate × Business Multipliers
                                   (profit load · expenses · competitive adjustments)
```

This separation means the model is portable across pricing strategies without
retraining, and the score can be explained independently of commercial decisions.

---

## Stage 0 — Entity Resolution

Before the feature store assembles any vector, all shared entities are resolved
by the `entities/` layer. This is a pre-computation step — not part of the
real-time scoring path.

| Entity | Resolved By | Output |
|---|---|---|
| Vehicle | `entity_vehicle.py` | VIN decode, MSRP, horsepower, ADAS efficacy flags |
| Person | `entity_person.py` | Deduplicated person with role flags (policyholder / driver / claimant) |
| Address | `entity_address.py` | Normalized + hashed address for graph edge reliability |
| Phone | `entity_phone.py` | Normalized + hashed phone number |
| Policy | `entity_policy.py` | Coverage validation, inception date, lapse calculation |

**Key rule:** `graph_builder.py` loads resolved entities as nodes and edges only.
Vehicle features (MSRP, ADAS efficacy, age) are computed from resolved entities
in the feature store pipeline — not inside the graph builder.

---

## Stage 1 — Feature Store (Versioned)

### Purpose
The Feature Store is the single source of truth for all inputs. Every feature
record is **frozen at quote time** and immutably stored with a timestamp.
All inputs are resolved entities from Stage 0 — raw generator output is not
fed directly into the feature store pipeline.

### Structure
Each record is stored as named key-value pairs — not raw vectors:

```
customer_id:                  C-00123
quote_id:                     Q-20260505-884
timestamp:                    2026-05-05T14:32:11.004Z
credit_score:                 712
prior_claims:                 2
telematics_distraction_score: 0.31
geohash:                      9q8yy
state:                        CA
credit_eligible:              false      ← regulatory mask applied
vin:                          1HGCM82633A123456   ← resolved by entity_vehicle.py
vehicle_age_years:            4                   ← resolved by entity_vehicle.py
msrp_usd:                     28000               ← resolved by entity_vehicle.py
horsepower:                   192                 ← resolved by entity_vehicle.py
msrp_to_power_ratio:          145.8               ← resolved by entity_vehicle.py
adas_aeb_efficacy:            0.74                ← resolved by entity_vehicle.py
```

### State Regulatory Mask
A `state_regulatory_profile` config gates feature eligibility before the vector
is assembled. Credit-based insurance scores are excluded for restricted
jurisdictions:

| State | Credit Score Eligible |
|---|---|
| California | ❌ No |
| Massachusetts | ❌ No |
| Michigan | ❌ No |
| Hawaii | ❌ No |
| All others | ✅ Yes |

**Important:** Excluded features are set to `null` — not imputed with a default
value. XGBoost's native missing-value handling routes null inputs through a learned
default path. Passing a synthetic value (e.g. `650`) risks acting as a credit
proxy, which regulators in restricted states prohibit.

### Versioning & Audit Trail
The frozen feature vector enables **snapshot replay**: if a customer contests
their rate, the exact inputs used at quote time can be reproduced for regulatory
review. This satisfies the "Right to Explanation" requirement without
reverse-engineering model internals.

---

## Stage 2 — Hurdle Model (Two-Stage Architecture)

### Stage 2a: Frequency (The "If")

| Property | Detail |
|---|---|
| Model | XGBoost with Binary Logistic Loss |
| Output | P(claim > 0) — probability a claim occurs |
| Class imbalance | Cost-sensitive weighting (not SMOTE) |

Cost-sensitive learning is preferred over SMOTE because it preserves the integrity
of real feature distributions. SMOTE creates synthetic records that may not reflect
actual policyholder behaviour.

### Stage 2b: Severity (The "How Much")

| Property | Detail |
|---|---|
| Model | XGBoost with Gamma Objective |
| Output | E[cost \| claim] — expected cost given a claim occurs |
| Distribution | Gamma — variance scales with the square of the mean, suited to long-tail loss data |

> **Gamma vs. Tweedie:** Use Gamma in Stage 2 only (severity given a claim).
> Use Tweedie only in a single-stage combined model where frequency and severity
> are not separated.

### Stage 2c: Integration

```
Risk Score = P(claim > 0) × E[cost | claim]
```

The risk score is the product of the two stages — a pure actuarial signal of
expected loss.

---

## Stage 3 — Calibration

Raw model outputs are not inherently probability-calibrated. Calibration ensures
that a predicted 10% risk corresponds to a 10% actual loss rate in that cohort.

| Method | Use Case |
|---|---|
| Isotonic Regression | Non-parametric; preferred for large datasets |
| Platt Scaling | Parametric; faster for smaller datasets |

Calibration output feeds directly into the audit trail — the calibrated score is
what is stored and explained to regulators, not the raw model logit.

---

## Stage 4 — Premium Calculation

```
Premium = Risk Score × Base Rate × Business Multipliers
```

Business multipliers include profit load, expense ratio, and competitive pricing
adjustments. These are owned by the pricing team, not the model.

---

## Top 10 Predictive Features

| Rank | Feature | Signal | Source |
|---|---|---|---|
| 1 | Credit-Based Insurance Score* | Moral hazard proxy — highest lift | Feature Store (state-gated) |
| 2 | Prior Loss Frequency (3–5yr) | At-fault carries 3× weight of not-at-fault | Feature Store |
| 3 | Telematics: Distracted Driving | Phone handling now outperforms hard braking | Feature Store (via entity_vehicle.py OBD-II linkage) |
| 4 | Territorial Risk (Geohash) | 1mile × 1mile resolution vs. zip code | Feature Store (derived) |
| 5 | Vehicle ADAS Efficacy | OEM-specific AEB effectiveness, not just presence | `entity_vehicle.py` → Feature Store |
| 6 | Insurance Continuity | Lapses >30 days linked to higher severity | Feature Store (via entity_policy.py) |
| 7 | Commute Entropy | Route variability as a cognitive load proxy | Feature Store (telematics) |
| 8 | Violation Severity Index | Weighted MVR — DUI weighted 10× tail light | Feature Store |
| 9 | Household Driver Density | Active drivers relative to vehicles | Feature Store (via entity_person.py) |
| 10 | Vehicle MSRP-to-Power Ratio | High horsepower × low experience | `entity_vehicle.py` → Feature Store |

*Jurisdiction-gated — excluded in CA, MA, MI, HI by statute.

---

## Drift Monitoring

Drift is the primary operational risk. Models do not fail suddenly — they drift,
leading to **adverse selection** where mispriced risks accumulate undetected.

### Population Stability Index (PSI)

PSI monitors the distribution of input features over time:

| PSI Value | Status | Action |
|---|---|---|
| < 0.1 | Stable | No action |
| 0.1 – 0.25 | Warning | Investigate |
| > 0.25 | Critical | Emergency retrain |

### Concept Drift

When the *relationship* between features and claims changes — not just the feature
distributions. Example: during a pandemic, mileage becomes less predictive but
speeding on empty roads becomes more predictive. PSI alone will not catch this;
monitor model performance metrics (Gini, loss ratio by tier) alongside PSI.

### SHAP Stability

Monitor global SHAP values regularly. If a single feature (e.g. hard braking)
comes to dominate all scores, the model has become brittle and is likely
overfitting to a specific sensor type.

---

## Champion-Challenger Framework

Never deploy a new model to 100% of traffic. The Champion-Challenger loop ensures
safe, evidence-based rollout:

```
1. Shadow Mode    → New challenger runs in background; predictions logged but not used
2. Gini Check     → After 3–6 months on holdout set: if Challenger Gini > Champion Gini
                    by 2%+, proceed
3. Phased Rollout → Gradual traffic shift (e.g. 10% → 25% → 50% → 100%)
4. Promote        → Challenger becomes the new Champion; previous Champion archived
```

The drift monitor feeds this loop: a PSI > 0.25 alert triggers a challenger
retrain, not an immediate Champion replacement.

---

## Architecture Summary

```
Customer quote
      │
      ▼
Entity Resolution                        ← Stage 0: VIN decode, person dedup,
      │                                             address normalize, policy lapse calc
      │  entity_vehicle.py → vehicles.parquet
      │  entity_person.py  → persons.parquet
      │  entity_address.py → addresses.parquet
      │  entity_policy.py  → policies.parquet
      │
      ▼
Feature Store ── state mask ── frozen vector ── audit trail
      │
      ├──────────────────────────────┐
      ▼                              ▼
Frequency model              Severity model
P(claim > 0)                 E(cost | claim)
      │                              │
      └──────────────┬───────────────┘
                     ▼
               Risk score = P × E
                     │
              ┌──────┴──────┐
              ▼             ▼
         Premium        Drift monitor
    (business layer)   PSI · concept drift
                             │
                             ▼
                    Champion-Challenger loop
                    Shadow → Gini → Promote
```

---

## Regulatory Checklist

- [ ] Entity resolution complete before feature store vector assembly
- [ ] Vehicle features (MSRP, ADAS, age) sourced from `entity_vehicle.py` — not
      looked up at scoring time
- [ ] Feature Store versioning enabled with millisecond-precision timestamps
- [ ] State regulatory mask applied before vector assembly — not at model level
- [ ] Credit score set to `null` (not imputed) for restricted states
- [ ] Calibrated scores stored in audit trail, not raw logits
- [ ] PSI monitoring active on all top-10 features
- [ ] Champion-Challenger shadow mode running before any model promotion
- [ ] SHAP global importance reviewed at each retraining cycle