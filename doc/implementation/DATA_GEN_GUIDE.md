
# Data Generation Implementation Guide
### Smart Oak Insurance — AI Platform
> Version: 2026-Q2 | Target: 20K quotes + 20K claims | Features: 20 per platform

---

## Core Principle

Generate data **top-down from archetypes**, never bottom-up from random values.

Random data produces no correlations between features. Your graph neural network,
anomaly detector, and ensemble model all rely on correlated feature patterns — a
staged-collision ring should have `attorney_present=True`, `reporting_delay_days`
drawn from a late-reporting distribution, AND `graph_hop_distance` close to known
fraud entities, all at once. Random generation breaks these correlations and produces
a dataset that teaches the model nothing meaningful.

---

## Dependency Order

Build in this exact order. Do not skip layers.

```
Layer 0 — config.py + feature_definitions.py           (no dependencies)
    ↓
Layer 1 — archetypes_underwriting.py
          archetypes_claims.py                          (imports feature_definitions)
    ↓
Layer 2 — generator.py                                  (imports archetypes, outputs raw records)
    ↓
Layer 3 — entity_vehicle.py                             (VIN decode, MSRP, ADAS — no graph dependency)
          entity_person.py                              (dedup, role assignment)
          entity_address.py                             (normalize + hash)
          entity_phone.py                               (normalize + hash)
          entity_policy.py                              (inception date, lapse calc — writes policy_inception_date)
    ↓
Layer 4 — validator.py                                  (validates resolved entities + raw data)
    ↓
Layer 5 — graph_builder.py                              (loads resolved entities as nodes/edges ONLY)
    ↓
Layer 6 — graph_features.py                             (queries Neo4j → enriches feature store)
          offline_pipeline.py                           (feature_definitions + resolved entities)
    ↓
Layer 7 — train_frequency.py + train_severity.py        (quotes, processed features)
          fraud_scoring/train.py                        (claims, processed + graph features)
```

**Why `feature_definitions.py` is Layer 0:** Archetypes must import feature names
from `feature_definitions.py`, not define their own. This prevents offline/online
skew from the first line of code. The same function that names
`telematics_distraction_score` in the offline pipeline must be what the archetype
populates. `feature_definitions.py` has no imports from archetypes, generator, or
`entity_*.py` modules — entity resolution outputs are passed as arguments to feature
computation functions, not imported as modules. This preserves Layer 0 independence.

**Why entity resolution is Layer 3, not Layer 5:** Vehicle, Person, Address, and
Phone are independent entities with no graph dependency. Base attributes (VIN decode,
MSRP, address normalization) must be resolved before the graph is built —
`graph_builder.py` loads resolved entities as nodes. Computing vehicle features
inside `graph_builder.py` creates a false dependency on the graph existing first.
Address and phone normalization must also happen before graph edges are created — a
`shares_address` edge between two unnormalized strings is unreliable and degrades
Louvain community detection silently.

**Why graph is Layer 5, not Layer 1:** Graph features cannot be baked into the
generator. At inference time you query Neo4j live. If you pre-join graph features
in the generator, you break the online serving path.

---

## Week 1 Build Order

```
Day 1   config.py, feature_definitions.py

Day 2   archetypes_claims.py, archetypes_underwriting.py

Day 3   generator.py — outputs data/raw/quotes.parquet + data/raw/claims.parquet
                       including all source datetime columns (see DEC-013)

Day 4   entity_vehicle.py  — VIN decode, MSRP lookup, ADAS efficacy flags
        entity_person.py   — dedup persons, assign roles (policyholder/driver/claimant)
        entity_address.py  — normalize + hash addresses for graph edge reliability
        entity_phone.py    — normalize + hash phone numbers
        entity_policy.py   — coverage validation, inception date, lapse calculation
                             writes policy_inception_date to data/entities/policies.parquet
        outputs → data/entities/

Day 5   validator.py        — verify entity resolution (null VINs, dedup counts,
                              address normalization coverage, null rates, fraud ratios,
                              temporal integrity checks — see Validator Checks section)
        graph_builder.py   — load resolved entities from data/entities/ as nodes
                              + edges into Neo4j (no feature computation here)
        offline_pipeline.py — compute features from resolved entities
                              → data/processed/
```

Do not touch model training until `data/processed/` has clean output from the
offline pipeline.

---

## When Is the Offline Pipeline Used?

The offline pipeline (`offline_pipeline.py`) runs in two distinct contexts:

**1. Training data preparation** — batch computation over all historical records.
It reads resolved entities from `data/entities/`, computes the full feature set,
and writes `data/processed/quotes_features.parquet` and `claims_features.parquet`
for model training.

**2. Nightly batch enrichment** — pre-computing features for known entities
(customers, vehicles, policies) and pushing them into the Redis online store, so
live inference can retrieve them in under 10ms without recomputing on the fly.

The offline pipeline is explicitly *not* used in the real-time sync path (<100ms).
At FNOL time, the online store (Redis) serves pre-computed features. Entity
resolution is also a pre-computation step completed offline — it does not run
inside either inference path.

---

## Archetype Definitions

### Quotes — 10 Driver/Vehicle Risk Profiles

Each archetype sets a **distribution per feature**, not a single value. The
generator samples from these distributions.

| Archetype | Volume | Annual Claim Rate | Telematics Opt-in | Credit Score Range |
|---|---|---|---|---|
| Young high-power driver | 1,500 | 18% | 45% | 580–650 |
| Urban commuter | 2,500 | 10% | 55% | 650–720 |
| Rural low-mileage | 2,000 | 5% | 40% | 700–780 |
| Senior driver | 1,500 | 8% | 30% | 720–800 |
| Multi-vehicle household | 2,500 | 7% | 60% | 680–750 |
| Lapsed coverage history | 1,500 | 14% | 25% | 560–630 |
| DUI on record | 1,000 | 20% | 20% | 580–660 |
| New driver (<2yr licensed) | 2,000 | 15% | 65% | 600–680 |
| Luxury vehicle | 1,500 | 9% | 70% | 740–820 |
| Preferred low-risk | 4,000 | 3% | 75% | 760–840 |
| **Total** | **20,000** | **~8% avg** | | |

```python
# archetypes_underwriting.py
from dataclasses import dataclass
from features.feature_definitions import FEATURE_NAMES  # import, don't redefine

@dataclass
class UnderwritingArchetype:
    name: str
    volume: int
    annual_claim_rate: float
    telematics_opt_in_rate: float        # drives telematics_available
    credit_score_dist: tuple             # (mean, std) for normal distribution
    prior_loss_frequency_dist: tuple
    violation_severity_index_dist: tuple
    insurance_lapse_days_dist: tuple
    # quote_requested_at is generated by generator.py from a realistic
    # date spread — not an archetype distribution. See generator.py notes.
    # ... all 20 features defined as distributions
```

### Claims — 10 Claim Archetypes

| Archetype | Volume | Fraud | Telematics Opt-in | Key Signals |
|---|---|---|---|---|
| Staged rear-end collision | 1,000 | ✅ Yes | 5% | `attorney_present=True`, `claimant_count` Poisson(3.2), assigned to fraud ring |
| Soft tissue exaggeration | 1,500 | ✅ Yes | 20% | `reporting_delay_days` Gamma(2,1.5), `reported_injury_count` high |
| VIN cloning | 800 | ✅ Yes | 3% | `policy_inception_days` <30, `device_fingerprint_match=False` |
| Inflated repair estimate | 1,200 | ✅ Yes | 15% | `attorney_centrality_score` high, `shared_attribute_count` >2 |
| Phantom passenger | 600 | ✅ Yes | 8% | `submission_hour` 22–4, `claimant_count` ≥3 |
| Coordinated fraud ring | 500 | ✅ Yes | 3% | assigned to dense fraud ring, `shared_attribute_count` >4 |
| Medical billing inflation | 700 | ✅ Yes | 10% | `reported_injury_count` high, `narrative_inconsistency_score` >0.7 |
| Legitimate fender-bender | 5,500 | ❌ No | 72% | `telematics_crash_match` >0.8, `reporting_delay_days` <2 |
| Legitimate major accident | 4,500 | ❌ No | 68% | Police report present, `risk_score_at_issuance` low |
| Total loss misrepresentation | 3,700 | ✅ Yes | 12% | `narrative_inconsistency_score` >0.6, `policy_inception_days` <60 |
| **Total** | **20,000** | **~33% fraud** | | |

> **Note on fraud rate:** 33% is higher than production (~15%) but correct for
> training. Use `class_weight` in XGBoost to correct for this during training,
> not by undersampling your minority class.

```python
# archetypes_claims.py
@dataclass
class ClaimArchetype:
    name: str
    volume: int
    is_fraud: bool
    telematics_opt_in_rate: float        # sampled to set telematics_available
    telematics_enrolled_rate: float      # of those with no telematics, how many were enrolled?
    # Each feature defined as a scipy distribution or fixed value
    reporting_delay_days_dist: tuple     # e.g. ("gamma", {"a": 2, "scale": 1.5})
                                         # loss_event_datetime + this offset → fnol_submitted_at
                                         # reporting_delay_days is derived from those two datetimes
                                         # (see DEC-013) — archetypes define the offset distribution
    attorney_present_prob: float
    claimant_count_dist: tuple
    fraud_ring_id: str | None            # assigns entity to a fraud ring;
                                         # graph_hop_distance is derived by
                                         # graph_features.py after graph is built —
                                         # archetypes do not set it directly
    shared_attribute_count_dist: tuple   # controls edge density in the graph;
                                         # hop distance emerges from graph structure
    loss_event_hour_dist: tuple          # e.g. ("uniform", {"low": 6, "high": 22})
                                         # drives submission_hour and loss_event_datetime
    claim_open_duration_days_dist: tuple # drives claim_closed_at offset from fnol_submitted_at;
                                         # fraud archetypes skew longer (SIU investigation time)
    fraud_confirmation_lag_days_dist: tuple | None
                                         # None for legitimate archetypes (fraud_confirmed_at = null)
                                         # fraud archetypes: ("gamma", {"a": 3, "scale": 30})
                                         # produces realistic 60–180 day SIU confirmation lag
    # ...
```

---

## Source Datetime Columns (DEC-013)

Source datetimes are added to raw parquet files to support PSI drift monitoring,
concept drift label windows, and temporal integrity validation. They are **pipeline
columns only** — they do not enter the feature vector. The feature store continues
to consume derived int features (`reporting_delay_days`, `policy_inception_days`).

See `feature_definitions.py` for the canonical derivation relationships:
```
policy_inception_days  = (quote_completed_at.date - policy_inception_date).days
reporting_delay_days   = (fnol_submitted_at - loss_event_datetime).days
```

### quotes.parquet — datetime columns

| Column | Type | Nullable | Written By | Purpose |
|---|---|---|---|---|
| `quote_requested_at` | datetime | No | `generator.py` | PSI current-period window anchor |
| `quote_completed_at` | datetime | No | `hurdle_model.py` | Source for `policy_inception_days` derivation |
| `policy_inception_date` | date | No | `entity_policy.py` | Source for `policy_inception_days` derivation |

**Generation notes:**
- `quote_requested_at` is spread across a realistic date range (e.g. 12-month
  window ending at generation time) using a uniform or slightly recency-weighted
  distribution. It is not an archetype distribution — it is a pipeline-level
  parameter in `generator.py` controlled by `QUOTE_DATE_RANGE_DAYS` in `config.py`.
- `quote_completed_at` is `quote_requested_at` + a small processing offset
  (seconds to minutes). Written by `hurdle_model.py` after risk scoring completes.
- `policy_inception_date` is written by `entity_policy.py` as an explicit date
  column. It is the authoritative source for `policy_inception_days` — do not
  recompute inception days from any other field.

### claims.parquet — datetime columns

| Column | Type | Nullable | Written By | Purpose |
|---|---|---|---|---|
| `loss_event_datetime` | datetime | No | `generator.py` | Source for `reporting_delay_days` derivation |
| `fnol_submitted_at` | datetime | No | `generator.py` | PSI current-period window anchor; source for `reporting_delay_days` |
| `claim_closed_at` | datetime | Yes — open claims | `generator.py` | Pipeline use only |
| `fraud_confirmed_at` | datetime | Yes — unconfirmed claims | Label store, post-SIU | 90/180-day concept drift label window (DEC-007) |

**Generation notes:**
- `loss_event_datetime` is sampled from the archetype's `loss_event_hour_dist`
  within the same date spread used for quotes. Fraud archetypes with late-night
  signals (phantom passenger) use a night-weighted hour distribution.
- `fnol_submitted_at` = `loss_event_datetime` + offset sampled from
  `reporting_delay_days_dist` (converted to timedelta). This ensures
  `reporting_delay_days` computed from the two datetimes exactly matches the
  archetype distribution — no rounding drift.
- `claim_closed_at` = `fnol_submitted_at` + offset sampled from
  `claim_open_duration_days_dist`. Null for a configurable fraction of records
  (open claims). Fraud archetypes skew longer due to SIU investigation time.
- `fraud_confirmed_at` is null for all legitimate archetypes and for fraud
  records that simulate open/unconfirmed investigations. For confirmed fraud
  records, it is `fnol_submitted_at` + offset sampled from
  `fraud_confirmation_lag_days_dist` (typically 60–180 days). This is the column
  that makes the delayed label pipeline from DEC-007 queryable in production.

---

## Generator Logic

```python
# data/synthetic/generator.py

import pandas as pd
import numpy as np
from faker import Faker
from features.feature_definitions import (
    build_telematics_features,
    apply_state_regulatory_mask,
)
from data.synthetic.archetypes_claims import CLAIM_ARCHETYPES
from data.synthetic.archetypes_underwriting import UNDERWRITING_ARCHETYPES

fake = Faker()

def generate_quotes(seed: int = 42) -> pd.DataFrame:
    """
    Outputs data/raw/quotes.parquet.
    Includes source datetime columns: quote_requested_at, quote_completed_at.
    policy_inception_date is written later by entity_policy.py.

    quote_requested_at is spread across QUOTE_DATE_RANGE_DAYS (config.py)
    ending at generation time — not an archetype distribution.
    quote_completed_at = quote_requested_at + small processing offset (seconds).
    """
    np.random.seed(seed)
    records = []

    # Date spread for PSI window anchor — pipeline-level, not per archetype
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=QUOTE_DATE_RANGE_DAYS)

    for archetype in UNDERWRITING_ARCHETYPES:
        for _ in range(archetype.volume):
            state = fake.state_abbr()
            has_telematics = np.random.random() < archetype.telematics_opt_in_rate
            telematics = sample_telematics(archetype) if has_telematics else None

            # Source datetimes — pipeline columns, do not enter feature vector
            quote_requested_at = start_date + pd.Timedelta(
                seconds=np.random.uniform(0, (end_date - start_date).total_seconds())
            )
            quote_completed_at = quote_requested_at + pd.Timedelta(
                seconds=np.random.uniform(1, 300)  # 1s–5min processing window
            )

            features = {
                "quote_id": fake.uuid4(),
                "customer_id": fake.uuid4(),
                "state": state,
                "archetype": archetype.name,
                # Source datetimes
                "quote_requested_at": quote_requested_at,
                "quote_completed_at": quote_completed_at,
                # policy_inception_date written by entity_policy.py — not here
                **sample_tabular_features(archetype),
                **build_telematics_features(telematics, policy=None),
            }
            features = apply_state_regulatory_mask(features, state)
            records.append(features)

    df = pd.DataFrame(records)
    df.to_parquet("data/raw/quotes.parquet", index=False)
    return df


def generate_claims(quotes_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Outputs data/raw/claims.parquet.
    Must be called AFTER quotes are risk-scored (risk_score_at_issuance required).
    Entity resolution (data/entities/) must be complete before offline_pipeline.py
    assembles the final feature vectors.

    Source datetime columns emitted:
      loss_event_datetime   — sampled from archetype loss_event_hour_dist
      fnol_submitted_at     — loss_event_datetime + reporting_delay offset
      claim_closed_at       — fnol_submitted_at + open_duration offset (nullable)
      fraud_confirmed_at    — fnol_submitted_at + confirmation_lag offset (nullable)

    reporting_delay_days in the feature vector is derived from these two datetimes
    in feature_definitions.py — it is NOT sampled independently here.
    """
    assert "risk_score_at_issuance" in quotes_df.columns, \
        "Run risk scoring on quotes before generating claims"

    np.random.seed(seed)
    records = []

    for archetype in CLAIM_ARCHETYPES:
        archetype_quotes = quotes_df.sample(
            n=archetype.volume, replace=True, random_state=seed
        )
        for _, quote in archetype_quotes.iterrows():
            has_telematics = np.random.random() < archetype.telematics_opt_in_rate
            telematics = sample_telematics_claims(archetype) if has_telematics else None

            # loss_event_datetime — within the quote date spread, after inception
            loss_event_datetime = sample_loss_event_datetime(
                archetype, quote["quote_requested_at"]
            )

            # fnol_submitted_at — derived from loss event + reporting delay offset
            reporting_delay_days = sample_from_dist(archetype.reporting_delay_days_dist)
            fnol_submitted_at = loss_event_datetime + pd.Timedelta(
                days=reporting_delay_days
            )

            # claim_closed_at — nullable for open claims
            open_duration_days = sample_from_dist(archetype.claim_open_duration_days_dist)
            claim_closed_at = (
                fnol_submitted_at + pd.Timedelta(days=open_duration_days)
                if np.random.random() > OPEN_CLAIM_RATE  # config.py
                else None
            )

            # fraud_confirmed_at — nullable for legitimate and unconfirmed claims
            fraud_confirmed_at = None
            if archetype.is_fraud and archetype.fraud_confirmation_lag_days_dist:
                if np.random.random() > UNCONFIRMED_FRAUD_RATE:  # config.py
                    lag_days = sample_from_dist(
                        archetype.fraud_confirmation_lag_days_dist
                    )
                    fraud_confirmed_at = fnol_submitted_at + pd.Timedelta(
                        days=lag_days
                    )

            features = {
                "claim_id": fake.uuid4(),
                "quote_id": quote["quote_id"],
                "customer_id": quote["customer_id"],
                "state": quote["state"],
                "archetype": archetype.name,
                "is_fraud": archetype.is_fraud,
                # Source datetimes — pipeline columns, do not enter feature vector
                "loss_event_datetime": loss_event_datetime,
                "fnol_submitted_at": fnol_submitted_at,
                "claim_closed_at": claim_closed_at,
                "fraud_confirmed_at": fraud_confirmed_at,
                # Derived int — computed from source datetimes for consistency
                # (feature_definitions.py owns this formula)
                "reporting_delay_days": reporting_delay_days,
                # Cross-platform shared feature from underwriting
                "risk_score_at_issuance": quote["risk_score_at_issuance"],
                **sample_claim_tabular_features(archetype),
                **build_telematics_features(telematics, policy=quote),
            }
            records.append(features)

    df = pd.DataFrame(records)
    df.to_parquet("data/raw/claims.parquet", index=False)
    return df
```

---

## Validator Checks

```python
# data/synthetic/validator.py

def validate(quotes_df, claims_df):
    checks = [
        # --- Null rate checks ---
        ("telematics_available null rate",
            lambda df: df["telematics_available"].isnull().mean() == 0),

        ("telematics_distraction_score null rate ~60%",
            lambda df: 0.55 < df["telematics_distraction_score"].isnull().mean() < 0.65),

        ("credit_score null in CA/MA/MI/HI",
            lambda df: df[df["state"].isin({"CA","MA","MI","HI"})]["credit_score"].isnull().all()),

        # --- Fraud ratio ---
        ("fraud rate 30-36%",
            lambda df: 0.30 < df["is_fraud"].mean() < 0.36),

        # --- Fraud signal checks ---
        ("enrolled_but_missing in 8-12% of fraud",
            lambda df: 0.08 < df[df["is_fraud"]]["telematics_enrolled_but_missing"].mean() < 0.12),

        # --- Cross-platform dependency ---
        ("risk_score_at_issuance populated",
            lambda df: df["risk_score_at_issuance"].isnull().mean() == 0),

        # --- Entity resolution checks ---
        ("no null VINs after entity resolution",
            lambda df: df["vin"].isnull().mean() == 0),

        ("address hash populated for all records",
            lambda df: df["address_hash"].isnull().mean() == 0),

        ("person dedup — no duplicate person_ids",
            lambda df: df["person_id"].duplicated().sum() == 0),

        # --- Temporal integrity checks (DEC-013) ---
        # Replaces the weaker "policy_inception_days >= 0" check
        ("policy inception before quote completed",
            lambda df: (df["policy_inception_date"]
                        >= df["quote_completed_at"].dt.date).all()),

        ("fnol after loss event",
            lambda df: (df["fnol_submitted_at"] >= df["loss_event_datetime"]).all()),

        ("claim closed after fnol (where closed)",
            lambda df: (
                df[df["claim_closed_at"].notna()]["claim_closed_at"]
                >= df[df["claim_closed_at"].notna()]["fnol_submitted_at"]
            ).all()),

        ("fraud confirmed after fnol (where confirmed)",
            lambda df: (
                df[df["fraud_confirmed_at"].notna()]["fraud_confirmed_at"]
                >= df[df["fraud_confirmed_at"].notna()]["fnol_submitted_at"]
            ).all()),

        # Confirms generator derives reporting_delay_days from datetimes,
        # not as an independent sample — catches any rounding drift
        ("reporting_delay_days consistent with datetimes",
            lambda df: (
                (df["fnol_submitted_at"] - df["loss_event_datetime"]).dt.days
                == df["reporting_delay_days"]
            ).all()),

        # Confirms fraud_confirmed_at is null for all legitimate archetypes
        ("fraud_confirmed_at null for legitimate claims",
            lambda df: df[~df["is_fraud"]]["fraud_confirmed_at"].isnull().all()),

        # Confirms PSI window anchor is populated and within expected date range
        ("quote_requested_at populated for all quotes",
            lambda df: df["quote_requested_at"].isnull().mean() == 0),

        ("fnol_submitted_at populated for all claims",
            lambda df: df["fnol_submitted_at"].isnull().mean() == 0),
    ]

    for name, check in checks:
        result = check(claims_df if "is_fraud" in claims_df.columns else quotes_df)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} — {name}")
```

---

## Severity Model Watch-out

At 20K quotes with ~8% annual claim rate → ~1,600 records with actual claims
for Gamma regression.

**Use 60/20/20 split** (not 80/20) for severity model:
- Train: ~960 claim records
- Val: ~320 claim records
- Test: ~320 claim records

If Gamma loss does not converge, bump quotes to 50K before adjusting model
hyperparameters. The bottleneck is data, not the model.

---

## Graph Density Watch-out

At 20K claims, fraud ring archetypes (staged collision, coordinated ring) produce
sparse graph clusters. If your GNN training loss does not converge:

1. Check Neo4j has loaded correctly: `MATCH (n) RETURN count(n)` — expect ~50K nodes
2. Check edge density: `MATCH ()-[r]->() RETURN count(r)` — expect ~200K edges
3. If clusters are isolated islands with no shared attributes, increase
   `shared_attribute_count` in ring archetypes
4. Bump to 50K claims before changing GNN architecture

Note: `graph_hop_distance` is a graph-derived metric computed by `graph_features.py`
after the graph is built. If hop distances are not clustering as expected per
archetype, verify `fraud_ring_id` assignments in archetypes_claims.py are loading
correctly into Neo4j before debugging GNN architecture.

---

## File Outputs

```
data/
├── raw/
│   ├── quotes.parquet           ← generator.py output
│   │                              model features + source datetime columns:
│   │                              quote_requested_at, quote_completed_at
│   │                              (policy_inception_date added by entity_policy.py)
│   └── claims.parquet           ← generator.py output (after risk scoring)
│                                  model features + source datetime columns:
│                                  loss_event_datetime, fnol_submitted_at,
│                                  claim_closed_at (nullable),
│                                  fraud_confirmed_at (nullable)
├── entities/                    ← entity resolution outputs (gitignored)
│   ├── vehicles.parquet         ← resolved VINs, MSRP, horsepower, ADAS efficacy
│   ├── persons.parquet          ← deduplicated persons with role flags
│   ├── addresses.parquet        ← normalized + hashed addresses
│   ├── phones.parquet           ← normalized + hashed phone numbers
│   └── policies.parquet         ← coverage validation, inception date,
│                                  lapse calc — writes policy_inception_date
└── processed/
    ├── quotes_features.parquet  ← offline_pipeline.py output
    │                              feature vector only — source datetimes
    │                              are NOT propagated here
    └── claims_features.parquet  ← offline_pipeline.py + graph_features.py output
                                   feature vector only — source datetimes
                                   are NOT propagated here
```

> **Source datetime propagation rule:** Source datetime columns (`quote_requested_at`,
> `fnol_submitted_at`, etc.) live in `data/raw/` and `data/entities/` only. They are
> consumed by `monitoring/psi_drift.py` directly from raw parquet — they are not
> forwarded into `data/processed/` or the feature vector. The feature vector contains
> only the derived ints (`reporting_delay_days`, `policy_inception_days`). This
> boundary enforces Option A from DEC-013 and prevents source datetimes from
> accidentally entering model training.

All files in `data/raw/`, `data/entities/`, and `data/processed/` are gitignored.
Regenerate from `generator.py` followed by `entity_*.py` resolution scripts.
