# Data Generation Implementation Guide
### Smart Oak Insurance — AI Platform
> Version: 2026-Q2 | Target: 20K quotes + 20K claims | Features: 20 per platform

---

## Core Principle

Generate data **top-down from archetypes**, never bottom-up from random values.

Random data produces no correlations between features. Your graph neural network, anomaly detector, and ensemble model all rely on correlated feature patterns — a staged-collision ring should have `attorney_present=True`, `reporting_delay_days` drawn from a late-reporting distribution, AND `graph_hop_distance` close to known fraud entities, all at once. Random generation breaks these correlations and produces a dataset that teaches the model nothing meaningful.

---

## Dependency Order

Build in this exact order. Do not skip layers.

```
Layer 0 — config.py + feature_definitions.py      (no dependencies)
    ↓
Layer 1 — archetypes_underwriting.py
          archetypes_claims.py                     (imports feature_definitions)
    ↓
Layer 2 — generator.py                             (imports archetypes)
    ↓
Layer 3 — validator.py                             (imports feature_definitions + raw data)
    ↓
Layer 4 — graph_builder.py                         (imports raw data → loads Neo4j)
          offline_pipeline.py                      (imports feature_definitions + raw data)
    ↓
Layer 5 — graph_features.py                        (queries Neo4j → enriches feature store)
    ↓
Layer 6 — train_frequency.py + train_severity.py   (quotes, processed features)
          fraud_scoring/train.py                   (claims, processed + graph features)
```

**Why `feature_definitions.py` is Layer 0:** Archetypes must import feature names from `feature_definitions.py`, not define their own. This prevents offline/online skew from the first line of code. The same function that names `telematics_distraction_score` in the offline pipeline must be what the archetype populates.

**Why graph is Layer 4, not Layer 1:** Graph features cannot be baked into the generator. At inference time you query Neo4j live. If you pre-join graph features in the generator, you break the online serving path.

---

## Week 1 Build Order

```
Day 1   config.py, feature_definitions.py
Day 2   archetypes_claims.py, archetypes_underwriting.py
Day 3   generator.py — outputs data/raw/quotes.parquet + data/raw/claims.parquet
Day 4   validator.py — verify null rates, fraud ratios, no claim before policy inception
Day 5   graph_builder.py — load Neo4j from raw data
        offline_pipeline.py — compute features → data/processed/
```

Do not touch model training until `data/processed/` has clean output from the offline pipeline.

---

## Archetype Definitions

### Quotes — 10 Driver/Vehicle Risk Profiles

Each archetype sets a **distribution per feature**, not a single value. The generator samples from these distributions.

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
    # ... all 20 features defined as distributions
```

### Claims — 10 Claim Archetypes

| Archetype | Volume | Fraud | Telematics Opt-in | Key Signals |
|---|---|---|---|---|
| Staged rear-end collision | 1,000 | ✅ Yes | 5% | `attorney_present=True`, `claimant_count` Poisson(3.2), `graph_hop_distance` ≤2 |
| Soft tissue exaggeration | 1,500 | ✅ Yes | 20% | `reporting_delay_days` Gamma(2,1.5), `reported_injury_count` high |
| VIN cloning | 800 | ✅ Yes | 3% | `policy_inception_days` <30, `device_fingerprint_match=False` |
| Inflated repair estimate | 1,200 | ✅ Yes | 15% | `attorney_centrality_score` high, `shared_attribute_count` >2 |
| Phantom passenger | 600 | ✅ Yes | 8% | `submission_hour` 22–4, `claimant_count` ≥3 |
| Coordinated fraud ring | 500 | ✅ Yes | 3% | `graph_hop_distance` =1, `shared_attribute_count` >4 |
| Medical billing inflation | 700 | ✅ Yes | 10% | `reported_injury_count` high, `narrative_inconsistency_score` >0.7 |
| Legitimate fender-bender | 5,500 | ❌ No | 72% | `telematics_crash_match` >0.8, `reporting_delay_days` <2 |
| Legitimate major accident | 4,500 | ❌ No | 68% | Police report present, `risk_score_at_issuance` low |
| Total loss misrepresentation | 3,700 | ✅ Yes | 12% | `narrative_inconsistency_score` >0.6, `policy_inception_days` <60 |
| **Total** | **20,000** | **~33% fraud** | | |

> **Note on fraud rate:** 33% is higher than production (~15%) but correct for training. Use `class_weight` in XGBoost to correct for this during training, not by undersampling your minority class.

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
    attorney_present_prob: float
    claimant_count_dist: tuple
    graph_hop_distance_dist: tuple
    # ...
```

---

## Generator Logic

```python
# data/synthetic/generator.py

import pandas as pd
import numpy as np
from faker import Faker
from features.feature_definitions import build_telematics_features, apply_state_regulatory_mask
from data.synthetic.archetypes_claims import CLAIM_ARCHETYPES
from data.synthetic.archetypes_underwriting import UNDERWRITING_ARCHETYPES

fake = Faker()

def generate_quotes(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    records = []
    for archetype in UNDERWRITING_ARCHETYPES:
        for _ in range(archetype.volume):
            state = fake.state_abbr()
            has_telematics = np.random.random() < archetype.telematics_opt_in_rate
            telematics = sample_telematics(archetype) if has_telematics else None

            features = {
                "quote_id": fake.uuid4(),
                "customer_id": fake.uuid4(),
                "state": state,
                "archetype": archetype.name,
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
    Claims are generated AFTER quotes are scored.
    risk_score_at_issuance must exist in quotes_df before calling this.
    """
    assert "risk_score_at_issuance" in quotes_df.columns, \
        "Run risk scoring on quotes before generating claims"
    # ... sampling logic
```

---

## Validator Checks

```python
# data/synthetic/validator.py

def validate(quotes_df, claims_df):
    checks = [
        # Null rate checks
        ("telematics_available null rate", lambda df: df["telematics_available"].isnull().mean() == 0),
        ("telematics_distraction_score null rate ~60%", lambda df:
            0.55 < df["telematics_distraction_score"].isnull().mean() < 0.65),
        ("credit_score null in CA/MA/MI/HI", lambda df:
            df[df["state"].isin({"CA","MA","MI","HI"})]["credit_score"].isnull().all()),

        # Fraud ratio
        ("fraud rate 30-36%", lambda df:
            0.30 < df["is_fraud"].mean() < 0.36),

        # Realism checks
        ("no claim before policy inception", lambda df:
            (df["policy_inception_days"] >= 0).all()),

        # Fraud signal checks
        ("enrolled_but_missing in 8-12% of fraud", lambda df:
            0.08 < df[df["is_fraud"]]["telematics_enrolled_but_missing"].mean() < 0.12),

        # Cross-platform dependency
        ("risk_score_at_issuance populated", lambda df:
            df["risk_score_at_issuance"].isnull().mean() == 0),
    ]

    for name, check in checks:
        result = check(claims_df if "is_fraud" in claims_df.columns else quotes_df)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} — {name}")
```

---

## Severity Model Watch-out

At 20K quotes with ~8% annual claim rate → ~1,600 records with actual claims for Gamma regression.

**Use 60/20/20 split** (not 80/20) for severity model:
- Train: ~960 claim records
- Val: ~320 claim records
- Test: ~320 claim records

If Gamma loss does not converge, bump quotes to 50K before adjusting model hyperparameters. The bottleneck is data, not the model.

---

## Graph Density Watch-out

At 20K claims, fraud ring archetypes (staged collision, coordinated ring) produce sparse graph clusters. If your GNN training loss does not converge:

1. Check Neo4j has loaded correctly: `MATCH (n) RETURN count(n)` — expect ~50K nodes
2. Check edge density: `MATCH ()-[r]->() RETURN count(r)` — expect ~200K edges
3. If clusters are isolated islands with no shared attributes, increase `shared_attribute_count` in ring archetypes
4. Bump to 50K claims before changing GNN architecture

---

## File Outputs

```
data/
├── raw/
│   ├── quotes.parquet          ← generator.py output (quotes)
│   └── claims.parquet          ← generator.py output (claims, after risk scoring)
└── processed/
    ├── quotes_features.parquet ← offline_pipeline.py output
    └── claims_features.parquet ← offline_pipeline.py + graph_features.py output
```

All files in `data/raw/` and `data/processed/` are gitignored. Regenerate from `generator.py`.