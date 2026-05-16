

# Hybrid Real-Time Fraud Decisioning Platform
### Personal Auto Insurance — ML Meetup Architecture Deep Dive

---

> **Core Framing:**
> Modern fraud detection is not a supervised classification problem.
> It is an **adversarial risk orchestration system** combining anomaly detection,
> graph intelligence, behavioral analytics, label-uncertain learning, and human
> investigation workflows.

---

## Table of Contents
1. [Architecture Flow](#1-architecture-flow)
2. [Signal Acquisition Layer](#2-signal-acquisition-layer)
3. [Feature Store & Graph Intelligence](#3-feature-store--graph-intelligence)
4. [Ensemble Fraud Scoring](#4-ensemble-fraud-scoring)
5. [Label Quality & PU Learning](#5-label-quality--pu-learning)
6. [Real-Time Decision Orchestration](#6-real-time-decision-orchestration)
7. [Inference Architecture: Sync vs Async](#7-inference-architecture-sync-vs-async)
8. [Adversarial Fraud Simulation](#8-adversarial-fraud-simulation)
9. [AI Governance & Explainability](#9-ai-governance--explainability)
10. [Data Reliability Controls](#10-data-reliability-controls)
11. [MLOps & Champion-Challenger](#11-mlops--champion-challenger)
12. [Investigator Copilot (HITL)](#12-investigator-copilot-hitl)
13. [Business KPIs](#13-business-kpis)
14. [Production vs Emerging Capabilities](#14-production-vs-emerging-capabilities)

---

## 1. Architecture Flow

```
FNOL Intake (Web / Mobile / Agent)
           │
           ▼
Real-Time Feature Enrichment  ◄──── Feature Store (online serving)
           │                         (pre-computed from resolved entities)
           ▼
  ┌────────┴─────────────────────┐
  │   Multi-Signal Fusion        │
  │  Graph · Device · Policy     │
  │  Telematics · NLP · Vision   │
  └────────┬─────────────────────┘
           │
           ▼
  Ensemble Fraud Scoring Engine
           │
           ▼
  Decision Orchestration Layer
  (rules + risk tiers + compliance)
           │
           ▼
  Adaptive Action Routing
  │            │           │
  ▼            ▼           ▼
STP       Evidence     SIU Escalation /
(auto)    Request      Payment Hold
           │
           ▼
  Investigator Copilot (HITL)
           │
           ▼
  Feedback Loop → Label Store → Retraining
```

---

## 2. Signal Acquisition Layer

| Signal Domain | Sources | Production Status |
|---|---|---|
| **Entity resolution** | VIN decode, address normalization, person dedup, phone normalization | ✅ Production |
| **Structured claim data** | FNOL form, policy system | ✅ Production |
| **Device fingerprint** | IP, browser/app metadata, geolocation | ✅ Production |
| **Telematics** | OBD-II, mobile SDK, crash event logs | ✅ Production |
| **Image/video** | Damage photos, repair estimates | ✅ Production |
| **Unstructured narrative** | Claim description, adjuster notes | ✅ Production |
| **Graph signals** | Shared addresses, phones, VINs, attorney networks | ✅ Production |
| **Third-party enrichment** | Motor vehicle records, credit bureau, LexisNexis | ✅ Production |
| **Federated fraud intelligence** | Cross-carrier shared fraud indicators | 🔬 Emerging |

> **Note:** Entity resolution is a pre-computation step completed offline in the
> `entities/` layer. It is not part of the real-time signal acquisition path at
> FNOL time. Resolved entities are consumed from `data/entities/` by the feature
> store pipeline and loaded into Neo4j by `graph_builder.py` before any inference
> runs.

---

## 3. Feature Store & Graph Intelligence

### Feature Store
- **Online store** (Redis / Feast): sub-10ms feature serving at FNOL time
- **Offline store** (S3 + Spark): training data, batch enrichment, backfill
- **Feature versioning**: immutable feature snapshots for regulatory reproducibility
- **Training-serving skew detection**: automated parity checks between offline
  and online pipelines
- **Entity dependency**: all feature store inputs are resolved entities from the
  `entities/` layer. Raw generator output is not fed directly into the feature
  store pipeline.

### Graph Database (Neo4j / Amazon Neptune)

```
Person ──[as_claimant]──► Claimant ──[filed]──► Claim
Person ──[as_driver]────► Driver ──[operates]──► Vehicle ──[involves]──► Claim
Person ──[as_policyholder]──► Policy ──[active_at_fnol]──► Claim

Address ──[shares_address]──► Person   ← normalized entity (entity_address.py)
Phone   ──[shares_phone]───► Person    ← normalized entity (entity_phone.py)
Vehicle ──[shares_vin]─────► Claim     ← fraud signal: same VIN, multiple claims
Claimant ──[represented_by]──► Attorney ──[linked_to]──► Fraud Ring
Claim ──[repaired_at]────────► Body Shop
```

**Person, Claimant, and Driver are distinct roles of the same Person entity.**
A person who appears as a claimant on policies they did not hold, or as a driver
on multiple unrelated claims, is a meaningful cross-role fraud signal. Collapsing
these roles into a single node loses that signal.

**Graph features extracted:**
- 1st–3rd degree connection to known fraud entities
- Shared attribute cluster density (address, phone, VIN, bank account)
- Attorney/body-shop network centrality scores
- Temporal submission burst patterns (coordinated ring behavior)
- Community detection via Louvain / label propagation

> **Reliability note:** `shares_address` and `shares_phone` edge reliability depends
> on normalized Address and Phone entities upstream (`entity_address.py`,
> `entity_phone.py`). If edges are built from unnormalized strings, Louvain
> community detection degrades silently — two records for "123 Main St" and
> "123 Main Street" will not share an edge and the ring will appear disconnected.

---

## 4. Ensemble Fraud Scoring

```
Tabular XGBoost          → P(fraud | structured features)
Graph Neural Network     → P(fraud | network context)        [GraphSAGE / GCN]
NLP Transformer          → P(fraud | narrative inconsistency)
Vision Model (ViT/CLIP)  → P(fraud | image tampering signals)
Anomaly Detector         → Isolation Forest / Autoencoder    [unsupervised]
          │
          ▼
  Stacking Meta-Learner  (logistic regression or LightGBM)
          │
          ▼
   Final Fraud Score  [0.0 – 1.0]  +  Confidence Interval
```

**Cost-sensitive learning:**
- Asymmetric loss functions: missed fraud >> false positive
- Class-weight tuning based on average indemnity per fraud type
- Threshold set to minimize expected dollar loss, not accuracy

**Vision layer (production-grade):**
- Vision Transformer (ViT) for image-level fraud probability
- CLIP-style embeddings for cross-modal consistency (photo ↔ narrative)
- Image tampering detection: metadata analysis, splicing, cloning artifacts
- Damage segmentation separate from fraud scoring (two distinct ML tasks)

---

## 5. Label Quality & PU Learning

> **The problem most fraud presentations ignore:**
> A meaningful fraction of your "legitimate" training labels are actually
> undetected fraud. You are training on a contaminated negative class.

### Why This Matters
- Fraud labels arrive late (SIU investigations take months)
- Historically missed fraud silently poisons the negatives
- Standard binary cross-entropy trains the model to replicate past blind spots

### Solutions in Pipeline

| Technique | Purpose |
|---|---|
| **PU Learning** (Positive-Unlabeled) | Treats negatives as "unlabeled," not confirmed clean |
| **Confident Learning** (cleanlab) | Identifies likely mislabeled examples via out-of-fold probability calibration |
| **Label smoothing** | Soft targets reduce overconfidence on noisy negatives |
| **Delayed label pipeline** | Re-trains on confirmed fraud labels post-SIU closure; models updated on rolling 90/180-day confirmed windows |
| **Semi-supervised augmentation** | Propagates fraud signal through graph neighborhoods |

---

## 6. Real-Time Decision Orchestration

> Moving beyond binary predict → score to **risk-tiered adaptive action routing**.

```
Score Tier    │ Threshold │ Action
──────────────┼───────────┼──────────────────────────────────────────────
Low Risk      │  < 0.25   │ Straight-through processing (STP)
Medium Risk   │  0.25–0.6 │ Request additional evidence (photos, EUO)
High Risk     │  0.6–0.85 │ SIU referral + payment delay notification
Extreme Risk  │  > 0.85   │ Payment hold + mandatory manual investigation
```

**Orchestration engine applies:**
- Business rules (policy exclusions, coverage verification)
- State-specific regulatory constraints (state DOI compliance)
- Explainability thresholds (adverse action triggers documentation)
- Feedback override tracking (investigator agrees/disagrees with tier)

**Adaptive friction principle:**
Legitimate claimants should experience minimal delay. Friction scales with risk
score and claim complexity, not uniformly applied.

---

## 7. Inference Architecture: Sync vs Async

> **Entity resolution is a pre-computation step — it is not part of the sync or
> async inference path.** VIN decode, address normalization, and person dedup are
> completed offline in the `entities/` layer before any inference runs. Resolved
> entities are pre-loaded into Neo4j and the feature store. At FNOL time, the sync
> path queries pre-resolved entities — it does not perform entity resolution within
> the latency budget.

### Synchronous Path (<100ms) — Blocks FNOL submission

| Component | Latency Budget |
|---|---|
| Online feature retrieval (Redis) | ~5ms |
| Device reputation lookup | ~10ms |
| Graph neighborhood query (1–2 hops) | ~20ms |
| Tabular XGBoost scoring | ~5ms |
| Lightweight image hash check | ~15ms |
| Decision orchestration + routing | ~10ms |
| **Total** | **~65ms** |

### Asynchronous Path (seconds–minutes) — Post-submission enrichment

- Deep ViT image analysis (damage estimation, tampering)
- Full graph traversal (3+ hop ring detection)
- LLM narrative reconciliation (statement consistency)
- Third-party data enrichment (MVR, credit)
- Cross-carrier fraud intelligence lookup

> **Why this matters:** Async enrichment results update the claim risk score in
> background, triggering escalation if the async score materially exceeds the sync
> score. This avoids blocking customer workflows while preserving investigative depth.

---

## 8. Adversarial Fraud Simulation

> Fraud is adversarial. Models must be stress-tested against tactics that haven't
> happened yet.

### Red-Team Pipeline
- **Synthetic identity generation**: LLM-generated claimant profiles with realistic
  but fabricated histories
- **Image spoofing simulation**: GAN-generated damage photos, copy-paste splicing,
  metadata stripping
- **Narrative adversarial examples**: LLM rewrites of known fraud narratives to
  evade NLP detection
- **Coordinated submission simulation**: synthetic fraud rings with graph structure
  mirroring known ring patterns

### Uses in Production
- Augment rare fraud class for training (GenAI oversampling vs. SMOTE)
- Evaluate model robustness before deployment
- Simulate emerging tactics (staged accidents, "sliding", vendor inflation)
- Champion-challenger adversarial benchmarking

---

## 9. AI Governance & Explainability

> For an ML audience: explainability is not just regulatory theater — it feeds the
> HITL loop and catches model failure modes.

### Per-Decision Explainability
- **SHAP values** on every fraud score: top-5 contributing features surfaced to
  investigators
- **Counterfactual explanations**: "score would drop from 0.82 → 0.31 if telematics
  data were consistent"
- **Graph path explanation**: "flagged because shares address with 3 confirmed fraud
  entities"
- Adverse action documentation auto-generated for regulatory compliance

### Governance Controls
- Bias monitoring: score distribution audited across protected class proxies
  (zip code, vehicle age as income proxies)
- Model lineage: every prediction tied to model version, feature snapshot,
  training cohort
- PII minimization: raw PII not stored in feature store; pseudonymized IDs
- Full audit log: every score, every decision, every investigator override
- Human override capability: investigators can contest and re-route; override
  rate tracked as model health signal

---

## 10. Data Reliability Controls

> Feature pipeline failures are silent killers. A missing telematics feed looks
> like "no fraud signal," not "broken pipeline."

### Monitoring Layer

| Check | Trigger |
|---|---|
| **Null rate spike** | Feature missing > baseline threshold |
| **Schema drift** | Unexpected column types or new categories |
| **Freshness SLA** | Feature timestamp stale beyond defined window |
| **Training-serving skew** | Distribution shift between offline training and online inference features |
| **Delayed event ingestion** | Telematics / third-party feed latency spike |
| **Entity resolution failure** | VIN decode miss rate spike, address normalization failure rate above threshold |
| **Entity dedup collision** | Unexpected merge of distinct persons or addresses — monitor dedup collision rate per batch |

**Fallback policy:**
If critical features are unavailable, the system automatically routes to a degraded
scoring policy (tabular-only) with a flag on the decision record. Investigators are
notified of degraded confidence. Entity resolution failures do not affect the sync
path directly — resolved entities are pre-loaded — but will cause stale graph
features on the next retraining cycle if undetected.

---

## 11. MLOps & Champion-Challenger

### Model Lifecycle
```
Training (offline)
    │  Feature Store snapshot (versioned)
    │  PU-corrected labels (90/180d confirmed window)
    │  Adversarial augmentation
    ▼
Validation
    │  Holdout AUC, Precision@K, Expected Cost
    │  PSI / CSI vs production population
    │  Adversarial robustness benchmarks
    ▼
Champion-Challenger Deployment
    │  5–10% traffic to challenger
    │  Business KPI gating (not just AUC)
    │  Auto-promote on sustained lift; auto-rollback on degradation
    ▼
Production Monitoring
    │  PSI drift alerts (feature + score distribution)
    │  SHAP drift: which features' importance is shifting?
    │  Label feedback latency tracking
    ▼
Retraining Triggers
       Scheduled (monthly)  +  Drift-triggered (PSI > threshold)
```

### What "Champion-Challenger" Actually Gates
- Not just AUC: gates on **fraud dollars captured per 1,000 claims scored**
- False positive rate on fast-track STP claims (customer friction metric)
- Investigator agreement rate on HITL escalations

---

## 12. Investigator Copilot (HITL)

> Not "AI replaces investigators." AI makes investigators 3x faster with better
> hit rates.

### Copilot Capabilities
- Auto-generated case summary: claim narrative, anomaly flags, network context,
  prior claim history
- Graph visualization: interactive fraud ring explorer with highlighted risk paths
- Evidence checklist: dynamically generated based on fraud type and score drivers
- Document verification: automated check of submitted photos against policy records
- Suggested next actions: based on similar historical cases that led to confirmed
  fraud

### Feedback Loop
```
Investigator Decision
    ├── Confirms fraud    → positive label reinforcement
    ├── Clears claim      → negative label (with confidence weight)
    └── Escalates further → pending label, re-scored on new evidence
         │
         ▼
    Label Store → Triggers retraining eligibility check
```

Override patterns are analyzed to detect systematic model blind spots.

---

## 13. Business KPIs

> For an ML audience: these are the optimization targets that translate model
> metrics into business reality.

| Metric | Target Direction | Proxy ML Metric |
|---|---|---|
| Fraud dollars prevented per 1,000 claims | ↑ | Expected cost reduction |
| False positive rate on STP claims | ↓ | Precision at low-risk threshold |
| SIU referral-to-confirmation rate | ↑ | Precision at high-risk threshold |
| Mean time from FNOL to SIU referral | ↓ | Sync path latency |
| Investigator case closure time | ↓ | Copilot adoption + accuracy |
| Model AUC degradation rate | ↓ | Drift monitoring |

> **Primary optimization target:** maximize fraud dollars prevented while minimizing
> false positive rate on legitimate claimants. Pure AUC optimization is insufficient
> — it ignores claim severity distribution.

---

## 14. Production vs Emerging Capabilities

### ✅ Production Today
- Tabular ML (XGBoost, LightGBM) with cost-sensitive learning
- Entity resolution (VIN decode, address normalization, person dedup)
- Graph analytics (Neo4j, fraud ring detection)
- Device fingerprinting and geolocation validation
- Telematics crash event validation
- Image tampering detection (metadata + pixel-level)
- NLP narrative consistency analysis
- SHAP-based explainability
- Champion-challenger deployment
- PSI/CSI drift monitoring
- Feature Store (online + offline)
- HITL investigator feedback loop

### 🔬 Emerging / Experimental
- Graph Neural Networks at scale (GraphSAGE inference latency challenges)
- LLM-powered narrative reconciliation in sync path
- GenAI synthetic fraud augmentation (red-team pipeline)
- Cross-carrier federated fraud intelligence network
- Physics-based damage validation (vehicle deformation modeling)
- Fully autonomous SIU triage agents

---

## Architecture Summary

| Dimension | Design Choice |
|---|---|
| **Framing** | Adversarial risk orchestration, not binary classification |
| **Entity foundation** | Independent entity resolution layer before feature store and graph build |
| **Scoring** | Multi-model ensemble + stacking meta-learner |
| **Label strategy** | PU Learning + Confident Learning + delayed confirmed labels |
| **Inference** | Hybrid sync (<100ms) + async (deep enrichment) |
| **Decisions** | Risk-tiered adaptive routing, not binary approve/deny |
| **Explainability** | SHAP per-decision + counterfactuals + graph path explanation |
| **Resilience** | Fallback scoring policy on feature pipeline failure |
| **Learning** | Continuous retraining from HITL feedback + drift triggers |
| **Adversarial** | Red-team pipeline + GenAI synthetic fraud simulation |

---

*Architecture version: 2026-Q2 | Audience: ML Practitioners | Format: Meetup Demo*
