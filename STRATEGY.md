# Smart Oak Insurance — AI Platform Strategy
### Intelligent Claims Lifecycle Platform | 2026

---

## Vision

> **"End-to-End Intelligent Insurance Operations Platform"**
> A unified AI platform spanning the full insurance lifecycle —
> from quote generation through claims settlement — through risk scoring,
> real-time fraud detection, and autonomous claims automation.

---

## Platform Narrative

Two distinct lifecycles. One shared data spine.

```
UNDERWRITING PLATFORM                    CLAIMS PLATFORM
─────────────────────                    ───────────────
Customer requests quote                  Accident happens
          ↓                                     ↓
    Risk Scoring                           FNOL Agent
  (P(claim), E[cost])                   (intake, triage)
          ↓                                     ↓
 Quote Generation Agent                  Fraud Scoring
          ↓                                     ↓
   Policy Issuance                      Claims Automation
                                               ↓
                         ┌─────────────────────┤
                         ↓                     ↓
                    STP (auto)         Investigator Copilot
                                          (HITL)

          └──────────── shared feature store ────────────┘
                  risk_score_at_issuance flows into
                  fraud scoring as a feature input
```

**Key insight:** Risk score is not a claims pipeline stage. It re-enters claims as a feature — high-risk policies filed shortly after issuance is one of the strongest fraud signals.

One cohesive platform. Not isolated models.

---

## Goals

- **C-Level Executive Appeal** — business outcomes, dollar impact, operational efficiency
- **ML Practitioner Credibility** — production-grade architecture, not demo toys
- **Meetup Demo** — compelling narrative with live scoring walkthrough
- **Technical Articles** — practitioner-grade content on LinkedIn + Towards Data Science

---

## Core Principle

> Modern insurance fraud detection is not a supervised classification problem.
> It is an **adversarial risk orchestration system** combining anomaly detection,
> graph intelligence, behavioral analytics, and human investigation workflows.

---

## Iteration Strategy

Planning is not linear. This platform is built through iterative feedback loops.

```
Build small slice
      ↓
Self-review: does this connect to the platform story?
      ↓
Multi-LLM review: technical correctness + narrative gaps
      ↓
Document lesson learned immediately
      ↓
Adjust next slice
      ↓
Repeat
```

### Multi-LLM Review Panel

| LLM | Role |
|---|---|
| **Claude** | Architecture review, narrative coherence, strategy |
| **GPT-4o** | Code generation, debugging, quick iterations |
| **Gemini** | Large context doc review, Google ecosystem |
| **Perplexity** | Industry benchmarks, real-world validation |

**Rule:** Max 2 LLM review cycles per component before moving forward. Working and documented beats perfect and unfinished.

---

## Two-Month Roadmap

```
Week 1-2   Synthetic data generator + shared feature store
             (separate archetypes for underwriting + claims)
Week 3     Fraud scoring (end-to-end, claims platform)
Week 4     Risk scoring + Quote Generation Agent (underwriting platform)
Week 5     FNOL Agent + Document Verification Agent (claims platform)
Week 6     Claims Automation + Subrogation Agent
Week 7     FastAPI + AWS deployment + architecture diagram
Week 8     Articles + meetup deck + repo polish + LinkedIn
```

---

## Phase 1 — Data Foundation (Week 1-2)

### Synthetic Data Strategy

**Lesson learned:** Generate data top-down from realistic archetypes, not bottom-up from random values.

Define 10 claim archetypes first:

| Archetype | Fraud Type | Key Signals |
|---|---|---|
| Staged rear-end collision | Organized ring | Multiple claimants, same attorney |
| Soft tissue exaggeration | Opportunistic | Delayed reporting, no ER visit |
| Legitimate fender-bender | None | Consistent narrative, telematics match |
| VIN cloning | Identity | Policy inception < 30 days |
| Inflated repair estimate | Vendor fraud | Body shop network centrality |
| Phantom passenger | Opportunistic | No witnesses, late-night claim |
| Coordinated fraud ring | Organized | Graph cluster, shared addresses |
| Medical billing inflation | Provider fraud | Billing code anomalies |
| Total loss misrepresentation | Opportunistic | Pre-existing damage photos |
| Legitimate major accident | None | Police report, ER visit, telematics |

Generate synthetic data from these archetypes using `faker` + domain-specific distributions. One shared dataset feeds ALL models.

---

## Phase 2 — Core Models (Week 3-5)

### Priority Order

```
UNDERWRITING PLATFORM
1. Risk scoring             ← Hurdle model (frequency + severity)
2. Quote Generation Agent   ← uses risk score for pricing

CLAIMS PLATFORM
3. Fraud scoring            ← most demo-able, highest business visibility
4. FNOL Agent               ← ties claims narrative together
5. Claims automation        ← straight-through processing for low-risk
```

**Note:** Risk score at policy issuance is stored in feature store and reused as a fraud scoring input feature — not a pipeline dependency.

### Fraud Scoring Architecture

```
Tabular XGBoost          → P(fraud | structured features)
Graph Neural Network     → P(fraud | network context)
NLP Transformer          → P(fraud | narrative inconsistency)
Vision Model (ViT)       → P(fraud | image tampering)
Anomaly Detector         → Isolation Forest (unsupervised)
          ↓
  Stacking Meta-Learner
          ↓
   Fraud Score [0.0–1.0] + SHAP Explanation
```

### Risk Scoring Architecture

```
Hurdle Model:
  Stage 1 (Frequency):  XGBoost classifier → P(claim)
  Stage 2 (Severity):   XGBoost Gamma      → E[cost | claim]
          ↓
  Combined Risk Score + Confidence Interval
```

### Agents

**Underwriting Platform**

| Agent | Business Value |
|---|---|
| **Quote Generation Agent** | Automated pricing using risk score, policy rules, market rates |

**Claims Platform**

| Agent | Business Value |
|---|---|
| **FNOL Agent** | Automated first notice intake, policy validation, initial triage |
| **Document Verification Agent** | Validates repair estimates, medical bills, police reports |
| **Subrogation Agent** | Identifies third-party recovery opportunities automatically |
| **Investigator Copilot** | HITL support — case summaries, evidence checklists, graph visualization |

---

## Phase 3 — Platform Layer (Week 6-7)

### Feature Store

```
Offline (Databricks/Spark)         Online (Redis)
──────────────────────────         ──────────────
Nightly batch computation    →     Pre-computed features
                                   served in <10ms

Shared feature_definitions.py ← single source of truth
(same functions, offline + online — prevents training-serving skew)
```

### Decision Orchestration Layer

```
Score Tier    │ Threshold │ Action
──────────────┼───────────┼──────────────────────────────
Low Risk      │  < 0.25   │ Straight-through processing
Medium Risk   │  0.25–0.6 │ Request additional evidence
High Risk     │  0.6–0.85 │ SIU referral
Extreme Risk  │  > 0.85   │ Payment hold + manual review
```

### Inference: Sync vs Async

```
Synchronous (<100ms)              Asynchronous (seconds–minutes)
────────────────────              ──────────────────────────────
Tabular scoring                   Deep ViT image analysis
Device reputation                 Full graph traversal
Graph 1-2 hop lookup              LLM narrative reconciliation
Lightweight image hash            Third-party enrichment
Decision routing                  Cross-carrier intelligence
```

---

## Phase 4 — Publish & Demo (Week 8)

### Article Strategy

| Article | Platform | Angle |
|---|---|---|
| "Why fraud detection is not a classification problem" | Towards Data Science | Contrarian, technically deep |
| "How we built a <100ms fraud scorer on AWS" | Towards Data Science | Concrete, deployable |
| "What C-suite gets wrong about claims AI" | LinkedIn | Executive reach |

### Meetup Demo Flow (20-25 min)

```
1. Open with adversarial framing (2 min)
2. Platform overview — one flow diagram (3 min)
3. Live fraud scoring walkthrough (5 min)
4. Sync vs async inference deep dive (5 min)
5. Decision orchestration layer (3 min)
6. Lessons learned + what's next (5 min)
7. Q&A
```

---

## Business KPIs

**Underwriting Platform**

| Metric | Target |
|---|---|
| Loss ratio improvement vs baseline | ↓ |
| Quote-to-bind rate on low-risk policies | ↑ |
| Risk score accuracy (actual vs predicted loss) | ↑ |

**Claims Platform**

| Metric | Target |
|---|---|
| Fraud dollars prevented per 1,000 claims | ↑ |
| False positive rate on legitimate claims | ↓ |
| SIU referral-to-confirmation rate | ↑ |
| Mean time FNOL → SIU referral | ↓ |
| Investigator case closure time | ↓ |
| Straight-through processing rate | ↑ |

---

## Lessons Learned

| Problem | Fix Applied |
|---|---|
| Random synthetic data wasted time | Define archetypes first, generate from realistic distributions |
| Siloed models, no shared story | Single data spine + feature store from day one |
| Month spent on basics without clear output | Time-box each component, working beats perfect |
| RAG felt disconnected | Wire RAG into FNOL Agent as policy document retrieval |
| Linear planning didn't reflect reality | Iterative loops with multi-LLM review panel |

---

## Git Repository Structure

```
smart-oak-insurance/
│
├── README.md
├── STRATEGY.md                          ← this document
├── ARCHITECTURE.md                      ← technical architecture deep dive
├── LESSONS_LEARNED.md                   ← living document, updated each iteration
│
├── docs/
│   ├── architecture_diagram.png
│   ├── data_dictionary.md
│   ├── api_reference.md
│   ├── decision_thresholds.md
│   ├── runbooks/
│   │   ├── local_setup.md
│   │   ├── aws_deployment.md
│   │   └── model_retraining.md
│   └── articles/
│       ├── fraud_not_classification.md
│       ├── sub100ms_fraud_scorer.md
│       └── csuite_claims_ai.md
│
├── data/
│   ├── synthetic/
│   │   ├── generator.py                 ← archetype-based synthetic data
│   │   ├── archetypes_claims.py         ← 10 claim archetypes (fraud + legit)
│   │   ├── archetypes_underwriting.py   ← driver/vehicle risk profiles
│   │   └── validator.py                 ← schema + realism checks
│   ├── raw/                             ← gitignored
│   └── processed/                       ← gitignored
│
├── features/
│   ├── feature_definitions.py           ← single source of truth, no skew
│   ├── offline_pipeline.py              ← Spark batch computation
│   ├── online_serving.py                ← Redis read/write
│   └── feature_store_client.py          ← unified interface for both platforms
│
├── graph/
│   ├── neo4j_client.py
│   ├── graph_builder.py                 ← load data → Neo4j
│   ├── graph_features.py                ← fraud ring detection
│   └── graph_queries.py                 ← Cypher query library
│
├── underwriting/                        ← UNDERWRITING PLATFORM
│   ├── models/
│   │   ├── risk_scoring/
│   │   │   ├── train_frequency.py       ← Stage 1: P(claim)
│   │   │   ├── train_severity.py        ← Stage 2: E[cost | claim]
│   │   │   └── hurdle_model.py          ← combined risk score
│   │   └── shared/
│   │       ├── shap_explainer.py
│   │       └── champion_challenger.py
│   ├── agents/
│   │   └── quote_generation_agent/
│   │       ├── agent.py
│   │       ├── tools.py                 ← pricing rules, market rate lookup
│   │       └── rag_retriever.py         ← underwriting guidelines RAG
│   └── api/
│       └── routers/
│           ├── risk.py
│           └── quote.py
│
├── claims/                              ← CLAIMS PLATFORM
│   ├── models/
│   │   ├── fraud_scoring/
│   │   │   ├── train.py
│   │   │   ├── ensemble.py
│   │   │   ├── pu_learning.py           ← contaminated label handling
│   │   │   └── evaluate.py
│   │   └── shared/
│   │       ├── cost_sensitive.py        ← asymmetric loss functions
│   │       ├── shap_explainer.py
│   │       └── champion_challenger.py
│   ├── agents/
│   │   ├── fnol_agent/
│   │   │   ├── agent.py
│   │   │   ├── tools.py                 ← policy lookup, coverage check
│   │   │   └── rag_retriever.py         ← policy document RAG
│   │   ├── document_verification_agent/
│   │   │   ├── agent.py
│   │   │   └── tools.py
│   │   ├── subrogation_agent/
│   │   │   ├── agent.py
│   │   │   └── tools.py
│   │   └── investigator_copilot/
│   │       ├── agent.py
│   │       ├── case_summary.py
│   │       └── evidence_checklist.py
│   ├── scoring/
│   │   ├── sync_scorer.py               ← <100ms path
│   │   ├── async_scorer.py              ← deep enrichment
│   │   └── decision_engine.py           ← risk tier routing
│   └── api/
│       └── routers/
│           ├── fnol.py
│           ├── fraud.py
│           └── claims.py
│
├── api/                                 ← unified FastAPI entry point
│   ├── main.py
│   ├── schemas.py                       ← Pydantic request/response models
│   └── middleware.py                    ← auth, logging, latency tracking
│
├── monitoring/
│   ├── psi_drift.py
│   ├── shap_drift.py
│   ├── data_quality.py                  ← null rates, schema, freshness
│   └── fallback_policy.py               ← degraded scoring on pipeline failure
│
├── tests/
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_fraud_scoring.py
│   │   ├── test_risk_scoring.py
│   │   ├── test_decision_engine.py
│   │   └── test_agents.py
│   ├── integration/
│   │   ├── test_neo4j_connection.py
│   │   ├── test_redis_serving.py
│   │   └── test_api_endpoints.py
│   └── adversarial/
│       ├── test_fraud_robustness.py     ← synthetic fraud stress tests
│       └── test_edge_cases.py
│
├── deployment/
│   ├── aws/
│   │   ├── cdk/                         ← AWS CDK infrastructure as code
│   │   │   ├── app.py
│   │   │   ├── underwriting_stack.py
│   │   │   ├── claims_stack.py
│   │   │   ├── feature_store_stack.py
│   │   │   └── api_stack.py
│   │   ├── sagemaker/
│   │   │   ├── training_pipeline.py
│   │   │   └── endpoint_config.py
│   │   └── lambda/
│   │       ├── sync_scorer_handler.py
│   │       └── async_scorer_handler.py
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.scoring
│   │   └── docker-compose.local.yml     ← Neo4j + Redis + API locally
│   └── configs/
│       ├── local.yaml
│       ├── staging.yaml
│       └── production.yaml
│
├── notebooks/                           ← Optional exploration only, never production
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_prototyping.ipynb
│
├── config.py                            ← env vars, thresholds, tier cutoffs
├── requirements.txt
├── requirements-dev.txt
└── .github/
    └── workflows/
        ├── ci.yml                       ← tests on PR
        └── deploy.yml                   ← deploy to AWS on demand
```

---

## Local → AWS Migration Map

```
Local                        AWS
─────                        ───
Redis (Docker)          →    ElastiCache
Neo4j Aura Free         →    Neptune
CSV / Parquet files     →    S3 + Delta Lake
FastAPI local           →    Lambda + API Gateway
Manual model runs       →    SageMaker Pipelines
Pandas offline          →    Spark on Glue / EMR
Docker Compose          →    ECS Fargate
```

Config swap, not code rewrite — because scripts are modular from day one.

---

## Production vs Emerging Capabilities

### ✅ Production Today
- Tabular ML (XGBoost, LightGBM) with cost-sensitive learning
- Graph analytics + fraud ring detection
- Device fingerprinting + geolocation validation
- Telematics crash event validation
- Image tampering detection
- NLP narrative consistency analysis
- SHAP explainability per decision
- Champion-challenger deployment
- PSI/CSI drift monitoring
- HITL investigator feedback loop

### 🔬 Emerging / Experimental
- Graph Neural Networks at inference scale
- LLM narrative reconciliation in sync path
- GenAI synthetic fraud augmentation
- Physics-based damage validation
- Fully autonomous SIU triage agents

---

*Strategy version: 2026-Q2 | Company: Smart Oak Insurance | Repo: smart-oak-insurance*