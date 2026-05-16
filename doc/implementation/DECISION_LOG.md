# Decision Log
### Smart Oak Insurance — AI Platform
> Living document. Update immediately when a decision is made. Decisions without rationale become mysteries.

---

## How to Use This Document

Each entry captures:
- **Decision** — what was decided
- **Rationale** — why, including the alternative considered
- **Trade-off** — what was accepted or rejected
- **Date + context** — when and what triggered it

---

## DEC-001 — Telematics nulls: null not default

**Date:** 2026-Q2
**Decision:** Telematics features (`telematics_distraction_score`, `telematics_hard_brake_rate`, `telematics_crash_match`, `commute_entropy`) are set to `null` for the ~60% of users without devices. No default or neutral value is imputed.

**Rationale:**
XGBoost's native missing-value handling learns the actual risk distribution for non-telematics users from training data. Non-telematics users skew slightly higher risk on average due to adverse selection — safer drivers are more likely to opt in to telematics programs. Imputing a neutral value (e.g. `distraction_score=0.5`) masks this learned signal and produces a systematically miscalibrated model for a majority of your user base.

**Alternative considered:** Mean imputation or a neutral midpoint value.

**Trade-off accepted:** Model behavior for the 60% null cohort is less interpretable to non-technical stakeholders ("why is the score higher when there's no data?"). Mitigated by SHAP explanation: `telematics_available=False` surfaces as a positive fraud/risk contributor in investigator copilot output.

**Applies to:** `feature_definitions.py` — `build_telematics_features()`, both platforms.

---

## DEC-002 — Credit score nulls: null not default (regulatory)

**Date:** 2026-Q2
**Decision:** `credit_score` is set to `null` for policies in CA, MA, MI, HI. It is never imputed with any value, including a state average or national average.

**Rationale:**
Imputing any value — even a neutral one — risks acting as a credit proxy in states where credit-based insurance scoring is prohibited by statute. Regulators in these states can interpret a model that produces different outputs based on a "synthetic" credit value as a de facto credit score violation. XGBoost's null path is the only safe route.

**Alternative considered:** Setting a fixed neutral value (e.g. national median score of 703).

**Trade-off accepted:** Slightly reduced model accuracy in restricted states vs. non-restricted states. Accepted — regulatory compliance is non-negotiable per `ai_ml_architect.agent.md` core principles.

**Applies to:** `feature_definitions.py` — `apply_state_regulatory_mask()`, underwriting platform.

**States affected:** CA, MA, MI, HI (see `CREDIT_RESTRICTED_STATES` in `config.py`).

---

## DEC-003 — Telematics trio convention

**Date:** 2026-Q2
**Decision:** Every nullable telematics signal is accompanied by two derived features: an availability flag (`telematics_available: bool`) and, for the claims platform, an enrolled-but-missing fraud signal (`telematics_enrolled_but_missing: bool`).

**Rationale:**
A single nullable float cannot express three distinct states:
1. User has a device and the signal is populated
2. User has no device (opted out or not enrolled)
3. User is enrolled in a telematics program but the feed is absent at claim time

State 3 is a meaningful fraud signal — organized rings frequently suppress telematics data. Without the explicit `telematics_enrolled_but_missing` feature, this signal is indistinguishable from state 2 in the model.

**Alternative considered:** A single categorical feature (`none` / `available` / `enrolled_missing`). Rejected because XGBoost handles boolean flags better than categorical encoding for this pattern, and the trio maps cleanly to the online serving path.

**Trade-off accepted:** Adds 2 derived features per telematics signal. Accepted — derivation is cheap and the signals are high-value.

**Convention:** Apply this trio pattern to any future nullable signal group added to the feature store.

---

## DEC-004 — feature_definitions.py as Layer 0

**Date:** 2026-Q2
**Decision:** `features/feature_definitions.py` is built before archetypes. Archetypes import feature names from `feature_definitions.py`. `feature_definitions.py` has no imports from archetypes or generator.

**Rationale:**
Training-serving skew — where the offline training pipeline and the online serving pipeline compute features differently — is the most common silent failure mode in production ML. The root cause is almost always feature names or logic defined in two places. Making `feature_definitions.py` the dependency root, not an output of data generation, forces both pipelines to share one implementation from day one.

**Alternative considered:** Define feature names in archetypes and import into `feature_definitions.py`. Rejected — this inverts the dependency and makes the data generation layer the source of truth for production serving logic.

**Trade-off accepted:** `feature_definitions.py` must be written before any data generation work begins. Small upfront cost that prevents a large debugging cost later.

---

## DEC-005 — Graph features as second-pass enrichment

**Date:** 2026-Q2
**Decision:** Graph features (`graph_hop_distance`, `shared_attribute_count`, `attorney_centrality_score`) are computed as a second pass after the offline pipeline runs — not baked into the generator.

**Rationale:**
At inference time (online serving), graph features are queried live from Neo4j. If graph features are pre-joined in the generator, the offline training pipeline and the online serving path use different computation logic — a form of training-serving skew. Keeping graph features as a separate enrichment step that queries Neo4j mirrors the production inference path exactly.

**Alternative considered:** Pre-computing and joining graph features inside `generator.py`. Rejected — breaks online/offline parity.

**Trade-off accepted:** Graph enrichment requires Neo4j to be loaded before `offline_pipeline.py` can produce a complete feature set. This adds a step to the local setup sequence but reflects the true production dependency.

---

## DEC-006 — 20K records, 20 features per platform

**Date:** 2026-Q2
**Decision:** Experiment dataset is 20K quotes + 20K claims with 20 features each (not 10K/10 as initially considered).

**Rationale:**
10 features was too few because several signals come in natural pairs or trios (e.g. `telematics_available` + `telematics_distraction_score` + `telematics_enrolled_but_missing`). Splitting a logical signal group across feature count limits produces a dataset that validates the archetype logic but not the multi-signal architecture. 20 features covers all four signal layers (tabular, telematics, graph, NLP/device) with enough features per layer to test the ensemble.

10K records was also borderline for the Gamma severity model — at ~8% claim rate that produces ~800 severity training records, which is thin. 20K produces ~1,600, sufficient for Gamma regression with a 60/20/20 split.

**Alternative considered:** Start with 10K/10 and expand. Rejected — rebuilding archetypes mid-experiment is more costly than defining them correctly upfront.

**Trade-off accepted:** Slightly more upfront work in archetype definitions. Accepted — total dataset is still ~10MB Parquet, trivially fits local Docker.

---

## DEC-007 — Fraud rate 33% in synthetic data (vs ~15% production)

**Date:** 2026-Q2
**Decision:** Synthetic claims dataset targets ~33% fraud rate across 10 archetypes, not the ~15% production rate.

**Rationale:**
The synthetic dataset is used for model development and pipeline validation, not for calibrating production thresholds. A 33% fraud rate provides sufficient positive class examples (~6,600 fraud records across 10 archetypes, ~660 per archetype) for XGBoost to learn meaningful patterns per archetype. At 15% fraud rate with 20K records, some archetypes would have fewer than 100 examples, which is too few to learn archetype-specific feature patterns.

Production class imbalance is corrected in the model via `scale_pos_weight` in XGBoost, not by adjusting the training dataset ratio.

**Alternative considered:** Match production ~15% fraud rate and oversample with SMOTE. Rejected — SMOTE creates synthetic records that may not reflect realistic feature distributions. Cost-sensitive weighting on real records is preferred (per `Risk_Scoring_Architecture.md`).

**Trade-off accepted:** Model thresholds calibrated on synthetic data will not match production thresholds. Accepted — threshold calibration is done on a held-out production sample, not on synthetic data.

---

## DEC-008 — No PostgreSQL

**Date:** 2026-Q2
**Decision:** PostgreSQL is not included in the platform stack.

**Rationale:**
Every data workload maps to a more appropriate technology:
- Online feature serving → Redis
- Offline training data → S3 + Parquet
- Graph relationships → Neo4j / Neptune
- Audit trail → JSON files on S3
- Model metadata → MLflow / SageMaker

The only scenario where Postgres would add value is investigator workflow state (case assignments, SIU queue, override history). If this need emerges, DynamoDB or Aurora Serverless is preferred over Postgres for AWS-native deployment compatibility.

**Alternative considered:** Postgres for operational claim/policy data. Rejected — this data lives in source systems (policy admin system, claims management system). The platform reads from those systems via API, not by owning the data.

**Trade-off accepted:** If a legacy policy system is Postgres-backed, a read-only connection will be needed for FNOL Agent policy lookup. This is a source system integration, not a platform database to own.

---

## DEC-009 — Pydantic models over standalone JSON Schema files

**Date:** 2026-Q2
**Decision:** Input validation schemas (`claim.schema`, `violation.schema`, etc.) are defined as Pydantic models in `api/schemas.py`, not as standalone `.json` schema files.

**Rationale:**
Pydantic models provide runtime validation, automatic serialization, and OpenAPI documentation generation for free. Maintaining both Pydantic models and JSON Schema files creates two sources of truth that drift apart. JSON Schema from Pydantic can be generated on demand via `model.schema_json()` when needed for external partner contracts.

**Alternative considered:** Standalone `claim.schema.json`, `violation.schema.json` etc. maintained manually. Rejected — dual maintenance is error-prone and adds no value over Pydantic-generated schema.

**Trade-off accepted:** External partners who need JSON Schema for their own validation receive a generated file, not a hand-maintained one. Acceptable — generated schema is more accurate than manually maintained.

---

## DEC-010 — risk_score_at_issuance as fraud feature (shared spine)

**Date:** 2026-Q2
**Decision:** The underwriting risk score at quote time is stored in the feature store and re-used as a fraud scoring input feature, not discarded after policy issuance.

**Rationale:**
High-risk policies filed shortly after issuance is one of the strongest fraud signals in personal auto. A legitimate claimant who happened to be a high-risk driver is distinguishable from a fraudulent claimant by the combination of `risk_score_at_issuance` + `policy_inception_days` + graph signals. Without `risk_score_at_issuance`, the fraud model has no knowledge of the underwriting context.

This is the architectural "shared data spine" concept from `STRATEGY.md` — risk score is not a claims pipeline stage, it re-enters claims as a feature.

**Dependency created:** Quotes must be scored by the risk model before claims can be generated. In week 1, stub with a rule-based score if the hurdle model is not yet trained. Replace with real model output in week 2.

**Trade-off accepted:** Tight coupling between underwriting and claims pipelines via the feature store. Accepted — this coupling is the intended architecture, not a side effect.


---

**T-01 — Amend DEC-003 applies-to**

```markdown
# BEFORE
**Applies to:** `feature_definitions.py` — `build_telematics_features()`, both platforms.

# AFTER
**Applies to:** `entity_vehicle.py` — OBD-II device → VIN linkage and enrollment status resolution.
`feature_definitions.py` — `build_telematics_features()`, both platforms.
```

---

**T-02 — Amend DEC-004 rationale**

```markdown
# BEFORE
**Decision:** `features/feature_definitions.py` is built before archetypes. Archetypes import
feature names from `feature_definitions.py`. `feature_definitions.py` has no imports from
archetypes or generator.

# AFTER
**Decision:** `features/feature_definitions.py` is built before archetypes. Archetypes import
feature names from `feature_definitions.py`. `feature_definitions.py` has no imports from
archetypes, generator, or `entity_*.py` modules. Entity resolution outputs are passed as
arguments to feature computation functions — not imported as modules. This preserves
Layer 0 independence.
```

---

**T-03 — Add DEC-011**

```markdown
## DEC-011 — Entity resolution as independent pre-graph layer

**Date:** 2026-Q2
**Decision:** Vehicle, Person, Address, and Phone are resolved as independent entities
in `entities/` before `graph_builder.py` runs. `graph_builder.py` loads resolved entities
as nodes and edges only — it does not compute features.

**Rationale:**
Vehicle base attributes (VIN decode, MSRP, ADAS efficacy) have no graph dependency.
Placing their computation in `graph_builder.py` created a false dependency: vehicle
features could not be computed until the graph was built, even though the graph
contributes nothing to those features. Additionally, address and phone normalization
must happen before graph edges are created — a `shares_address` edge between two
unnormalized strings is unreliable and degrades Louvain community detection silently.

**Alternative considered:** Compute vehicle and address features inside `graph_builder.py`
as it loads data. Rejected — conflates entity persistence (loading nodes/edges) with
feature engineering, and creates an unnecessary graph build dependency for non-graph
features.

**Trade-off accepted:** Adds an explicit entity resolution step to the build order and
a new `entities/` directory to the repo. Accepted — the separation prevents a class of
silent failures where a graph build error takes down base feature computation, breaking
the tabular-only fallback scoring policy defined in `Fraud_Detection_Architecture.md`.

**Applies to:** `entities/` layer (new), `graph_builder.py` (scope reduced to node/edge
loading only), `offline_pipeline.py` (reads from `data/entities/` not raw generator
output directly).
```

---
