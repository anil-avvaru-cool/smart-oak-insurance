# Gap Analysis — Post-DEC-013 Documentation Update
### Smart Oak Insurance — AI Platform
> Generated: 2026-05-22 | Based on: DECISION_LOG.md, DATA_GEN_GUIDE.md, FEATURE_STORE_GUIDE.md, Risk_Scoring_Architecture.md

The four docs are internally consistent and well-aligned with each other. The gaps are all between the docs and the **current codebase**. Every relevant source file has been read; what follows is what is missing, organized by implementation phase.

---

## Legend

- **[BREAK]** — code will produce wrong output silently if not fixed before the next run
- **[NEW]** — module or file that does not exist yet
- **[SCHEMA]** — dataclass / config field missing (blocks downstream implementation)
- **[DOC-DRIFT]** — doc and code disagree on a name or formula; one must win

---

## Phase 1 — Foundation (schema + config, no data yet)

These unblock everything else. Do these before writing a single line of generator logic.

| # | File | Gap | Type |
|---|---|---|---|
| 1.1 | `data/config.py` | `QUOTE_DATE_RANGE_DAYS`, `OPEN_CLAIM_RATE`, `UNCONFIRMED_FRAUD_RATE` are absent — generator cannot reference them | [SCHEMA] |
| 1.2 | `data/config.py` | PSI constants missing: `PSI_REFERENCE_VERSION`, `PSI_CURRENT_WINDOW_DAYS`, `PSI_MIN_RECORDS` | [SCHEMA] |
| 1.3 | `data/archetypes_claims.py` | `ClaimArchetype` missing three datetime distribution fields: `loss_event_hour_dist`, `claim_open_duration_days_dist`, `fraud_confirmation_lag_days_dist` | [SCHEMA] |
| 1.4 | `data/archetypes_claims.py` | `ClaimArchetype` missing `fraud_ring_id: str \| None` — graph_builder.py has no ring membership to load | [SCHEMA] |
| 1.5 | `data/archetypes_claims.py` | `ClaimArchetype` missing `telematics_enrolled_rate: float` — required to generate the `telematics_enrolled_but_missing` signal correctly | [SCHEMA] |
| 1.6 | `data/entities/entity_policy.py` | Writes column named `inception_date`; DEC-013 canonical name is `policy_inception_date`. Every downstream consumer (feature_definitions.py derivation, validator.py) must agree on one name | [DOC-DRIFT] |
| 1.7 | `features/feature_definitions.py` | Derivation formula comments absent. DEC-013 requires canonical `policy_inception_days` and `reporting_delay_days` formulas to live here as the authoritative reference | [SCHEMA] |

**Tasks:**
- [ ] Add `QUOTE_DATE_RANGE_DAYS = 365`, `OPEN_CLAIM_RATE = 0.15`, `UNCONFIRMED_FRAUD_RATE = 0.20` to `config.py`
- [ ] Add `PSI_REFERENCE_VERSION`, `PSI_CURRENT_WINDOW_DAYS = 14`, `PSI_MIN_RECORDS = 500` to `config.py`
- [ ] Add `loss_event_hour_dist`, `claim_open_duration_days_dist`, `fraud_confirmation_lag_days_dist` fields to `ClaimArchetype` dataclass and populate all 10 archetype instances
- [ ] Add `fraud_ring_id: str | None` and `telematics_enrolled_rate: float` to `ClaimArchetype` dataclass
- [ ] Rename `inception_date` → `policy_inception_date` in `entity_policy.py` output and any validators that read it
- [ ] Add canonical derivation doc-comments to `feature_definitions.py` for `policy_inception_days` and `reporting_delay_days`

---

## Phase 2 — Generator: emit source datetime columns (DEC-013 core)

The current `generator.py` has zero datetime columns. `reporting_delay_days` and
`policy_inception_days` are **independently sampled** from normal distributions — this violates
DEC-013's Option A contract that derived ints must be computed from the datetimes, not the
other way around.

| # | File | Gap | Type |
|---|---|---|---|
| 2.1 | `data/generator.py` | `generate_quotes()` emits no `quote_requested_at` — PSI has no time axis | [BREAK] |
| 2.2 | `data/generator.py` | `generate_quotes()` emits no `quote_completed_at` — source for `policy_inception_days` derivation is absent | [BREAK] |
| 2.3 | `data/generator.py` | `generate_claims()` emits no `loss_event_datetime` — source for `reporting_delay_days` derivation is absent | [BREAK] |
| 2.4 | `data/generator.py` | `generate_claims()` emits no `fnol_submitted_at` — PSI claims window anchor is missing | [BREAK] |
| 2.5 | `data/generator.py` | `generate_claims()` emits no `claim_closed_at` (nullable) | [NEW] |
| 2.6 | `data/generator.py` | `generate_claims()` emits no `fraud_confirmed_at` (nullable) — 90/180-day label window is unqueryable | [NEW] |
| 2.7 | `data/generator.py` | `reporting_delay_days` is sampled independently from `Normal(mean, std)` — must be derived from `(fnol_submitted_at − loss_event_datetime).days` per DEC-013 | [BREAK] |
| 2.8 | `data/generator.py` | `policy_inception_days` is sampled independently — must be derived from `(quote_completed_at.date − policy_inception_date).days` per DEC-013 | [BREAK] |

**Tasks:**
- [ ] Add date-spread generation block to `generate_quotes()` using `QUOTE_DATE_RANGE_DAYS` from `config.py`; emit `quote_requested_at` (uniform spread) and `quote_completed_at` (small processing offset, 1s–5min)
- [ ] In `generate_claims()`, sample `loss_event_datetime` from `archetype.loss_event_hour_dist` within the same date spread; derive `fnol_submitted_at = loss_event_datetime + Timedelta(days=reporting_delay_sample)`
- [ ] Derive `reporting_delay_days` from `(fnol_submitted_at − loss_event_datetime).dt.days` — remove the independent normal sample
- [ ] Generate `claim_closed_at` from `archetype.claim_open_duration_days_dist`; null at `OPEN_CLAIM_RATE` probability
- [ ] Generate `fraud_confirmed_at` from `archetype.fraud_confirmation_lag_days_dist`; null for all legitimate archetypes and at `UNCONFIRMED_FRAUD_RATE` probability for fraud archetypes
- [ ] Derive `policy_inception_days` from `entity_policy.py`'s `policy_inception_date` (stub at generation time and re-derive in `offline_pipeline.py` after entity resolution)

---

## Phase 3 — hurdle_model.py: write `quote_completed_at`

| # | File | Gap | Type |
|---|---|---|---|
| 3.1 | `underwriting/models/risk_scoring/hurdle_model.py` | After risk scoring, `quote_completed_at = now()` is never written back to `quotes.parquet`. Without it, `policy_inception_days` cannot be derived per DEC-013 | [NEW] |

**Task:**
- [ ] After `score()` / `predict()` completes, write `quote_completed_at = pd.Timestamp.now()` into `quotes.parquet` alongside `risk_score_at_issuance`

---

## Phase 4 — Validator: DEC-013 temporal integrity checks

All seven temporal checks specified in DEC-013 and DATA_GEN_GUIDE.md are absent. The current
validator only checks `reporting_delay_days >= 0` and `policy_inception_days >= 0` — weaker guards
that cannot catch datetime arithmetic rounding drift.

| # | File | Gap | Type |
|---|---|---|---|
| 4.1 | `data/validator.py` | `fnol after loss event` check missing | [NEW] |
| 4.2 | `data/validator.py` | `claim closed after fnol (where closed)` check missing | [NEW] |
| 4.3 | `data/validator.py` | `fraud confirmed after fnol (where confirmed)` check missing | [NEW] |
| 4.4 | `data/validator.py` | `reporting_delay_days consistent with datetimes` check missing — this is the critical drift guard | [NEW] |
| 4.5 | `data/validator.py` | `fraud_confirmed_at null for legitimate claims` check missing | [NEW] |
| 4.6 | `data/validator.py` | `quote_requested_at populated for all quotes` check missing | [NEW] |
| 4.7 | `data/validator.py` | `fnol_submitted_at populated for all claims` check missing | [NEW] |
| 4.8 | `data/validator.py` | `policy inception before quote_completed_at` check missing | [NEW] |

**Task:**
- [ ] Add `_check_temporal_integrity(quotes_df, claims_df)` function to `validator.py` with all eight checks, called from `validate_feature_correlations()`

---

## Phase 5 — Feature store snapshot: audit datetime fields

DEC-013 and FEATURE_STORE_GUIDE.md specify that `fnol_submitted_at`, `loss_event_datetime`,
and a `_datetime_note` block appear as **top-level envelope fields** in claim snapshots (not inside
`features`). These are audit fields for regulatory replay. They are currently absent from the
snapshot writer and from the snapshot envelope validator.

| # | File | Gap | Type |
|---|---|---|---|
| 5.1 | `features/offline_pipeline.py` | Claim snapshots don't emit `fnol_submitted_at` or `loss_event_datetime` at the envelope level | [NEW] |
| 5.2 | `features/offline_pipeline.py` | Quote snapshots don't emit `quote_requested_at` or `quote_completed_at` at the envelope level | [NEW] |
| 5.3 | `data/validator.py` → `_SNAPSHOT_ENVELOPE_KEYS` | Frozenset doesn't include `fnol_submitted_at`, `loss_event_datetime` for claims; `quote_requested_at`, `quote_completed_at` for quotes | [BREAK] |

**Tasks:**
- [ ] Add audit datetime fields to `build_claim_snapshot()` and `build_quote_snapshot()` in `offline_pipeline.py`
- [ ] Add `_datetime_note` block to snapshot JSON per FEATURE_STORE_GUIDE.md schema
- [ ] Update `_SNAPSHOT_ENVELOPE_KEYS` in `validator.py` to check for these fields

---

## Phase 6 — monitoring/psi_drift.py (new module)

Referenced in three docs (DECISION_LOG DEC-013, FEATURE_STORE_GUIDE, Risk_Scoring_Architecture).
Does not exist.

| # | File | Gap | Type |
|---|---|---|---|
| 6.1 | `monitoring/psi_drift.py` | Module does not exist. PSI monitoring has no time axis without it — DEC-013 rationale item #1 | [NEW] |

**Tasks:**
- [ ] Create `monitoring/` package with `__init__.py`
- [ ] Implement `psi_drift.py` with a rolling 14-day current-period cohort keyed on `quote_requested_at` (quotes) and `fnol_submitted_at` (claims), read directly from `data/raw/` — not from the feature store
- [ ] Implement PSI computation per feature with null-as-own-bin treatment (do not filter nulls before PSI)
- [ ] Implement `fraud_confirmed_at` window query for the 90/180-day concept drift label window
- [ ] Wire `PSI_REFERENCE_VERSION`, `PSI_CURRENT_WINDOW_DAYS`, `PSI_MIN_RECORDS` from `config.py`
- [ ] Add `tests/unit/test_psi_drift.py`

---

## Phase 7 — DEC-005 graph pre-baking (architectural debt)

Lower urgency; gated on Neo4j infrastructure being healthy.

| # | File | Gap | Type |
|---|---|---|---|
| 7.1 | `data/generator.py` | `graph_hop_distance`, `shared_attribute_count`, `attorney_centrality_score` are pre-sampled from archetype distributions — violates DEC-005 second-pass enrichment contract and creates offline/online skew | [BREAK] |
| 7.2 | `data/archetypes_claims.py` | `graph_hop_distance_lambda` drives the pre-baked sampling; it should drive `fraud_ring_id` assignment instead | [SCHEMA] |
| 7.3 | `data/graph_features.py` | Presumably overwrites pre-baked values on `--compute-graph-features`, creating an implicit ordering dependency that is not enforced | [BREAK] |

**Recommended approach:** In `generator.py`, emit `graph_hop_distance = 999` (sentinel), `shared_attribute_count = 0`, `attorney_centrality_score = 0.0` as stubs. Let `graph_features.py` overwrite them after Neo4j is built. Remove `graph_hop_distance_lambda` from `ClaimArchetype`; replace with `fraud_ring_id`.

**Tasks:**
- [ ] Remove `graph_hop_distance_lambda`, `shared_attribute_count_mean`, `attorney_centrality_mean` from `ClaimArchetype` dataclass
- [ ] Emit sentinel stubs for all three graph features in `generator.py`
- [ ] Add `fraud_ring_id` instances to all 10 archetype definitions (fraud archetypes share ring IDs; legitimate archetypes get `None`)
- [ ] Verify `graph_features.py` overwrites stubs correctly and validator sentinel-rate check still fires appropriately

---

## Phase 8 — Tests

| # | Gap | Type |
|---|---|---|
| 8.1 | No tests for DEC-013 temporal ordering or datetime arithmetic consistency | [NEW] |
| 8.2 | No tests for `psi_drift.py` (PSI value computation, window slicing, null-bin handling) | [NEW] |
| 8.3 | No tests verifying snapshot audit datetime fields are written and validated by envelope checker | [NEW] |
| 8.4 | `telematics_enrolled_rate` → `telematics_enrolled_but_missing` signal path has no test coverage | [NEW] |

---

## Execution order summary

| Phase | Effort | Blocks |
|---|---|---|
| **1 — Foundation (config + schema)** | Small | All other phases |
| **2 — Generator datetimes** | Medium | Validator, PSI, snapshot audit |
| **3 — hurdle_model quote_completed_at** | Tiny | `policy_inception_days` derivation |
| **4 — Validator temporal checks** | Small | Catching regressions in Phase 2 |
| **5 — Snapshot audit datetimes** | Small | Regulatory replay, snapshot validator |
| **6 — monitoring/psi_drift.py** | Large | PSI → champion-challenger loop |
| **7 — DEC-005 graph pre-baking** | Medium | Online/offline parity (needs Neo4j) |
| **8 — Tests** | Medium | Regression confidence |

Phases 1 → 2 → 3 → 4 are tightly coupled and should run as one continuous sprint — none
produces a working artifact without the others. Phases 5 and 6 can be parallelized once Phase 2
is stable and the datetime columns exist in raw parquet. Phase 7 is gated on Neo4j
infrastructure being healthy.
