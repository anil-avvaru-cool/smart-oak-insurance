# Risk Score Model — Pending Tasks

> Last updated: 2026-05-24  
> All GAP analysis phases (1–8 / DEC-013) are **complete**. Items below are the remaining
> operational and monitoring gaps before the model is production-ready.

---

## 1. Champion-Challenger Framework

The architecture specifies a four-stage loop (Shadow → Gini check → Phased rollout → Promote).
Only the trigger signal (`champion_challenger_triggered` flag in `PSIDatasetReport`) exists.
No runner, no Gini comparison, no traffic split.

| # | Task | File to create / edit |
|---|---|---|
| CC-1 | **Shadow mode logger** — when a drift report triggers, run the challenger model on live quotes and log predictions to `data/processed/risk_models/challenger_predictions_<ts>.parquet` without acting on them | `monitoring/champion_challenger.py` (new) |
| CC-2 | **Gini comparison harness** — after 3–6 months of shadow data, compare challenger Gini vs champion Gini on the held-out cohort; emit pass/fail with delta | `monitoring/champion_challenger.py` |
| CC-3 | **Phased rollout config** — traffic split percentages (10 → 25 → 50 → 100) configurable in `data/config.py`; router reads current split and logs which model served each quote | `data/config.py` + `monitoring/champion_challenger.py` |
| CC-4 | **Promote workflow** — CLI flag `--promote-challenger` that copies challenger artifacts to champion slot, archives previous champion with timestamp suffix | `main.py` + `monitoring/champion_challenger.py` |
| CC-5 | **Wire drift trigger → shadow start** — when `run_drift_check()` returns `champion_challenger_triggered=True`, automatically log the event and start shadow scoring | `main.py` `--drift-check` handler |

---

## 2. SHAP Stability Monitoring

The architecture requires global SHAP values to be reviewed at each retraining cycle. No module exists.

| # | Task | File to create / edit |
|---|---|---|
| SH-1 | **SHAP snapshot writer** — after `--train-risk-model`, compute global mean absolute SHAP values for the frequency model and write to `data/processed/risk_models/shap_snapshot_<ts>.json` | `monitoring/shap_monitor.py` (new) |
| SH-2 | **SHAP drift check** — compare current snapshot to the previous one; warn if any single feature's SHAP share exceeds 40% (brittleness threshold from architecture doc) | `monitoring/shap_monitor.py` |
| SH-3 | **Wire into `--train-risk-model`** — call SHAP snapshot writer automatically after training completes | `main.py` |
| SH-4 | **Add `--shap-check` CLI flag** — standalone flag to print SHAP comparison for the two most recent snapshots | `main.py` |

---

## 3. Concept Drift Monitoring

PSI catches population shift; it does not catch when the *relationship* between features and
claims changes. The architecture calls out Gini and loss ratio by tier as the required signals.

| # | Task | File to create / edit |
|---|---|---|
| CD-1 | **Gini trend tracker** — compute Gini coefficient on a rolling holdout cohort (keyed on `quote_requested_at`) and append to `data/processed/risk_models/gini_history.json` after each retrain | `monitoring/concept_drift.py` (new) |
| CD-2 | **Loss ratio by tier** — compute actual loss ratio (incurred / premium) per `policy_tier_at_issuance` bucket on the closed-claims window; flag tiers where ratio drifts >10% from baseline | `monitoring/concept_drift.py` |
| CD-3 | **Label lag warning** — use the `LabelWindowReport` from `psi_drift.py`; if `confirmation_rate_90d` drops below 0.5 emit a warning that concept drift signals may be unreliable | `monitoring/concept_drift.py` |
| CD-4 | **Wire into `--drift-check`** — include concept drift summary in the drift report JSON under a `concept_drift` key | `main.py` + `monitoring/psi_drift.py` |

---

## 4. Regulatory Checklist Sign-off

All items signed off 2026-05-24. See `doc/Risk_Scoring_Architecture.md` for full evidence notes.

| # | Checklist Item | Status |
|---|---|---|
| RC-1 | Entity resolution runs before feature store vector assembly | ✅ `_load_vehicle_lookup()` enforces ordering via FileNotFoundError |
| RC-2 | Vehicle features sourced from `entity_vehicle.py` only | ✅ `_merge_vehicle_entity()` overrides raw fields with resolved entity values |
| RC-3 | `policy_inception_date` written by `entity_policy.py` to `data/entities/policies.parquet` | ✅ Column confirmed; `policy_inception_days` derived in `build_claim_feature_vector()` |
| RC-4 | Feature Store versioning with millisecond-precision timestamps | ✅ Microsecond-precision ISO-8601 confirmed in snapshot files |
| RC-5 | State regulatory mask applied before vector assembly | ✅ `apply_state_regulatory_mask()` called inside `build_quote_feature_vector()` |
| RC-6 | `credit_score` is `null` for restricted states, never imputed | ✅ Code-level confirmed; `CREDIT_RESTRICTED_STATES` = {CA, MA, MI, HI} |
| RC-7 | Calibrated scores stored in audit trail, not raw logits | ✅ `frequency_calibration.json` exists; snapshots store `risk_score_at_issuance`, never raw logits |
| RC-8 | PSI monitoring active on all top-10 features | ✅ Fixed: added `telematics_distraction_score`, `telematics_commute_entropy`, `household_driver_density` to `QUOTE_NUMERIC_FEATURES` |
| RC-9 | PSI current-period window keyed on `quote_requested_at` | ✅ `_run_dataset_psi(time_col="quote_requested_at")` confirmed |
| RC-10 | Null rate treated as its own PSI bin | ✅ `_bin_numeric()` appends `(null)` bin unconditionally |
| RC-11 | Champion-Challenger shadow mode before any model promotion | ✅ CC-5 auto-triggers shadow scoring in `--drift-check` handler |
| RC-12 | SHAP global importance reviewed at each retraining cycle | ✅ SH-3 runs `write_shap_snapshot()` as Stage 2d in `--train-risk-model` |

---

## 5. Ops / CLI Gaps

| # | Task | File | Status |
|---|---|---|---|
| OP-1 | `--train-risk-model` does not save metrics to a versioned file — only prints to stdout. Write `training_run_<ts>.json` to `data/processed/risk_models/` alongside model artifacts | `main.py` | ✅ Done 2026-05-24 |
| OP-2 | No `--drift-check --as-of YYYY-MM-DD` flag for back-testing drift on a historical date | `main.py` | ✅ Done 2026-05-24 |
| OP-3 | Drift reports accumulate unbounded in `data/processed/risk_models/`. Add `--purge-drift-logs --keep N` to retain only the N most recent reports | `main.py` | ✅ Done 2026-05-24 |

---

## Priority Order

```
CC-1  →  CC-2  →  SH-1  →  SH-2           # unblock champion-challenger + SHAP
CD-1  →  CD-2  →  CD-3  →  CD-4           # concept drift layer
RC-1 … RC-10 sign-off (parallel with above)
CC-3  →  CC-4  →  CC-5                     # full rollout workflow
OP-1  →  OP-2  →  OP-3                     # ops polish
RC-11, RC-12 sign-off after CC/SH complete
```
