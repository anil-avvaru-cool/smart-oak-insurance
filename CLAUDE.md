# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Refer @.github\agents\ai_ml_architect.agent.md

## Commands

**Setup:**
```bash
uv venv && source .venv/bin/activate
uv sync
cp example.env .env  # then edit .env
```

**Data pipeline (run in order):**
```bash
uv run -m main --generate-data          # generates data/raw/quotes.parquet + claims.parquet
uv run -m main --resolve-entities       # writes data/raw/entities/*.parquet
uv run -m main --validate-data          # validates all outputs

# Requires Neo4j running (see Docker section):
uv run -m main --build-graph
uv run -m main --compute-graph-features
```

**Tests:**
```bash
uv run pytest                           # all tests
uv run pytest tests/unit/test_feature_store.py  # single file
```

**Lint / format:**
```bash
uv run ruff check .
uv run black .
```

**Docker (for Neo4j + Redis locally):**
```bash
docker compose up -d neo4j redis
docker compose run --rm app python main.py --build-graph --validate-data
docker compose run --rm app python main.py --compute-graph-features --validate-data
export $(cat .env | xargs)   # expose env vars to host shell
docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "MATCH (n) RETURN count(n);"
```

## Architecture

This is a **synthetic insurance data platform** — no real customer data exists. Everything flows from archetype definitions through generation, entity resolution, graph construction, and into a feature store consumed by both an underwriting and a claims platform.

### Data pipeline order

```
archetypes_*.py  →  generator.py  →  quotes.parquet / claims.parquet
                                               ↓
                             data/entities/  (resolve-entities step)
                          vehicles / persons / addresses / phones / policies
                                               ↓
                                    graph_builder.py → Neo4j
                                               ↓
                                   graph_features.py → updates claims.parquet
                                               ↓
                              features/  (feature store — offline JSON + Redis)
```

### Three-layer dependency rule

1. **`features/feature_definitions.py` is Layer 0.** It has zero imports from `data/`. Archetypes reference feature names from it, not the reverse. Both offline and online serving use the same functions — this is the anti-skew contract.

2. **Entity resolution runs before graph building.** `data/entities/` resolves Vehicle (VIN, MSRP, ADAS), Person, Address, and Phone into canonical parquet files. `graph_builder.py` loads these as nodes/edges only — it computes no features.

3. **Graph features are second-pass enrichment.** `graph_features.py` queries Neo4j after the graph is built and writes `graph_hop_distance`, `shared_attribute_count`, and `attorney_centrality_score` back into `claims.parquet`. This mirrors the online inference path (live Neo4j query), preventing offline/online skew.

### Shared data spine

`risk_score_at_issuance` is computed during underwriting and stored in the feature store. It re-enters the claims platform as a fraud input feature — high-risk policy + short `policy_inception_days` is one of the strongest fraud signals. This cross-platform flow is intentional, not incidental.

### Neo4j graph model

Nodes: `Claim`, `Claimant`, `Attorney`, `Vehicle`, `Entity` (shared phone/address).  
Key relationships: `(Claimant)-[:FILED]->(Claim)`, `(Claimant)-[:REPRESENTED_BY]->(Attorney)`, `(Claim|Claimant)-[:SHARES {type}]->(Entity)`.  
Fraud ring detection relies on shared `Entity` nodes connecting otherwise unrelated claims.

### Feature store

`FeatureStoreClient` (`features/feature_store_client.py`) wraps two backends:
- **Offline**: JSON snapshots in a configurable directory (local dev) / S3 Parquet (production)
- **Online**: Redis (`OnlineFeatureStore` in `features/online_serving.py`)

`build_quote_snapshot` and `build_claim_snapshot` in `offline_pipeline.py` call the shared functions in `feature_definitions.py`.

## Key design decisions

**Telematics nulls:** Telematics features are `null` for non-enrolled users — never imputed. XGBoost's native null path learns the actual risk distribution. The trio convention always accompanies nullable signals: raw value + `telematics_available` (bool) + `telematics_enrolled_but_missing` (bool, fraud signal for suppressed feeds).

**Credit score nulls:** `credit_score` is `null` in CA, MA, MI, HI (`CREDIT_RESTRICTED_STATES` in `feature_definitions.py`). Never imputed — regulatory compliance, non-negotiable.

**Fraud rate in synthetic data:** ~33% fraud rate in generated data (vs ~15% production). This is intentional to provide enough per-archetype examples for XGBoost. Production class imbalance is corrected via `scale_pos_weight`, not by adjusting the dataset.

**Sentinel value 999:** `graph_hop_distance = 999` means no path found to any fraud seed claim within 6 hops. If >50% of records are 999 after `--compute-graph-features`, the graph likely has no fraud `Claim` nodes (check `is_fraud` column propagation in `graph_builder.py`).

**No PostgreSQL:** Online features → Redis. Offline training data → Parquet. Graphs → Neo4j. There is no relational DB in the stack.
