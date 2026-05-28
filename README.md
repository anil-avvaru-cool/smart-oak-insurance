# Smart Oak Insurance

Synthetic insurance risk-and-fraud platform for underwriting and claims modeling.

## Setup

**Local dev** (for tests and linting):
```bash
uv venv && source .venv/bin/activate
uv sync
cp example.env .env        # then edit .env
uv export --format requirements-txt --no-hashes --no-emit-workspace > requirements.txt
```

**Docker** (required for the full pipeline — Neo4j + Redis):
```bash
docker compose build --no-cache
docker compose up -d neo4j redis   # wait ~30 s for Neo4j to become healthy
```

## Data pipeline

```bash
# Wipe previous outputs (optional)
sudo rm -rf ./data/raw/* ./data/processed/*

docker compose run --rm app python main.py \
  --generate-data --resolve-entities \
  --build-graph --compute-graph-features \
  --run-offline-pipeline --validate-data
```

## Training

```bash
docker compose run --rm app python main.py --train-risk-model
docker compose run --rm app python main.py --calibrate-risk-model
```

## Champion-Challenger drift test

```bash
# 1. Seed challenger from current champion (one-time)
cp data/processed/risk_models/frequency_model.json \
   data/processed/risk_models/frequency_model_challenger.json
cp data/processed/risk_models/frequency_calibration.json \
   data/processed/risk_models/frequency_calibration_challenger.json

# 2. Inject synthetic drift, run checks, then restore
docker compose run --rm app python scripts/inject_drift.py
docker compose run --rm app python main.py --drift-check    # triggers CC shadow scoring
docker compose run --rm app python main.py --compare-gini
docker compose run --rm app python scripts/inject_drift.py --restore
```

## Monitoring

```bash
docker compose run --rm app python main.py --drift-check
docker compose run --rm app python main.py --drift-check --as-of 2026-05-24
docker compose run --rm app python main.py --compare-gini
docker compose run --rm app python main.py --shap-check
docker compose run --rm app python main.py --purge-drift-logs
```

## Maintenance

```bash
# Fix ownership after Docker writes files as root
sudo chown -R $USER:$USER ./data && chmod -R u+rwx ./data

# Reset Neo4j graph
docker compose run --rm app python main.py --reset-graph

# Query Neo4j directly
export $(cat .env | xargs)
docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "MATCH (n) RETURN count(n);"
docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "MATCH (n) RETURN n LIMIT 10;" > neo4jQueryResult.txt

docker compose logs neo4j

# Teardown
docker compose down
docker compose down -v
docker compose rm -s -f -v app
docker compose down --rmi all -v
```

## Project layout

- `data/` — synthetic data generator, archetypes, validation, graph bootstrapping
- `features/` — shared feature engineering and feature store persistence
- `tests/` — unit tests for feature store and generator workflow
