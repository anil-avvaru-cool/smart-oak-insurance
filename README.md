# Smart Oak Insurance

Synthetic insurance risk-and-fraud platform for underwriting and claims modeling.

## Setup

1. Create a Python virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```
2. Install runtime dependencies:
   ```bash
   uv sync
   ```
3. Configure environment variables:
   ```bash
   cp example.env .env
   # Edit .env with your configuration
   ```


## Generate synthetic data

Run locally:
```bash
uv run -m main --generate-data
uv run -m main --resolve-entities
```

Validate generated data:
```bash
uv run -m main --validate-data
```

Generated outputs are written to `data/raw/quotes.parquet`, `data/raw/claims.parquet`, and resolved entities to `data/entities/`.

## Docker

Docker commands:
```bash
docker compose up -d --build
docker compose up -d neo4j redis
sudo rm -rf ./data/raw ./data/processed

docker compose run --rm app python main.py --generate-data
docker compose run --rm app python main.py --resolve-entities

# Delete existing and start from scratch
docker compose run --rm app python main.py --build-graph
docker compose run --rm app python main.py --compute-graph-features
docker compose run --rm app python main.py --run-offline-pipeline
docker compose run --rm app python main.py --validate-data

# Maintenance
# Delete existing graph with constraints
docker compose run --rm app python main.py --reset-graph
# 1. Export variables to your current host terminal(only once)
export $(cat .env | xargs)

# 2. Run your original command (it will now find $NEO4J_PASSWORD)
docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "MATCH (n) RETURN count(n);"
docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
"MATCH (n) 
 RETURN n 
 LIMIT 10;" > neo4jQueryResult.txt

docker compose logs neo4j
docker compose down
docker compose down -v
docker compose down --rmi all -v
```

## Project layout

- `data/` — synthetic data generator, archetypes, validation, graph bootstrapping
- `features/` — shared feature engineering and feature store persistence
- `tests/` — unit tests for feature store and generator workflow
