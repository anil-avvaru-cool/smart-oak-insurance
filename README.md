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

   ```bash
   # Configure environment variables:
   cp example.env .env
   # Edit .env with your configuration

   # Copy dependencies to requirements.txt to use in docker
   uv export --format requirements-txt --no-hashes --no-emit-workspace > requirements.txt

   ```
## Generate synthetic data

Run locally:
```bash
uv run -m main --generate-data --resolve-entities --build-graph --compute-graph-features --run-offline-pipeline --validate-data
```

Generated outputs are written to `data/raw/quotes.parquet`, `data/raw/claims.parquet`, and resolved entities to `data/entities/`.

## Docker

Docker commands:
```bash
docker compose build --no-cache
docker compose up -d neo4j redis app
sudo rm -rf ./data/raw/* ./data/processed/*
# Run all in single command
docker compose run --rm app python main.py --generate-data --resolve-entities --build-graph --compute-graph-features --run-offline-pipeline --validate-data
docker compose run --rm app python main.py --train-risk-model

uv run -m main --generate-data --resolve-entities --build-graph --compute-graph-features --run-offline-pipeline --validate-data
uv run -m main --train-risk-model
uv run -m main --calibrate-risk-model

# Maintenance

# To create permissions for current user
sudo chown -R $USER:$USER /path/to/directory
chmod -R u+rwx /path/to/directory


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
docker compose rm -s -f -v app
docker compose down --rmi all -v
```

## Project layout

- `data/` — synthetic data generator, archetypes, validation, graph bootstrapping
- `features/` — shared feature engineering and feature store persistence
- `tests/` — unit tests for feature store and generator workflow
