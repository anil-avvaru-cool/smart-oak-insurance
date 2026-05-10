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
```

Validate generated data:
```bash
uv run -m main --validate-data
```

Generated outputs are written to `data/raw/quotes.parquet` and `data/raw/claims.parquet`.

## Docker

Docker commands:
```bash
docker compose up -d --build
#Run the generator inside Docker:
docker compose run --rm app python main.py --generate-data
#Run validation inside Docker:
docker compose run --rm app python main.py --validate-data

# Maintenance
docker compose down
docker compose down --rmi all -v
```

## Project layout

- `data/` — synthetic data generator, archetypes, validation, graph bootstrapping
- `features/` — shared feature engineering and feature store persistence
- `tests/` — unit tests for feature store and generator workflow
