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

Build the container and start services:
```powershell
docker compose up --build
```

Run the generator inside Docker:
```powershell
docker compose run --rm app python main.py --generate-data
```

Run validation inside Docker:
```powershell
docker compose run --rm app python main.py --validate-data
```

## Project layout

- `data/` — synthetic data generator, archetypes, validation, graph bootstrapping
- `features/` — shared feature engineering and feature store persistence
- `tests/` — unit tests for feature store and generator workflow
