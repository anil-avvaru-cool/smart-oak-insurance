"""Inject distribution-shifted rows into quotes.parquet to force PSI > 0.25.

Shifts `driver_age` into the 75-85 range for injected rows (reference mean ~40),
which produces a simulated PSI of ~0.90 — well above the 0.25 CC trigger.

Usage:
    uv run python scripts/inject_drift.py           # inject drift
    uv run python scripts/inject_drift.py --restore # restore backup

Backup is written to data/raw/quotes.bak.parquet before any mutation.
Running --restore is idempotent if the backup does not exist.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUOTES_PATH = Path("data/raw/quotes.parquet")
BACKUP_PATH = Path("data/raw/quotes.bak.parquet")

N_INJECT = 600
RANDOM_SEED = 42
# driver_age reference mean ~40, std ~13, max 80 — injecting 75-85 guarantees major drift
DRIFT_AGE_LOW = 75
DRIFT_AGE_HIGH = 85
PSI_CURRENT_WINDOW_DAYS = 14


def _verify_psi(df_orig: pd.DataFrame, df_new: pd.DataFrame) -> float:
    """Return the driver_age PSI between reference and new current window."""
    from monitoring.psi_drift import PSI_CURRENT_WINDOW_DAYS as CFG_DAYS, compute_psi_series

    now = pd.Timestamp.now()
    window_start = now - pd.Timedelta(days=CFG_DAYS)

    ref = df_orig[pd.to_datetime(df_orig["quote_requested_at"]) < window_start]["driver_age"]
    curr = df_new[pd.to_datetime(df_new["quote_requested_at"]) >= window_start]["driver_age"]

    psi, _ = compute_psi_series(ref, curr)
    return psi


def inject(quotes_path: Path = QUOTES_PATH, backup_path: Path = BACKUP_PATH) -> None:
    if not quotes_path.exists():
        print(f"ERROR: {quotes_path} not found. Run --generate-data first.", file=sys.stderr)
        sys.exit(1)

    if backup_path.exists():
        print(f"Backup already exists at {backup_path} — skipping overwrite.")
        print("If you want a fresh injection, run --restore first.")
        sys.exit(1)

    df = pd.read_parquet(quotes_path)
    rng = np.random.default_rng(RANDOM_SEED)

    now = pd.Timestamp.now()

    # Clone N_INJECT rows from existing data as the base for injected records
    inject_rows = df.sample(N_INJECT, random_state=RANDOM_SEED).copy().reset_index(drop=True)

    # Assign new unique quote_ids so they don't collide with originals
    inject_rows["quote_id"] = [f"drift_inject_{i:05d}" for i in range(N_INJECT)]

    # Stamp into the current 14-day window, evenly spaced
    inject_rows["quote_requested_at"] = pd.date_range(
        end=now - pd.Timedelta(minutes=5),
        periods=N_INJECT,
        freq="30min",
    )
    # quote_completed_at a few minutes after requested
    inject_rows["quote_completed_at"] = inject_rows["quote_requested_at"] + pd.Timedelta(minutes=3)

    # Apply the distribution shift: driver_age → 75–85 (reference mean ~40)
    inject_rows["driver_age"] = rng.integers(DRIFT_AGE_LOW, DRIFT_AGE_HIGH + 1, size=N_INJECT)

    # Keep all other columns as-is (realistic values from real rows)

    df_drifted = pd.concat([df, inject_rows], ignore_index=True)

    # Verify PSI before writing
    psi = _verify_psi(df, df_drifted)
    print(f"Simulated driver_age PSI: {psi:.4f}  (trigger threshold: 0.25)")
    if psi < 0.25:
        print("WARNING: PSI did not exceed threshold — injection may be insufficient.", file=sys.stderr)

    # Write backup then drifted file
    df.to_parquet(backup_path)
    df_drifted.to_parquet(quotes_path)

    print(f"Backup written  → {backup_path}  ({len(df)} rows)")
    print(f"Drifted dataset → {quotes_path}  ({len(df_drifted)} rows, +{N_INJECT} injected)")
    print()
    print("Next steps:")
    print("  uv run -m main --drift-check    # should print champion_challenger_triggered: true")
    print("  uv run -m main --compare-gini")
    print("  uv run python scripts/inject_drift.py --restore")


def restore(quotes_path: Path = QUOTES_PATH, backup_path: Path = BACKUP_PATH) -> None:
    if not backup_path.exists():
        print(f"No backup found at {backup_path} — nothing to restore.")
        return

    backup_path.rename(quotes_path)
    print(f"Restored {quotes_path} from backup. Backup removed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--restore", action="store_true", help="Restore original quotes.parquet from backup")
    args = parser.parse_args()

    if args.restore:
        restore()
    else:
        inject()


if __name__ == "__main__":
    main()
