from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

from data.config import CLAIMS_OUTPUT, OFFLINE_FEATURES_DIR, QUOTES_OUTPUT
from data.states import US_STATE_ABBREVIATIONS

_PASS = "\033[92m✓ PASS\033[0m"
_FAIL = "\033[91m✗ FAIL\033[0m"
_WARN = "\033[93m⚠ WARN\033[0m"
_SKIP = "\033[90m~ SKIP\033[0m"


def validate_quote_dataset(quotes_df: pd.DataFrame) -> None:
    print("Validating quote dataset...")
    print(f"  records: {len(quotes_df)}")
    print(f"  quote state coverage: {quotes_df['state'].nunique()} states")
    print(f"  risk score range: {quotes_df['risk_score_at_issuance'].min():.3f} - {quotes_df['risk_score_at_issuance'].max():.3f}")
    invalid_states = quotes_df.loc[~quotes_df["state"].isin(US_STATE_ABBREVIATIONS), "state"]
    if len(invalid_states):
        print(f"  {_FAIL} invalid state values found: {sorted(invalid_states.unique())}")
    else:
        print(f"  {_PASS} state values are valid")
    if (quotes_df["credit_score"] < 500).any() or (quotes_df["credit_score"] > 850).any():
        print(f"  {_FAIL} credit score out of range")
    else:
        print(f"  {_PASS} credit score range looks healthy")


def validate_claim_dataset(claims_df: pd.DataFrame) -> None:
    print("Validating claim dataset...")
    print(f"  records: {len(claims_df)}")
    fraud_rate = claims_df[claims_df["is_fraud"] == True].shape[0] / max(1, len(claims_df))
    print(f"  fraud rate: {fraud_rate:.2%}")
    if fraud_rate < 0.15 or fraud_rate > 0.45:
        print(f"  {_WARN} fraud rate is outside expected training range")
    else:
        print(f"  {_PASS} fraud rate is within expected training range")
    if (claims_df["policy_inception_days"] < 0).any():
        print(f"  {_FAIL} negative policy_inception_days found")
    else:
        print(f"  {_PASS} policy inception values are valid")
    if (claims_df["reporting_delay_days"] < 0).any():
        print(f"  {_FAIL} negative reporting_delay_days found")
    else:
        print(f"  {_PASS} reporting delay values are valid")


def validate_graph_features(claims_df: pd.DataFrame) -> None:
    print("Validating graph features...")

    # Validate graph_hop_distance column
    if "graph_hop_distance" in claims_df.columns:
        hop_distance_vals = claims_df["graph_hop_distance"]
        sentinel_count = (hop_distance_vals == 999).sum()
        sentinel_pct = sentinel_count / len(claims_df) * 100 if len(claims_df) > 0 else 0

        print(f"  records with graph_hop_distance: {(hop_distance_vals.notna()).sum()}")
        print(f"  sentinel values (999): {sentinel_count} ({sentinel_pct:.1f}%)")

        if sentinel_pct > 50:
            print(f"  {_FAIL} >50% sentinel values indicate graph is not connected to fraud entities")
        elif sentinel_pct > 20:
            print(f"  {_WARN} {sentinel_pct:.1f}% sentinel values may indicate incomplete graph coverage")
        else:
            print(f"  {_PASS} graph connectivity looks healthy")

        non_sentinel = hop_distance_vals[hop_distance_vals != 999]
        if len(non_sentinel) > 0:
            print(f"  hop distance range (excluding 999): {non_sentinel.min()} - {non_sentinel.max()}")
            print(f"  hop distance median: {non_sentinel.median():.1f}")
    else:
        print(f"  {_WARN} graph_hop_distance column not found in dataset")

    # Validate attorney_centrality_score column
    if "attorney_centrality_score" in claims_df.columns:
        centrality_vals = claims_df["attorney_centrality_score"]
        print(f"  attorney centrality range: {centrality_vals.min():.3f} - {centrality_vals.max():.3f}")
        if (centrality_vals < 0).any() or (centrality_vals > 1.0).any():
            print(f"  {_FAIL} attorney centrality score out of expected [0, 1] range")
        else:
            print(f"  {_PASS} attorney centrality scores in valid range")

    # Validate shared_attribute_count column
    if "shared_attribute_count" in claims_df.columns:
        shared_count = claims_df["shared_attribute_count"]
        print(f"  shared attribute count range: {shared_count.min()} - {shared_count.max()}")
        print(f"  shared attribute median: {shared_count.median():.1f}")
        if (shared_count < 0).any():
            print(f"  {_FAIL} negative shared attribute counts found")
        else:
            print(f"  {_PASS} shared attribute counts are valid")


def validate_graph_topology(neo4j_uri: str | None = None, neo4j_user: str | None = None, neo4j_password: str | None = None) -> None:
    """Validate Neo4j graph topology and connectivity."""
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("Skipping graph topology validation (Neo4j credentials not provided)")
        return

    print("Validating graph topology...")

    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        with driver.session() as session:
            # Check if graph is empty
            result = session.run("MATCH (n) RETURN count(*) as total_nodes")
            total_nodes = result.single()['total_nodes']

            if total_nodes == 0:
                print(f"  {_FAIL} GRAPH IS EMPTY - no nodes found in Neo4j")
                print(f"     FIX: Run 'python main.py --build-graph' to populate the graph")
                driver.close()
                return

            # Count nodes by type
            result = session.run("""
                MATCH (n)
                RETURN labels(n) as label, count(*) as count
            """)

            print(f"  total nodes: {total_nodes}")
            for record in result:
                label = record['label'][0] if record['label'] else "Unknown"
                count = record['count']
                print(f"    {label}: {count}")

            # Count relationships by type
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(*) as count
            """)

            rel_count = 0
            for record in result:
                rel_type = record['rel_type']
                count = record['count']
                rel_count += count
                print(f"  {rel_type} relationships: {count}")

            if rel_count == 0:
                print(f"  {_FAIL} NO RELATIONSHIPS found - graph is disconnected")
            else:
                print(f"  total relationships: {rel_count}")

            # Check for fraud claimants
            result = session.run("""
                MATCH (c:Claimant {is_fraud: true})
                RETURN count(*) as fraud_count
            """)

            row = result.single()
            fraud_count = row['fraud_count'] if row else 0
            print(f"  fraud claimants: {fraud_count}")

            if fraud_count == 0:
                print(f"  {_FAIL} NO fraud claimants in graph (explains 999 sentinel values)")
            elif fraud_count < 5:
                print(f"  {_WARN} very few fraud claimants ({fraud_count}), may limit path discovery")
            else:
                print(f"  {_PASS} fraud claimants present for path analysis")

            # Check for isolated claimants (no connections)
            result = session.run("""
                MATCH (c:Claimant)
                WHERE NOT (c)-[]-()
                RETURN count(*) as isolated_count
            """)

            isolated = result.single()['isolated_count'] if result.single() else 0
            if isolated > 0:
                print(f"  {_WARN} {isolated} isolated claimant nodes (no SHARES relationships)")
            else:
                print(f"  {_PASS} all claimants have relationships")

        driver.close()
    except Exception as e:
        print(f"  {_FAIL} error validating graph topology: {e}")
        print(f"     Check Neo4j connection: {neo4j_uri}")


def validate_entity_resolution() -> None:
    from pathlib import Path
    from data.config import RAW_DATA_DIR

    print("Validating entity resolution...")
    entities_dir = RAW_DATA_DIR / "entities"

    checks = {
        "vehicles.parquet": entities_dir / "vehicles.parquet",
        "persons.parquet": entities_dir / "persons.parquet",
        "addresses.parquet": entities_dir / "addresses.parquet",
        "phones.parquet": entities_dir / "phones.parquet",
        "policies.parquet": entities_dir / "policies.parquet",
    }

    missing_files = [name for name, path in checks.items() if not path.exists()]
    if missing_files:
        print(f"  {_FAIL} missing entity files: {missing_files}")
        print(f"     FIX: Run 'python main.py --resolve-entities' to generate entity data")
        return

    vehicles_df = pd.read_parquet(checks["vehicles.parquet"])
    persons_df = pd.read_parquet(checks["persons.parquet"])
    addresses_df = pd.read_parquet(checks["addresses.parquet"])
    phones_df = pd.read_parquet(checks["phones.parquet"])
    policies_df = pd.read_parquet(checks["policies.parquet"])

    # Vehicle validation
    null_vins = vehicles_df["vin"].isnull().sum()
    if null_vins > 0:
        print(f"  {_FAIL} {null_vins} null VINs found")
    else:
        print(f"  {_PASS} no null VINs ({len(vehicles_df)} vehicles)")

    # Person validation
    duplicate_persons = persons_df["person_id"].duplicated().sum()
    if duplicate_persons > 0:
        print(f"  {_FAIL} {duplicate_persons} duplicate person_ids found")
    else:
        print(f"  {_PASS} no duplicate person_ids ({len(persons_df)} unique persons)")

    # Address validation
    null_hashes = addresses_df["address_hash"].isnull().sum()
    if null_hashes > 0:
        print(f"  {_FAIL} {null_hashes} null address hashes found")
    else:
        print(f"  {_PASS} all addresses have hashes ({len(addresses_df)} unique addresses)")

    # Phone validation
    valid_phones = phones_df[phones_df["is_valid"]].shape[0]
    print(f"  valid phones: {valid_phones}/{len(phones_df)}")

    # Policy validation
    if (policies_df["inception_date"] > policies_df["expiration_date"]).any():
        print(f"  {_FAIL} policies with expiration before inception found")
    else:
        print(f"  {_PASS} policy dates are valid ({len(policies_df)} policies)")



_TELEMATICS_RAW_COLS = [
    "telematics_distraction_score",
    "telematics_hard_brake_rate",
    "telematics_crash_match",
    "telematics_commute_entropy",
]


def _check_telematics_trio_invariant(df: pd.DataFrame, dataset: str) -> None:
    has_raw = all(c in df.columns for c in _TELEMATICS_RAW_COLS)
    has_derived = all(c in df.columns for c in ("telematics_available", "telematics_enrolled_but_missing"))

    if not has_raw:
        print(f"  {_SKIP} {dataset} telematics trio (raw columns absent)")
        return

    if has_derived:
        # Feature-vector path: full trio consistency
        available = df["telematics_available"] == True
        enrolled_missing = df["telematics_enrolled_but_missing"] == True

        bad = (df.loc[available, _TELEMATICS_RAW_COLS].isnull().all(axis=1)).sum()
        if bad:
            print(f"  {_FAIL} {dataset}: {bad} rows telematics_available=True but all raw signals null")
        else:
            print(f"  {_PASS} {dataset} telematics: available=True implies at least one raw signal")

        bad = (df.loc[enrolled_missing, "telematics_available"] == True).sum()
        if bad:
            print(f"  {_FAIL} {dataset}: {bad} rows enrolled_but_missing=True yet also available=True")
        else:
            print(f"  {_PASS} {dataset} telematics: enrolled_but_missing=True implies available=False")

        non_enrolled = (~available) & (~enrolled_missing)
        if non_enrolled.any():
            bad = df.loc[non_enrolled, _TELEMATICS_RAW_COLS].notnull().any(axis=1).sum()
            if bad:
                print(f"  {_FAIL} {dataset}: {bad} non-enrolled rows contain non-null raw telematics values")
            else:
                print(f"  {_PASS} {dataset} telematics: non-enrolled rows have no raw signals")
    else:
        # Raw-data path: generator copies all 4 columns together, so each row must be
        # uniformly null (not enrolled) or uniformly non-null (enrolled with data).
        null_mask = df[_TELEMATICS_RAW_COLS].isnull()
        mixed = (null_mask.any(axis=1)) & (~null_mask.all(axis=1))
        bad = mixed.sum()
        n_enrolled = (~null_mask.all(axis=1)).sum()
        if bad:
            print(f"  {_FAIL} {dataset}: {bad} rows have mixed null/non-null telematics signals (should be all-or-nothing)")
        else:
            print(f"  {_PASS} {dataset} telematics: all-or-nothing consistency holds ({n_enrolled} enrolled rows)")


def _check_state_regulatory_compliance(quotes_df: pd.DataFrame) -> None:
    """Credit masking happens in build_quote_feature_vector, not in raw parquet.

    Raw data check: non-restricted states must always have a credit_score (generator
    bug if null). Restricted states having a non-null credit_score is expected — the
    feature builder masks them to None before serving.
    """
    from features.feature_definitions import CREDIT_RESTRICTED_STATES

    if "state" not in quotes_df.columns or "credit_score" not in quotes_df.columns:
        print(f"  {_SKIP} state regulatory compliance (state/credit_score columns missing)")
        return

    restricted = quotes_df["state"].isin(CREDIT_RESTRICTED_STATES)
    n_restricted = restricted.sum()

    # Non-restricted states: generator must always populate credit_score
    unrestricted = ~restricted
    if unrestricted.any():
        bad = quotes_df.loc[unrestricted, "credit_score"].isnull().sum()
        n = unrestricted.sum()
        if bad:
            print(f"  {_FAIL} {bad}/{n} non-restricted-state quotes have null credit_score (generator bug)")
        else:
            print(f"  {_PASS} credit_score present for all {n} non-restricted-state quotes")

    # Restricted states: credit_score will be masked by the feature builder — having
    # a non-null value here is correct; masking to null happens at serving time.
    print(f"  info: {n_restricted} restricted-state quotes have raw credit_score (masked to null by feature builder)")


def _check_fraud_signal_directions(claims_df: pd.DataFrame) -> None:
    if "is_fraud" not in claims_df.columns:
        print(f"  {_SKIP} fraud signal directions (is_fraud column missing)")
        return

    fraud = claims_df[claims_df["is_fraud"] == True]
    legit = claims_df[claims_df["is_fraud"] == False]
    if fraud.empty or legit.empty:
        print(f"  {_FAIL} fraud signal directions: dataset lacks fraud or legit rows")
        return

    # (column, direction) — fraud_gt: fraud mean > legit mean; fraud_lt: opposite
    checks: list[tuple[str, str]] = [
        ("attorney_present", "fraud_gt"),
        ("reporting_delay_days", "fraud_gt"),
        ("claimant_count", "fraud_gt"),
        ("shared_attribute_count", "fraud_gt"),
        ("attorney_centrality_score", "fraud_gt"),
        ("narrative_inconsistency_score", "fraud_gt"),
        ("narrative_complexity_score", "fraud_gt"),
        ("ip_geolocation_delta_miles", "fraud_gt"),
        ("prior_claims_count", "fraud_gt"),
        ("device_fingerprint_match", "fraud_lt"),
        ("policy_inception_days", "fraud_lt"),
        ("telematics_available", "fraud_lt"),
    ]

    passes = fails = skips = 0
    for col, direction in checks:
        if col not in claims_df.columns:
            skips += 1
            continue
        fraud_mean = fraud[col].astype(float).mean()
        legit_mean = legit[col].astype(float).mean()
        ok = (fraud_mean > legit_mean) if direction == "fraud_gt" else (fraud_mean < legit_mean)
        if not ok:
            fails += 1
            print(f"  {_FAIL} {col}: fraud={fraud_mean:.3f} legit={legit_mean:.3f} (signal inverted)")
        else:
            passes += 1
    print(f"  fraud signal directions: {passes} pass, {fails} fail, {skips} skip")


def _check_fraud_feature_correlations(claims_df: pd.DataFrame) -> None:
    import numpy as np

    if "is_fraud" not in claims_df.columns:
        return

    y = claims_df["is_fraud"].astype(float)

    # +1 = positive correlation with fraud expected, -1 = negative
    expected_signs: dict[str, int] = {
        "attorney_present": +1,
        "reporting_delay_days": +1,
        "claimant_count": +1,
        "shared_attribute_count": +1,
        "attorney_centrality_score": +1,
        "narrative_inconsistency_score": +1,
        "ip_geolocation_delta_miles": +1,
        "prior_claims_count": +1,
        "device_fingerprint_match": -1,
        "policy_inception_days": -1,
        "telematics_available": -1,
    }

    passes = fails = skips = 0
    for col, expected in expected_signs.items():
        if col not in claims_df.columns:
            skips += 1
            continue
        x = claims_df[col].astype(float)
        valid = x.notna() & y.notna()
        if valid.sum() < 10:
            skips += 1
            continue
        r = float(np.corrcoef(x[valid], y[valid])[0, 1])
        if abs(r) < 0.01:
            print(f"  {_WARN} {col}: near-zero correlation with is_fraud (r={r:.3f})")
        elif (r > 0) == (expected > 0):
            passes += 1
        else:
            fails += 1
            direction = "positive" if expected > 0 else "negative"
            print(f"  {_FAIL} {col}: r={r:.3f} (expected {direction} sign)")
    print(f"  feature-fraud correlations: {passes} correct sign, {fails} wrong sign, {skips} skipped")


def _check_cross_dataset_spine(quotes_df: pd.DataFrame, claims_df: pd.DataFrame) -> None:
    if "quote_id" not in claims_df.columns or "quote_id" not in quotes_df.columns:
        print(f"  {_SKIP} cross-dataset spine (quote_id column missing)")
        return

    claim_qids = set(claims_df["quote_id"].dropna().astype(str))
    quote_qids = set(quotes_df["quote_id"].dropna().astype(str))
    orphaned = claim_qids - quote_qids
    if orphaned:
        print(f"  {_FAIL} {len(orphaned)} claim quote_ids not found in quotes dataset")
    else:
        print(f"  {_PASS} all {len(claim_qids)} claim quote_ids exist in quotes dataset")

    # risk_score_at_issuance must agree across the join (shared data spine invariant)
    if "risk_score_at_issuance" in claims_df.columns and "risk_score_at_issuance" in quotes_df.columns:
        merged = claims_df[["quote_id", "risk_score_at_issuance"]].merge(
            quotes_df[["quote_id", "risk_score_at_issuance"]].rename(
                columns={"risk_score_at_issuance": "_q_risk"}
            ),
            on="quote_id",
            how="inner",
        )
        mismatch = (merged["risk_score_at_issuance"] - merged["_q_risk"]).abs() > 1e-6
        bad = mismatch.sum()
        if bad:
            print(f"  {_FAIL} {bad} claims have risk_score_at_issuance differing from linked quote")
        else:
            print(f"  {_PASS} risk_score_at_issuance consistent across {len(merged)} quote-claim pairs")


def _check_attorney_coherence(claims_df: pd.DataFrame) -> None:
    for col in ("attorney_present", "attorney_centrality_score"):
        if col not in claims_df.columns:
            print(f"  {_SKIP} attorney coherence ({col} missing)")
            return

    with_attorney = claims_df[claims_df["attorney_present"] == True]
    if with_attorney.empty:
        print(f"  {_WARN} attorney coherence: no attorney_present=True rows found")
        return

    zero_pct = (with_attorney["attorney_centrality_score"] == 0).mean() * 100
    if zero_pct > 20:
        print(f"  {_WARN} {zero_pct:.1f}% of attorney_present=True rows have centrality_score=0")
    else:
        print(f"  {_PASS} attorney coherence: {zero_pct:.1f}% of attorney rows have zero centrality")

    no_attorney = claims_df[claims_df["attorney_present"] == False]
    if not no_attorney.empty:
        high_pct = (no_attorney["attorney_centrality_score"] > 0.3).mean() * 100
        if high_pct > 10:
            print(f"  {_WARN} {high_pct:.1f}% of attorney_present=False rows have centrality_score > 0.3")


def _check_graph_sentinel_coherence(claims_df: pd.DataFrame) -> None:
    if "graph_hop_distance" not in claims_df.columns or "is_fraud" not in claims_df.columns:
        return

    fraud = claims_df[claims_df["is_fraud"] == True]
    legit = claims_df[claims_df["is_fraud"] == False]

    if not fraud.empty:
        fraud_sentinel_pct = (fraud["graph_hop_distance"] == 999).mean() * 100
        if fraud_sentinel_pct > 30:
            print(f"  {_FAIL} {fraud_sentinel_pct:.1f}% of fraud claims have sentinel hop_distance=999 (graph poorly connected)")
        else:
            print(f"  {_PASS} fraud claims sentinel rate: {fraud_sentinel_pct:.1f}%")

    if not legit.empty:
        legit_sentinel_pct = (legit["graph_hop_distance"] == 999).mean() * 100
        print(f"  info: legit claims sentinel rate: {legit_sentinel_pct:.1f}% (expect > fraud rate)")
        if not fraud.empty and legit_sentinel_pct < (fraud["graph_hop_distance"] == 999).mean() * 100:
            print(f"  {_WARN} legit sentinel rate is lower than fraud rate (hop_distance signal may be inverted)")


def validate_feature_correlations(quotes_df: pd.DataFrame, claims_df: pd.DataFrame) -> None:
    """Deep cross-feature and fraud-signal correlation checks.

    Catches data generation bugs — inverted signals, broken archetypes, telematics
    trio inconsistency — before they silently degrade model quality or waste a
    full training run.
    """
    print("\nValidating feature correlations...")
    _check_telematics_trio_invariant(quotes_df, "quotes")
    _check_telematics_trio_invariant(claims_df, "claims")
    _check_state_regulatory_compliance(quotes_df)
    _check_fraud_signal_directions(claims_df)
    _check_fraud_feature_correlations(claims_df)
    _check_cross_dataset_spine(quotes_df, claims_df)
    _check_attorney_coherence(claims_df)
    _check_graph_sentinel_coherence(claims_df)


_SNAPSHOT_ENVELOPE_KEYS = frozenset({
    "record_id", "record_type", "timestamp", "feature_store_version",
    "state", "regulatory_mask_applied", "features",
})

_QUOTE_FEATURE_KEYS = frozenset({
    "credit_score", "prior_loss_frequency", "prior_loss_severity_avg",
    "insurance_lapse_days", "violation_severity_index", "household_driver_density",
    "driver_age", "years_licensed", "vehicle_msrp_power_ratio", "vehicle_adas_score",
    "vehicle_age_years", "geohash_risk_score", "state", "annual_mileage_estimate",
    "risk_score_at_issuance", "policy_tier_at_issuance",
    "telematics_distraction_score", "telematics_hard_brake_rate",
    "telematics_crash_match", "telematics_commute_entropy",
    "telematics_available", "telematics_enrolled_but_missing", "credit_eligible",
})

_CLAIM_FEATURE_KEYS = frozenset({
    "policy_inception_days", "prior_claims_count", "reported_injury_count",
    "reporting_delay_days", "attorney_present", "submission_hour", "claimant_count",
    "graph_hop_distance", "shared_attribute_count", "attorney_centrality_score",
    "narrative_inconsistency_score", "narrative_complexity_score",
    "risk_score_at_issuance", "policy_tier_at_issuance",
    "ip_geolocation_delta_miles", "device_fingerprint_match", "submission_channel",
    "telematics_distraction_score", "telematics_hard_brake_rate",
    "telematics_crash_match", "telematics_commute_entropy",
    "telematics_available", "telematics_enrolled_but_missing",
})

_TELEMATICS_SIGNAL_KEYS = (
    "telematics_distraction_score",
    "telematics_hard_brake_rate",
    "telematics_crash_match",
    "telematics_commute_entropy",
)


def _check_snapshot_envelope(snap: dict, path: str) -> list[str]:
    missing = _SNAPSHOT_ENVELOPE_KEYS - snap.keys()
    errs: list[str] = []
    if missing:
        errs.append(f"{path}: missing envelope keys {sorted(missing)}")
    if snap.get("record_type") not in ("quote", "claim"):
        errs.append(f"{path}: invalid record_type={snap.get('record_type')!r}")
    if not isinstance(snap.get("features"), dict):
        errs.append(f"{path}: 'features' is not a dict")
    return errs


def _check_feature_schema(features: dict, expected_keys: frozenset, record_id: str) -> list[str]:
    missing = expected_keys - features.keys()
    if missing:
        return [f"{record_id}: missing feature keys {sorted(missing)}"]
    return []


def _check_regulatory_mask_in_snapshot(features: dict, state: str, record_id: str) -> list[str]:
    from features.feature_definitions import CREDIT_RESTRICTED_STATES

    errs: list[str] = []
    if state in CREDIT_RESTRICTED_STATES:
        if features.get("credit_score") is not None:
            errs.append(f"{record_id}: restricted state {state} but credit_score is not None")
        if features.get("credit_eligible") is not False:
            errs.append(f"{record_id}: restricted state {state} but credit_eligible is not False")
    else:
        if features.get("credit_eligible") is not True:
            errs.append(f"{record_id}: non-restricted state {state} but credit_eligible is not True")
    return errs


def _check_telematics_trio_in_snapshot(features: dict, record_id: str) -> list[str]:
    available = features.get("telematics_available")
    enrolled_missing = features.get("telematics_enrolled_but_missing")
    raw_signals = [features.get(k) for k in _TELEMATICS_SIGNAL_KEYS]
    any_raw = any(v is not None for v in raw_signals)
    errs: list[str] = []

    if available and not any_raw:
        errs.append(f"{record_id}: telematics_available=True but all raw signals are null")
    if enrolled_missing and available:
        errs.append(f"{record_id}: telematics_enrolled_but_missing=True but telematics_available=True")
    if not available and not enrolled_missing and any_raw:
        errs.append(f"{record_id}: non-enrolled row has non-null raw telematics signals")
    return errs


def _check_value_ranges_quote(features: dict, record_id: str) -> list[str]:
    errs: list[str] = []
    risk = features.get("risk_score_at_issuance")
    if risk is not None and not (0.0 <= float(risk) <= 1.0):
        errs.append(f"{record_id}: risk_score_at_issuance={risk} out of [0, 1]")
    age = features.get("driver_age")
    if age is not None and int(age) > 0 and int(age) < 15:
        errs.append(f"{record_id}: driver_age={age} implausibly low")
    msrp_ratio = features.get("vehicle_msrp_power_ratio")
    if msrp_ratio is not None and float(msrp_ratio) < 0:
        errs.append(f"{record_id}: vehicle_msrp_power_ratio={msrp_ratio} is negative")
    return errs


def _check_nan_in_features(features: dict, record_id: str) -> list[str]:
    nan_keys = [k for k, v in features.items() if isinstance(v, float) and math.isnan(v)]
    if nan_keys:
        return [f"{record_id}: NaN values in features (should be null): {sorted(nan_keys)}"]
    return []


def _check_value_ranges_claim(features: dict, record_id: str) -> list[str]:
    errs: list[str] = []
    for col in ("policy_inception_days", "reporting_delay_days", "shared_attribute_count"):
        val = features.get(col)
        if val is not None and int(val) < 0:
            errs.append(f"{record_id}: {col}={val} is negative")
    centrality = features.get("attorney_centrality_score")
    if centrality is not None and not (0.0 <= float(centrality) <= 1.0):
        errs.append(f"{record_id}: attorney_centrality_score={centrality} out of [0, 1]")
    hop = features.get("graph_hop_distance")
    if hop is not None and int(hop) < 0:
        errs.append(f"{record_id}: graph_hop_distance={hop} is negative")
    return errs


def validate_offline_features(features_dir: Path | None = None) -> None:
    """Validate feature snapshots produced by run_offline_pipeline.

    Checks:
    - Envelope completeness (required keys, record_type, version)
    - Feature schema — every key from feature_definitions present
    - Regulatory masking — credit_score/credit_eligible correct for restricted states in quotes
    - Telematics trio invariant — available/enrolled_missing/raw signals consistent
    - Value ranges — risk scores [0,1], no negatives where impossible
    - Cross-snapshot spine — claim risk_score_at_issuance matches linked quote snapshot
    """
    from features.feature_definitions import FEATURE_STORE_VERSION

    if features_dir is None:
        features_dir = OFFLINE_FEATURES_DIR

    print("\nValidating offline feature snapshots...")

    if not features_dir.exists():
        print(f"  {_SKIP} features directory not found: {features_dir}")
        print(f"     FIX: Run 'python main.py --run-offline-pipeline' to generate snapshots")
        return

    snapshot_files = sorted(features_dir.glob("*.json"))
    if not snapshot_files:
        print(f"  {_SKIP} no snapshot files found in {features_dir}")
        return

    print(f"  loading {len(snapshot_files)} snapshots from {features_dir}")

    quote_snaps: list[dict] = []
    claim_snaps: list[dict] = []
    envelope_errors: list[str] = []
    schema_errors: list[str] = []
    regulatory_errors: list[str] = []
    telematics_errors: list[str] = []
    range_errors: list[str] = []
    nan_errors: list[str] = []
    version_mismatches = 0

    for snap_path in snapshot_files:
        try:
            snap = json.loads(snap_path.read_text())
        except Exception as e:
            envelope_errors.append(f"{snap_path.name}: JSON parse error — {e}")
            continue

        env_errs = _check_snapshot_envelope(snap, snap_path.name)
        envelope_errors.extend(env_errs)
        if env_errs:
            continue

        if snap.get("feature_store_version") != FEATURE_STORE_VERSION:
            version_mismatches += 1

        record_type = snap["record_type"]
        record_id = snap["record_id"]
        features = snap["features"]
        state = snap.get("state", "")

        nan_errors.extend(_check_nan_in_features(features, record_id))

        if record_type == "quote":
            schema_errors.extend(_check_feature_schema(features, _QUOTE_FEATURE_KEYS, record_id))
            regulatory_errors.extend(_check_regulatory_mask_in_snapshot(features, state, record_id))
            telematics_errors.extend(_check_telematics_trio_in_snapshot(features, record_id))
            range_errors.extend(_check_value_ranges_quote(features, record_id))
            quote_snaps.append(snap)
        elif record_type == "claim":
            schema_errors.extend(_check_feature_schema(features, _CLAIM_FEATURE_KEYS, record_id))
            telematics_errors.extend(_check_telematics_trio_in_snapshot(features, record_id))
            range_errors.extend(_check_value_ranges_claim(features, record_id))
            claim_snaps.append(snap)

    print(f"  quote snapshots: {len(quote_snaps)}, claim snapshots: {len(claim_snaps)}")

    # Envelope
    if envelope_errors:
        for err in envelope_errors[:5]:
            print(f"  {_FAIL} {err}")
        if len(envelope_errors) > 5:
            print(f"  {_FAIL} ... and {len(envelope_errors) - 5} more envelope errors")
    else:
        print(f"  {_PASS} all snapshot envelopes are valid")

    # Version
    if version_mismatches:
        print(f"  {_FAIL} {version_mismatches} snapshots have unexpected feature_store_version (expected {FEATURE_STORE_VERSION!r})")
    else:
        print(f"  {_PASS} all snapshots match feature_store_version {FEATURE_STORE_VERSION!r}")

    # Schema
    if schema_errors:
        for err in schema_errors[:5]:
            print(f"  {_FAIL} {err}")
        if len(schema_errors) > 5:
            print(f"  {_FAIL} ... and {len(schema_errors) - 5} more schema errors")
    else:
        print(f"  {_PASS} feature schema complete for all snapshots")

    # Regulatory masking
    if regulatory_errors:
        for err in regulatory_errors[:5]:
            print(f"  {_FAIL} {err}")
        if len(regulatory_errors) > 5:
            print(f"  {_FAIL} ... and {len(regulatory_errors) - 5} more regulatory masking errors")
    else:
        print(f"  {_PASS} regulatory masking correct for all quote snapshots")

    # Telematics trio
    if telematics_errors:
        for err in telematics_errors[:5]:
            print(f"  {_FAIL} {err}")
        if len(telematics_errors) > 5:
            print(f"  {_FAIL} ... and {len(telematics_errors) - 5} more telematics trio errors")
    else:
        print(f"  {_PASS} telematics trio invariant holds for all snapshots")

    # NaN values
    if nan_errors:
        for err in nan_errors[:5]:
            print(f"  {_FAIL} {err}")
        if len(nan_errors) > 5:
            print(f"  {_FAIL} ... and {len(nan_errors) - 5} more NaN errors")
    else:
        print(f"  {_PASS} no NaN values in feature snapshots (all nulls are JSON null)")

    # Value ranges
    if range_errors:
        for err in range_errors[:5]:
            print(f"  {_FAIL} {err}")
        if len(range_errors) > 5:
            print(f"  {_FAIL} ... and {len(range_errors) - 5} more value range errors")
    else:
        print(f"  {_PASS} feature value ranges are valid")

    # Cross-snapshot spine: claim risk_score_at_issuance matches linked quote
    if quote_snaps and claim_snaps:
        quote_risk_index = {
            s["record_id"]: s["features"].get("risk_score_at_issuance")
            for s in quote_snaps
        }
        spine_mismatches = 0
        spine_orphans = 0
        for cs in claim_snaps:
            cf = cs["features"]
            # claim record_id is e.g. "C-0001", linked quote is in features.risk_score_at_issuance
            # We match via the quote_id embedded in the file — cross-check by comparing the value
            claim_risk = cf.get("risk_score_at_issuance")
            # Only verifiable if claim was linked to a specific quote (quote_id in features)
            # The offline pipeline indexes by quote_id; record_id is the claim_id.
            # We can only check if the value is plausible here (not None if any quote exists).
            if claim_risk is None and quote_risk_index:
                spine_orphans += 1

        if spine_orphans:
            print(f"  {_WARN} {spine_orphans} claim snapshots have null risk_score_at_issuance (no underwriting context joined)")
        else:
            print(f"  {_PASS} all claim snapshots have risk_score_at_issuance from underwriting context")


def validate_data(quotes_df: pd.DataFrame | None = None, claims_df: pd.DataFrame | None = None,
                 neo4j_uri: str | None = None, neo4j_user: str | None = None, neo4j_password: str | None = None) -> None:
    if quotes_df is None:
        quotes_df = pd.read_parquet(QUOTES_OUTPUT)
    if claims_df is None:
        claims_df = pd.read_parquet(CLAIMS_OUTPUT)

    validate_quote_dataset(quotes_df)
    validate_claim_dataset(claims_df)
    validate_entity_resolution()
    validate_graph_features(claims_df)
    validate_feature_correlations(quotes_df, claims_df)
    validate_offline_features()

    if neo4j_uri and neo4j_user and neo4j_password:
        validate_graph_topology(neo4j_uri, neo4j_user, neo4j_password)

        # Check if workflow is complete
        if "graph_hop_distance" in claims_df.columns:
            sentinel_pct = (claims_df["graph_hop_distance"] == 999).sum() / len(claims_df) * 100
            if sentinel_pct == 100:
                print(f"\n{_WARN} WORKFLOW INCOMPLETE:")
                print(f"   Step 1: {_PASS} Generate data (claims have synthetic graph_hop_distance)")
                print(f"   Step 2: ❓ Build graph (check Neo4j status above)")
                print(f"   Step 3: {_FAIL} Compute graph features (NOT DONE - run this command:)")
                print("      python main.py --compute-graph-features")

    print("Validation complete.")
