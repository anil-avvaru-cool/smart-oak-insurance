from __future__ import annotations

import pandas as pd
from neo4j import GraphDatabase

from data.config import CLAIMS_OUTPUT, QUOTES_OUTPUT
from data.states import US_STATE_ABBREVIATIONS


def validate_quote_dataset(quotes_df: pd.DataFrame) -> None:
    print("Validating quote dataset...")
    print(f"  records: {len(quotes_df)}")
    print(f"  quote state coverage: {quotes_df['state'].nunique()} states")
    print(f"  risk score range: {quotes_df['risk_score_at_issuance'].min():.3f} - {quotes_df['risk_score_at_issuance'].max():.3f}")
    invalid_states = quotes_df.loc[~quotes_df["state"].isin(US_STATE_ABBREVIATIONS), "state"]
    if len(invalid_states):
        print(f"  [FAIL] invalid state values found: {sorted(invalid_states.unique())}")
    else:
        print("  [PASS] state values are valid")
    if (quotes_df["credit_score"] < 500).any() or (quotes_df["credit_score"] > 850).any():
        print("  [FAIL] credit score out of range")
    else:
        print("  [PASS] credit score range looks healthy")


def validate_claim_dataset(claims_df: pd.DataFrame) -> None:
    print("Validating claim dataset...")
    print(f"  records: {len(claims_df)}")
    fraud_rate = claims_df[claims_df["is_fraud"] == True].shape[0] / max(1, len(claims_df))
    print(f"  fraud rate: {fraud_rate:.2%}")
    if fraud_rate < 0.15 or fraud_rate > 0.45:
        print("  [WARN] fraud rate is outside expected training range")
    else:
        print("  [PASS] fraud rate is within expected training range")
    if (claims_df["policy_inception_days"] < 0).any():
        print("  [FAIL] negative policy_inception_days found")
    else:
        print("  [PASS] policy inception values are valid")
    if (claims_df["reporting_delay_days"] < 0).any():
        print("  [FAIL] negative reporting_delay_days found")
    else:
        print("  [PASS] reporting delay values are valid")


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
            print(f"  [FAIL] >50% sentinel values indicate graph is not connected to fraud entities")
        elif sentinel_pct > 20:
            print(f"  [WARN] {sentinel_pct:.1f}% sentinel values may indicate incomplete graph coverage")
        else:
            print(f"  [PASS] graph connectivity looks healthy")
        
        non_sentinel = hop_distance_vals[hop_distance_vals != 999]
        if len(non_sentinel) > 0:
            print(f"  hop distance range (excluding 999): {non_sentinel.min()} - {non_sentinel.max()}")
            print(f"  hop distance median: {non_sentinel.median():.1f}")
    else:
        print("  [WARN] graph_hop_distance column not found in dataset")
    
    # Validate attorney_centrality_score column
    if "attorney_centrality_score" in claims_df.columns:
        centrality_vals = claims_df["attorney_centrality_score"]
        print(f"  attorney centrality range: {centrality_vals.min():.3f} - {centrality_vals.max():.3f}")
        if (centrality_vals < 0).any() or (centrality_vals > 1.0).any():
            print(f"  [FAIL] attorney centrality score out of expected [0, 1] range")
        else:
            print(f"  [PASS] attorney centrality scores in valid range")
    
    # Validate shared_attribute_count column
    if "shared_attribute_count" in claims_df.columns:
        shared_count = claims_df["shared_attribute_count"]
        print(f"  shared attribute count range: {shared_count.min()} - {shared_count.max()}")
        print(f"  shared attribute median: {shared_count.median():.1f}")
        if (shared_count < 0).any():
            print(f"  [FAIL] negative shared attribute counts found")
        else:
            print(f"  [PASS] shared attribute counts are valid")


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
                print(f"  [FAIL] GRAPH IS EMPTY - no nodes found in Neo4j")
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
                print(f"  [FAIL] NO RELATIONSHIPS found - graph is disconnected")
            else:
                print(f"  total relationships: {rel_count}")
            
            # Check for fraud claimants
            result = session.run("""
                MATCH (c:Claimant {is_fraud: true})
                RETURN count(*) as fraud_count
            """)
            
            fraud_count = result.single()['fraud_count'] if result.single() else 0
            print(f"  fraud claimants: {fraud_count}")
            
            if fraud_count == 0:
                print(f"  [FAIL] NO fraud claimants in graph (explains 999 sentinel values)")
            elif fraud_count < 5:
                print(f"  [WARN] very few fraud claimants ({fraud_count}), may limit path discovery")
            else:
                print(f"  [PASS] fraud claimants present for path analysis")
            
            # Check for isolated claimants (no connections)
            result = session.run("""
                MATCH (c:Claimant)
                WHERE NOT (c)-[]-()
                RETURN count(*) as isolated_count
            """)
            
            isolated = result.single()['isolated_count'] if result.single() else 0
            if isolated > 0:
                print(f"  [WARN] {isolated} isolated claimant nodes (no SHARES relationships)")
            else:
                print(f"  [PASS] all claimants have relationships")
        
        driver.close()
    except Exception as e:
        print(f"  [FAIL] error validating graph topology: {e}")
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
        print(f"  [FAIL] missing entity files: {missing_files}")
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
        print(f"  [FAIL] {null_vins} null VINs found")
    else:
        print(f"  [PASS] no null VINs ({len(vehicles_df)} vehicles)")

    # Person validation
    duplicate_persons = persons_df["person_id"].duplicated().sum()
    if duplicate_persons > 0:
        print(f"  [FAIL] {duplicate_persons} duplicate person_ids found")
    else:
        print(f"  [PASS] no duplicate person_ids ({len(persons_df)} unique persons)")

    # Address validation
    null_hashes = addresses_df["address_hash"].isnull().sum()
    if null_hashes > 0:
        print(f"  [FAIL] {null_hashes} null address hashes found")
    else:
        print(f"  [PASS] all addresses have hashes ({len(addresses_df)} unique addresses)")

    # Phone validation
    valid_phones = phones_df[phones_df["is_valid"]].shape[0]
    print(f"  valid phones: {valid_phones}/{len(phones_df)}")

    # Policy validation
    if (policies_df["inception_date"] > policies_df["expiration_date"]).any():
        print(f"  [FAIL] policies with expiration before inception found")
    else:
        print(f"  [PASS] policy dates are valid ({len(policies_df)} policies)")



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
        print(f"  [SKIP] {dataset} telematics trio (raw columns absent)")
        return

    if has_derived:
        # Feature-vector path: full trio consistency
        available = df["telematics_available"] == True
        enrolled_missing = df["telematics_enrolled_but_missing"] == True

        bad = (df.loc[available, _TELEMATICS_RAW_COLS].isnull().all(axis=1)).sum()
        if bad:
            print(f"  [FAIL] {dataset}: {bad} rows telematics_available=True but all raw signals null")
        else:
            print(f"  [PASS] {dataset} telematics: available=True implies at least one raw signal")

        bad = (df.loc[enrolled_missing, "telematics_available"] == True).sum()
        if bad:
            print(f"  [FAIL] {dataset}: {bad} rows enrolled_but_missing=True yet also available=True")
        else:
            print(f"  [PASS] {dataset} telematics: enrolled_but_missing=True implies available=False")

        non_enrolled = (~available) & (~enrolled_missing)
        if non_enrolled.any():
            bad = df.loc[non_enrolled, _TELEMATICS_RAW_COLS].notnull().any(axis=1).sum()
            if bad:
                print(f"  [FAIL] {dataset}: {bad} non-enrolled rows contain non-null raw telematics values")
            else:
                print(f"  [PASS] {dataset} telematics: non-enrolled rows have no raw signals")
    else:
        # Raw-data path: generator copies all 4 columns together, so each row must be
        # uniformly null (not enrolled) or uniformly non-null (enrolled with data).
        null_mask = df[_TELEMATICS_RAW_COLS].isnull()
        mixed = (null_mask.any(axis=1)) & (~null_mask.all(axis=1))
        bad = mixed.sum()
        n_enrolled = (~null_mask.all(axis=1)).sum()
        if bad:
            print(f"  [FAIL] {dataset}: {bad} rows have mixed null/non-null telematics signals (should be all-or-nothing)")
        else:
            print(f"  [PASS] {dataset} telematics: all-or-nothing consistency holds ({n_enrolled} enrolled rows)")


def _check_state_regulatory_compliance(quotes_df: pd.DataFrame) -> None:
    """Credit masking happens in build_quote_feature_vector, not in raw parquet.

    Raw data check: non-restricted states must always have a credit_score (generator
    bug if null). Restricted states having a non-null credit_score is expected — the
    feature builder masks them to None before serving.
    """
    from features.feature_definitions import CREDIT_RESTRICTED_STATES

    if "state" not in quotes_df.columns or "credit_score" not in quotes_df.columns:
        print("  [SKIP] state regulatory compliance (state/credit_score columns missing)")
        return

    restricted = quotes_df["state"].isin(CREDIT_RESTRICTED_STATES)
    n_restricted = restricted.sum()

    # Non-restricted states: generator must always populate credit_score
    unrestricted = ~restricted
    if unrestricted.any():
        bad = quotes_df.loc[unrestricted, "credit_score"].isnull().sum()
        n = unrestricted.sum()
        if bad:
            print(f"  [FAIL] {bad}/{n} non-restricted-state quotes have null credit_score (generator bug)")
        else:
            print(f"  [PASS] credit_score present for all {n} non-restricted-state quotes")

    # Restricted states: credit_score will be masked by the feature builder — having
    # a non-null value here is correct; masking to null happens at serving time.
    print(f"  info: {n_restricted} restricted-state quotes have raw credit_score (masked to null by feature builder)")


def _check_fraud_signal_directions(claims_df: pd.DataFrame) -> None:
    if "is_fraud" not in claims_df.columns:
        print("  [SKIP] fraud signal directions (is_fraud column missing)")
        return

    fraud = claims_df[claims_df["is_fraud"] == True]
    legit = claims_df[claims_df["is_fraud"] == False]
    if fraud.empty or legit.empty:
        print("  [FAIL] fraud signal directions: dataset lacks fraud or legit rows")
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
        tag = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1
            print(f"  [{tag}] {col}: fraud={fraud_mean:.3f} legit={legit_mean:.3f} (signal inverted)")
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
            print(f"  [WARN] {col}: near-zero correlation with is_fraud (r={r:.3f})")
        elif (r > 0) == (expected > 0):
            passes += 1
        else:
            fails += 1
            direction = "positive" if expected > 0 else "negative"
            print(f"  [FAIL] {col}: r={r:.3f} (expected {direction} sign)")
    print(f"  feature-fraud correlations: {passes} correct sign, {fails} wrong sign, {skips} skipped")


def _check_cross_dataset_spine(quotes_df: pd.DataFrame, claims_df: pd.DataFrame) -> None:
    if "quote_id" not in claims_df.columns or "quote_id" not in quotes_df.columns:
        print("  [SKIP] cross-dataset spine (quote_id column missing)")
        return

    claim_qids = set(claims_df["quote_id"].dropna().astype(str))
    quote_qids = set(quotes_df["quote_id"].dropna().astype(str))
    orphaned = claim_qids - quote_qids
    if orphaned:
        print(f"  [FAIL] {len(orphaned)} claim quote_ids not found in quotes dataset")
    else:
        print(f"  [PASS] all {len(claim_qids)} claim quote_ids exist in quotes dataset")

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
            print(f"  [FAIL] {bad} claims have risk_score_at_issuance differing from linked quote")
        else:
            print(f"  [PASS] risk_score_at_issuance consistent across {len(merged)} quote-claim pairs")


def _check_attorney_coherence(claims_df: pd.DataFrame) -> None:
    for col in ("attorney_present", "attorney_centrality_score"):
        if col not in claims_df.columns:
            print(f"  [SKIP] attorney coherence ({col} missing)")
            return

    with_attorney = claims_df[claims_df["attorney_present"] == True]
    if with_attorney.empty:
        print("  [WARN] attorney coherence: no attorney_present=True rows found")
        return

    zero_pct = (with_attorney["attorney_centrality_score"] == 0).mean() * 100
    if zero_pct > 20:
        print(f"  [WARN] {zero_pct:.1f}% of attorney_present=True rows have centrality_score=0")
    else:
        print(f"  [PASS] attorney coherence: {zero_pct:.1f}% of attorney rows have zero centrality")

    no_attorney = claims_df[claims_df["attorney_present"] == False]
    if not no_attorney.empty:
        high_pct = (no_attorney["attorney_centrality_score"] > 0.3).mean() * 100
        if high_pct > 10:
            print(f"  [WARN] {high_pct:.1f}% of attorney_present=False rows have centrality_score > 0.3")


def _check_graph_sentinel_coherence(claims_df: pd.DataFrame) -> None:
    if "graph_hop_distance" not in claims_df.columns or "is_fraud" not in claims_df.columns:
        return

    fraud = claims_df[claims_df["is_fraud"] == True]
    legit = claims_df[claims_df["is_fraud"] == False]

    if not fraud.empty:
        fraud_sentinel_pct = (fraud["graph_hop_distance"] == 999).mean() * 100
        if fraud_sentinel_pct > 30:
            print(f"  [FAIL] {fraud_sentinel_pct:.1f}% of fraud claims have sentinel hop_distance=999 (graph poorly connected)")
        else:
            print(f"  [PASS] fraud claims sentinel rate: {fraud_sentinel_pct:.1f}%")

    if not legit.empty:
        legit_sentinel_pct = (legit["graph_hop_distance"] == 999).mean() * 100
        print(f"  info: legit claims sentinel rate: {legit_sentinel_pct:.1f}% (expect > fraud rate)")
        if not fraud.empty and legit_sentinel_pct < (fraud["graph_hop_distance"] == 999).mean() * 100:
            print(f"  [WARN] legit sentinel rate is lower than fraud rate (hop_distance signal may be inverted)")


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

    if neo4j_uri and neo4j_user and neo4j_password:
        validate_graph_topology(neo4j_uri, neo4j_user, neo4j_password)

        # Check if workflow is complete
        if "graph_hop_distance" in claims_df.columns:
            sentinel_pct = (claims_df["graph_hop_distance"] == 999).sum() / len(claims_df) * 100
            if sentinel_pct == 100:
                print("\n[WARN] WORKFLOW INCOMPLETE:")
                print("   Step 1: [PASS] Generate data (claims have synthetic graph_hop_distance)")
                print("   Step 2: ❓ Build graph (check Neo4j status above)")
                print("   Step 3: [FAIL] Compute graph features (NOT DONE - run this command:)")
                print("      python main.py --compute-graph-features")

    print("Validation complete.")

