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

