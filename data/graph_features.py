from __future__ import annotations

import pandas as pd
from neo4j import GraphDatabase
from pathlib import Path


def compute_graph_features(claims_path: Path, neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> pd.DataFrame:
    """Compute graph-based features for each claim by querying Neo4j."""
    claims_df = pd.read_parquet(claims_path)
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    features = []
    
    with driver.session() as session:
        for _, claim in claims_df.iterrows():
            claim_id = claim['claim_id']
            claimant_id = f"claimant_{claim_id}"
            
            # Hop distance to known fraud claimant
            result = session.run("""
                MATCH (c:Claimant {id: $claimant_id})
                MATCH (f:Claimant {is_fraud: true})
                WHERE c <> f
                MATCH path = shortestPath((c)-[*]-(f))
                RETURN length(path) as hop_distance
                ORDER BY hop_distance
                LIMIT 1
            """, claimant_id=claimant_id)
            
            hop_distance = result.single()
            hop_distance = hop_distance['hop_distance'] if hop_distance else 999  # large number if no path
            
            # Attorney centrality (degree)
            attorney_centrality = 0
            if claim['attorney_present']:
                attorney_id = f"attorney_{claim_id}"
                result = session.run("""
                    MATCH (a:Attorney {id: $attorney_id})<-[:REPRESENTED_BY]-(c:Claimant)
                    RETURN count(c) as degree
                """, attorney_id=attorney_id)
                degree = result.single()
                attorney_centrality = degree['degree'] if degree else 0
            
            # Shared attribute count
            result = session.run("""
                MATCH (c:Claimant {id: $claimant_id})-[:SHARES]->(e:Entity)<-[:SHARES]-(other:Claimant)
                WHERE c <> other
                RETURN count(DISTINCT other) as shared_count
            """, claimant_id=claimant_id)
            
            shared_count = result.single()['shared_count']
            
            features.append({
                'claim_id': claim_id,
                'graph_hop_distance': hop_distance,
                'attorney_centrality_score': attorney_centrality / 100.0,  # normalize
                'shared_attribute_count': shared_count
            })
    
    driver.close()
    return pd.DataFrame(features)