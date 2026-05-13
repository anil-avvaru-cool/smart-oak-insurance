from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd
from neo4j import GraphDatabase, Session

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("neo4j").setLevel(logging.WARNING)

SENTINEL_HOP = 999
DEFAULT_MAX_HOPS = 6
BATCH_SIZE = 1000


def _compute_attorney_centrality(tx) -> None:
    """
    Compute normalized degree centrality for Attorney nodes and store as a.centrality.
    Uses COUNT instead of size(pattern) to be compatible with Neo4j 5+.
    """
    logger.info("Computing attorney degree centrality...")
    # Get degree per attorney using OPTIONAL MATCH and COUNT
    res = tx.run(
        """
        MATCH (a:Attorney)
        OPTIONAL MATCH (a)--(n)
        WITH a, COUNT(n) AS deg
        RETURN a.id AS id, deg
        """
    )
    rows = [r for r in res]
    if not rows:
        logger.info("No attorneys found; skipping centrality.")
        return

    degrees = [(r["id"], r["deg"]) for r in rows]
    max_deg = max(d for _, d in degrees) or 1
    for att_id, deg in degrees:
        centrality = float(deg) / float(max_deg)
        tx.run(
            """
            MATCH (a:Attorney {id: $id})
            SET a.centrality = $centrality
            """,
            id=att_id,
            centrality=centrality,
        )
    logger.info("Attorney centrality stored.")


def _get_all_claim_ids(session: Session) -> List[str]:
    res = session.run("MATCH (cl:Claim) RETURN cl.id AS id ORDER BY cl.id")
    return [r["id"] for r in res]


def _compute_features_for_batch(tx, claim_ids: List[str], max_hops: int = DEFAULT_MAX_HOPS) -> None:
    logger.info(f"Computing features for batch of {len(claim_ids)} claims...")
    for claim_id in claim_ids:
        shared_count_row = tx.run(
            """
            MATCH (cl:Claim {id: $claim_id})-[:SHARES]->(e:Entity)
            RETURN count(DISTINCT e) AS shared_count
            """,
            claim_id=claim_id,
        ).single()
        shared_count = int(shared_count_row["shared_count"]) if shared_count_row and shared_count_row["shared_count"] is not None else 0

        att_row = tx.run(
            """
            MATCH (cl:Claim {id: $claim_id})
            OPTIONAL MATCH (cl)<-[:FILED]-(c:Claimant)-[:REPRESENTED_BY]->(a:Attorney)
            RETURN a.centrality AS centrality
            LIMIT 1
            """,
            claim_id=claim_id,
        ).single()
        attorney_centrality = float(att_row["centrality"]) if att_row and att_row["centrality"] is not None else 0.0

        hop_row = tx.run(
            f"""
            MATCH (c:Claim {{id: $claim_id}}), (f:Claim)
            WHERE f.is_fraud = true AND c <> f
            WITH c, f, shortestPath((c)-[*..{max_hops}]-(f)) AS p
            WHERE p IS NOT NULL
            RETURN min(length(p)) AS dist
            """,
            claim_id=claim_id,
        ).single()

        if hop_row is None or hop_row.get("dist") is None:
            graph_hop_distance = SENTINEL_HOP
        else:
            graph_hop_distance = int(hop_row["dist"])

        tx.run(
            """
            MATCH (cl:Claim {id: $claim_id})
            SET cl.shared_attribute_count = $shared_count,
                cl.graph_hop_distance = $graph_hop_distance,
                cl.attorney_centrality = $attorney_centrality
            """,
            claim_id=claim_id,
            shared_count=shared_count,
            graph_hop_distance=graph_hop_distance,
            attorney_centrality=attorney_centrality,
        )

    logger.info("Batch feature computation complete.")


def compute_graph_features(claims_path: Path, neo4j_uri: str, neo4j_user: str, neo4j_password: str, max_hops: int = DEFAULT_MAX_HOPS, batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    claims_df = pd.read_parquet(claims_path)
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    with driver.session() as session:
        logger.info("Starting graph feature computation...")

        fraud_count = session.run("MATCH (f:Claim) WHERE f.is_fraud = true RETURN count(f) AS fraud_count").single().get("fraud_count", 0)
        logger.info(f"Fraud seed claims: {fraud_count}")
        if fraud_count == 0:
            logger.warning("No fraud seed claims found. graph_hop_distance will be sentinel for all claims.")

        # compute attorney centrality using execute_write to persist centrality
        session.execute_write(lambda tx: _compute_attorney_centrality(tx))

        claim_ids = _get_all_claim_ids(session)
        total = len(claim_ids)
        logger.info(f"Found {total} claims to compute features for.")

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = claim_ids[start:end]
            logger.info(f"Processing claims {start + 1}..{end} / {total}")
            session.execute_write(lambda tx, ids=batch, mh=max_hops: _compute_features_for_batch(tx, ids, mh))

        rows = session.run(
            """
            MATCH (cl:Claim)
            RETURN cl.id AS claim_id, cl.shared_attribute_count AS shared_attribute_count,
                   cl.graph_hop_distance AS graph_hop_distance, cl.attorney_centrality AS attorney_centrality
            ORDER BY cl.id
            """
        )
        features = []
        for r in rows:
            features.append({
                "claim_id": r["claim_id"],
                "shared_attribute_count": int(r["shared_attribute_count"]) if r["shared_attribute_count"] is not None else 0,
                "graph_hop_distance": int(r["graph_hop_distance"]) if r["graph_hop_distance"] is not None else SENTINEL_HOP,
                "attorney_centrality_score": float(r["attorney_centrality"]) if r["attorney_centrality"] is not None else 0.0,
            })

    driver.close()
    return pd.DataFrame(features)
