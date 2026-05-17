from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd
from neo4j import GraphDatabase, Session

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.getLogger("neo4j").setLevel(logging.WARNING)

SENTINEL_HOP = 999
DEFAULT_MAX_HOPS = 6
BATCH_SIZE = 1000


def _compute_attorney_centrality(tx) -> None:
    """Fetch all attorney degrees in one query, then bulk-SET centrality via UNWIND."""
    logger.info("Computing attorney centrality...")

    rows = list(
        tx.run(
            """
            MATCH (a:Attorney)
            OPTIONAL MATCH (a)--(n)
            WITH a, count(n) AS deg
            RETURN a.id AS id, deg
            """
        )
    )

    if not rows:
        logger.warning("No attorneys found")
        return

    max_deg = max(r["deg"] for r in rows) or 1

    data = [
        {"id": r["id"], "centrality": float(r["deg"]) / float(max_deg)}
        for r in rows
    ]

    tx.run(
        """
        UNWIND $rows AS row
        MATCH (a:Attorney {id: row.id})
        SET a.centrality = row.centrality
        """,
        rows=data,
    )

    logger.info("Attorney centrality complete (%d attorneys)", len(data))


def _compute_shared_attribute_counts(session: Session) -> None:
    """Bulk-compute shared entity counts for all claims in two queries."""
    logger.info("Computing shared attribute counts...")

    session.run(
        """
        MATCH (cl:Claim)-[:SHARES]->(e:Entity)<-[:SHARES]-(other:Claim)
        WHERE cl <> other
        WITH cl, count(DISTINCT e) AS cnt
        SET cl.shared_attribute_count = cnt
        """
    )
    session.run(
        """
        MATCH (cl:Claim)
        WHERE cl.shared_attribute_count IS NULL
        SET cl.shared_attribute_count = 0
        """
    )

    logger.info("Shared attribute counts complete")


def _compute_per_claim_attorney_centrality(session: Session) -> None:
    """Single query: propagate max attorney centrality to each Claim node."""
    logger.info("Computing per-claim attorney centrality...")

    session.run(
        """
        MATCH (cl:Claim)
        OPTIONAL MATCH (cl)<-[:FILED]-(c:Claimant)-[:REPRESENTED_BY]->(a:Attorney)
        WITH cl, max(a.centrality) AS centrality
        SET cl.attorney_centrality = CASE
            WHEN centrality IS NULL THEN 0.0
            ELSE centrality
        END
        """
    )

    logger.info("Per-claim attorney centrality complete")


def _compute_hop_distances(session: Session, max_hops: int) -> None:
    """
    Flood-fill from all fraud seed nodes simultaneously.

    One shortestPath query over all (fraud_seed, claim) pairs replaces the
    original N-per-claim shortestPath loop.  Neo4j executes the traversal
    server-side; Python sees a single round trip.
    """
    logger.info(
        "Computing fraud hop distances (flood-fill, max_hops=%d)...", max_hops
    )

    session.run(
        f"""
        MATCH (f:Claim {{is_fraud: true}}), (cl:Claim)
        WHERE f <> cl
        MATCH p = shortestPath((cl)-[*..{max_hops}]-(f))
        WITH cl, min(length(p)) AS hop_dist
        SET cl.graph_hop_distance = hop_dist
        """
    )

    # Claims with no path to any fraud seed get the sentinel value.
    session.run(
        f"""
        MATCH (cl:Claim)
        WHERE cl.graph_hop_distance IS NULL
        SET cl.graph_hop_distance = {SENTINEL_HOP}
        """
    )

    logger.info("Hop distance flood-fill complete")


def compute_graph_features(
    claims_path: Path,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    max_hops: int = DEFAULT_MAX_HOPS,
    batch_size: int = BATCH_SIZE,  # kept for API compatibility; no longer drives a loop
) -> pd.DataFrame:

    if not claims_path.exists():
        raise FileNotFoundError(
            f"Claims file not found: {claims_path}. Run --generate-data first."
        )
    expected_count = len(pd.read_parquet(claims_path))
    logger.info("Claims parquet: %d records to enrich", expected_count)

    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password),
    )

    with driver.session() as session:

        logger.info("Starting graph feature computation...")

        fraud_count = session.run(
            """
            MATCH (f:Claim)
            WHERE f.is_fraud = true
            RETURN count(f) AS fraud_count
            """
        ).single()["fraud_count"]

        logger.info("Fraud seed claims: %s", fraud_count)

        if fraud_count == 0:
            logger.warning(
                "No fraud claims found. "
                "All hop distances will become sentinel."
            )

        # 1. Attorney node centrality: 1 fetch + 1 UNWIND SET
        session.execute_write(lambda tx: _compute_attorney_centrality(tx))

        # 2. Shared attribute count: 2 queries (SET + NULL fill)
        _compute_shared_attribute_counts(session)

        # 3. Per-claim attorney centrality: 1 query (reads Attorney.centrality set above)
        _compute_per_claim_attorney_centrality(session)

        # 4. Hop distances: 1 flood-fill + 1 sentinel fill
        _compute_hop_distances(session, max_hops)

        # 5. Read all enriched features back to Python
        total = session.run(
            "MATCH (cl:Claim) RETURN count(cl) AS cnt"
        ).single()["cnt"]
        logger.info("Reading features for %s claims", total)

        rows = session.run(
            """
            MATCH (cl:Claim)

            RETURN
                cl.id AS claim_id,
                cl.shared_attribute_count AS shared_attribute_count,
                cl.graph_hop_distance AS graph_hop_distance,
                cl.attorney_centrality AS attorney_centrality

            ORDER BY cl.id
            """
        )

        features = []

        for r in rows:

            features.append(
                {
                    "claim_id": r["claim_id"],
                    "shared_attribute_count": (
                        int(r["shared_attribute_count"])
                        if r["shared_attribute_count"] is not None
                        else 0
                    ),
                    "graph_hop_distance": (
                        int(r["graph_hop_distance"])
                        if r["graph_hop_distance"] is not None
                        else SENTINEL_HOP
                    ),
                    "attorney_centrality_score": (
                        float(r["attorney_centrality"])
                        if r["attorney_centrality"] is not None
                        else 0.0
                    ),
                }
            )

    driver.close()

    return pd.DataFrame(features)
