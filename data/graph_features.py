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

    logger.info("Computing attorney centrality...")

    rows = list(
        tx.run(
            """
            MATCH (a:Attorney)
            OPTIONAL MATCH (a)--(n)
            WITH a, COUNT(n) AS deg
            RETURN a.id AS id, deg
            """
        )
    )

    if not rows:
        logger.warning("No attorneys found")
        return

    max_deg = max(r["deg"] for r in rows) or 1

    for row in rows:

        centrality = float(row["deg"]) / float(max_deg)

        tx.run(
            """
            MATCH (a:Attorney {id: $id})
            SET a.centrality = $centrality
            """,
            id=row["id"],
            centrality=centrality,
        )

    logger.info("Attorney centrality complete")


def _get_all_claim_ids(session: Session) -> List[str]:

    rows = session.run(
        """
        MATCH (cl:Claim)
        RETURN cl.id AS id
        ORDER BY cl.id
        """
    )

    return [r["id"] for r in rows]


def _compute_features_for_batch(
    tx,
    claim_ids: List[str],
    max_hops: int = DEFAULT_MAX_HOPS,
) -> None:

    logger.info(
        "Computing graph features for %s claims",
        len(claim_ids),
    )

    for claim_id in claim_ids:

        # --------------------------------------------------------------
        # Shared attribute count
        # --------------------------------------------------------------

        shared_row = tx.run(
            """
            MATCH (cl:Claim {id: $claim_id})
                  -[:SHARES]->
                  (e:Entity)
                  <-[:SHARES]-
                  (other:Claim)

            WHERE other.id <> cl.id

            RETURN count(DISTINCT e) AS shared_count
            """,
            claim_id=claim_id,
        ).single()

        shared_count = (
            int(shared_row["shared_count"])
            if shared_row and shared_row["shared_count"] is not None
            else 0
        )

        # --------------------------------------------------------------
        # Attorney centrality
        # --------------------------------------------------------------

        attorney_row = tx.run(
            """
            MATCH (cl:Claim {id: $claim_id})

            OPTIONAL MATCH
                (cl)<-[:FILED]-(c:Claimant)
                -[:REPRESENTED_BY]->
                (a:Attorney)

            RETURN max(a.centrality) AS centrality
            """,
            claim_id=claim_id,
        ).single()

        attorney_centrality = (
            float(attorney_row["centrality"])
            if attorney_row
            and attorney_row["centrality"] is not None
            else 0.0
        )

        # --------------------------------------------------------------
        # Fraud hop distance
        # --------------------------------------------------------------

        hop_row = tx.run(
            f"""
            MATCH (c:Claim {{id: $claim_id}}),
                  (f:Claim)

            WHERE f.is_fraud = true
              AND c.id <> f.id

            MATCH p = shortestPath(
                (c)-[*..{max_hops}]-(f)
            )

            RETURN min(length(p)) AS dist
            """,
            claim_id=claim_id,
        ).single()

        graph_hop_distance = (
            int(hop_row["dist"])
            if hop_row and hop_row["dist"] is not None
            else SENTINEL_HOP
        )

        # --------------------------------------------------------------
        # Persist features
        # --------------------------------------------------------------

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

    logger.info("Batch complete")


def compute_graph_features(
    claims_path: Path,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    max_hops: int = DEFAULT_MAX_HOPS,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:

    pd.read_parquet(claims_path)

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

        session.execute_write(
            lambda tx: _compute_attorney_centrality(tx)
        )

        claim_ids = _get_all_claim_ids(session)

        total = len(claim_ids)

        logger.info("Found %s claims", total)

        for start in range(0, total, batch_size):

            end = min(start + batch_size, total)

            logger.info(
                "Processing claims %s..%s / %s",
                start + 1,
                end,
                total,
            )

            batch = claim_ids[start:end]

            session.execute_write(
                lambda tx, ids=batch, hops=max_hops:
                _compute_features_for_batch(
                    tx,
                    ids,
                    hops,
                )
            )

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