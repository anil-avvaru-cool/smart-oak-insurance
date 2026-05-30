from __future__ import annotations

import logging
from pathlib import Path

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


def _compute_hop_distances(session: Session, max_hops: int, seed_ids: list[str] | None = None) -> None:
    """
    Multi-source BFS from fraud seed claims simultaneously.

    Seeds fraud Claim nodes at distance 0 — either a caller-supplied subset
    (temporal seeding to avoid label leakage) or all is_fraud=true claims.
    Expands one hop at a time across the full graph. O(V+E) total.
    """
    logger.info(
        "Computing fraud hop distances (multi-source BFS, max_hops=%d)...", max_hops
    )

    # Clean up any leftover tmp_hop from a previous interrupted run.
    session.run("MATCH (n) WHERE n.tmp_hop IS NOT NULL REMOVE n.tmp_hop")

    if seed_ids is not None:
        # Temporal seeding: only claims confirmed fraudulent before the cutoff date.
        # Prevents graph_hop_distance=0 from leaking the is_fraud label for claims
        # in the evaluation period.
        seed_count = session.run(
            """
            MATCH (f:Claim)
            WHERE f.id IN $ids
            SET f.tmp_hop = 0
            RETURN count(f) AS cnt
            """,
            ids=seed_ids,
        ).single()["cnt"]
        logger.info("BFS: seeded %d fraud claims (temporal subset) at distance 0", seed_count)
    else:
        seed_count = session.run(
            """
            MATCH (f:Claim {is_fraud: true})
            SET f.tmp_hop = 0
            RETURN count(f) AS cnt
            """
        ).single()["cnt"]
        logger.info("BFS: seeded %d fraud claims at distance 0", seed_count)

    for hop in range(1, max_hops + 1):
        result = session.run(
            """
            MATCH (known)--(neighbor)
            WHERE known.tmp_hop = $prev_hop AND neighbor.tmp_hop IS NULL
            SET neighbor.tmp_hop = $hop
            RETURN count(neighbor) AS cnt
            """,
            prev_hop=hop - 1,
            hop=hop,
        )
        new_nodes = result.single()["cnt"]
        logger.info("BFS hop %d: %d new nodes reached", hop, new_nodes)
        if new_nodes == 0:
            break

    # Assign graph_hop_distance to Claims from BFS result; sentinel for unreachable.
    session.run(
        f"""
        MATCH (cl:Claim)
        SET cl.graph_hop_distance = coalesce(cl.tmp_hop, {SENTINEL_HOP})
        REMOVE cl.tmp_hop
        """
    )

    # Remove tmp_hop from all remaining non-Claim nodes.
    session.run("MATCH (n) WHERE n.tmp_hop IS NOT NULL REMOVE n.tmp_hop")

    # Null out distance for seed claims: they were assigned 0 because they seeded the BFS,
    # not because they are "0 hops from fraud" in a meaningful sense. Keeping 0 makes the
    # feature a direct label encoding (distance=0 ↔ is_fraud=True) that XGBoost collapses onto.
    if seed_ids is not None:
        session.run(
            f"MATCH (f:Claim) WHERE f.id IN $ids SET f.graph_hop_distance = {SENTINEL_HOP}",
            ids=seed_ids,
        )
    else:
        session.run(
            f"MATCH (f:Claim {{is_fraud: true}}) SET f.graph_hop_distance = {SENTINEL_HOP}"
        )
    logger.info("Seed claim hop distances reset to sentinel (prevents label encoding)")

    logger.info("Hop distance flood-fill complete")


def compute_graph_features(
    claims_path: Path,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    max_hops: int = DEFAULT_MAX_HOPS,
    batch_size: int = BATCH_SIZE,  # kept for API compatibility; no longer drives a loop
    seed_cutoff_dt: str | None = None,
) -> pd.DataFrame:

    if not claims_path.exists():
        raise FileNotFoundError(
            f"Claims file not found: {claims_path}. Run --generate-data first."
        )
    claims_raw = pd.read_parquet(claims_path)
    expected_count = len(claims_raw)
    logger.info("Claims parquet: %d records to enrich", expected_count)

    seed_ids: list[str] | None = None
    if seed_cutoff_dt is not None:
        cutoff = pd.Timestamp(seed_cutoff_dt)
        loss_dt = pd.to_datetime(claims_raw["loss_event_datetime"])
        if loss_dt.dt.tz is not None and cutoff.tz is None:
            cutoff = cutoff.tz_localize("UTC")
        mask = claims_raw["is_fraud"] & (loss_dt < cutoff)
        seed_ids = claims_raw.loc[mask, "claim_id"].tolist()
        logger.info(
            "Temporal BFS seeding: %d fraud claims before %s (of %d total fraud)",
            len(seed_ids),
            seed_cutoff_dt,
            int(claims_raw["is_fraud"].sum()),
        )

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

        # 4. Hop distances: seed from temporal subset if cutoff provided, else all fraud
        _compute_hop_distances(session, max_hops, seed_ids=seed_ids)

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
