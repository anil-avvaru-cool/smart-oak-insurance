from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path
from typing import Dict

import pandas as pd
from faker import Faker
from neo4j import GraphDatabase, Session
from neo4j.exceptions import Neo4jError

from data.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.getLogger("neo4j").setLevel(logging.WARNING)


def _create_constraints(session: Session) -> None:
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Claimant) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Attorney) REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Vehicle) REQUIRE v.vin IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (cl:Claim) REQUIRE cl.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    ]

    for q in constraints:
        try:
            session.run(q)
        except Neo4jError as e:
            logger.debug("Constraint issue: %s -- %s", q, e)


def _normalize_key(value: str) -> str:
    if value is None:
        return ""

    key = value.strip().lower()
    key = (
        key.replace("(", "")
        .replace(")", "")
        .replace("-", "")
        .replace(".", "")
        .replace(" ", "")
    )

    return key


def build_graph_from_claims(
    claims_path: Path,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    batch_size: int = 1000,
) -> None:

    claims_df = pd.read_parquet(claims_path)
    logger.info("Loaded %s claims", len(claims_df))

    # Load resolved vehicle entities (DEC-011: graph_builder loads nodes/edges only)
    vehicles_path = RAW_DATA_DIR / "entities" / "vehicles.parquet"
    if not vehicles_path.exists():
        raise FileNotFoundError(
            f"Resolved vehicle entities not found at {vehicles_path}. "
            "Run --resolve-entities before --build-graph."
        )
    vehicles_df = pd.read_parquet(vehicles_path)
    quote_to_vehicle: Dict[str, Dict] = (
        vehicles_df.set_index("quote_id")[["vin", "make", "model", "year"]]
        .to_dict("index")
    )
    logger.info("Loaded %s resolved vehicles", len(vehicles_df))

    faker = Faker()

    Faker.seed(42)
    random.seed(42)

    # ------------------------------------------------------------------
    # Shared entity pools
    # ------------------------------------------------------------------

    shared_phone_pool = [
        faker.phone_number()
        for _ in range(300)
    ]

    shared_address_pool = [
        faker.address().replace("\n", ", ")
        for _ in range(200)
    ]

    fraud_phone_pool = shared_phone_pool[:40]
    fraud_address_pool = shared_address_pool[:25]

    fraud_attorney_pool = [
        f"fraud_attorney_{i}"
        for i in range(20)
    ]

    legit_attorney_pool = [
        f"legit_attorney_{i}"
        for i in range(200)
    ]

    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password),
    )

    with driver.session() as session:

        logger.info("Clearing Neo4j graph...")
        session.run("MATCH (n) DETACH DELETE n")

        _create_constraints(session)

        # Cross-batch entity dedup: key -> entity_id (deterministic hash)
        entities: Dict[str, str] = {}

        def _process_batch(tx, batch_df: pd.DataFrame) -> None:
            """
            Build parameter lists for all node/relationship types, then fire
            one UNWIND Cypher per type.  Reduces ~7-9*N round trips to 7-9
            regardless of batch size.
            """
            claimants = []
            claims = []
            filed_rels = []
            attorneys: Dict[str, str] = {}   # id -> name (dedup within batch)
            rep_by_rels = []
            vehicles: Dict[str, Dict] = {}   # vin -> props (dedup within batch)
            involves_rels = []
            new_entities = []
            claimant_shares = []
            claim_shares = []

            for _, claim in batch_df.iterrows():

                claim_id = claim["claim_id"]
                is_fraud = bool(claim["is_fraud"])
                claimant_id = f"claimant_{claim_id}"

                # ----------------------------------------------------------
                # Shared entity reuse logic
                # ----------------------------------------------------------

                if is_fraud:

                    claimant_phone = (
                        random.choice(fraud_phone_pool)
                        if random.random() < 0.85
                        else faker.phone_number()
                    )
                    claimant_address = (
                        random.choice(fraud_address_pool)
                        if random.random() < 0.80
                        else faker.address().replace("\n", ", ")
                    )
                    attorney_id = (
                        random.choice(fraud_attorney_pool)
                        if claim["attorney_present"]
                        else None
                    )

                else:

                    claimant_phone = (
                        random.choice(shared_phone_pool)
                        if random.random() < 0.03
                        else faker.phone_number()
                    )
                    claimant_address = (
                        random.choice(shared_address_pool)
                        if random.random() < 0.05
                        else faker.address().replace("\n", ", ")
                    )
                    attorney_id = (
                        random.choice(legit_attorney_pool)
                        if (
                            claim["attorney_present"]
                            and random.random() < 0.15
                        )
                        else (
                            f"attorney_{claim_id}"
                            if claim["attorney_present"]
                            else None
                        )
                    )

                claimants.append(
                    {
                        "id": claimant_id,
                        "name": faker.name(),
                        "phone": claimant_phone,
                        "address": claimant_address,
                        "is_fraud": is_fraud,
                    }
                )
                claims.append(
                    {
                        "id": claim_id,
                        "is_fraud": is_fraud,
                        "state": claim["state"],
                    }
                )
                filed_rels.append(
                    {"claimant_id": claimant_id, "claim_id": claim_id}
                )

                if attorney_id:
                    if attorney_id not in attorneys:
                        attorneys[attorney_id] = faker.name()
                    rep_by_rels.append(
                        {
                            "claimant_id": claimant_id,
                            "attorney_id": attorney_id,
                        }
                    )

                vehicle_info = quote_to_vehicle.get(str(claim["quote_id"]))
                if vehicle_info is not None:
                    vin = vehicle_info["vin"]
                    if vin not in vehicles:
                        vehicles[vin] = vehicle_info
                    involves_rels.append(
                        {"claim_id": claim_id, "vin": vin}
                    )
                else:
                    logger.warning(
                        "No resolved vehicle for quote_id %s — Vehicle node skipped",
                        claim["quote_id"],
                    )

                for attr, raw_value in [
                    ("phone", claimant_phone),
                    ("address", claimant_address),
                ]:
                    key = f"{attr}:{_normalize_key(raw_value)}"
                    if key not in entities:
                        entity_id = (
                            "e_"
                            + hashlib.sha256(key.encode()).hexdigest()[:16]
                        )
                        entities[key] = entity_id
                    entity_id = entities[key]
                    new_entities.append(
                        {"id": entity_id, "type": attr, "value": raw_value}
                    )
                    claimant_shares.append(
                        {
                            "claimant_id": claimant_id,
                            "entity_id": entity_id,
                            "type": attr,
                        }
                    )
                    claim_shares.append(
                        {
                            "claim_id": claim_id,
                            "entity_id": entity_id,
                            "type": attr,
                        }
                    )

            # --------------------------------------------------------------
            # UNWIND writes — one Cypher per node/relationship type
            # --------------------------------------------------------------

            tx.run(
                """
                UNWIND $rows AS row
                MERGE (c:Claimant {id: row.id})
                SET c.name     = row.name,
                    c.phone    = row.phone,
                    c.address  = row.address,
                    c.is_fraud = row.is_fraud
                """,
                rows=claimants,
            )

            tx.run(
                """
                UNWIND $rows AS row
                MERGE (cl:Claim {id: row.id})
                SET cl.is_fraud = row.is_fraud,
                    cl.state    = row.state
                """,
                rows=claims,
            )

            tx.run(
                """
                UNWIND $rows AS row
                MATCH (c:Claimant {id: row.claimant_id}),
                      (cl:Claim   {id: row.claim_id})
                MERGE (c)-[:FILED]->(cl)
                """,
                rows=filed_rels,
            )

            if attorneys:
                tx.run(
                    """
                    UNWIND $rows AS row
                    MERGE (a:Attorney {id: row.id})
                    ON CREATE SET a.name = row.name
                    """,
                    rows=[{"id": k, "name": v} for k, v in attorneys.items()],
                )
                tx.run(
                    """
                    UNWIND $rows AS row
                    MATCH (c:Claimant {id: row.claimant_id}),
                          (a:Attorney {id: row.attorney_id})
                    MERGE (c)-[:REPRESENTED_BY]->(a)
                    """,
                    rows=rep_by_rels,
                )

            if vehicles:
                tx.run(
                    """
                    UNWIND $rows AS row
                    MERGE (v:Vehicle {vin: row.vin})
                    SET v.make  = row.make,
                        v.model = row.model,
                        v.year  = row.year
                    """,
                    rows=[
                        {"vin": vin, **info}
                        for vin, info in vehicles.items()
                    ],
                )
                tx.run(
                    """
                    UNWIND $rows AS row
                    MATCH (cl:Claim   {id:  row.claim_id}),
                          (v:Vehicle  {vin: row.vin})
                    MERGE (cl)-[:INVOLVES]->(v)
                    """,
                    rows=involves_rels,
                )

            if new_entities:
                tx.run(
                    """
                    UNWIND $rows AS row
                    MERGE (e:Entity {id: row.id})
                    SET e.type  = row.type,
                        e.value = row.value
                    """,
                    rows=new_entities,
                )

            tx.run(
                """
                UNWIND $rows AS row
                MATCH (c:Claimant {id: row.claimant_id}),
                      (e:Entity   {id: row.entity_id})
                MERGE (c)-[:SHARES {type: row.type}]->(e)
                """,
                rows=claimant_shares,
            )

            tx.run(
                """
                UNWIND $rows AS row
                MATCH (cl:Claim {id: row.claim_id}),
                      (e:Entity  {id: row.entity_id})
                MERGE (cl)-[:SHARES {type: row.type}]->(e)
                """,
                rows=claim_shares,
            )

        total = len(claims_df)

        for start in range(0, total, batch_size):

            end = min(start + batch_size, total)

            logger.info(
                "Importing claims %s..%s / %s",
                start + 1,
                end,
                total,
            )

            batch_df = claims_df.iloc[start:end]

            session.execute_write(
                lambda tx, df=batch_df: _process_batch(tx, df)
            )

        # --------------------------------------------------------------
        # Validation stats
        # --------------------------------------------------------------

        shares_count = session.run(
            """
            MATCH ()-[r:SHARES]->()
            RETURN count(r) AS shares
            """
        ).single()["shares"]

        connected_entities = session.run(
            """
            MATCH (e:Entity)<-[:SHARES]-(c:Claim)
            WITH e, count(DISTINCT c) AS claim_count
            WHERE claim_count > 1
            RETURN count(e) AS connected_entities
            """
        ).single()["connected_entities"]

        logger.info("✅ Graph build complete")
        logger.info("Claims: %s", total)
        logger.info("Unique entities: %s", len(entities))
        logger.info("SHARES relationships: %s", shares_count)
        logger.info(
            "Entities shared across claims: %s",
            connected_entities,
        )

    driver.close()

    print(f"Loaded {len(claims_df)} claims into Neo4j graph.")
