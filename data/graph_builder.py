from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict

import pandas as pd
from faker import Faker
from neo4j import GraphDatabase, Session
from neo4j.exceptions import Neo4jError

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

        entities: Dict[str, str] = {}

        def _process_batch(tx, batch_df: pd.DataFrame) -> None:

            for _, claim in batch_df.iterrows():

                claim_id = claim["claim_id"]
                is_fraud = bool(claim["is_fraud"])

                claimant_id = f"claimant_{claim_id}"

                # ------------------------------------------------------
                # Shared entity reuse logic
                # ------------------------------------------------------

                if is_fraud:

                    if random.random() < 0.85:
                        claimant_phone = random.choice(fraud_phone_pool)
                    else:
                        claimant_phone = faker.phone_number()

                    if random.random() < 0.80:
                        claimant_address = random.choice(fraud_address_pool)
                    else:
                        claimant_address = faker.address().replace("\n", ", ")

                    attorney_id = (
                        random.choice(fraud_attorney_pool)
                        if claim["attorney_present"]
                        else None
                    )

                else:

                    if random.random() < 0.03:
                        claimant_phone = random.choice(shared_phone_pool)
                    else:
                        claimant_phone = faker.phone_number()

                    if random.random() < 0.05:
                        claimant_address = random.choice(shared_address_pool)
                    else:
                        claimant_address = faker.address().replace("\n", ", ")

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

                claimant_name = faker.name()

                # ------------------------------------------------------
                # Claimant
                # ------------------------------------------------------

                tx.run(
                    """
                    MERGE (c:Claimant {id: $id})
                    SET c.name = $name,
                        c.phone = $phone,
                        c.address = $address,
                        c.is_fraud = $is_fraud
                    """,
                    id=claimant_id,
                    name=claimant_name,
                    phone=claimant_phone,
                    address=claimant_address,
                    is_fraud=is_fraud,
                )

                # ------------------------------------------------------
                # Claim
                # ------------------------------------------------------

                tx.run(
                    """
                    MERGE (cl:Claim {id: $id})
                    SET cl.is_fraud = $is_fraud,
                        cl.state = $state
                    """,
                    id=claim_id,
                    is_fraud=is_fraud,
                    state=claim["state"],
                )

                # ------------------------------------------------------
                # FILED relationship
                # ------------------------------------------------------

                tx.run(
                    """
                    MATCH (c:Claimant {id: $claimant_id}),
                          (cl:Claim {id: $claim_id})

                    MERGE (c)-[:FILED]->(cl)
                    """,
                    claimant_id=claimant_id,
                    claim_id=claim_id,
                )

                # ------------------------------------------------------
                # Attorney
                # ------------------------------------------------------

                if attorney_id:

                    tx.run(
                        """
                        MERGE (a:Attorney {id: $id})
                        ON CREATE SET a.name = $name
                        """,
                        id=attorney_id,
                        name=faker.name(),
                    )

                    tx.run(
                        """
                        MATCH (c:Claimant {id: $claimant_id}),
                              (a:Attorney {id: $attorney_id})

                        MERGE (c)-[:REPRESENTED_BY]->(a)
                        """,
                        claimant_id=claimant_id,
                        attorney_id=attorney_id,
                    )

                # ------------------------------------------------------
                # Vehicle
                # ------------------------------------------------------

                vehicle_vin = faker.vin()

                tx.run(
                    """
                    MERGE (v:Vehicle {vin: $vin})
                    SET v.make = $make,
                        v.model = $model,
                        v.year = $year
                    """,
                    vin=claim["vehicle_vin"],
                    make=faker.company(),
                    model=faker.word(),
                    year=faker.year(),
                )

                tx.run(
                    """
                    MATCH (cl:Claim {id: $claim_id}),
                          (v:Vehicle {vin: $vin})

                    MERGE (cl)-[:INVOLVES]->(v)
                    """,
                    claim_id=claim_id,
                    vin=vehicle_vin,
                )

                # ------------------------------------------------------
                # Shared entities
                # ------------------------------------------------------

                for attr, raw_value in [
                    ("phone", claimant_phone),
                    ("address", claimant_address),
                ]:

                    key = f"{attr}:{_normalize_key(raw_value)}"

                    if key not in entities:

                        entity_id = f"entity_{len(entities)}"

                        entities[key] = entity_id

                        tx.run(
                            """
                            MERGE (e:Entity {id: $id})
                            SET e.type = $type,
                                e.value = $value
                            """,
                            id=entity_id,
                            type=attr,
                            value=raw_value,
                        )

                    entity_id = entities[key]

                    tx.run(
                        """
                        MATCH (c:Claimant {id: $claimant_id}),
                              (e:Entity {id: $entity_id})

                        MERGE (c)-[:SHARES {type: $type}]->(e)
                        """,
                        claimant_id=claimant_id,
                        entity_id=entity_id,
                        type=attr,
                    )

                    tx.run(
                        """
                        MATCH (cl:Claim {id: $claim_id}),
                              (e:Entity {id: $entity_id})

                        MERGE (cl)-[:SHARES {type: $type}]->(e)
                        """,
                        claim_id=claim_id,
                        entity_id=entity_id,
                        type=attr,
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