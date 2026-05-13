from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from faker import Faker
from neo4j import GraphDatabase, Session
from neo4j.exceptions import Neo4jError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("neo4j").setLevel(logging.WARNING)


def _create_constraints(session: Session) -> None:
    """
    Create uniqueness constraints using Neo4j 5+ syntax.
    """
    logger.info("Creating uniqueness constraints (Neo4j 5+ syntax)...")
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
            logger.debug("Constraint statement returned: %s -- %s", q, e)


def _normalize_key(value: str) -> str:
    if value is None:
        return ""
    key = value.strip().lower()
    key = key.replace("(", "").replace(")", "").replace("-", "").replace(".", "").replace(" ", "")
    return key


def build_graph_from_claims(claims_path: Path, neo4j_uri: str, neo4j_user: str, neo4j_password: str, batch_size: int = 1000) -> None:
    claims_df = pd.read_parquet(claims_path)
    logger.info(f"Loaded {len(claims_df)} claims from {claims_path}")

    faker = Faker()
    Faker.seed(42)

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    with driver.session() as session:
        logger.info("Clearing existing Neo4j data...")
        session.run("MATCH (n) DETACH DELETE n")

        _create_constraints(session)

        entities: Dict[str, str] = {}
        attorney_count = 0

        def _process_batch(tx, batch_df: pd.DataFrame) -> None:
            nonlocal attorney_count
            for _, claim in batch_df.iterrows():
                claim_id = claim["claim_id"]
                claimant_id = f"claimant_{claim_id}"
                attorney_id = f"attorney_{claim_id}" if claim["attorney_present"] else None

                # Claimant
                claimant_name = faker.name()
                claimant_phone = faker.phone_number()
                claimant_address = faker.address().replace("\n", ", ")
                tx.run(
                    """
                    MERGE (c:Claimant {id: $id})
                    SET c.name = $name, c.phone = $phone, c.address = $address, c.is_fraud = $is_fraud
                    """,
                    id=claimant_id,
                    name=claimant_name,
                    phone=claimant_phone,
                    address=claimant_address,
                    is_fraud=claim["is_fraud"],
                )

                # Claim
                tx.run(
                    """
                    MERGE (cl:Claim {id: $id})
                    SET cl.is_fraud = $is_fraud, cl.state = $state
                    """,
                    id=claim_id,
                    is_fraud=claim["is_fraud"],
                    state=claim["state"],
                )

                # FILED
                tx.run(
                    """
                    MATCH (c:Claimant {id: $claimant_id}), (cl:Claim {id: $claim_id})
                    MERGE (c)-[:FILED]->(cl)
                    """,
                    claimant_id=claimant_id,
                    claim_id=claim_id,
                )

                # Attorney
                if attorney_id:
                    attorney_count += 1
                    attorney_name = faker.name()
                    tx.run(
                        """
                        MERGE (a:Attorney {id: $id})
                        SET a.name = $name
                        """,
                        id=attorney_id,
                        name=attorney_name,
                    )
                    tx.run(
                        """
                        MATCH (c:Claimant {id: $claimant_id}), (a:Attorney {id: $attorney_id})
                        MERGE (c)-[:REPRESENTED_BY]->(a)
                        """,
                        claimant_id=claimant_id,
                        attorney_id=attorney_id,
                    )

                # Vehicle
                vehicle_vin = faker.vin()
                tx.run(
                    """
                    MERGE (v:Vehicle {vin: $vin})
                    SET v.make = $make, v.model = $model, v.year = $year
                    """,
                    vin=vehicle_vin,
                    make=faker.company(),
                    model=faker.word(),
                    year=faker.year(),
                )
                tx.run(
                    """
                    MATCH (cl:Claim {id: $claim_id}), (v:Vehicle {vin: $vin})
                    MERGE (cl)-[:INVOLVES]->(v)
                    """,
                    claim_id=claim_id,
                    vin=vehicle_vin,
                )

                # Shared entities (deduplicate by normalized key)
                for attr, raw_value in [("phone", claimant_phone), ("address", claimant_address)]:
                    key = _normalize_key(raw_value)
                    if key not in entities:
                        entity_id = f"entity_{len(entities)}"
                        entities[key] = entity_id
                        tx.run(
                            """
                            MERGE (e:Entity {id: $id})
                            SET e.type = $type, e.value = $value
                            """,
                            id=entity_id,
                            type=attr,
                            value=raw_value,
                        )
                    else:
                        entity_id = entities[key]

                    # Claimant -> Entity
                    tx.run(
                        """
                        MATCH (c:Claimant {id: $claimant_id}), (e:Entity {id: $entity_id})
                        MERGE (c)-[:SHARES {type: $type}]->(e)
                        """,
                        claimant_id=claimant_id,
                        entity_id=entity_id,
                        type=attr,
                    )

                    # Claim -> Entity
                    tx.run(
                        """
                        MATCH (cl:Claim {id: $claim_id}), (e:Entity {id: $entity_id})
                        MERGE (cl)-[:SHARES {type: $type}]->(e)
                        """,
                        claim_id=claim_id,
                        entity_id=entity_id,
                        type=attr,
                    )

        total = len(claims_df)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_df = claims_df.iloc[start:end]
            logger.info(f"  Importing claims {start + 1}..{end} / {total}")
            # Use execute_write (driver v5)
            session.execute_write(lambda tx, df=batch_df: _process_batch(tx, df))

        # Post-import counts
        shares_count = session.run("MATCH ()-[r:SHARES]->() RETURN count(r) AS shares").single().get("shares", 0)
        represented_count = session.run("MATCH ()-[r:REPRESENTED_BY]->() RETURN count(r) AS reps").single().get("reps", 0)
        filed_count = session.run("MATCH ()-[r:FILED]->() RETURN count(r) AS filed").single().get("filed", 0)

        logger.info("✅ Graph build complete:")
        logger.info(f"   • Claims processed: {len(claims_df)}")
        logger.info(f"   • Attorneys added: {attorney_count}")
        logger.info(f"   • Shared entities (unique): {len(entities)}")
        logger.info(f"   • SHARES relationships: {shares_count}")
        logger.info(f"   • REPRESENTED_BY relationships: {represented_count}")
        logger.info(f"   • FILED relationships: {filed_count}")

    driver.close()
    print(f"Loaded {len(claims_df)} claims into Neo4j graph.")
