from __future__ import annotations

import logging
import pandas as pd
from faker import Faker
from neo4j import GraphDatabase
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress verbose third-party logs
logging.getLogger("neo4j").setLevel(logging.WARNING)


def build_graph_from_claims(claims_path: Path, neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> None:
    """Load claim records into Neo4j for graph feature enrichment."""
    claims_df = pd.read_parquet(claims_path)
    logger.info(f"Loaded {len(claims_df)} claims from {claims_path}")
    faker = Faker()
    Faker.seed(42)  # for reproducibility

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    with driver.session() as session:
        # Clear existing data
        logger.info("Clearing existing Neo4j data...")
        session.run("MATCH (n) DETACH DELETE n")
        
        # Create constraints for uniqueness
        logger.info("Creating uniqueness constraints...")
        session.run("CREATE CONSTRAINT claimant_id IF NOT EXISTS FOR (c:Claimant) REQUIRE c.id IS UNIQUE")
        session.run("CREATE CONSTRAINT attorney_id IF NOT EXISTS FOR (a:Attorney) REQUIRE a.id IS UNIQUE")
        session.run("CREATE CONSTRAINT vehicle_vin IF NOT EXISTS FOR (v:Vehicle) REQUIRE v.vin IS UNIQUE")
        session.run("CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (cl:Claim) REQUIRE cl.id IS UNIQUE")
        session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        logger.info("Constraints created successfully")
        
        entities = {}  # cache for shared entities
        attorney_count = 0
        shared_entity_count = 0
        
        logger.info("Processing claims and building graph relationships...")
        for idx, claim in claims_df.iterrows():
            if (idx + 1) % 1000 == 0:
                logger.info(f"  Processed {idx + 1}/{len(claims_df)} claims")
            
            claimant_id = f"claimant_{claim['claim_id']}"
            attorney_id = f"attorney_{claim['claim_id']}" if claim['attorney_present'] else None
            vehicle_vin = faker.vin()
            claim_id = claim['claim_id']
            
            # Create Claimant
            claimant_name = faker.name()
            claimant_phone = faker.phone_number()
            claimant_address = faker.address().replace('\n', ', ')
            
            session.run("""
                MERGE (c:Claimant {id: $id})
                SET c.name = $name,
                    c.phone = $phone,
                    c.address = $address,
                    c.is_fraud = $is_fraud
            """, id=claimant_id, name=claimant_name, phone=claimant_phone, address=claimant_address, is_fraud=claim['is_fraud'])
            
            # Create Attorney if present
            if attorney_id:
                attorney_count += 1
                attorney_name = faker.name()
                session.run("""
                    MERGE (a:Attorney {id: $id})
                    SET a.name = $name
                """, id=attorney_id, name=attorney_name)
                
                # Relationship Claimant -[:REPRESENTED_BY]-> Attorney
                session.run("""
                    MATCH (c:Claimant {id: $claimant_id}), (a:Attorney {id: $attorney_id})
                    MERGE (c)-[:REPRESENTED_BY]->(a)
                """, claimant_id=claimant_id, attorney_id=attorney_id)
            
            # Create Vehicle
            session.run("""
                MERGE (v:Vehicle {vin: $vin})
                SET v.make = $make,
                    v.model = $model,
                    v.year = $year
            """, vin=vehicle_vin, make=faker.company(), model=faker.word(), year=faker.year())
            
            # Create Claim
            session.run("""
                MERGE (cl:Claim {id: $id})
                SET cl.is_fraud = $is_fraud,
                    cl.state = $state
            """, id=claim_id, is_fraud=claim['is_fraud'], state=claim['state'])
            
            # Relationships
            session.run("""
                MATCH (c:Claimant {id: $claimant_id}), (cl:Claim {id: $claim_id})
                MERGE (c)-[:FILED]->(cl)
            """, claimant_id=claimant_id, claim_id=claim_id)
            
            session.run("""
                MATCH (cl:Claim {id: $claim_id}), (v:Vehicle {vin: $vin})
                MERGE (cl)-[:INVOLVES]->(v)
            """, claim_id=claim_id, vin=vehicle_vin)
            
            # Shared entities
            for attr, value in [('phone', claimant_phone), ('address', claimant_address)]:
                if value not in entities:
                    entity_id = f"entity_{len(entities)}"
                    entities[value] = entity_id
                    session.run("""
                        MERGE (e:Entity {id: $id})
                        SET e.type = $type,
                            e.value = $value
                    """, id=entity_id, type=attr, value=value)
                else:
                    entity_id = entities[value]
                    shared_entity_count += 1
                
                session.run("""
                    MATCH (c:Claimant {id: $claimant_id}), (e:Entity {id: $entity_id})
                    MERGE (c)-[:SHARES {type: $type}]->(e)
                """, claimant_id=claimant_id, entity_id=entity_id, type=attr)
    
    driver.close()
    logger.info(f"✅ Graph build complete:")
    logger.info(f"   • Claims processed: {len(claims_df)}")
    logger.info(f"   • Attorneys added: {attorney_count}")
    logger.info(f"   • Shared entities: {len(entities)}")
    logger.info(f"   • Shared entity connections: {shared_entity_count}")
    print(f"Loaded {len(claims_df)} claims into Neo4j graph.")
