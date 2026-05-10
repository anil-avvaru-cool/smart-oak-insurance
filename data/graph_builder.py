from __future__ import annotations

from pathlib import Path

from neo4j import GraphDatabase


def build_graph_from_claims(claims_path: Path, neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> None:
    """Load claim records into Neo4j for graph feature enrichment.

    This is a minimal bootstrap implementation. The graph schema and
    enrichment queries can be extended once the fraud archetypes are defined.
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session() as session:
        session.run("RETURN 1")
    driver.close()
