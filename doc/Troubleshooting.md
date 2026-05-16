docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
"// 1. Total SHARES relationships
MATCH ()-[r:SHARES]->() RETURN count(r) AS total_shares;
// 2. SHARES relationships from Claim nodes specifically
MATCH (c:Claim)-[r:SHARES]->(e:Entity) RETURN count(r) AS claim_to_entity_shares;
// 3. SHARES relationships from Claimant nodes
MATCH (p:Claimant)-[r:SHARES]->(e:Entity) RETURN count(r) AS claimant_to_entity_shares;
// 4. Sample Claim degrees (use OPTIONAL MATCH + COUNT)
MATCH (cl:Claim)
OPTIONAL MATCH (cl)-[r]-()
WITH cl, COUNT(r) AS deg
RETURN cl.id AS id, deg
LIMIT 10;
// 5. Entities shared by multiple claims (should be >0 if dedup worked)
MATCH (e:Entity)
OPTIONAL MATCH (e)<-[:SHARES]-(c:Claim)
WITH e, COUNT(DISTINCT c) AS claims_count
WHERE claims_count > 1
RETURN COUNT(e) AS entities_shared_across_claims, claims_count
LIMIT 5;
// 6. For fraud claims, check connected entities
MATCH (f:Claim) WHERE f.is_fraud = true
OPTIONAL MATCH (f)-[:SHARES]->(e:Entity)
RETURN COUNT(DISTINCT f) AS fraud_claims, COUNT(DISTINCT e) AS fraud_connected_entities;
" > neo4jQueryResult.txt