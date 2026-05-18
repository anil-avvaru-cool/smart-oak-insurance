import argparse
import os

from dotenv import load_dotenv

from data.generator import generate_data
from data.entities import resolve_vehicles, resolve_persons, resolve_addresses, resolve_phones, resolve_policies
from data.graph_builder import build_graph_from_claims, clear_graph
from data.graph_features import compute_graph_features
from data.validator import validate_data
from data.config import CLAIMS_OUTPUT, OFFLINE_FEATURES_DIR, QUOTES_OUTPUT

load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart Oak Insurance data tools")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic quote and claim datasets")
    parser.add_argument("--resolve-entities", action="store_true", help="Resolve and normalize entity data (vehicles, persons, addresses, phones, policies)")
    parser.add_argument("--reset-graph", action="store_true", help="Delete all graph nodes/relationships and drop all constraints (use before --build-graph for a clean reload)")
    parser.add_argument("--build-graph", action="store_true", help="Build graph from claims data in Neo4j")
    parser.add_argument("--compute-graph-features", action="store_true", help="Compute graph features and update claims data")
    parser.add_argument("--run-offline-pipeline", action="store_true", help="Build offline feature snapshots from quotes and claims parquet files")
    parser.add_argument("--validate-data", action="store_true", help="Validate generated datasets")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.generate_data:
        quotes_df, claims_df = generate_data()
        print(f"Generated {len(quotes_df)} quotes and {len(claims_df)} claims")
        return

    if args.resolve_entities:
        resolve_vehicles()
        resolve_persons()
        resolve_addresses()
        resolve_phones()
        resolve_policies()
        print("Entity resolution complete")
        if not args.validate_data:
            return

    if args.reset_graph:
        clear_graph(os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
        print("Graph cleared")
        if not args.build_graph:
            return

    if args.build_graph:
        build_graph_from_claims(CLAIMS_OUTPUT, os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
        print("Graph built successfully")
        if not args.validate_data:
            return

    if args.compute_graph_features:
        graph_features_df = compute_graph_features(CLAIMS_OUTPUT, os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])

        # Merge with claims and update parquet
        import pandas as pd
        claims_df = pd.read_parquet(CLAIMS_OUTPUT)
        claims_df = claims_df.drop(columns=['graph_hop_distance', 'attorney_centrality_score', 'shared_attribute_count'], errors='ignore')
        claims_df = claims_df.merge(graph_features_df, on='claim_id', how='left')
        claims_df.to_parquet(CLAIMS_OUTPUT, index=False)
        print("Graph features computed and updated")
        if not args.validate_data:
            return

    if args.run_offline_pipeline:
        from features.offline_pipeline import run_offline_pipeline
        quotes_written, claims_written = run_offline_pipeline(
            quotes_path=QUOTES_OUTPUT,
            claims_path=CLAIMS_OUTPUT,
            output_dir=OFFLINE_FEATURES_DIR,
        )
        print(f"Offline pipeline complete: {quotes_written} quote snapshots, {claims_written} claim snapshots → {OFFLINE_FEATURES_DIR}")
        if not args.validate_data:
            return

    if args.validate_data:
        validate_data()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
