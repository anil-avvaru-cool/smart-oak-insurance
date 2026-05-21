import argparse
import os

from dotenv import load_dotenv

from data.generator import generate_data
from data.entities import resolve_vehicles, resolve_persons, resolve_addresses, resolve_phones, resolve_policies
from data.graph_builder import build_graph_from_claims, clear_graph
from data.graph_features import compute_graph_features
from data.validator import validate_data
from data.config import CLAIMS_OUTPUT, OFFLINE_FEATURES_DIR, QUOTES_OUTPUT, RISK_MODELS_DIR

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
    parser.add_argument("--train-risk-model", action="store_true", help="Train Stage 2 Hurdle Model (frequency + severity) and write models to data/processed/risk_models/")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.generate_data:
        quotes_df, claims_df = generate_data()
        print(f"Generated {len(quotes_df)} quotes and {len(claims_df)} claims")
        if not (args.resolve_entities or args.reset_graph or args.build_graph or args.compute_graph_features or args.run_offline_pipeline or args.validate_data):
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

    if args.train_risk_model:
        from underwriting.models.risk_scoring.train_frequency import train as train_frequency
        from underwriting.models.risk_scoring.train_severity import train as train_severity
        from underwriting.models.risk_scoring.hurdle_model import evaluate as evaluate_hurdle

        print("Stage 2a — training frequency model (P(claim))…")
        freq_metrics = train_frequency(QUOTES_OUTPUT, RISK_MODELS_DIR)
        print(f"  AUC={freq_metrics['auc']}  Gini={freq_metrics['gini']}  log_loss={freq_metrics['log_loss']}")

        print("Stage 2b — training severity model (E[cost | claim])…")
        sev_metrics = train_severity(QUOTES_OUTPUT, CLAIMS_OUTPUT, RISK_MODELS_DIR)
        print(f"  MAE=${sev_metrics['mae_usd']:,.0f}  RMSE=${sev_metrics['rmse_usd']:,.0f}  MAPE={sev_metrics['mape']:.1%}")

        print("Stage 2c — evaluating hurdle model (risk score = P × E)…")
        hurdle_metrics = evaluate_hurdle(QUOTES_OUTPUT, RISK_MODELS_DIR)
        print(f"  Hurdle Gini={hurdle_metrics['hurdle_gini']}  mean_risk=${hurdle_metrics['risk_score_mean']:,.0f}  P95=${hurdle_metrics['risk_score_p95']:,.0f}")
        print(f"Models written to {RISK_MODELS_DIR}")
        if not args.validate_data:
            return

    if args.validate_data:
        validate_data()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
