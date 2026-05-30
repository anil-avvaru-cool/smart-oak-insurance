import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from data.generator import generate_data
from data.entities import resolve_vehicles, resolve_persons, resolve_addresses, resolve_phones, resolve_policies
from data.graph_builder import build_graph_from_claims, clear_graph
from data.graph_features import compute_graph_features
from data.validator import validate_data
from data.config import CLAIMS_OUTPUT, FRAUD_MODELS_DIR, OFFLINE_FEATURES_DIR, QUOTES_OUTPUT, RISK_MODELS_DIR

load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart Oak Insurance data tools")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic quote and claim datasets")
    parser.add_argument("--resolve-entities", action="store_true", help="Resolve and normalize entity data (vehicles, persons, addresses, phones, policies)")
    parser.add_argument("--reset-graph", action="store_true", help="Delete all graph nodes/relationships and drop all constraints (use before --build-graph for a clean reload)")
    parser.add_argument("--build-graph", action="store_true", help="Build graph from claims data in Neo4j")
    parser.add_argument("--compute-graph-features", action="store_true", help="Compute graph features and update claims data")
    parser.add_argument("--graph-seed-cutoff", metavar="YYYY-MM-DD", help="ISO date: seed BFS hop distances only from fraud claims before this date (prevents label leakage; use the same date as your train/test split)")
    parser.add_argument("--run-offline-pipeline", action="store_true", help="Build offline feature snapshots from quotes and claims parquet files")
    parser.add_argument("--validate-data", action="store_true", help="Validate generated datasets")
    parser.add_argument("--drift-check", action="store_true", help="Compute PSI drift for quotes and claims; print JSON report to stdout; auto-starts shadow scoring if champion-challenger is triggered")
    parser.add_argument("--train-risk-model", action="store_true", help="Train Stage 2 Hurdle Model (frequency + severity) and write models to data/processed/risk_models/")
    parser.add_argument("--calibrate-risk-model", action="store_true", help="Stage 3: fit Platt scaling on frequency model + bias correction on severity model")
    parser.add_argument("--compare-gini", action="store_true", help="CC-2: compare challenger vs champion Gini on accumulated shadow cohort; prints pass/fail with delta")
    parser.add_argument("--promote-challenger", action="store_true", help="CC-4: promote challenger model artifacts to champion slot; archives previous champion with timestamp suffix")
    parser.add_argument("--shap-check", action="store_true", help="SH-4: print SHAP share comparison for the two most recent snapshots")
    parser.add_argument("--as-of", metavar="YYYY-MM-DD", help="OP-2: back-test --drift-check as of a historical date (ISO format)")
    parser.add_argument("--purge-drift-logs", action="store_true", help="OP-3: delete old drift reports, keeping only the N most recent")
    parser.add_argument("--keep", type=int, default=10, metavar="N", help="OP-3: number of drift reports to retain (default: 10)")
    parser.add_argument("--score-quote", metavar="JSON_PATH", help="Score a single quote from a JSON feature file; prints p_claim, e_cost_usd, risk_score, and top-5 SHAP reason codes")
    parser.add_argument("--train-fraud-models", action="store_true", help="Train Section-4 ensemble fraud scoring: XGBoost + Isolation Forest + stacking meta-learner")
    parser.add_argument("--label-quality", action="store_true", help="Stage 5: run PU Learning + Confident Learning + label smoothing; writes pu_adjusted_labels.parquet + label_quality_report.json")
    parser.add_argument("--train-fraud-pu", action="store_true", help="Stage 5b: train XGBoost fraud model on PU-adjusted sample weights (requires --label-quality to have been run first)")
    parser.add_argument("--confirmed-window", type=int, default=90, metavar="DAYS", help="SIU confirmation window in days for --label-quality (default: 90)")
    parser.add_argument("--score-claim", metavar="JSON_PATH", help="Score a single claim JSON; prints fraud_score, CI, base model scores, and top-5 SHAP reason codes")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.generate_data:
        quotes_df, claims_df = generate_data()
        print(f"Generated {len(quotes_df)} quotes and {len(claims_df)} claims")
        if not (args.resolve_entities or args.reset_graph or args.build_graph or args.compute_graph_features or args.run_offline_pipeline or args.validate_data):
            return

    if args.purge_drift_logs:
        import glob as _glob
        _archived_dir = RISK_MODELS_DIR / "archived"
        _patterns = ["drift_report_*.json", "challenger_predictions_*.parquet"]
        total_deleted = 0
        for _pat in _patterns:
            _files = sorted(
                _glob.glob(str(RISK_MODELS_DIR / _pat)) +
                (_glob.glob(str(_archived_dir / _pat)) if _archived_dir.exists() else [])
            )
            _to_delete = _files[: max(0, len(_files) - args.keep)]
            for p in _to_delete:
                Path(p).unlink()
                total_deleted += 1
            print(f"  {_pat}: {len(_to_delete)} deleted, {len(_files) - len(_to_delete)} retained")
        print(f"Purged {total_deleted} file(s) total ({args.keep} most recent of each type kept)")
        return

    if args.drift_check:
        from monitoring.psi_drift import run_drift_check, report_to_dict
        from monitoring.concept_drift import run_concept_drift_check
        import pandas as pd
        as_of = pd.Timestamp(args.as_of) if args.as_of else None
        reports = run_drift_check(as_of=as_of)
        payload = {name: report_to_dict(r) for name, r in reports.items()}

        # CD-4: include concept drift summary (non-fatal; reuses label_window from claims report)
        try:
            concept_drift = run_concept_drift_check(
                label_window=reports["claims"].label_window,
            )
            payload["concept_drift"] = concept_drift
        except Exception as exc:
            payload["concept_drift"] = {"error": str(exc)}

        RISK_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _archived_dir = RISK_MODELS_DIR / "archived"
        _archived_dir.mkdir(exist_ok=True)
        for _old in RISK_MODELS_DIR.glob("drift_report_*.json"):
            _old.rename(_archived_dir / _old.name)

        ts = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
        out_path = RISK_MODELS_DIR / f"drift_report_{ts}.json"
        out_path.write_text(json.dumps(payload, indent=2))

        # Print concise summary instead of full JSON
        print("Drift check summary")
        print("=" * 40)
        for name, rep in reports.items():
            status = "TRIGGERED" if rep.champion_challenger_triggered else "OK"
            top = sorted(rep.feature_results, key=lambda r: r.psi, reverse=True)[:3]
            top_str = ", ".join(f"{r.feature}={r.psi:.3f}" for r in top)
            print(f"  [{name}] cc={status}  top PSI: {top_str}")
        cd = payload.get("concept_drift", {})
        if "error" not in cd:
            lag = cd.get("label_lag") or {}
            gini = cd.get("gini_trend") or {}
            print(f"  [concept_drift] label_lag={lag.get('lag_days', 'n/a')}d  gini={gini.get('gini', 'n/a')}")
        print(f"\nFull report → {out_path}")

        # CC-5: auto-start shadow scoring when drift triggers the champion-challenger loop
        cc_triggered = any(r.champion_challenger_triggered for r in reports.values())
        if cc_triggered:
            chal_model_path = RISK_MODELS_DIR / "frequency_model_challenger.json"
            if chal_model_path.exists():
                from monitoring.champion_challenger import score_shadow
                try:
                    shadow_path = score_shadow(
                        quotes_path=QUOTES_OUTPUT,
                        models_dir=RISK_MODELS_DIR,
                        output_dir=RISK_MODELS_DIR,
                        triggered_by="drift_check",
                    )
                    print(f"\nChampion-challenger triggered — shadow scoring complete → {shadow_path}")
                except Exception as exc:
                    print(f"\nChampion-challenger triggered — shadow scoring failed: {exc}")
            else:
                print(
                    "\nChampion-challenger triggered — no challenger model found. "
                    "Train a new model and save as frequency_model_challenger.json to begin shadow scoring."
                )
        return

    if args.compare_gini:
        from monitoring.champion_challenger import compare_gini
        result = compare_gini(
            quotes_path=QUOTES_OUTPUT,
            models_dir=RISK_MODELS_DIR,
            output_dir=RISK_MODELS_DIR,
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"Gini comparison [{status}]")
        print(f"  champion_gini  = {result['champion_gini']}")
        print(f"  challenger_gini = {result['challenger_gini']}")
        print(f"  delta           = {result['delta']:+.4f}  (threshold >= {result['gini_pass_delta_threshold']})")
        print(f"  n_records       = {result['n_records']}  ({result['n_shadow_files']} shadow files)")
        return

    if args.promote_challenger:
        from monitoring.champion_challenger import promote_challenger
        result = promote_challenger(models_dir=RISK_MODELS_DIR, output_dir=RISK_MODELS_DIR)
        print(f"Promotion complete (ts={result['ts']})")
        print(f"  Promoted : {result['promoted']}")
        print(f"  Archived : {result['archived']}")
        if result["missing_challenger"]:
            print(f"  WARNING  — missing challenger artifacts skipped: {result['missing_challenger']}")
        return

    if args.shap_check:
        from monitoring.shap_monitor import compare_shap_snapshots
        result = compare_shap_snapshots(output_dir=RISK_MODELS_DIR)
        print(f"SHAP check: {result['previous_snapshot']}  →  {result['current_snapshot']}")
        print(f"  max feature : {result['max_shap_share_feature']}  ({result['max_shap_share']:.1%} share)")
        print(f"  brittleness : {'WARNING' if result['brittleness_warning'] else 'OK'}")
        if result["warnings"]:
            for w in result["warnings"]:
                print(f"  ⚠  {w}")
        print()
        print(f"  {'Feature':<40} {'Prev':>8} {'Curr':>8} {'Delta':>8}  {'Rank shift':>10}")
        print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}  {'-'*10}")
        for row in result["feature_comparison"]:
            shift = row["rank_shift"]
            shift_str = f"+{shift}" if shift > 0 else str(shift)
            print(
                f"  {row['feature']:<40} {row['prev_share']:>8.1%} {row['curr_share']:>8.1%}"
                f" {row['delta']:>+8.1%}  {shift_str:>10}"
            )
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
        import pandas as pd
        # Auto-derive seed cutoff from data (80th-percentile loss date) so the BFS
        # seed window always matches the temporal train/test split in train_xgb_fraud.
        seed_cutoff = args.graph_seed_cutoff
        if seed_cutoff is None:
            _dates = pd.to_datetime(pd.read_parquet(CLAIMS_OUTPUT, columns=["loss_event_datetime"])["loss_event_datetime"])
            if _dates.dt.tz is not None:
                _dates = _dates.dt.tz_localize(None)
            seed_cutoff = str(_dates.quantile(0.8).date())
            print(f"Auto-derived graph seed cutoff: {seed_cutoff} (80th-percentile loss date)")
        graph_features_df = compute_graph_features(CLAIMS_OUTPUT, os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"], seed_cutoff_dt=seed_cutoff)

        # Merge with claims and update parquet
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

        # OP-1: write versioned training run metrics file
        import pandas as pd
        ts = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
        training_run = {
            "ts": ts,
            "frequency": freq_metrics,
            "severity": sev_metrics,
            "hurdle": hurdle_metrics,
        }
        RISK_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        run_path = RISK_MODELS_DIR / f"training_run_{ts}.json"
        run_path.write_text(json.dumps(training_run, indent=2))
        print(f"Training run metrics → {run_path}")

        # SH-3: auto-compute SHAP snapshot after every training run
        print("Stage 2d — computing SHAP snapshot…")
        try:
            from monitoring.shap_monitor import write_shap_snapshot
            shap_path = write_shap_snapshot(
                quotes_path=QUOTES_OUTPUT,
                models_dir=RISK_MODELS_DIR,
                output_dir=RISK_MODELS_DIR,
            )
            print(f"  SHAP snapshot → {shap_path}")
        except Exception as exc:
            print(f"  SHAP snapshot failed (non-fatal): {exc}")

        if not (args.calibrate_risk_model or args.validate_data):
            return

    if args.calibrate_risk_model:
        from underwriting.models.risk_scoring.calibrate import (
            calibrate_frequency,
            calibrate_severity,
        )

        print("Stage 3a — calibrating frequency model (Platt scaling)…")
        freq_cal = calibrate_frequency(QUOTES_OUTPUT, RISK_MODELS_DIR, RISK_MODELS_DIR)
        print(
            f"  Brier: {freq_cal['brier_before']:.4f} → {freq_cal['brier_after']:.4f}"
            f"  ECE: {freq_cal['ece_before']:.4f} → {freq_cal['ece_after']:.4f}"
        )
        print(
            f"  p_claim mean: {freq_cal['p_claim_mean_before']:.4f} → {freq_cal['p_claim_mean_after']:.4f}"
            f"  (actual rate: {freq_cal['actual_claim_rate']:.4f})"
        )

        print("Stage 3b — calibrating severity model (multiplicative bias correction)…")
        sev_cal = calibrate_severity(QUOTES_OUTPUT, CLAIMS_OUTPUT, RISK_MODELS_DIR, RISK_MODELS_DIR)
        print(
            f"  Bias correction: ×{sev_cal['bias_correction']:.4f}"
            f"  MAE: ${sev_cal['mae_before_usd']:,.0f} → ${sev_cal['mae_after_usd']:,.0f}"
        )
        print(f"Calibration artifacts written to {RISK_MODELS_DIR}")
        if not args.validate_data:
            return

    if args.score_quote:
        import pandas as pd
        from underwriting.models.risk_scoring.hurdle_model import HurdleModel

        quote_path = Path(args.score_quote)
        if not quote_path.exists():
            print(f"Error: {quote_path} not found")
            return

        payload = json.loads(quote_path.read_text())
        df = pd.DataFrame([payload])
        model = HurdleModel(RISK_MODELS_DIR)
        result = model.score_with_components(df).iloc[0]
        reason_codes = model.explain(df, top_n=5)[0]

        print(f"p_claim      : {result['p_claim']:.4f}")
        print(f"e_cost_usd   : ${result['e_cost_usd']:,.2f}")
        print(f"risk_score   : ${result['risk_score']:,.2f}")
        print(f"calibrated   : {model.calibration_applied}")
        print("\nTop-5 SHAP reason codes (frequency model):")
        for rc in reason_codes:
            direction = "+" if rc["shap_pct"] > 0 else ""
            print(f"  {rc['feature']:<40} {direction}{rc['shap_pct']:.1f}%")
        return

    if args.train_fraud_models:
        from fraud.ensemble_fraud_scorer import train_ensemble

        print("Stage 4a — training XGBoost fraud classifier (P(fraud | claim features))…")
        print("Stage 4b — training Isolation Forest anomaly detector…")
        print("Stage 4c — training stacking meta-learner (5-fold OOF)…")
        all_metrics = train_ensemble(CLAIMS_OUTPUT, QUOTES_OUTPUT, FRAUD_MODELS_DIR)

        xm = all_metrics["xgb_fraud"]
        am = all_metrics["anomaly_detector"]
        sm = all_metrics["stacking_meta_learner"]
        print(f"\n  [4a XGBoost]  AUC={xm['auc']}  Gini={xm['gini']}  AP={xm['average_precision']}")
        print(f"  [4b IF]       contamination={am['contamination']}  anomaly_p95={am['anomaly_score_p95']}")
        print(f"  [4c Stacking] OOF AUC={sm['oof_auc']}  OOF Gini={sm['oof_gini']}")
        print(f"\nFraud models written to {FRAUD_MODELS_DIR}")

        import pandas as _pd
        ts = _pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
        FRAUD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        run_path = FRAUD_MODELS_DIR / f"fraud_training_run_{ts}.json"
        run_path.write_text(json.dumps(all_metrics, indent=2))
        print(f"Training run metrics → {run_path}")
        if not (args.label_quality or args.train_fraud_pu or args.validate_data):
            return

    if args.label_quality:
        from fraud.label_quality import build_pu_adjusted_dataset

        print("Stage 5 — Label Quality & PU Learning")
        print("  Techniques: PU Learning (Elkan-Noto) + Confident Learning + Label Smoothing")
        report = build_pu_adjusted_dataset(
            CLAIMS_OUTPUT,
            QUOTES_OUTPUT,
            FRAUD_MODELS_DIR,
            confirmed_window_days=args.confirmed_window,
        )
        print(f"\nLabel quality report:")
        print(f"  records           : {report['n_records']}")
        print(f"  positives         : {report['n_positives']}  negatives: {report['n_negatives']}")
        print(f"  confirmed fraud   : {report['n_confirmed_fraud']} (within {report['confirmed_window_days']}d window)")
        print(f"  c_estimate        : {report['c_estimate']}  (P(labeled|true fraud) — PU label frequency)")
        print(f"  noise rate (neg.) : {report['estimated_noise_rate_negatives']:.2%}  ({report['n_suspected_mislabel_negatives']} suspected mislabeled negatives)")
        print(f"  high PU-score neg.: {report['n_high_pu_score_negatives']} (PU score > 0.5 despite label=0)")
        print(f"  OOF AUC           : {report['oof_auc']}")
        print(f"\nArtifacts written to {FRAUD_MODELS_DIR}")
        print(f"  pu_adjusted_labels.parquet")
        print(f"  label_quality_report.json")
        if not (args.train_fraud_pu or args.validate_data):
            return

    if args.train_fraud_pu:
        from fraud.train_xgb_fraud import train_with_pu_labels

        pu_labels_path = FRAUD_MODELS_DIR / "pu_adjusted_labels.parquet"
        if not pu_labels_path.exists():
            print(f"Error: {pu_labels_path} not found — run --label-quality first")
            return

        print("Stage 5b — Training XGBoost on PU-adjusted sample weights…")
        pu_metrics = train_with_pu_labels(CLAIMS_OUTPUT, QUOTES_OUTPUT, pu_labels_path, FRAUD_MODELS_DIR)
        print(f"  [5b PU XGBoost]  AUC={pu_metrics['auc']}  Gini={pu_metrics['gini']}  AP={pu_metrics['average_precision']}")
        print(f"  Model written to {FRAUD_MODELS_DIR / 'fraud_model_pu.json'}")

        # Compare with standard model if it exists
        std_metrics_path = FRAUD_MODELS_DIR / "fraud_metrics.json"
        if std_metrics_path.exists():
            std = json.loads(std_metrics_path.read_text())
            delta_auc = pu_metrics["auc"] - std["auc"]
            delta_gini = pu_metrics["gini"] - std["gini"]
            sign = "+" if delta_auc >= 0 else ""
            print(f"\n  vs. standard model:  ΔAUC={sign}{delta_auc:.4f}  ΔGini={sign}{delta_gini:.4f}")
        if not args.validate_data:
            return

    if args.score_claim:
        import pandas as pd
        from fraud.ensemble_fraud_scorer import EnsembleFraudScorer

        claim_path = Path(args.score_claim)
        if not claim_path.exists():
            print(f"Error: {claim_path} not found")
            return

        payload = json.loads(claim_path.read_text())
        df = pd.DataFrame([payload])
        scorer = EnsembleFraudScorer(FRAUD_MODELS_DIR)
        result = scorer.score(df).iloc[0]
        reason_codes = scorer.explain(df, top_n=5)[0]

        print(f"fraud_score  : {result['fraud_score']:.4f}")
        print(f"CI           : [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        print(f"  p_xgb      : {result['p_xgb']:.4f}")
        print(f"  p_anomaly  : {result['p_anomaly']:.4f}")
        print(f"  p_nlp      : {result['p_nlp']:.4f}  (narrative proxy)")
        print(f"  p_gnn      : {result['p_gnn']:.4f}  (stub — emerging)")
        print(f"  p_vision   : {result['p_vision']:.4f}  (stub — emerging)")
        print("\nTop-5 SHAP reason codes (XGBoost fraud model):")
        for rc in reason_codes:
            direction = "+" if rc["shap_pct"] > 0 else ""
            print(f"  {rc['feature']:<40} {direction}{rc['shap_pct']:.1f}%")
        return

    if args.validate_data:
        validate_data()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
