"""
Main entry point for CIIS School Intelligence AI.

Usage:
    python main.py              # Run the server
    python main.py --train      # Train models only
    python main.py --demo-data  # Generate demo data only
"""

import sys
import argparse
import uvicorn
from app.config import get_settings


def main():
    parser = argparse.ArgumentParser(description="CIIS School Intelligence AI")
    parser.add_argument("--train", action="store_true", help="Train models only")
    parser.add_argument("--demo-data", action="store_true", help="Generate demo data only")
    parser.add_argument("--host", default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    args = parser.parse_args()

    settings = get_settings()

    if args.demo_data:
        from app.demo_data import generate_all_demo_data, get_data_summary
        import json
        data = generate_all_demo_data()
        summary = get_data_summary(data)
        print("\n" + json.dumps(summary, indent=2))
        return

    if args.train:
        from app.demo_data import generate_all_demo_data
        from app.ml_pipeline import run_full_training_pipeline
        import json

        data = generate_all_demo_data()
        results = run_full_training_pipeline(data["students"], data["attendance"], data["grades"])

        print("\n" + "=" * 60)
        print("TRAINING RESULTS")
        print("=" * 60)
        print(f"Best model: {results['best_model']}")
        for name, eval_data in results["evaluations"].items():
            print(f"\n{name}:")
            print(f"  F1:        {eval_data['f1']:.4f}")
            print(f"  Precision: {eval_data['precision']:.4f}")
            print(f"  Recall:    {eval_data['recall']:.4f}")
            print(f"  ROC-AUC:   {eval_data.get('roc_auc', 'N/A')}")
            print(f"  Accuracy:  {eval_data['accuracy']:.4f}")
        print("=" * 60)
        return

    # Run the server
    host = args.host or settings.app_host
    port = args.port or settings.app_port

    print(f"\n{'='*60}")
    print(f"  CIIS School Intelligence AI")
    print(f"  Server: http://{host}:{port}")
    print(f"  API Docs: http://{host}:{port}/docs")
    print(f"  Dashboard: http://{host}:{port}/")
    print(f"  Environment: {settings.app_env}")
    print(f"{'='*60}\n")

    uvicorn.run(
        "app.api:app",
        host=host,
        port=port,
        reload=settings.app_debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
