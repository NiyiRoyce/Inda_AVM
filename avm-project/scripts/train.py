"""
CLI script for training the AVM model.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging
from src.auth.gcp_auth import authenticate_gcp
from pipelines.train_pipeline import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(description="Train AVM model")
    
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file (optional, defaults to BigQuery)"
    )
    
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="GCP project ID"
    )
    
    parser.add_argument(
        "--use-colab",
        action="store_true",
        help="Use Colab authentication"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save trained models"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Authenticate if not using CSV
    if not args.csv:
        authenticate_gcp(use_colab=args.use_colab)
    
    # Create and run pipeline
    pipeline = TrainingPipeline(project_id=args.project_id)
    
    metrics, diagnostics = pipeline.run(
        from_csv=args.csv,
        save_models=not args.no_save
    )
    
    print("\n✅ Training completed successfully!")
    
    if not args.no_save:
        print("\n📦 Models saved to artifacts/models/")


if __name__ == "__main__":
    main()