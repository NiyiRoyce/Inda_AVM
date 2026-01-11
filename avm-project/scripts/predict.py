"""
CLI script for making predictions with trained models.
"""
import argparse
import sys
import json
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging
from src.models.ensemble import EnsemblePredictor


def main():
    parser = argparse.ArgumentParser(description="Make predictions with AVM model")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file with features"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (optional)"
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
    
    # Load ensemble predictor
    print("Loading models...")
    ensemble = EnsemblePredictor.load_from_artifacts()
    print("✅ Models loaded")
    
    # Load input data
    print(f"\nLoading input data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"✅ Loaded {len(df)} rows")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = ensemble.predict(df)
    
    # Add predictions to dataframe
    df["predicted_price"] = predictions
    
    # Save or display
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n✅ Predictions saved to {args.output}")
    else:
        print("\nPredictions:")
        print(df[["predicted_price"]].head(10))
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Prediction Summary:")
    print("=" * 60)
    print(f"Mean:   ₦{predictions.mean():,.0f}")
    print(f"Median: ₦{pd.Series(predictions).median():,.0f}")
    print(f"Min:    ₦{predictions.min():,.0f}")
    print(f"Max:    ₦{predictions.max():,.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()