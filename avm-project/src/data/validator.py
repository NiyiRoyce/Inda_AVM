"""
Data validation and quality checks.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List

from config.features import TARGET_VARIABLE, NUMERIC_FEATURES
from config.settings import GEOGRAPHIC_BOUNDS

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and schema."""
    
    def __init__(self):
        self.required_columns = [TARGET_VARIABLE] + NUMERIC_FEATURES
    
    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Run all validation checks on DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results:
                - is_valid: bool
                - issues: List[str]
                - invalid_indices: List[int]
        """
        issues = []
        invalid_indices = []
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for negative/zero prices
        if TARGET_VARIABLE in df.columns:
            invalid_price_mask = df[TARGET_VARIABLE] <= 0
            num_invalid_prices = invalid_price_mask.sum()
            if num_invalid_prices > 0:
                issues.append(
                    f"{num_invalid_prices} rows with non-positive prices"
                )
                invalid_indices.extend(df[invalid_price_mask].index.tolist())
        
        # Check geographic bounds
        if "latitude" in df.columns:
            invalid_lat = (
                (df["latitude"] < GEOGRAPHIC_BOUNDS["latitude_min"]) |
                (df["latitude"] > GEOGRAPHIC_BOUNDS["latitude_max"])
            )
            num_invalid_lat = invalid_lat.sum()
            if num_invalid_lat > 0:
                issues.append(
                    f"{num_invalid_lat} rows with invalid latitude "
                    f"(outside {GEOGRAPHIC_BOUNDS['latitude_min']}-"
                    f"{GEOGRAPHIC_BOUNDS['latitude_max']})"
                )
        
        if "longitude" in df.columns:
            invalid_lon = (
                (df["longitude"] < GEOGRAPHIC_BOUNDS["longitude_min"]) |
                (df["longitude"] > GEOGRAPHIC_BOUNDS["longitude_max"])
            )
            num_invalid_lon = invalid_lon.sum()
            if num_invalid_lon > 0:
                issues.append(
                    f"{num_invalid_lon} rows with invalid longitude "
                    f"(outside {GEOGRAPHIC_BOUNDS['longitude_min']}-"
                    f"{GEOGRAPHIC_BOUNDS['longitude_max']})"
                )
        
        # Check for extreme outliers in bed/bath counts
        for col in ["list_beds", "detail_beds", "list_baths", "detail_baths"]:
            if col in df.columns:
                extreme_values = df[col] > 50  # Arbitrary threshold
                num_extreme = extreme_values.sum()
                if num_extreme > 0:
                    issues.append(
                        f"{num_extreme} rows with extreme {col} values (>50)"
                    )
        
        # Remove duplicates from invalid_indices
        invalid_indices = list(set(invalid_indices))
        
        is_valid = len(issues) == 0
        
        return {
            "is_valid": is_valid,
            "issues": issues,
            "invalid_indices": invalid_indices,
        }
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for the dataset.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary of summary statistics
        """
        stats = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }
        
        if TARGET_VARIABLE in df.columns:
            stats["price_stats"] = {
                "mean": df[TARGET_VARIABLE].mean(),
                "median": df[TARGET_VARIABLE].median(),
                "min": df[TARGET_VARIABLE].min(),
                "max": df[TARGET_VARIABLE].max(),
                "std": df[TARGET_VARIABLE].std(),
            }
        
        return stats