"""
Data cleaning operations: outlier handling, invalid coordinates.
"""
import logging
import pandas as pd
import numpy as np
from typing import List

from config.settings import GEOGRAPHIC_BOUNDS, BED_BATH_CAP
from config.features import CAPPABLE_FEATURES, TARGET_VARIABLE

logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles data cleaning operations."""
    
    def __init__(self):
        self.geo_bounds = GEOGRAPHIC_BOUNDS
        self.bed_bath_cap = BED_BATH_CAP
    
    def remove_invalid_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with non-positive prices.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        initial_len = len(df)
        df = df[df[TARGET_VARIABLE] > 0].copy()
        removed = initial_len - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} rows with non-positive prices")
        
        # Assert no non-positive prices remain
        assert (df[TARGET_VARIABLE] > 0).all(), "Invalid non-positive prices detected"
        
        return df
    
    def cap_bed_bath_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cap bed and bath counts to realistic upper bounds.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with capped values
        """
        df = df.copy()
        
        for col in CAPPABLE_FEATURES:
            if col in df.columns:
                original_max = df[col].max()
                df[col] = df[col].clip(lower=0, upper=self.bed_bath_cap)
                new_max = df[col].max()
                
                if original_max > self.bed_bath_cap:
                    logger.info(
                        f"Capped {col} from max {original_max} to {new_max}"
                    )
        
        return df
    
    def clean_geographic_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Set invalid geographic coordinates to NaN.
        Nigeria bounds: latitude ≈ 4–14, longitude ≈ 2–15.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned coordinates
        """
        df = df.copy()
        
        # Clean latitude
        if "latitude" in df.columns:
            invalid_lat = (
                (df["latitude"] < self.geo_bounds["latitude_min"]) |
                (df["latitude"] > self.geo_bounds["latitude_max"])
            )
            num_invalid = invalid_lat.sum()
            
            if num_invalid > 0:
                df.loc[invalid_lat, "latitude"] = np.nan
                logger.info(f"Set {num_invalid} invalid latitude values to NaN")
        
        # Clean longitude
        if "longitude" in df.columns:
            invalid_lon = (
                (df["longitude"] < self.geo_bounds["longitude_min"]) |
                (df["longitude"] > self.geo_bounds["longitude_max"])
            )
            num_invalid = invalid_lon.sum()
            
            if num_invalid > 0:
                df.loc[invalid_lon, "longitude"] = np.nan
                logger.info(f"Set {num_invalid} invalid longitude values to NaN")
        
        return df
    
    def clean_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all cleaning operations in sequence.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Fully cleaned DataFrame
        """
        logger.info("Starting data cleaning pipeline...")
        
        df = self.remove_invalid_prices(df)
        df = self.cap_bed_bath_values(df)
        df = self.clean_geographic_coordinates(df)
        
        logger.info("Data cleaning completed")
        return df