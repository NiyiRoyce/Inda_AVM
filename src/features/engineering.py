"""
Feature engineering and creation.
"""
import logging
import pandas as pd
import numpy as np
from typing import List

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates derived features from raw data."""
    
    def __init__(self):
        pass
    
    def create_room_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregate room-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new room features
        """
        df = df.copy()
        
        # Total rooms (beds + baths + toilets)
        df["rooms_total"] = (
            df.get("list_beds", 0) +
            df.get("list_baths", 0) +
            df.get("detail_toilets", 0)
        )
        
        # Total bathrooms (baths + toilets)
        df["total_bathrooms"] = (
            df.get("detail_baths", 0) +
            df.get("detail_toilets", 0)
        )
        
        logger.debug("Created room aggregate features")
        return df
    
    def create_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features measuring consistency between list and detail fields.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with consistency features
        """
        df = df.copy()
        
        # Difference between listed and detailed beds
        if "list_beds" in df.columns and "detail_beds" in df.columns:
            df["diff_beds"] = df["list_beds"] - df["detail_beds"]
        
        # Difference between listed and detailed baths
        if "list_baths" in df.columns and "detail_baths" in df.columns:
            df["diff_baths"] = df["list_baths"] - df["detail_baths"]
        
        logger.debug("Created consistency features")
        return df
    
    def create_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create accessibility and distance-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with distance features
        """
        df = df.copy()
        
        # Accessibility score (inverse of key amenity distances)
        distance_cols = [
            "school_distance_meters",
            "hospital_distance_meters", 
            "clinic_distance_meters"
        ]
        
        available_cols = [col for col in distance_cols if col in df.columns]
        
        if available_cols:
            # Sum of distances (with small epsilon to avoid division by zero)
            total_distance = df[available_cols].sum(axis=1) + 1
            df["accessibility_score"] = 1 / total_distance
            logger.debug("Created accessibility_score feature")
        
        return df
    
    def get_distance_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Automatically detect distance columns by suffix.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of distance column names
        """
        distance_cols = [
            col for col in df.columns 
            if col.endswith("_distance_meters")
        ]
        logger.info(f"Detected {len(distance_cols)} distance columns")
        return distance_cols
    
    def engineer_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all feature engineering steps.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering...")
        
        df = self.create_room_features(df)
        df = self.create_consistency_features(df)
        df = self.create_distance_features(df)
        
        logger.info("Feature engineering completed")
        return df