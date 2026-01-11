"""
Data transformations: log transforms, scaling, encoding.
"""
import logging
import pandas as pd
import numpy as np

from config.features import TARGET_VARIABLE, LOG_TARGET_VARIABLE

logger = logging.getLogger(__name__)


class DataTransformer:
    """Handles data transformations."""
    
    def __init__(self):
        pass
    
    def create_log_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create log-transformed target variable.
        
        Args:
            df: Input DataFrame with price_naira column
            
        Returns:
            DataFrame with log_price column added
        """
        df = df.copy()
        
        # Ensure no non-positive prices
        if (df[TARGET_VARIABLE] <= 0).any():
            raise ValueError(
                "Cannot log-transform non-positive prices. "
                "Run data cleaning first."
            )
        
        df[LOG_TARGET_VARIABLE] = np.log(df[TARGET_VARIABLE])
        logger.info(f"Created {LOG_TARGET_VARIABLE} column")
        
        return df
    
    def log_transform_distances(
        self, 
        df: pd.DataFrame, 
        distance_cols: list
    ) -> pd.DataFrame:
        """
        Apply log(1 + x) transformation to distance features.
        
        Args:
            df: Input DataFrame
            distance_cols: List of distance column names
            
        Returns:
            DataFrame with log-transformed distance columns
        """
        df = df.copy()
        
        for col in distance_cols:
            if col in df.columns:
                log_col = f"log_{col}"
                df[log_col] = np.log1p(df[col])  # log(1 + x)
                logger.debug(f"Created {log_col}")
        
        logger.info(f"Log-transformed {len(distance_cols)} distance features")
        return df
    
    def transform_all(
        self, 
        df: pd.DataFrame, 
        distance_cols: list = None
    ) -> pd.DataFrame:
        """
        Run all transformations.
        
        Args:
            df: Input DataFrame
            distance_cols: List of distance columns to transform
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Starting data transformations...")
        
        df = self.create_log_target(df)
        
        if distance_cols:
            df = self.log_transform_distances(df, distance_cols)
        
        logger.info("Data transformations completed")
        return df