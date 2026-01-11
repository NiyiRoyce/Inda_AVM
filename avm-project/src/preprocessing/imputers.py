"""
Missing value imputation.
"""
import logging
import pandas as pd
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer

from config.model_config import IMPUTATION_STRATEGY
from config.settings import MODEL_ARTIFACTS

logger = logging.getLogger(__name__)


class FeatureImputer:
    """Handles missing value imputation."""
    
    def __init__(self, strategy: str = IMPUTATION_STRATEGY):
        """
        Initialize imputer.
        
        Args:
            strategy: Imputation strategy ('median', 'mean', 'most_frequent')
        """
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy)
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, feature_cols: list) -> "FeatureImputer":
        """
        Fit imputer on training data.
        
        Args:
            df: Training DataFrame
            feature_cols: List of feature column names
            
        Returns:
            Self (for method chaining)
        """
        self.feature_names = feature_cols
        
        # Check for missing columns
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        
        X = df[feature_cols]
        self.imputer.fit(X)
        self.is_fitted = True
        
        # Log statistics
        n_features_with_missing = sum(df[feature_cols].isnull().any())
        logger.info(
            f"Fitted imputer (strategy={self.strategy}) on {len(feature_cols)} features. "
            f"{n_features_with_missing} features had missing values."
        )
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted imputer.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            DataFrame with imputed values
        """
        if not self.is_fitted:
            raise RuntimeError("Imputer must be fitted before transform")
        
        df = df.copy()
        X_imputed = self.imputer.transform(df[self.feature_names])
        df[self.feature_names] = X_imputed
        
        logger.debug(f"Imputed missing values for {len(self.feature_names)} features")
        return df
    
    def fit_transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            feature_cols: List of feature column names
            
        Returns:
            Transformed DataFrame
        """
        self.fit(df, feature_cols)
        return self.transform(df)
    
    def save(self, filepath: Path = None) -> None:
        """
        Save fitted imputer to disk.
        
        Args:
            filepath: Path to save imputer. If None, uses default from config.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted imputer")
        
        filepath = filepath or MODEL_ARTIFACTS["imputer"]
        
        # Save both imputer and feature names
        artifact = {
            "imputer": self.imputer,
            "feature_names": self.feature_names,
            "strategy": self.strategy,
        }
        
        joblib.dump(artifact, filepath)
        logger.info(f"Imputer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path = None) -> "FeatureImputer":
        """
        Load fitted imputer from disk.
        
        Args:
            filepath: Path to load imputer from. If None, uses default from config.
            
        Returns:
            Loaded FeatureImputer instance
        """
        filepath = filepath or MODEL_ARTIFACTS["imputer"]
        
        artifact = joblib.load(filepath)
        
        instance = cls(strategy=artifact["strategy"])
        instance.imputer = artifact["imputer"]
        instance.feature_names = artifact["feature_names"]
        instance.is_fitted = True
        
        logger.info(f"Imputer loaded from {filepath}")
        return instance