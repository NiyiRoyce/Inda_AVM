"""
Feature selection utilities.
"""
import logging
import json
from typing import List
from pathlib import Path

from config.features import NUMERIC_FEATURES, DISTANCE_SUFFIX
from config.settings import MODEL_ARTIFACTS

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Manages feature selection and storage."""
    
    def __init__(self):
        self.selected_features = None
    
    def select_features(self, df, include_distance: bool = True) -> List[str]:
        """
        Select features for modeling.
        
        Args:
            df: Input DataFrame
            include_distance: Whether to include distance features
            
        Returns:
            List of selected feature names
        """
        features = []
        
        # Add numeric features
        for feat in NUMERIC_FEATURES:
            if feat in df.columns:
                features.append(feat)
        
        # Add distance features if requested
        if include_distance:
            distance_features = [
                col for col in df.columns 
                if col.endswith(DISTANCE_SUFFIX)
            ]
            features.extend(distance_features)
        
        # Store selected features
        self.selected_features = features
        
        logger.info(f"Selected {len(features)} features for modeling")
        logger.debug(f"Features: {features}")
        
        return features
    
    def save_feature_names(self, filepath: Path = None) -> None:
        """
        Save selected feature names to JSON.
        
        Args:
            filepath: Path to save feature names
        """
        if self.selected_features is None:
            raise RuntimeError("No features selected yet")
        
        filepath = filepath or MODEL_ARTIFACTS["feature_names"]
        
        with open(filepath, 'w') as f:
            json.dump({"features": self.selected_features}, f, indent=2)
        
        logger.info(f"Feature names saved to {filepath}")
    
    @staticmethod
    def load_feature_names(filepath: Path = None) -> List[str]:
        """
        Load feature names from JSON.
        
        Args:
            filepath: Path to load feature names from
            
        Returns:
            List of feature names
        """
        filepath = filepath or MODEL_ARTIFACTS["feature_names"]
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        features = data["features"]
        logger.info(f"Loaded {len(features)} feature names from {filepath}")
        
        return features