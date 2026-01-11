"""
Vertex AI custom predictor for deployment.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from src.models.ensemble import EnsemblePredictor

logger = logging.getLogger(__name__)


class VertexAIPredictor:
    """
    Custom predictor for Vertex AI deployment.
    Loads models from GCS and serves predictions.
    """
    
    def __init__(self):
        """Initialize predictor by loading models from GCS."""
        self.ensemble = None
        self._load_models()
    
    def _load_models(self):
        """Load models from model directory."""
        # Vertex AI sets AIP_STORAGE_URI environment variable
        model_dir = os.environ.get("AIP_STORAGE_URI", ".")
        
        logger.info(f"Loading models from: {model_dir}")
        
        try:
            # Load ensemble predictor
            # Note: In production, models should be in the container or GCS
            self.ensemble = EnsemblePredictor.load_from_artifacts()
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def preprocess(self, instances: List[Dict]) -> pd.DataFrame:
        """
        Preprocess raw input instances.
        
        Args:
            instances: List of feature dictionaries
            
        Returns:
            DataFrame ready for prediction
        """
        df = pd.DataFrame(instances)
        
        # Any additional preprocessing can go here
        # The ensemble predictor handles imputation
        
        return df
    
    def predict(self, instances: List[Dict]) -> List[Dict]:
        """
        Generate predictions for input instances.
        
        Args:
            instances: List of feature dictionaries
            
        Returns:
            List of prediction dictionaries
        """
        try:
            # Preprocess
            X = self.preprocess(instances)
            
            # Predict
            predictions = self.ensemble.predict(X)
            
            # Format output
            results = []
            for pred in predictions:
                results.append({
                    "predicted_price": float(pred),
                    "currency": "NGN"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


# For Vertex AI compatibility
if __name__ == "__main__":
    # This allows the predictor to be used as a serving application
    predictor = VertexAIPredictor()