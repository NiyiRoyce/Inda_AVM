"""
Inference pipeline for making predictions on new data.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union

from src.models.ensemble import EnsemblePredictor
from src.preprocessing.cleaners import DataCleaner
from src.preprocessing.transformers import DataTransformer
from src.features.engineering import FeatureEngineer
from src.features.extractors import FeatureExtractor

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Orchestrates the prediction workflow for new data."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Optional path to model artifacts
        """
        self.ensemble = None
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.engineer = FeatureEngineer()
        self.extractor = FeatureExtractor()
        
        # Load models
        self.load_models(model_path)
    
    def load_models(self, model_path: Optional[Path] = None):
        """
        Load trained models.
        
        Args:
            model_path: Optional path to model directory
        """
        logger.info("Loading trained models...")
        
        if model_path:
            # Load from custom path (not implemented in this version)
            logger.warning("Custom model path loading not implemented, using defaults")
        
        self.ensemble = EnsemblePredictor.load_from_artifacts()
        logger.info("Models loaded successfully")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing input data...")
        
        # Clean data (but don't remove rows)
        df = self.cleaner.cap_bed_bath_values(df)
        df = self.cleaner.clean_geographic_coordinates(df)
        
        # Extract text features if columns exist
        df = self.extractor.extract_all(df)
        
        # Transform
        distance_cols = self.engineer.get_distance_columns(df)
        if distance_cols:
            df = self.transformer.log_transform_distances(df, distance_cols)
        
        # Engineer features
        df = self.engineer.engineer_all(df)
        
        logger.info("Preprocessing completed")
        return df
    
    def predict(
        self, 
        df: pd.DataFrame,
        include_preprocessing: bool = True
    ) -> np.ndarray:
        """
        Generate predictions for input data.
        
        Args:
            df: Input DataFrame
            include_preprocessing: Whether to preprocess data
            
        Returns:
            Array of predicted prices
        """
        if include_preprocessing:
            df = self.preprocess_data(df)
        
        logger.info(f"Generating predictions for {len(df)} properties...")
        predictions = self.ensemble.predict(df)
        
        logger.info("Predictions generated successfully")
        return predictions
    
    def predict_with_metadata(
        self,
        df: pd.DataFrame,
        include_preprocessing: bool = True
    ) -> pd.DataFrame:
        """
        Generate predictions and return DataFrame with metadata.
        
        Args:
            df: Input DataFrame
            include_preprocessing: Whether to preprocess data
            
        Returns:
            DataFrame with original data and predictions
        """
        # Make predictions
        predictions = self.predict(df, include_preprocessing)
        
        # Create result DataFrame
        result = df.copy()
        result["predicted_price"] = predictions
        result["currency"] = "NGN"
        
        # Add confidence indicators (placeholder for future enhancement)
        result["prediction_confidence"] = "medium"
        
        return result
    
    def predict_from_csv(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load CSV, predict, and optionally save results.
        
        Args:
            input_path: Path to input CSV
            output_path: Optional path to save predictions
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Predict
        results = self.predict_with_metadata(df)
        
        # Save if requested
        if output_path:
            results.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        
        return results
    
    def predict_single(self, features: dict) -> float:
        """
        Predict for a single property.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Predicted price
        """
        df = pd.DataFrame([features])
        predictions = self.predict(df)
        return predictions[0]
    
    def get_prediction_summary(self, predictions: np.ndarray) -> dict:
        """
        Generate summary statistics for predictions.
        
        Args:
            predictions: Array of predicted prices
            
        Returns:
            Dictionary of summary statistics
        """
        return {
            "count": len(predictions),
            "mean": float(predictions.mean()),
            "median": float(np.median(predictions)),
            "min": float(predictions.min()),
            "max": float(predictions.max()),
            "std": float(predictions.std()),
            "percentiles": {
                "p25": float(np.percentile(predictions, 25)),
                "p50": float(np.percentile(predictions, 50)),
                "p75": float(np.percentile(predictions, 75)),
                "p90": float(np.percentile(predictions, 90)),
                "p95": float(np.percentile(predictions, 95)),
            }
        }