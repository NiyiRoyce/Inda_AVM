"""
Ensemble predictor combining linear and residual models.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from src.models.linear import LinearAVMModel
from src.models.residual import ResidualAVMModel
from src.preprocessing.imputers import FeatureImputer
from src.features.selectors import FeatureSelector

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor that combines linear and residual models.
    Handles preprocessing and feature selection.
    """
    
    def __init__(
        self,
        linear_model: LinearAVMModel = None,
        residual_model: ResidualAVMModel = None,
        imputer: FeatureImputer = None,
        feature_names: list = None
    ):
        """
        Initialize ensemble predictor.
        
        Args:
            linear_model: Fitted linear model
            residual_model: Fitted residual model
            imputer: Fitted imputer
            feature_names: List of feature names
        """
        self.linear_model = linear_model
        self.residual_model = residual_model
        self.imputer = imputer
        self.feature_names = feature_names
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input features.
        
        Args:
            X: Raw feature DataFrame
            
        Returns:
            Preprocessed features
        """
        if self.imputer is None:
            logger.warning("No imputer available, skipping imputation")
            return X
        
        return self.imputer.transform(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions (linear + residual).
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Ensemble price predictions
        """
        if self.linear_model is None or self.residual_model is None:
            raise RuntimeError("Both linear and residual models must be loaded")
        
        # Preprocess
        X_proc = self.preprocess(X)
        
        # Ensure correct features
        if self.feature_names:
            X_proc = X_proc[self.feature_names]
        
        # Linear prediction (price-space with smearing)
        linear_price = self.linear_model.predict_price(X_proc)
        
        # Add linear signal as feature for residual model
        X_residual = X_proc.copy()
        X_residual["linreg_signal"] = linear_price
        
        # Residual prediction
        residual = self.residual_model.predict(X_residual)
        
        # Final ensemble prediction
        final_pred = linear_price + residual
        
        return final_pred
    
    @classmethod
    def load_from_artifacts(cls) -> "EnsemblePredictor":
        """
        Load all components from saved artifacts.
        
        Returns:
            Loaded EnsemblePredictor instance
        """
        logger.info("Loading ensemble predictor from artifacts...")
        
        # Load models
        linear_model = LinearAVMModel.load()
        residual_model = ResidualAVMModel.load()
        
        # Load imputer
        try:
            imputer = FeatureImputer.load()
        except FileNotFoundError:
            logger.warning("Imputer not found, predictions will assume clean data")
            imputer = None
        
        # Load feature names
        try:
            feature_names = FeatureSelector.load_feature_names()
        except FileNotFoundError:
            logger.warning("Feature names not found")
            feature_names = None
        
        instance = cls(
            linear_model=linear_model,
            residual_model=residual_model,
            imputer=imputer,
            feature_names=feature_names
        )
        
        logger.info("Ensemble predictor loaded successfully")
        return instance