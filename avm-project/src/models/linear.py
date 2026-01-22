"""
Linear regression model with smearing correction.
"""
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression

from src.models.base import BaseAVMModel
from config.model_config import LINEAR_MODEL_CONFIG
from config.settings import MODEL_ARTIFACTS

logger = logging.getLogger(__name__)


class LinearAVMModel(BaseAVMModel):
    """Linear regression model for AVM with smearing correction."""
    
    def __init__(self, **kwargs):
        """
        Initialize linear model.
        
        Args:
            **kwargs: Arguments passed to LinearRegression
        """
        super().__init__()
        model_config = {**LINEAR_MODEL_CONFIG, **kwargs}
        self.model = LinearRegression(**model_config)
        self.smearing_factor = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit linear regression and compute smearing factor.
        
        Args:
            X: Feature DataFrame
            y: Log-transformed target prices
        """
        logger.info("Fitting linear regression model...")
        
        # Fit model
        self.model.fit(X, y)
        
        # Compute smearing factor from training residuals
        y_pred_log = self.model.predict(X)
        residuals_log = y.to_numpy() - y_pred_log
        
        # Smearing factor = mean of exp(residuals)
        self.smearing_factor = np.mean(np.exp(residuals_log))
        
        self.is_fitted = True
        
        logger.info(f"Linear model fitted. Smearing factor: {self.smearing_factor:.4f}")
        logger.info(f"Model coefficients: {len(self.model.coef_)} features")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict in log-space (use predict_price for price-space).
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Log-space predictions
        """
        self._check_fitted()
        return self.model.predict(X)
    
    def predict_price(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict in price-space with smearing correction.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Price-space predictions
        """
        self._check_fitted()
        
        log_pred = self.predict(X)
        price_pred = np.exp(log_pred) * self.smearing_factor
        
        return price_pred
    
    def get_residuals(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Compute log-space residuals for training residual model.
        
        Args:
            X: Feature DataFrame
            y: True log prices
            
        Returns:
            Log-space residuals
        """
        self._check_fitted()
        
        y_pred = self.predict(X)
        residuals = y.to_numpy() - y_pred
        
        return residuals
    
    def save(self, model_path: Path = None, smearing_path: Path = None):
        """
        Save model and smearing factor separately.
        
        Args:
            model_path: Path to save model
            smearing_path: Path to save smearing factor
        """
        self._check_fitted()
        
        model_path = model_path or MODEL_ARTIFACTS["linear_model"]
        smearing_path = smearing_path or MODEL_ARTIFACTS["smearing_factor"]
        
        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"Linear model saved to {model_path}")
        
        # Save smearing factor
        joblib.dump(self.smearing_factor, smearing_path)
        logger.info(f"Smearing factor saved to {smearing_path}")
    
    @classmethod
    def load(cls, model_path: Path = None, smearing_path: Path = None):
        """
        Load model and smearing factor.
        
        Args:
            model_path: Path to load model from
            smearing_path: Path to load smearing factor from
            
        Returns:
            Loaded LinearAVMModel instance
        """
        model_path = model_path or MODEL_ARTIFACTS["linear_model"]
        smearing_path = smearing_path or MODEL_ARTIFACTS["smearing_factor"]
        
        instance = cls()
        instance.model = joblib.load(model_path)
        instance.smearing_factor = joblib.load(smearing_path)
        instance.is_fitted = True
        
        logger.info(f"Linear model loaded from {model_path}")
        logger.info(f"Smearing factor loaded: {instance.smearing_factor:.4f}")
        
        return instance