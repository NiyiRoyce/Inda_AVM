"""
LightGBM residual model for capturing non-linear patterns.
"""
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import lightgbm as lgb

from src.models.base import BaseAVMModel
from config.model_config import LIGHTGBM_PARAMS, LIGHTGBM_TRAINING
from config.settings import MODEL_ARTIFACTS

logger = logging.getLogger(__name__)


class ResidualAVMModel(BaseAVMModel):
    """LightGBM model for learning residuals from linear model."""
    
    def __init__(self, params: dict = None, training_params: dict = None):
        """
        Initialize residual model.
        
        Args:
            params: LightGBM parameters (overrides defaults)
            training_params: Training parameters (overrides defaults)
        """
        super().__init__()
        self.params = {**LIGHTGBM_PARAMS, **(params or {})}
        self.training_params = {**LIGHTGBM_TRAINING, **(training_params or {})}
        self.best_iteration = None
    
    def fit(self, X: pd.DataFrame, y_residuals: np.ndarray):
        """
        Fit LightGBM on residuals.
        
        Args:
            X: Feature DataFrame (including linear signal)
            y_residuals: Residuals from linear model
        """
        logger.info("Fitting residual LightGBM model...")
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X, y_residuals)
        
        # Train model with early stopping
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=self.training_params["num_boost_round"],
            valid_sets=[train_data],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.training_params["early_stopping_rounds"]
                ),
                lgb.log_evaluation(period=0),  # Suppress verbose output
            ]
        )
        
        self.best_iteration = self.model.best_iteration
        self.is_fitted = True
        
        logger.info(
            f"Residual model fitted. Best iteration: {self.best_iteration}"
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict residual corrections.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predicted residuals
        """
        self._check_fitted()
        
        predictions = self.model.predict(
            X,
            num_iteration=self.best_iteration
        )
        
        return predictions
    
    def save(self, filepath: Path = None):
        """
        Save LightGBM model.
        
        Args:
            filepath: Path to save model
        """
        self._check_fitted()
        
        filepath = filepath or MODEL_ARTIFACTS["residual_model"]
        
        # Save model using joblib (compatible with deployment)
        joblib.dump(self.model, filepath)
        
        logger.info(f"Residual model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path = None):
        """
        Load LightGBM model.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded ResidualAVMModel instance
        """
        filepath = filepath or MODEL_ARTIFACTS["residual_model"]
        
        instance = cls()
        instance.model = joblib.load(filepath)
        instance.best_iteration = instance.model.best_iteration
        instance.is_fitted = True
        
        logger.info(f"Residual model loaded from {filepath}")
        
        return instance