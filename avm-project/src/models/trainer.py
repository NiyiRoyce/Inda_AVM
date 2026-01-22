"""
Model training orchestration.
"""
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict

from src.models.linear import LinearAVMModel
from src.models.residual import ResidualAVMModel
from src.preprocessing.imputers import FeatureImputer
from src.features.selectors import FeatureSelector

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrates training of linear and residual models."""
    
    def __init__(self):
        self.linear_model = None
        self.residual_model = None
        self.imputer = None
        self.feature_selector = None
        self.feature_names = None
    
    def prepare_features(
        self, 
        X_train: pd.DataFrame,
        X_val: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features: select and impute.
        
        Args:
            X_train: Training features
            X_val: Validation features
            
        Returns:
            Tuple of (X_train_proc, X_val_proc)
        """
        logger.info("Preparing features...")
        
        # Select features
        self.feature_selector = FeatureSelector()
        self.feature_names = self.feature_selector.select_features(
            X_train, 
            include_distance=True
        )
        
        # Fit and transform imputer on training data
        self.imputer = FeatureImputer()
        X_train_proc = self.imputer.fit_transform(X_train, self.feature_names)
        
        # Transform validation data
        X_val_proc = self.imputer.transform(X_val)
        
        # Select only needed features
        X_train_proc = X_train_proc[self.feature_names]
        X_val_proc = X_val_proc[self.feature_names]
        
        logger.info(f"Features prepared. Shape: {X_train_proc.shape}")
        
        return X_train_proc, X_val_proc
    
    def train_linear_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> LinearAVMModel:
        """
        Train linear regression model.
        
        Args:
            X_train: Training features
            y_train: Training target (log-space)
            
        Returns:
            Trained linear model
        """
        logger.info("Training linear model...")
        
        self.linear_model = LinearAVMModel()
        self.linear_model.fit(X_train, y_train)
        
        logger.info("Linear model training completed")
        return self.linear_model
    
    def prepare_residual_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Prepare features for residual model.
        Adds linear signal and computes residual targets.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training target (log-space)
            
        Returns:
            Tuple of (X_residual_train, X_residual_val, y_residual_train)
        """
        logger.info("Preparing residual model features...")
        
        # Get linear predictions (price-space)
        linear_train_price = self.linear_model.predict_price(X_train)
        linear_val_price = self.linear_model.predict_price(X_val)
        
        # Add linear signal as feature
        X_residual_train = X_train.copy()
        X_residual_train["linreg_signal"] = linear_train_price
        
        X_residual_val = X_val.copy()
        X_residual_val["linreg_signal"] = linear_val_price
        
        # Compute residual targets (price-space)
        y_train_price = np.exp(y_train.to_numpy())
        y_residual_train = y_train_price - linear_train_price
        
        logger.info(
            f"Residual features prepared. "
            f"Mean residual: {y_residual_train.mean():.2f}"
        )
        
        return X_residual_train, X_residual_val, y_residual_train
    
    def train_residual_model(
        self,
        X_residual_train: pd.DataFrame,
        y_residual_train: np.ndarray
    ) -> ResidualAVMModel:
        """
        Train residual LightGBM model.
        
        Args:
            X_residual_train: Training features with linear signal
            y_residual_train: Residual targets
            
        Returns:
            Trained residual model
        """
        logger.info("Training residual model...")
        
        self.residual_model = ResidualAVMModel()
        self.residual_model.fit(X_residual_train, y_residual_train)
        
        logger.info("Residual model training completed")
        return self.residual_model
    
    def train_all(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict:
        """
        Train complete ensemble pipeline.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training target (log-space)
            
        Returns:
            Dictionary containing all trained components
        """
        logger.info("=" * 60)
        logger.info("Starting full model training pipeline")
        logger.info("=" * 60)
        
        # 1. Prepare features
        X_train_proc, X_val_proc = self.prepare_features(X_train, X_val)
        
        # 2. Train linear model
        self.train_linear_model(X_train_proc, y_train)
        
        # 3. Prepare residual features
        X_res_train, X_res_val, y_res_train = self.prepare_residual_features(
            X_train_proc, X_val_proc, y_train
        )
        
        # 4. Train residual model
        self.train_residual_model(X_res_train, y_res_train)
        
        logger.info("=" * 60)
        logger.info("Model training pipeline completed successfully")
        logger.info("=" * 60)
        
        return {
            "linear_model": self.linear_model,
            "residual_model": self.residual_model,
            "imputer": self.imputer,
            "feature_selector": self.feature_selector,
            "feature_names": self.feature_names,
            "X_val_processed": X_val_proc,
            "X_residual_val": X_res_val,
        }
    
    def save_all(self):
        """Save all trained components."""
        logger.info("Saving all model artifacts...")
        
        if self.linear_model:
            self.linear_model.save()
        
        if self.residual_model:
            self.residual_model.save()
        
        if self.imputer:
            self.imputer.save()
        
        if self.feature_selector:
            self.feature_selector.save_feature_names()
        
        logger.info("All artifacts saved successfully")