"""
Base model class for AVM models.
"""
import logging
import joblib
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseAVMModel(ABC):
    """Abstract base class for AVM models."""
    
    def __init__(self):
        """Initialize base model."""
        self.model = None
        self.is_fitted = False
    
    def _check_fitted(self):
        """
        Check if model is fitted before making predictions.
        
        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before making predictions"
            )
    
    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model.
        
        Args:
            X: Training features
            y: Training target
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def save(self, filepath: Path = None):
        """
        Save fitted model to disk.
        
        Args:
            filepath: Path to save model
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: Path = None):
        """
        Load fitted model from disk.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded model instance
        """
        pass