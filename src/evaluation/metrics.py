"""
Model evaluation metrics.
"""
import logging
import numpy as np
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Computes evaluation metrics for AVM models."""
    
    @staticmethod
    def compute_price_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute price-space metrics (MAE, RMSE, R²).
        
        Args:
            y_true: True prices
            y_pred: Predicted prices
            
        Returns:
            Dictionary of metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
    
    @staticmethod
    def compute_relative_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute relative error metrics (MRE, MAPE).
        
        Args:
            y_true: True prices
            y_pred: Predicted prices
            
        Returns:
            Dictionary of relative metrics
        """
        relative_errors = np.abs(y_pred - y_true) / y_true
        
        # Median Relative Error
        mre = np.median(relative_errors)
        
        # Mean Absolute Percentage Error
        mape = np.mean(relative_errors)
        
        return {
            "mre": mre,
            "mape": mape,
        }
    
    @staticmethod
    def compute_log_metrics(
        y_true_log: np.ndarray, 
        y_pred_log: np.ndarray
    ) -> Dict:
        """
        Compute log-space RMSE.
        
        Args:
            y_true_log: True log prices
            y_pred_log: Predicted log prices
            
        Returns:
            Dictionary of log metrics
        """
        rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
        
        return {
            "rmse_log": rmse_log,
        }
    
    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_true_log: np.ndarray = None,
        y_pred_log: np.ndarray = None
    ) -> Dict:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: True prices
            y_pred: Predicted prices
            y_true_log: True log prices (optional)
            y_pred_log: Predicted log prices (optional)
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Price-space metrics
        price_metrics = ModelEvaluator.compute_price_metrics(y_true, y_pred)
        metrics.update(price_metrics)
        
        # Relative metrics
        relative_metrics = ModelEvaluator.compute_relative_metrics(y_true, y_pred)
        metrics.update(relative_metrics)
        
        # Log-space metrics (if provided)
        if y_true_log is not None and y_pred_log is not None:
            log_metrics = ModelEvaluator.compute_log_metrics(y_true_log, y_pred_log)
            metrics.update(log_metrics)
        
        return metrics
    
    @staticmethod
    def format_metrics(metrics: Dict) -> str:
        """
        Format metrics for display.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Formatted string
        """
        lines = ["Model Evaluation Metrics:", "=" * 40]
        
        if "mae" in metrics:
            lines.append(f"MAE:        ₦{metrics['mae']:,.0f}")
        if "rmse" in metrics:
            lines.append(f"RMSE:       ₦{metrics['rmse']:,.0f}")
        if "r2" in metrics:
            lines.append(f"R²:         {metrics['r2']:.3f}")
        if "mre" in metrics:
            lines.append(f"MRE:        {metrics['mre']:.3%}")
        if "mape" in metrics:
            lines.append(f"MAPE:       {metrics['mape']:.3%}")
        if "rmse_log" in metrics:
            lines.append(f"Log-RMSE:   {metrics['rmse_log']:.3f}")
        
        lines.append("=" * 40)
        
        return "\n".join(lines)