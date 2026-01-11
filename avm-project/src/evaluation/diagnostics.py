"""
Model diagnostics and error analysis.
"""
import logging
import numpy as np
from typing import Dict, List

from config.model_config import (
    EVALUATION_PERCENTILES,
    EXTREME_PREDICTION_THRESHOLDS,
    PRICE_TIER_QUANTILES
)

logger = logging.getLogger(__name__)


class ModelDiagnostics:
    """Performs diagnostic analysis on model predictions."""
    
    @staticmethod
    def analyze_prediction_ranges(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        percentiles: List[int] = None
    ) -> Dict:
        """
        Analyze prediction and truth ranges.
        
        Args:
            y_true: True prices
            y_pred: Predicted prices
            percentiles: Percentiles to compute
            
        Returns:
            Dictionary of range statistics
        """
        percentiles = percentiles or EVALUATION_PERCENTILES
        
        return {
            "true_min": y_true.min(),
            "true_max": y_true.max(),
            "pred_min": y_pred.min(),
            "pred_max": y_pred.max(),
            "true_percentiles": np.percentile(y_true, percentiles).tolist(),
            "pred_percentiles": np.percentile(y_pred, percentiles).tolist(),
        }
    
    @staticmethod
    def analyze_prediction_ratios(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        percentiles: List[int] = None
    ) -> Dict:
        """
        Analyze prediction/truth ratios.
        
        Args:
            y_true: True prices
            y_pred: Predicted prices
            percentiles: Percentiles to compute
            
        Returns:
            Dictionary of ratio statistics
        """
        percentiles = percentiles or EVALUATION_PERCENTILES
        
        price_ratios = y_pred / y_true
        
        return {
            "ratio_percentiles": np.percentile(price_ratios, percentiles).tolist(),
            "ratio_mean": price_ratios.mean(),
            "ratio_median": np.median(price_ratios),
        }
    
    @staticmethod
    def detect_extreme_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        low_threshold: float = None,
        high_threshold: float = None
    ) -> Dict:
        """
        Detect extreme over/under predictions.
        
        Args:
            y_true: True prices
            y_pred: Predicted prices
            low_threshold: Flag if pred < threshold * true
            high_threshold: Flag if pred > threshold * true
            
        Returns:
            Dictionary of extreme prediction statistics
        """
        low_threshold = low_threshold or EXTREME_PREDICTION_THRESHOLDS["low"]
        high_threshold = high_threshold or EXTREME_PREDICTION_THRESHOLDS["high"]
        
        extreme_low = y_pred < low_threshold * y_true
        extreme_high = y_pred > high_threshold * y_true
        
        num_extreme = np.sum(extreme_low | extreme_high)
        pct_extreme = num_extreme / len(y_true)
        
        return {
            "num_extreme_low": np.sum(extreme_low),
            "num_extreme_high": np.sum(extreme_high),
            "num_extreme_total": num_extreme,
            "pct_extreme": pct_extreme,
        }
    
    @staticmethod
    def analyze_by_price_tier(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantiles: List[float] = None
    ) -> Dict:
        """
        Analyze error metrics by price tier.
        
        Args:
            y_true: True prices
            y_pred: Predicted prices
            quantiles: Quantiles to define tiers
            
        Returns:
            Dictionary of tier-based metrics
        """
        quantiles = quantiles or PRICE_TIER_QUANTILES
        
        # Define price bins
        price_bins = np.quantile(y_true, quantiles)
        tiers = np.digitize(y_true, price_bins)
        
        # Compute MRE for each tier
        relative_errors = np.abs(y_pred - y_true) / y_true
        
        tier_metrics = {}
        for tier in np.unique(tiers):
            tier_mask = tiers == tier
            tier_mre = np.median(relative_errors[tier_mask])
            tier_count = np.sum(tier_mask)
            
            tier_metrics[f"tier_{tier}"] = {
                "mre": tier_mre,
                "count": tier_count,
                "price_min": y_true[tier_mask].min(),
                "price_max": y_true[tier_mask].max(),
            }
        
        return tier_metrics
    
    @staticmethod
    def run_full_diagnostics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Run complete diagnostic suite.
        
        Args:
            y_true: True prices
            y_pred: Predicted prices
            
        Returns:
            Dictionary of all diagnostic results
        """
        logger.info("Running full diagnostic analysis...")
        
        diagnostics = {}
        
        # Range analysis
        diagnostics["ranges"] = ModelDiagnostics.analyze_prediction_ranges(
            y_true, y_pred
        )
        
        # Ratio analysis
        diagnostics["ratios"] = ModelDiagnostics.analyze_prediction_ratios(
            y_true, y_pred
        )
        
        # Extreme predictions
        diagnostics["extremes"] = ModelDiagnostics.detect_extreme_predictions(
            y_true, y_pred
        )
        
        # Tier-based analysis
        diagnostics["tiers"] = ModelDiagnostics.analyze_by_price_tier(
            y_true, y_pred
        )
        
        logger.info("Diagnostic analysis completed")
        
        return diagnostics
    
    @staticmethod
    def format_diagnostics(diagnostics: Dict) -> str:
        """
        Format diagnostics for display.
        
        Args:
            diagnostics: Dictionary of diagnostic results
            
        Returns:
            Formatted string
        """
        lines = ["Model Diagnostics:", "=" * 60]
        
        # Ranges
        if "ranges" in diagnostics:
            r = diagnostics["ranges"]
            lines.append("\nPrediction Ranges:")
            lines.append(f"  True:      ₦{r['true_min']:,.0f} - ₦{r['true_max']:,.0f}")
            lines.append(f"  Predicted: ₦{r['pred_min']:,.0f} - ₦{r['pred_max']:,.0f}")
        
        # Extremes
        if "extremes" in diagnostics:
            e = diagnostics["extremes"]
            lines.append(f"\nExtreme Predictions:")
            lines.append(f"  Total: {e['num_extreme_total']} ({e['pct_extreme']:.2%})")
            lines.append(f"  Under-predictions: {e['num_extreme_low']}")
            lines.append(f"  Over-predictions: {e['num_extreme_high']}")
        
        # Tiers
        if "tiers" in diagnostics:
            lines.append("\nError by Price Tier:")
            for tier_name, tier_data in diagnostics["tiers"].items():
                lines.append(
                    f"  {tier_name}: MRE={tier_data['mre']:.3%} "
                    f"(n={tier_data['count']})"
                )
        
        lines.append("=" * 60)
        
        return "\n".join(lines)