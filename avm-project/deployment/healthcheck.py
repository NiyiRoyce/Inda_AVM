"""
Health check endpoint for deployment.
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class HealthChecker:
    """Performs health checks for the prediction service."""
    
    def __init__(self, predictor):
        """
        Initialize health checker.
        
        Args:
            predictor: Predictor instance to check
        """
        self.predictor = predictor
    
    def check_models_loaded(self) -> bool:
        """
        Check if models are loaded.
        
        Returns:
            True if models are loaded, False otherwise
        """
        try:
            return (
                self.predictor.ensemble is not None and
                self.predictor.ensemble.linear_model is not None and
                self.predictor.ensemble.residual_model is not None
            )
        except Exception as e:
            logger.error(f"Model check failed: {e}")
            return False
    
    def check_prediction_capability(self) -> bool:
        """
        Check if prediction is working with dummy data.
        
        Returns:
            True if prediction works, False otherwise
        """
        try:
            # Create minimal dummy instance
            dummy_instance = {
                'list_beds': 4,
                'list_baths': 3,
                'detail_beds': 4,
                'detail_baths': 3,
                'detail_toilets': 4,
                'latitude': 6.5,
                'longitude': 3.4,
                'school_distance_meters': 1000,
                'hospital_distance_meters': 1500,
                'clinic_distance_meters': 2000,
                'mall_distance_meters': 1200,
                'pharmacy_distance_meters': 800,
                'police_station_distance_meters': 1800,
                'aerodrome_distance_meters': 20000,
            }
            
            # Try prediction
            result = self.predictor.predict([dummy_instance])
            
            # Check result is valid
            return (
                result is not None and
                len(result) > 0 and
                isinstance(result[0], dict) and
                'predicted_price' in result[0]
            )
            
        except Exception as e:
            logger.error(f"Prediction check failed: {e}")
            return False
    
    def get_health_status(self) -> Dict:
        """
        Get comprehensive health status.
        
        Returns:
            Dictionary with health status information
        """
        models_loaded = self.check_models_loaded()
        prediction_works = self.check_prediction_capability()
        
        is_healthy = models_loaded and prediction_works
        
        return {
            'status': 'healthy' if is_healthy else 'unhealthy',
            'checks': {
                'models_loaded': models_loaded,
                'prediction_works': prediction_works,
            },
            'details': {
                'linear_model': self.predictor.ensemble.linear_model is not None if self.predictor.ensemble else False,
                'residual_model': self.predictor.ensemble.residual_model is not None if self.predictor.ensemble else False,
            }
        }