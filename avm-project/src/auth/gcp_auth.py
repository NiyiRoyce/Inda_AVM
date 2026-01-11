"""
GCP authentication handling for different environments.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def authenticate_gcp(use_colab: bool = False) -> None:
    """
    Authenticate with GCP based on environment.
    
    Args:
        use_colab: If True, use Colab authentication. 
                   If False, use application default credentials.
    
    Raises:
        RuntimeError: If authentication fails
    """
    try:
        if use_colab:
            logger.info("Authenticating using Google Colab...")
            from google.colab import auth
            auth.authenticate_user()
            logger.info("Colab authentication successful")
        else:
            logger.info("Using Application Default Credentials")
            # ADC will be used automatically by google-cloud libraries
            # Ensure GOOGLE_APPLICATION_CREDENTIALS is set or gcloud auth is configured
            logger.info("Authentication configured via ADC")
            
    except ImportError as e:
        if use_colab:
            raise RuntimeError(
                "Colab authentication requested but google.colab not available. "
                "Are you running in Colab?"
            ) from e
        logger.warning("Could not import Colab auth (expected outside Colab)")
    except Exception as e:
        raise RuntimeError(f"GCP authentication failed: {e}") from e


def get_credentials(use_colab: bool = False):
    """
    Get GCP credentials object.
    
    Args:
        use_colab: Whether to use Colab authentication
        
    Returns:
        Credentials object or None (uses ADC)
    """
    if use_colab:
        try:
            from google.colab import auth
            auth.authenticate_user()
            import google.auth
            credentials, project = google.auth.default()
            return credentials
        except ImportError:
            logger.warning("Colab not available, falling back to ADC")
    
    # Use Application Default Credentials
    return None