"""
GCP authentication handling for different environments.
Supports: service account files, ADC, Colab, and gcloud CLI.
"""
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def setup_gcp_credentials() -> None:
    """
    Setup GCP credentials from environment variables.
    
    Priority order:
    1. GOOGLE_APPLICATION_CREDENTIALS env var (points to service account JSON)
    2. SERVICE_ACCOUNT_JSON_PATH env var (points to service account JSON)
    3. ADC via gcloud CLI
    4. Colab environment
    
    Raises:
        RuntimeError: If neither GOOGLE_APPLICATION_CREDENTIALS nor SERVICE_ACCOUNT_JSON_PATH are set
    """
    # Load env vars if dotenv available
    try:
        from dotenv import load_dotenv
        env_local = Path(__file__).resolve().parent.parent.parent / ".env.local"
        env_shared = Path(__file__).resolve().parent.parent.parent / ".env"
        
        if env_local.exists():
            load_dotenv(env_local, override=False)
        elif env_shared.exists():
            load_dotenv(env_shared, override=False)
    except ImportError:
        pass
    
    # Check if GOOGLE_APPLICATION_CREDENTIALS is already set
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    service_account_path = os.environ.get("SERVICE_ACCOUNT_JSON_PATH")
    
    # If neither is set, try to locate the default service account file
    if not gac and not service_account_path:
        default_path = Path(__file__).resolve().parent.parent / "config" / "service_account.json"
        if default_path.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(default_path)
            logger.info(f"Using default service account at: {default_path}")
        else:
            logger.warning(
                "No GCP credentials configured. "
                "Set GOOGLE_APPLICATION_CREDENTIALS or SERVICE_ACCOUNT_JSON_PATH environment variable, "
                "or place service_account.json in src/config/"
            )
    elif service_account_path and not gac:
        # If SERVICE_ACCOUNT_JSON_PATH is set but not GOOGLE_APPLICATION_CREDENTIALS, set it
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
        logger.info(f"Using service account from SERVICE_ACCOUNT_JSON_PATH: {service_account_path}")
    
    # Validate the credentials file exists if set
    gac_final = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_final and not Path(gac_final).exists():
        logger.error(f"Credentials file not found: {gac_final}")
        raise RuntimeError(f"Service account credentials file not found: {gac_final}")


def authenticate_gcp(use_colab: bool = False) -> None:
    """
    Authenticate with GCP based on environment.
    
    Args:
        use_colab: If True, use Colab authentication. 
                   If False, use application default credentials or service account file.
    
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
            logger.info("Configuring GCP credentials...")
            setup_gcp_credentials()
            logger.info("GCP credentials configured successfully")
            
    except ImportError as e:
        if use_colab:
            raise RuntimeError(
                "Colab authentication requested but google.colab not available. "
                "Are you running in Colab?"
            ) from e
        logger.info("Could not import Colab auth (expected outside Colab)")
    except Exception as e:
        raise RuntimeError(f"GCP authentication failed: {e}") from e


def get_credentials(use_colab: bool = False):
    """
    Get GCP credentials object.
    
    Args:
        use_colab: Whether to use Colab authentication
        
    Returns:
        Credentials object or None (uses ADC/GOOGLE_APPLICATION_CREDENTIALS)
        
    Raises:
        RuntimeError: If credentials cannot be loaded
    """
    if use_colab:
        try:
            from google.colab import auth
            auth.authenticate_user()
            import google.auth
            credentials, project = google.auth.default()
            logger.info(f"Using Colab credentials for project: {project}")
            return credentials
        except ImportError:
            logger.warning("Colab not available, falling back to ADC")
    
    # Setup credentials from environment
    setup_gcp_credentials()
    
    # Let google-cloud libraries handle ADC automatically
    # If GOOGLE_APPLICATION_CREDENTIALS is set, it will be used
    import google.auth
    try:
        credentials, project = google.auth.default()
        logger.info(f"Using default credentials for project: {project}")
        return credentials
    except Exception as e:
        logger.error(f"Failed to get default credentials: {e}")
        raise RuntimeError(f"Could not load GCP credentials: {e}") from e
