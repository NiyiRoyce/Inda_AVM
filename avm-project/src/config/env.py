"""
Environment validation utilities.

Note: validation checks whether required environment variables are present
in the environment (not whether the settings module has non-default values).
"""

import os
from pathlib import Path


def validate_environment(require_gcp_credentials: bool = False) -> None:
    """Raise RuntimeError if required environment variables are not set.

    This function checks the presence of variables in `os.environ` so that
    defaults in `settings` don't falsely satisfy validation.
    
    Args:
        require_gcp_credentials: If True, ensures GOOGLE_APPLICATION_CREDENTIALS 
                               or SERVICE_ACCOUNT_JSON_PATH is properly set
    
    Raises:
        RuntimeError: If required environment variables are missing
    """
    # Load env variables if dotenv is available
    try:
        from dotenv import load_dotenv
        env_local = Path(__file__).resolve().parent.parent.parent / ".env.local"
        env_shared = Path(__file__).resolve().parent.parent.parent / ".env"
        
        if env_local.exists():
            load_dotenv(env_local, override=True)
        elif env_shared.exists():
            load_dotenv(env_shared, override=True)
    except ImportError:
        pass
    
    required_keys = [
        "GCP_PROJECT_ID",
        "BIGQUERY_DATASET",
        "BIGQUERY_TRAIN_TABLE",
        "GCS_BUCKET",
    ]

    missing = [k for k in required_keys if not os.environ.get(k)]

    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    
    # Check GCP credentials are accessible if required
    if require_gcp_credentials:
        gcp_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        service_account_path = os.environ.get("SERVICE_ACCOUNT_JSON_PATH")
        
        if not gcp_creds and not service_account_path:
            raise RuntimeError(
                "GCP credentials required but not configured. "
                "Set GOOGLE_APPLICATION_CREDENTIALS or SERVICE_ACCOUNT_JSON_PATH"
            )
        
        # Validate the path exists
        credentials_path = gcp_creds or service_account_path
        if credentials_path and not Path(credentials_path).exists():
            raise RuntimeError(
                f"Credentials file not found at: {credentials_path}"
            )

