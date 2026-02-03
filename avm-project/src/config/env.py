"""
Environment validation utilities.

Note: validation checks whether required environment variables are present
in the environment (not whether the settings module has non-default values).
"""

import os
from pathlib import Path
import structlog

logger = structlog.get_logger()


def validate_environment(
    require_gcp_credentials: bool = False,
    strict: bool = False
) -> None:
    """Validate required environment variables.
    
    This function checks the presence of variables in `os.environ` so that
    defaults in `settings` don't falsely satisfy validation.
    
    In strict mode (local/dev), raises RuntimeError on missing variables.
    In non-strict mode (Cloud Run), logs warnings and allows degraded startup.
    
    Args:
        require_gcp_credentials: If True, ensures GOOGLE_APPLICATION_CREDENTIALS 
                               or SERVICE_ACCOUNT_JSON_PATH is properly set
        strict: If True, raises RuntimeError. If False, logs warning.
    
    Raises:
        RuntimeError: Only if strict=True and required environment variables are missing
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
    warnings = []

    # Check GCP credentials are accessible if required
    if require_gcp_credentials:
        gcp_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        service_account_path = os.environ.get("SERVICE_ACCOUNT_JSON_PATH")
        
        # Check if credentials exist (either explicit or ADC)
        has_explicit_credentials = gcp_creds or service_account_path
        has_adc = Path.home().joinpath(".config/gcloud/application_default_credentials.json").exists()
        
        if not has_explicit_credentials and not has_adc:
            missing.append("GCP_credentials (GOOGLE_APPLICATION_CREDENTIALS, SERVICE_ACCOUNT_JSON_PATH, or ADC)")
        elif has_explicit_credentials:
            # Validate the path exists if explicitly provided
            credentials_path = gcp_creds or service_account_path
            if credentials_path and not Path(credentials_path).exists():
                missing.append(f"Credentials file at {credentials_path}")

    # Handle validation results
    if missing:
        msg = f"Missing required environment variables: {', '.join(missing)}"
        
        if strict:
            logger.error(
                "Environment validation failed (strict mode)",
                missing_vars=missing,
                warnings=warnings,
            )
            raise RuntimeError(msg)
        else:
            logger.warning(
                "Environment validation failed - running in degraded mode",
                missing_vars=missing,
                warnings=warnings,
            )
    
    # Log warnings for optional issues
    if warnings and not missing:
        logger.info(
            "Environment validation completed with warnings",
            warnings=warnings,
        )
    
    if not missing and not warnings:
        logger.info("Environment validation successful")