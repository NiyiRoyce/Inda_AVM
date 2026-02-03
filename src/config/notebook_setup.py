"""
Notebook initialization utilities for consistent GCP/environment setup.
Use this in your notebooks to ensure proper authentication and configuration.
"""

import sys
import os
from pathlib import Path


def setup_notebook_environment(load_credentials: bool = True):
    """
    Initialize notebook environment with proper imports and GCP authentication.
    
    This function should be called at the beginning of each notebook that uses GCP.
    It handles:
    - Adding project root to Python path
    - Loading environment variables from .env files
    - Setting up GCP authentication
    - Initializing settings module
    
    Args:
        load_credentials: If True, load GCP credentials from service account
    
    Example:
        >>> from src.config.notebook_setup import setup_notebook_environment
        >>> setup_notebook_environment()
    """
    # Get the notebook's directory and project root
    notebook_dir = Path.cwd()
    project_root = None
    
    # Find project root by looking for requirements.txt or setup.py
    current = notebook_dir
    for _ in range(10):  # Search up to 10 levels
        if (current / "requirements.txt").exists() or (current / "setup.py").exists():
            project_root = current
            break
        current = current.parent
    
    if not project_root:
        # Fallback: assume notebooks/ directory exists
        if (notebook_dir / "requirements.txt").exists():
            project_root = notebook_dir
        else:
            project_root = notebook_dir.parent
    
    # Add project root and src to path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))
    
    # Load environment variables from .env files
    try:
        from dotenv import load_dotenv
        env_local = project_root / ".env.local"
        env_shared = project_root / ".env"
        
        if env_local.exists():
            load_dotenv(env_local, override=True)
            print(f"✓ Loaded environment from {env_local}")
        elif env_shared.exists():
            load_dotenv(env_shared, override=True)
            print(f"✓ Loaded environment from {env_shared}")
        else:
            print(f"⚠ No .env or .env.local found in {project_root}")
    except ImportError:
        print("⚠ python-dotenv not installed; environment variables must be set externally")
    
    # Validate environment variables are set
    try:
        # Import directly without importing full src module
        from config.env import validate_environment
        validate_environment(require_gcp_credentials=load_credentials)
        print("✓ Environment variables validated")
    except RuntimeError as e:
        print(f"✗ Environment validation failed: {e}")
        raise
    
    # Setup GCP authentication if needed
    if load_credentials:
        try:
            from auth.gcp_auth import authenticate_gcp
            authenticate_gcp(use_colab=False)
            print("✓ GCP authentication configured")
        except Exception as e:
            print(f"✗ GCP authentication failed: {e}")
            raise
    
    # Initialize settings (creates artifact directories, loads credentials)
    try:
        from config import settings
        settings.initialize(load_credentials=load_credentials)
        print("✓ Settings initialized")
        print(f"  - GCP Project: {settings.GCP_PROJECT_ID}")
        print(f"  - BigQuery Dataset: {settings.BIGQUERY_DATASET}")
        print(f"  - GCS Bucket: {settings.GCS_BUCKET}")
        if load_credentials and settings.GCP_CREDENTIALS:
            print(f"  - Credentials: Loaded")
        else:
            print(f"  - Credentials: Using ADC/Environment")
    except Exception as e:
        print(f"✗ Settings initialization failed: {e}")
        raise


def get_bigquery_client():
    """
    Create and return an authenticated BigQuery client.
    
    Requires setup_notebook_environment() to have been called first.
    
    Returns:
        google.cloud.bigquery.Client: Authenticated BigQuery client
        
    Raises:
        RuntimeError: If GCP credentials not configured
    """
    from google.cloud import bigquery
    from config import settings
    
    try:
        client = bigquery.Client(
            project=settings.GCP_PROJECT_ID,
            credentials=settings.GCP_CREDENTIALS
        )
        print(f"✓ BigQuery client created for project: {settings.GCP_PROJECT_ID}")
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to create BigQuery client: {e}") from e


def get_gcs_client():
    """
    Create and return an authenticated Google Cloud Storage client.
    
    Requires setup_notebook_environment() to have been called first.
    
    Returns:
        google.cloud.storage.Client: Authenticated GCS client
        
    Raises:
        RuntimeError: If GCP credentials not configured
    """
    from google.cloud import storage
    from config import settings
    
    try:
        client = storage.Client(
            project=settings.GCP_PROJECT_ID,
            credentials=settings.GCP_CREDENTIALS
        )
        print(f"✓ GCS client created for project: {settings.GCP_PROJECT_ID}")
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to create GCS client: {e}") from e

