"""
Environment validation utilities.

Note: validation checks whether required environment variables are present
in the environment (not whether the settings module has non-default values).
"""

import os


def validate_environment() -> None:
    """Raise RuntimeError if required environment variables are not set.

    This function checks the presence of variables in `os.environ` so that
    defaults in `settings` don't falsely satisfy validation.
    """
    required_keys = [
        "GCP_PROJECT_ID",
        "BIGQUERY_DATASET",
        "BIGQUERY_TRAIN_TABLE",
        "GCS_BUCKET",
    ]

    missing = [k for k in required_keys if not os.environ.get(k)]

    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
