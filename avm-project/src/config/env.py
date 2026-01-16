"""
Environment validation utilities.
"""

from config import settings


def validate_environment() -> None:
    required = {
        "GCP_PROJECT_ID": settings.GCP_PROJECT_ID,
        "BIGQUERY_DATASET": settings.BIGQUERY_DATASET,
        "BIGQUERY_TRAIN_TABLE": settings.BIGQUERY_TRAIN_TABLE,
        "GCS_BUCKET": settings.GCS_BUCKET,
    }

    missing = [k for k, v in required.items() if not v]

    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
