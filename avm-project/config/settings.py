"""
Central configuration for AVM project.
Environment variables override defaults.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREPROCESSORS_DIR = ARTIFACTS_DIR / "preprocessors"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, PREPROCESSORS_DIR, METADATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# GCP Configuration
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "primal-result-478707-k2")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "wed_scrape_sync")
BIGQUERY_TABLE = os.getenv("BIGQUERY_TABLE", "master_listings")

# BigQuery query
BIGQUERY_QUERY = f"""
SELECT *
FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}`
"""

# Data validation bounds
GEOGRAPHIC_BOUNDS = {
    "latitude_min": 4.0,
    "latitude_max": 14.0,
    "longitude_min": 2.0,
    "longitude_max": 15.0,
}

# Feature engineering constants
BED_BATH_CAP = 10

# Training configuration
TRAIN_TEST_SPLIT = {
    "test_size": 0.2,
    "random_state": 42,
}

# Model artifacts paths
MODEL_ARTIFACTS = {
    "linear_model": MODELS_DIR / "linreg_model.pkl",
    "residual_model": MODELS_DIR / "residual_model.pkl",
    "smearing_factor": MODELS_DIR / "smearing_factor.pkl",
    "imputer": PREPROCESSORS_DIR / "imputer.pkl",
    "feature_names": METADATA_DIR / "feature_names.json",
    "training_stats": METADATA_DIR / "training_stats.json",
}

# GCS Configuration (for deployment)
GCS_BUCKET = os.getenv("GCS_BUCKET", "linear_regression_model")
GCS_MODEL_PATH = f"gs://{GCS_BUCKET}/models/"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"