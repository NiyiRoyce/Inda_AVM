"""
Central configuration for AVM project.
Environment variables override defaults.
"""

import os
from pathlib import Path

# ==================================================
# Environment
# ==================================================
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")  # local | staging | production
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

APP_NAME = os.getenv("APP_NAME", "avm-project")


# ==================================================
# Project Paths
# ==================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ARTIFACTS_DIR / "models"))
PREPROCESSORS_DIR = Path(os.getenv("PREPROCESSORS_DIR", ARTIFACTS_DIR / "preprocessors"))
METADATA_DIR = Path(os.getenv("METADATA_DIR", ARTIFACTS_DIR / "metadata"))

for dir_path in [ARTIFACTS_DIR, MODELS_DIR, PREPROCESSORS_DIR, METADATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ==================================================
# GCP Configuration
# ==================================================
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "primal-result-478707-k2")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")

# ADC is preferred; this is optional (local only)
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


# ==================================================
# BigQuery Configuration
# ==================================================
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "wed_scrape_sync")
BIGQUERY_TRAIN_TABLE = os.getenv("BIGQUERY_TRAIN_TABLE", "master_listings")
BIGQUERY_PREDICTIONS_TABLE = os.getenv("BIGQUERY_PREDICTIONS_TABLE", "predictions")

BIGQUERY_TRAIN_QUERY = f"""
SELECT *
FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TRAIN_TABLE}`
"""


# ==================================================
# Google Cloud Storage
# ==================================================
GCS_BUCKET = os.getenv("GCS_BUCKET", "linear_regression_model")
GCS_MODELS_PREFIX = "models/"
GCS_PREPROCESSORS_PREFIX = "preprocessors/"
GCS_METADATA_PREFIX = "metadata/"


# ==================================================
# API Configuration
# ==================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8080))
WORKERS = int(os.getenv("WORKERS", 1))

ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() == "true"
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")
API_KEY = os.getenv("API_KEY", "")


# ==================================================
# Logging
# ==================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ==================================================
# Feature Engineering / Validation
# ==================================================
GEOGRAPHIC_BOUNDS = {
    "latitude_min": 4.0,
    "latitude_max": 14.0,
    "longitude_min": 2.0,
    "longitude_max": 15.0,
}

BED_BATH_CAP = 10


# ==================================================
# Training Configuration
# ==================================================
TRAIN_TEST_SPLIT = float(os.getenv("TRAIN_TEST_SPLIT", 0.2))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))


# ==================================================
# Model Artifacts (local paths)
# ==================================================
MODEL_ARTIFACTS = {
    "linear_model": MODELS_DIR / "linear_model.pkl",
    "residual_model": MODELS_DIR / "residual_model.pkl",
    "smearing_factor": MODELS_DIR / "smearing_factor.pkl",
    "imputer": PREPROCESSORS_DIR / "imputer.pkl",
    "feature_names": METADATA_DIR / "feature_names.json",
    "training_stats": METADATA_DIR / "training_stats.json",
}


# ==================================================
# Feature Flags
# ==================================================
ENABLE_EXPERIMENTAL_FEATURES = os.getenv(
    "ENABLE_EXPERIMENTAL_FEATURES", "false"
).lower() == "true"

ENABLE_REQUEST_LOGGING = os.getenv(
    "ENABLE_REQUEST_LOGGING", "true"
).lower() == "true"

ENABLE_PROFILING = os.getenv(
    "ENABLE_PROFILING", "false"
).lower() == "true"
