"""
Central configuration for AVM project with safer, lazy initialization.
Environment variables override defaults. Avoid heavy work at import time.
"""

import os
from pathlib import Path
import json
from typing import Optional


# --- helpers for robust env parsing ---------------------------------
def getenv_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


def getenv_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def getenv_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def getenv_path(name: str, default: Path) -> Path:
    val = os.getenv(name)
    return Path(val) if val else default


# ==================================================
# Basic environment
# ==================================================
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")  # local | staging | production
DEBUG = getenv_bool("DEBUG", False)
APP_NAME = os.getenv("APP_NAME", "avm-project")


# ==================================================
# Project Paths (do not create dirs on import)
# ==================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = getenv_path("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
MODELS_DIR = getenv_path("MODELS_DIR", ARTIFACTS_DIR / "models")
PREPROCESSORS_DIR = getenv_path("PREPROCESSORS_DIR", ARTIFACTS_DIR / "preprocessors")
METADATA_DIR = getenv_path("METADATA_DIR", ARTIFACTS_DIR / "metadata")


def ensure_artifact_dirs() -> None:
    """Create artifact directories on demand (safe to call multiple times)."""
    for d in (ARTIFACTS_DIR, MODELS_DIR, PREPROCESSORS_DIR, METADATA_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            # keep import-time behavior safe: don't raise here
            pass


# ==================================================
# GCP Service Account and credentials (lazy)
# ==================================================
# Default path: config/service_account.json relative to this file
SERVICE_ACCOUNT_JSON_PATH = getenv_path(
    "SERVICE_ACCOUNT_JSON_PATH", Path(__file__).resolve().parent / "service_account.json"
)


def get_gcp_credentials() -> Optional[object]:
    """Return credentials object if available, otherwise None.

    Priority:
    1. `GOOGLE_APPLICATION_CREDENTIALS` file path (service account file)
    2. `SERVICE_ACCOUNT_JSON_PATH` packaged file
    3. Application Default Credentials (client libraries will handle)
    """
    # Avoid importing google.auth modules at top-level; import only when needed.
    # If a file path was provided via env, prefer that.
    gac = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if gac:
        try:
            from google.oauth2 import service_account

            return service_account.Credentials.from_service_account_file(gac)
        except Exception:
            return None

    if SERVICE_ACCOUNT_JSON_PATH and SERVICE_ACCOUNT_JSON_PATH.exists():
        try:
            from google.oauth2 import service_account

            with open(SERVICE_ACCOUNT_JSON_PATH) as f:
                key_dict = json.load(f)
            return service_account.Credentials.from_service_account_info(key_dict)
        except Exception:
            return None

    return None


# A module-level placeholder that callers can use; remains None until explicitly loaded
GCP_CREDENTIALS = None


# ==================================================
# GCP Configuration
# ==================================================
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "primal-result-478707-k2")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # optional, ADC


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
# API Configuration
# ==================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = getenv_int("API_PORT", 8080)
WORKERS = getenv_int("WORKERS", 1)
API_VERSION = os.getenv("API_VERSION", "1.0.0")
MAX_BATCH_SIZE = getenv_int("MAX_BATCH_SIZE", 100)

ENABLE_AUTH = getenv_bool("ENABLE_AUTH", False)
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")
API_KEY = os.getenv("API_KEY", "")

# Model artifacts path (for inference pipeline initialization)
MODEL_ARTIFACTS_PATH = os.getenv("MODEL_ARTIFACTS_PATH", None)

# Geographic bounds for validation
MIN_LATITUDE = float(os.getenv("MIN_LATITUDE", GEOGRAPHIC_BOUNDS["latitude_min"]))
MAX_LATITUDE = float(os.getenv("MAX_LATITUDE", GEOGRAPHIC_BOUNDS["latitude_max"]))
MIN_LONGITUDE = float(os.getenv("MIN_LONGITUDE", GEOGRAPHIC_BOUNDS["longitude_min"]))
MAX_LONGITUDE = float(os.getenv("MAX_LONGITUDE", GEOGRAPHIC_BOUNDS["longitude_max"]))


# ==================================================
# Logging
# ==================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ==================================================
# Training Configuration
# ==================================================
TRAIN_TEST_SPLIT = {
    "test_size": getenv_float("TRAIN_TEST_SPLIT_TEST_SIZE", 0.2),
    "random_state": getenv_int("TRAIN_TEST_SPLIT_RANDOM_STATE", 42)
}
RANDOM_SEED = getenv_int("RANDOM_SEED", 42)


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
ENABLE_EXPERIMENTAL_FEATURES = getenv_bool("ENABLE_EXPERIMENTAL_FEATURES", False)

ENABLE_REQUEST_LOGGING = getenv_bool("ENABLE_REQUEST_LOGGING", True)

ENABLE_PROFILING = getenv_bool("ENABLE_PROFILING", False)


def initialize(load_credentials: bool = False) -> None:
    """Perform safe initialization steps that were previously done at import time.

    - create artifact directories
    - optionally load GCP credentials into `GCP_CREDENTIALS`
    """
    ensure_artifact_dirs()
    if load_credentials:
        global GCP_CREDENTIALS
        GCP_CREDENTIALS = get_gcp_credentials()

