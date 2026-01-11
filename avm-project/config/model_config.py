"""
Model hyperparameters and training configurations.
"""

# Linear Regression Configuration
LINEAR_MODEL_CONFIG = {
    "n_jobs": -1,  # Use all available cores
}

# LightGBM Residual Model Configuration
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1,
    "force_col_wise": True,  # Suppress warnings
}

LIGHTGBM_TRAINING = {
    "num_boost_round": 1000,
    "early_stopping_rounds": 50,
}

# Imputation strategy
IMPUTATION_STRATEGY = "median"

# Evaluation percentiles
EVALUATION_PERCENTILES = [1, 5, 25, 50, 75, 95, 99]

# Extreme prediction thresholds
EXTREME_PREDICTION_THRESHOLDS = {
    "low": 0.2,   # Flag if prediction < 20% of true value
    "high": 5.0,  # Flag if prediction > 500% of true value
}

# Price tier quantiles for tier-based evaluation
PRICE_TIER_QUANTILES = [0, 0.25, 0.5, 0.75, 0.9]