"""
End-to-end training pipeline.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import DataLoader
from src.preprocessing.cleaners import DataCleaner
from src.preprocessing.transformers import DataTransformer
from src.features.engineering import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.diagnostics import ModelDiagnostics

from src.config.settings import TRAIN_TEST_SPLIT
from src.config.features import LOG_TARGET_VARIABLE, NUMERIC_FEATURES

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates the complete training workflow."""

    def __init__(self, project_id: str | None = None):
        """
        Initialize training pipeline.

        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.data_loader = DataLoader(project_id=project_id)
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.engineer = FeatureEngineer()
        self.trainer = ModelTrainer()

        self.df: pd.DataFrame | None = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    def load_data(self, from_csv: str | None = None) -> pd.DataFrame:
        """
        Load data from BigQuery or CSV.
        """
        if from_csv:
            logger.info(f"Loading data from CSV: {from_csv}")
            self.df = self.data_loader.load_from_csv(from_csv, validate=True)
        else:
            logger.info("Loading data from BigQuery")
            self.df = self.data_loader.load_and_validate(
                validate=True,
                remove_invalid=True,
            )

        return self.df

    def preprocess_data(self) -> pd.DataFrame:
        """
        Clean and transform data.
        """
        if self.df is None:
            raise ValueError("Data must be loaded before preprocessing")

        logger.info("Starting preprocessing...")

        # Clean
        self.df = self.cleaner.clean_all(self.df)

        # Transform
        distance_cols = self.engineer.get_distance_columns(self.df)
        self.df = self.transformer.transform_all(self.df, distance_cols)

        # Engineer features
        self.df = self.engineer.engineer_all(self.df)

        logger.info("Preprocessing completed")
        return self.df

    def split_data(self):
        """Split data into train and validation sets."""
        if self.df is None:
            raise ValueError("Data must be preprocessed before splitting")

        if LOG_TARGET_VARIABLE not in self.df.columns:
            raise KeyError(f"Target column '{LOG_TARGET_VARIABLE}' not found in data")

        logger.info("Splitting data into train/validation sets...")

        # Feature selection AFTER feature engineering
        distance_cols = self.engineer.get_distance_columns(self.df)
        all_features = list(set(NUMERIC_FEATURES + distance_cols))
        all_features = [f for f in all_features if f in self.df.columns]

        X = self.df[all_features]
        y = self.df[LOG_TARGET_VARIABLE]

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X,
            y,
            test_size=TRAIN_TEST_SPLIT["test_size"],
            random_state=TRAIN_TEST_SPLIT["random_state"],
        )

        logger.info(
            f"Train: {len(self.X_train)} samples, "
            f"Validation: {len(self.X_val)} samples"
        )

    def train_models(self) -> dict:
        """
        Train linear and residual models.
        """
        if any(v is None for v in [self.X_train, self.X_val, self.y_train, self.y_val]):
            raise ValueError("Data must be split before training")

        logger.info("Training models...")

        trained = self.trainer.train_all(
            X_train=self.X_train,
            X_val=self.X_val,
            y_train=self.y_train,
            y_val=self.y_val,
        )

        return trained

    def evaluate_models(self, trained: dict):
        """
        Evaluate trained models on validation set.
        """
        required_keys = {
            "linear_model",
            "residual_model",
            "X_val_processed",
            "X_residual_val",
        }
        missing = required_keys - trained.keys()
        if missing:
            raise KeyError(f"Missing trained artifacts: {missing}")

        logger.info("Evaluating models...")

        # Prepared validation features
        X_val_proc = trained["X_val_processed"]
        X_val_residual = trained["X_residual_val"]

        # Predict
        linear_price = trained["linear_model"].predict_price(X_val_proc)
        residual = trained["residual_model"].predict(X_val_residual)
        final_pred = linear_price + residual

        # Guard against invalid values
        final_pred = np.clip(final_pred, a_min=1e-8, a_max=None)

        # True values
        y_val_true = np.exp(self.y_val.to_numpy())

        # Metrics
        metrics = ModelEvaluator.compute_all_metrics(
            y_true=y_val_true,
            y_pred=final_pred,
            y_true_log=self.y_val.to_numpy(),
            y_pred_log=np.log(final_pred),
        )

        print("\n" + ModelEvaluator.format_metrics(metrics))

        diagnostics = ModelDiagnostics.run_full_diagnostics(
            y_val_true,
            final_pred,
        )

        print("\n" + ModelDiagnostics.format_diagnostics(diagnostics))

        return metrics, diagnostics

    def save_models(self):
        """Save all trained models and artifacts."""
        self.trainer.save_all()

    def run(self, from_csv: str | None = None, save_models: bool = True):
        """
        Run the complete training pipeline.
        """
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)

        self.load_data(from_csv=from_csv)
        self.preprocess_data()
        self.split_data()
        trained = self.train_models()
        metrics, diagnostics = self.evaluate_models(trained)

        if save_models:
            self.save_models()

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("=" * 60)

        return metrics, diagnostics
