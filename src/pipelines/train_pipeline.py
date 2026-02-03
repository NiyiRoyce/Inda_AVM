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
        logger.info("Loading data...")

        if from_csv:
            self.df = self.data_loader.load_from_csv(from_csv, validate=True)
        else:
            self.df = self.data_loader.load_and_validate(
                validate=True,
                remove_invalid=False
            )

        return self.df

    def preprocess_data(self) -> pd.DataFrame:
        logger.info("Starting preprocessing")

        self.df = self.cleaner.clean_all(self.df)

        distance_cols = self.engineer.get_distance_columns(self.df)
        self.df = self.transformer.transform_all(self.df, distance_cols)

        self.df = self.engineer.engineer_all(self.df)

        logger.info("Preprocessing completed")
        return self.df

    def split_data(self) -> None:
        logger.info("Splitting data")

        distance_cols = self.engineer.get_distance_columns(self.df)
        all_features = list(set(NUMERIC_FEATURES + distance_cols))
        all_features = [f for f in all_features if f in self.df.columns]

        X = self.df[all_features]
        y = self.df[LOG_TARGET_VARIABLE]

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X,
            y,
            test_size=TRAIN_TEST_SPLIT["test_size"],
            random_state=TRAIN_TEST_SPLIT["random_state"]
        )

        logger.info(
            "Train samples: %d | Validation samples: %d",
            len(self.X_train),
            len(self.X_val)
        )

    def train_models(self) -> dict:
        logger.info("Training models")

        return self.trainer.train_all(
            X_train=self.X_train,
            X_val=self.X_val,
            y_train=self.y_train,
            y_val=self.y_val
        )

    def evaluate_models(self, trained: dict):
        logger.info("Evaluating models")

        X_val_proc = trained["X_val_processed"]
        X_val_residual = trained["X_residual_val"]

        linear_price = trained["linear_model"].predict_price(X_val_proc)
        residual = trained["residual_model"].predict(X_val_residual)

        final_pred = linear_price + residual
        final_pred = np.clip(final_pred, 1e-6, None)

        y_true = np.exp(self.y_val.to_numpy())

        metrics = ModelEvaluator.compute_all_metrics(
            y_true=y_true,
            y_pred=final_pred,
            y_true_log=self.y_val.to_numpy(),
            y_pred_log=np.log(final_pred)
        )

        print("\n" + ModelEvaluator.format_metrics(metrics))

        diagnostics = ModelDiagnostics.run_full_diagnostics(
            y_true,
            final_pred
        )

        print("\n" + ModelDiagnostics.format_diagnostics(diagnostics))

        return metrics, diagnostics

    def save_models(self) -> None:
        self.trainer.save_all()

    def run(self, from_csv: str | None = None, save_models: bool = True):
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)

        self.load_data(from_csv)
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
