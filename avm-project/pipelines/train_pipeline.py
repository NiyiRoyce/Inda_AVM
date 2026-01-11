"""
End-to-end training pipeline.
"""
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import DataLoader
from src.preprocessing.cleaners import DataCleaner
from src.preprocessing.transformers import DataTransformer
from src.features.engineering import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.diagnostics import ModelDiagnostics
from config.settings import TRAIN_TEST_SPLIT
from config.features import LOG_TARGET_VARIABLE

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates the complete training workflow."""
    
    def __init__(self, project_id: str = None):
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
        
        self.df = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
    
    def load_data(self, from_csv: str = None) -> pd.DataFrame:
        """
        Load data from BigQuery or CSV.
        
        Args:
            from_csv: Optional path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        if from_csv:
            logger.info(f"Loading data from CSV: {from_csv}")
            self.df = self.data_loader.load_from_csv(from_csv, validate=True)
        else:
            logger.info("Loading data from BigQuery")
            self.df = self.data_loader.load_and_validate(
                validate=True,
                remove_invalid=False  # We'll clean manually
            )
        
        return self.df
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Clean and transform data.
        
        Returns:
            Preprocessed DataFrame
        """
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
        logger.info("Splitting data into train/validation sets...")
        
        # Prepare feature matrix and target
        feature_cols = self.engineer.get_distance_columns(self.df)
        from config.features import NUMERIC_FEATURES
        all_features = list(set(NUMERIC_FEATURES + feature_cols))
        
        # Filter to existing columns
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
            f"Train: {len(self.X_train)} samples, "
            f"Validation: {len(self.X_val)} samples"
        )
    
    def train_models(self) -> dict:
        """
        Train linear and residual models.
        
        Returns:
            Dictionary of trained components
        """
        logger.info("Training models...")
        
        trained = self.trainer.train_all(
            self.X_train,
            self.X_val,
            self.y_train
        )
        
        return trained
    
    def evaluate_models(self, trained: dict):
        """
        Evaluate trained models on validation set.
        
        Args:
            trained: Dictionary of trained components
        """
        logger.info("Evaluating models...")
        
        # Get validation predictions
        from src.models.ensemble import EnsemblePredictor
        
        ensemble = EnsemblePredictor(
            linear_model=trained["linear_model"],
            residual_model=trained["residual_model"],
            imputer=trained.get("imputer"),
            feature_names=trained.get("feature_names")
        )
        
        # Prepare validation features
        X_val_proc = trained["X_val_processed"]
        X_val_residual = trained["X_residual_val"]
        
        # Predict
        import numpy as np
        linear_price = trained["linear_model"].predict_price(X_val_proc)
        residual = trained["residual_model"].predict(X_val_residual)
        final_pred = linear_price + residual
        
        # True values
        y_val_true = np.exp(self.y_val.to_numpy())
        
        # Compute metrics
        metrics = ModelEvaluator.compute_all_metrics(
            y_true=y_val_true,
            y_pred=final_pred,
            y_true_log=self.y_val.to_numpy(),
            y_pred_log=np.log(final_pred)
        )
        
        # Display metrics
        print("\n" + ModelEvaluator.format_metrics(metrics))
        
        # Run diagnostics
        diagnostics = ModelDiagnostics.run_full_diagnostics(
            y_val_true,
            final_pred
        )
        
        print("\n" + ModelDiagnostics.format_diagnostics(diagnostics))
        
        return metrics, diagnostics
    
    def save_models(self):
        """Save all trained models and artifacts."""
        self.trainer.save_all()
    
    def run(self, from_csv: str = None, save_models: bool = True):
        """
        Run the complete training pipeline.
        
        Args:
            from_csv: Optional CSV file path
            save_models: Whether to save models after training
            
        Returns:
            Tuple of (metrics, diagnostics)
        """
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)
        
        # 1. Load data
        self.load_data(from_csv=from_csv)
        
        # 2. Preprocess
        self.preprocess_data()
        
        # 3. Split
        self.split_data()
        
        # 4. Train
        trained = self.train_models()
        
        # 5. Evaluate
        metrics, diagnostics = self.evaluate_models(trained)
        
        # 6. Save
        if save_models:
            self.save_models()
        
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("=" * 60)
        
        return metrics, diagnostics