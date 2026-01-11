"""
Data loading orchestration.
"""
import logging
import pandas as pd
from typing import Optional

from src.data.bigquery_client import BigQueryClient
from src.data.validator import DataValidator

logger = logging.getLogger(__name__)


class DataLoader:
    """Orchestrates data loading and validation."""
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            project_id: GCP project ID
        """
        self.bq_client = BigQueryClient(project_id=project_id)
        self.validator = DataValidator()
    
    def load_and_validate(
        self, 
        validate: bool = True,
        remove_invalid: bool = True
    ) -> pd.DataFrame:
        """
        Load data from BigQuery and optionally validate.
        
        Args:
            validate: Whether to run validation checks
            remove_invalid: Whether to remove invalid rows
            
        Returns:
            Loaded (and optionally validated) DataFrame
        """
        # Load data
        df = self.bq_client.load_master_listings()
        logger.info(f"Loaded {len(df)} rows from BigQuery")
        
        if validate:
            # Run validation
            validation_results = self.validator.validate(df)
            
            if not validation_results["is_valid"]:
                logger.warning("Data validation found issues:")
                for issue in validation_results["issues"]:
                    logger.warning(f"  - {issue}")
            
            # Remove invalid rows if requested
            if remove_invalid and validation_results["invalid_indices"]:
                original_len = len(df)
                df = df.drop(index=validation_results["invalid_indices"])
                df = df.reset_index(drop=True)
                logger.info(
                    f"Removed {original_len - len(df)} invalid rows. "
                    f"Remaining: {len(df)}"
                )
        
        return df
    
    def load_from_csv(self, filepath: str, validate: bool = True) -> pd.DataFrame:
        """
        Load data from CSV file (for testing/development).
        
        Args:
            filepath: Path to CSV file
            validate: Whether to run validation
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from CSV: {filepath}")
        df = pd.read_csv(filepath)
        
        if validate:
            validation_results = self.validator.validate(df)
            if not validation_results["is_valid"]:
                logger.warning("CSV validation found issues:")
                for issue in validation_results["issues"]:
                    logger.warning(f"  - {issue}")
        
        return df