"""
BigQuery client and query execution.
"""
import logging
from typing import Optional
import pandas as pd
from google.cloud import bigquery

from config.settings import GCP_PROJECT_ID, BIGQUERY_QUERY

logger = logging.getLogger(__name__)


class BigQueryClient:
    """Client for executing BigQuery queries."""
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize BigQuery client.
        
        Args:
            project_id: GCP project ID. If None, uses default from config.
        """
        self.project_id = project_id or GCP_PROJECT_ID
        self.client = bigquery.Client(project=self.project_id)
        logger.info(f"BigQuery client initialized for project: {self.project_id}")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a BigQuery SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            
        Returns:
            Query results as pandas DataFrame
            
        Raises:
            Exception: If query execution fails
        """
        try:
            logger.info("Executing BigQuery query...")
            logger.debug(f"Query: {query[:200]}...")  # Log first 200 chars
            
            query_job = self.client.query(query)
            df = query_job.to_dataframe()
            
            logger.info(f"Query executed successfully. Rows retrieved: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"BigQuery query execution failed: {e}")
            raise
    
    def load_master_listings(self) -> pd.DataFrame:
        """
        Load master listings data using default query.
        
        Returns:
            Master listings DataFrame
        """
        logger.info("Loading master listings from BigQuery...")
        return self.execute_query(BIGQUERY_QUERY)
    
    def get_table_schema(self, dataset_id: str, table_id: str) -> list:
        """
        Get schema of a BigQuery table.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            
        Returns:
            List of schema fields
        """
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        table = self.client.get_table(table_ref)
        return table.schema