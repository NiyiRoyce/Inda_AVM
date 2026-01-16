"""
BigQuery client and query execution.
"""

import logging
from typing import Optional, List

import pandas as pd
from google.cloud import bigquery

from config.settings import (
    GCP_PROJECT_ID,
    BIGQUERY_TRAIN_QUERY,
    GCP_CREDENTIALS,
)

logger = logging.getLogger(__name__)


class BigQueryClient:
    """Client for executing BigQuery queries."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        credentials=None,
    ):
        """
        Initialize BigQuery client.

        Args:
            project_id: GCP project ID. Defaults to config value.
            credentials: Optional google.auth.credentials.Credentials.
                         If None, uses Application Default Credentials (ADC).
        """
        self.project_id = project_id or GCP_PROJECT_ID
        self.credentials = credentials or GCP_CREDENTIALS

        if self.credentials:
            self.client = bigquery.Client(
                project=self.project_id,
                credentials=self.credentials,
            )
            logger.info("BigQuery client initialized with explicit credentials.")
        else:
            self.client = bigquery.Client(project=self.project_id)
            logger.info("BigQuery client initialized using ADC.")

        logger.info(f"BigQuery project: {self.project_id}")

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a BigQuery SQL query and return results as DataFrame.

        Args:
            query: SQL query string

        Returns:
            Query results as pandas DataFrame
        """
        try:
            logger.info("Executing BigQuery query...")
            logger.debug("Query preview:\n%s", query[:300])

            query_job = self.client.query(query)
            df = query_job.result().to_dataframe()

            logger.info("Query successful. Rows retrieved: %d", len(df))
            return df

        except Exception:
            logger.exception("BigQuery query execution failed.")
            raise

    def load_master_listings(self) -> pd.DataFrame:
        """
        Load master listings using the default training query.
        """
        logger.info("Loading master listings from BigQuery...")
        return self.execute_query(BIGQUERY_TRAIN_QUERY)

    def get_table_schema(self, dataset_id: str, table_id: str) -> List[bigquery.SchemaField]:
        """
        Retrieve the schema of a BigQuery table.

        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID

        Returns:
            List of BigQuery SchemaField objects
        """
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        table = self.client.get_table(table_ref)
        return table.schema
