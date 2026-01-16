"""
Vertex AI predictor wrapper that exposes the project's InferencePipeline.

This module provides a minimal interface compatible with Vertex custom
prediction containers while delegating the heavy lifting to
`pipelines.inference_pipeline.InferencePipeline`.
"""
import logging
from typing import List, Dict

import pandas as pd

from src.pipelines.inference_pipeline import InferencePipeline

logger = logging.getLogger(__name__)


class VertexAIPredictor:
    """Wrapper around `InferencePipeline` for Vertex AI deployment."""

    def __init__(self, model_path: str = None):
        """Initialize and load the inference pipeline.

        Args:
            model_path: Optional path to model artifacts (not required)
        """
        self.pipeline = InferencePipeline()
        logger.info("InferencePipeline initialized for VertexAIPredictor")

    def preprocess(self, instances: List[Dict]) -> pd.DataFrame:
        """Convert incoming instances to a DataFrame and preprocess.

        Vertex expects a `preprocess(instances)` method when using custom
        predictors; keep the signature simple.
        """
        df = pd.DataFrame(instances)
        return self.pipeline.preprocess_data(df)

    def predict(self, instances: List[Dict]) -> List[Dict]:
        """Run prediction for a batch of instances.

        Returns a list of JSON-serializable dicts.
        """
        df = pd.DataFrame(instances)
        predictions = self.pipeline.predict(df)

        results = []
        for p in predictions:
            results.append({"predicted_price": float(p), "currency": "NGN"})

        return results


if __name__ == "__main__":
    # Simple local smoke entrypoint
    predictor = VertexAIPredictor()
    print("VertexAIPredictor ready")
