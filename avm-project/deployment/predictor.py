"""Compatibility shim: delegate to the explicit Vertex AI wrapper.

This file preserves the previous `deployment.predictor` module path but
delegates implementation to `deployment.vertex_predictor.VertexAIPredictor`.
"""
from deployment.vertex_predictor import VertexAIPredictor


if __name__ == "__main__":
    # Simple local smoke entrypoint
    predictor = VertexAIPredictor()
    print("VertexAIPredictor ready (shim)")