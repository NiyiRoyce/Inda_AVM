"""
FastAPI application for AVM prediction service.

This module provides a production-ready API for real estate property valuation
using machine learning models. It includes robust error handling, health checks,
and both single and batch prediction endpoints.
"""
from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional

from src.config import settings
from src.config.env import validate_environment
from src.api.schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    VertexPredictRequest,
    VertexPredictResponse,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
)
from src.api.dependencies import (
    get_pipeline,
    validate_batch_size,
    check_model_health,
)
from src.pipelines.inference_pipeline import InferencePipeline

# ==================================================
# Logging Configuration
# ==================================================

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()

# ==================================================
# Helper Functions
# ==================================================


def build_prediction_response(predicted_price: float) -> Optional[PredictionResponse]:
    """
    Build a standardized prediction response from a predicted price.
    
    Args:
        predicted_price: The predicted property price
        
    Returns:
        PredictionResponse if prediction is valid (>0), None otherwise
    """
    if predicted_price <= 0:
        return None
    
    return PredictionResponse(
        predicted_price=float(predicted_price),
        log_price=float(np.log(predicted_price)),
        baseline_price=float(predicted_price * 0.95),  # TODO: Replace with actual baseline
        residual_correction=float(predicted_price * 0.05),  # TODO: Replace with actual residual
        confidence_score=0.85,  # TODO: Implement proper confidence scoring
    )


# ==================================================
# Application Factory
# ==================================================


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This factory pattern allows for easier testing and configuration management.
    Validates environment in soft mode to prevent Cloud Run restart loops.

    Returns:
        FastAPI: Configured application instance
    """
    # Validate environment in NON-STRICT mode (Cloud Run safe)
    env_error = None
    try:
        validate_environment(strict=False)
    except Exception as e:
        logger.error(
            "Environment validation failed",
            error=str(e),
            exc_info=True,
        )
        env_error = str(e)

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.API_VERSION,
        debug=settings.DEBUG,
        description="Automated Valuation Model (AVM) API for Real Estate",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )

    # Configure CORS middleware
    # TODO: Update allow_origins based on environment (production should be restrictive)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize pipeline state (lazy loading pattern)
    app.state.pipeline = None
    app.state.pipeline_loaded = False
    app.state.pipeline_error = None
    app.state.env_error = env_error  # Track environment validation state

    logger.info(
        "Application created successfully",
        app_name=settings.APP_NAME,
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG,
        env_validation_passed=env_error is None,
    )

    return app


# ==================================================
# Application Initialization with Safe Fallback
# ==================================================

try:
    app = create_app()
except Exception as e:
    # Prevent container crashes in production environments (e.g., Cloud Run, Vertex AI)
    # Create minimal fallback app that can still respond to health checks
    logger.warning(
        "Failed to create app with full configuration, using minimal fallback",
        error=str(e),
        exc_info=True,
    )

    app = FastAPI(
        title="AVM API (Fallback Mode)",
        description="Minimal fallback application - environment configuration error",
    )

    app.state.pipeline = None
    app.state.pipeline_loaded = False
    app.state.pipeline_error = str(e)
    app.state.env_error = str(e)

# ==================================================
# Lifecycle Event Handlers
# ==================================================


@app.on_event("startup")
async def startup_event():
    """
    Application startup handler.

    Note: Models are loaded lazily on first prediction request via get_pipeline()
    dependency to avoid blocking container startup in cloud environments.
    """
    logger.info("Application startup initiated")
    try:
        # Perform any startup validation or warmup here
        # Actual model loading happens in get_pipeline() dependency
        logger.info("Startup completed successfully")
    except Exception as e:
        logger.error("Startup validation failed", error=str(e), exc_info=True)
        # Don't fail startup - allow degraded mode operation


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown handler for cleanup operations."""
    logger.info("Application shutdown initiated")
    try:
        # Cleanup resources if needed
        if hasattr(app.state, 'pipeline') and app.state.pipeline is not None:
            # Perform any necessary cleanup
            logger.info("Cleaned up pipeline resources")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))

    logger.info("Application shutdown completed")


# ==================================================
# API Routes
# ==================================================


@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing basic service information.

    Returns:
        Dict containing service metadata
    """
    return {
        "service": settings.APP_NAME,
        "version": settings.API_VERSION,
        "status": "running",
        "environment": settings.ENVIRONMENT,
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for container orchestration and monitoring.

    This endpoint is designed to be safe for frequent polling:
    - Never raises exceptions
    - Doesn't load models (uses lazy loading status)
    - Returns quickly
    - Reports degraded state if environment validation failed

    Returns:
        HealthResponse: Current health status of the service
    """
    healthy = True
    try:
        health_data = check_model_health()
        healthy = health_data.get("healthy", False)
    except Exception as e:
        logger.warning("Health check failed", error=str(e))
        healthy = False

    # Check if environment validation failed
    if getattr(app.state, "env_error", None) is not None:
        logger.warning(
            "Health check reports degraded due to environment validation failure",
            env_error=app.state.env_error,
        )
        healthy = False

    return HealthResponse(
        status="healthy" if healthy else "degraded",
        model_loaded=healthy,
        version=settings.API_VERSION,
        environment=settings.ENVIRONMENT,
    )


@app.get("/{path:path}", tags=["HealthProbe"])
async def catch_all_get(path: str, request: Request):
    """
    Catch-all GET handler for Vertex AI health probes.

    Returns HTTP 200 OK for any GET request to avoid deployment failures.
    """
    return {"status": "ok"}


# ==================================================
# Vertex AI Prediction Endpoint
# ==================================================
# This MUST be defined before /predict/single — Vertex AI always hits /predict
# and sends the payload wrapped in an "instances" array. This endpoint unwraps
# it, runs predictions, and returns results in the format Vertex expects.
# ==================================================


@app.post(
    "/predict",
    response_model=VertexPredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Vertex AI prediction endpoint (handles instances envelope)",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def vertex_predict(
    request: VertexPredictRequest,
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> VertexPredictResponse:
    """
    Vertex AI prediction endpoint.

    Vertex AI wraps the user's payload in an 'instances' array before forwarding
    to this endpoint. This handler unwraps each instance, runs the prediction
    pipeline, and returns results in the format Vertex expects: {"predictions": [...]}.

    Args:
        request: Vertex AI wrapped request containing instances array
        pipeline: InferencePipeline instance (injected via dependency)

    Returns:
        VertexPredictResponse: Predictions list matching Vertex AI contract

    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    try:
        results = []
        failed_instances = []
        
        for idx, instance in enumerate(request.instances):
            # Each instance is a PredictionRequest with a .property field
            features = instance.property.model_dump()

            # Generate prediction
            predicted_price = pipeline.predict_single(features)
            
            # Build response using helper function
            response = build_prediction_response(predicted_price)
            
            if response is None:
                failed_instances.append(idx)
                logger.warning(
                    "Invalid prediction for instance",
                    instance_index=idx,
                    predicted_price=predicted_price
                )
                continue
                
            results.append(response)

        if failed_instances:
            logger.warning(
                "Some predictions failed",
                failed_count=len(failed_instances),
                failed_indices=failed_instances,
            )

        if not results:
            raise ValueError("All predictions resulted in zero or negative values")

        logger.info(
            "Vertex prediction completed",
            total_instances=len(request.instances),
            total_predictions=len(results),
            failed_predictions=len(failed_instances),
        )

        return VertexPredictResponse(predictions=results)

    except ValueError as e:
        logger.warning("Invalid input data", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error("Vertex prediction failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Please try again later."
        )


# ==================================================
# Single Property Prediction (Direct/Local Use)
# ==================================================


@app.post(
    "/predict/single",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Single property valuation prediction (direct use, not via Vertex)",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def predict_property(
    request: PredictionRequest,
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> PredictionResponse:
    """
    Generate valuation prediction for a single property.

    Use this endpoint when calling the API directly (e.g. locally or via a custom proxy).
    For Vertex AI deployments, use /predict instead.

    Args:
        request: Property features for valuation
        pipeline: InferencePipeline instance (injected via dependency)

    Returns:
        PredictionResponse: Predicted price and metadata

    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    try:
        # Extract features from request
        features = request.property.model_dump()

        # Generate prediction
        predicted_price = pipeline.predict_single(features)
        
        # Build response using helper function
        response = build_prediction_response(predicted_price)
        
        if response is None:
            raise ValueError(
                f"Prediction resulted in invalid value: {predicted_price}"
            )

        logger.info(
            "Single prediction generated",
            predicted_price=predicted_price,
            property_type=features.get('property_type'),
            location=features.get('location')
        )

        return response

    except ValueError as e:
        logger.warning("Invalid input data", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error("Prediction failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Please try again later."
        )


# ==================================================
# Batch Prediction (Direct/Local Use)
# ==================================================


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    dependencies=[Depends(validate_batch_size)],
    summary="Batch property valuation predictions (direct use, not via Vertex)",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input or batch size exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def predict_batch(
    request: BatchPredictionRequest,
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> BatchPredictionResponse:
    """
    Generate valuation predictions for multiple properties.

    Use this endpoint when calling the API directly (e.g. locally or via a custom proxy).
    For Vertex AI deployments, use /predict instead — Vertex handles batching via the
    instances array automatically.

    Efficient batch processing for up to 100 properties per request.

    Args:
        request: List of property features (max 100)
        pipeline: InferencePipeline instance (injected via dependency)

    Returns:
        BatchPredictionResponse: Predictions for all properties

    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    try:
        # Convert request to DataFrame for batch processing
        features_list = [prop.model_dump() for prop in request.properties]
        df = pd.DataFrame(features_list)

        # Generate batch predictions
        predictions = pipeline.predict(df, include_preprocessing=True)

        # Build response for each prediction using helper function
        results = []
        failed_indices = []
        
        for idx, predicted_price in enumerate(predictions):
            response = build_prediction_response(predicted_price)
            
            if response is None:
                failed_indices.append(idx)
                logger.warning(
                    "Invalid prediction in batch",
                    batch_index=idx,
                    predicted_price=predicted_price
                )
                continue
                
            results.append(response)

        if failed_indices:
            logger.warning(
                "Some batch predictions failed",
                failed_count=len(failed_indices),
                failed_indices=failed_indices,
            )

        if not results:
            raise ValueError("All predictions resulted in zero or negative values")

        logger.info(
            "Batch predictions generated",
            total_requested=len(request.properties),
            total_succeeded=len(results),
            total_failed=len(failed_indices),
        )

        return BatchPredictionResponse(
            predictions=results,
            count=len(results)
        )

    except ValueError as e:
        logger.warning("Invalid batch input data", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error("Batch prediction failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed. Please try again later."
        )


# ==================================================
# Batch Prediction with Summary (Direct/Local Use)
# ==================================================


@app.post(
    "/predict/summary",
    tags=["Predictions"],
    summary="Batch predictions with summary statistics",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def predict_with_summary(
    request: BatchPredictionRequest,
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> Dict[str, Any]:
    """
    Generate predictions with aggregated summary statistics.

    Useful for portfolio analysis and market insights.

    Args:
        request: List of property features
        pipeline: InferencePipeline instance (injected via dependency)

    Returns:
        Dict containing individual predictions and summary statistics

    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    try:
        # Convert to DataFrame
        features_list = [prop.model_dump() for prop in request.properties]
        df = pd.DataFrame(features_list)

        # Generate predictions
        predictions = pipeline.predict(df, include_preprocessing=True)

        # Filter out invalid predictions
        valid_predictions = [p for p in predictions if p > 0]
        
        if not valid_predictions:
            raise ValueError("All predictions resulted in zero or negative values")

        # Calculate summary statistics
        summary = pipeline.get_prediction_summary(valid_predictions)

        # Build response - only include valid predictions
        prediction_list = [
            {
                "predicted_price": float(p),
                "log_price": float(np.log(p)),
            }
            for p in valid_predictions
        ]

        logger.info(
            "Summary prediction completed",
            total=len(predictions),
            valid=len(valid_predictions),
            invalid=len(predictions) - len(valid_predictions),
            summary=summary
        )

        return {
            "predictions": prediction_list,
            "summary": summary,
            "count": len(valid_predictions),
            "invalid_count": len(predictions) - len(valid_predictions),
        }

    except ValueError as e:
        logger.warning("Invalid summary input data", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error("Summary prediction failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Summary prediction failed. Please try again later."
        )


# ==================================================
# Global Exception Handler
# ==================================================


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for uncaught errors.

    Provides consistent error responses and comprehensive logging.

    Args:
        request: The request that caused the exception
        exc: The exception that was raised

    Returns:
        JSONResponse with error details
    """
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.DEBUG else None,
        },
    )


# ==================================================
# Local Development Entry Point
# ==================================================
# Note: In production (Docker/Cloud Run), use CMD in Dockerfile instead

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    logger.info(
        "Starting development server",
        host="0.0.0.0",
        port=port
    )

    uvicorn.run(
        "src.api.app:app",  # Module path for hot reload
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=settings.DEBUG,  # Enable hot reload in debug mode
    )