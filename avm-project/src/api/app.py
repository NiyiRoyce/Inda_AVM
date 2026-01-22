"""
FastAPI application for AVM prediction service.
"""
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import pandas as pd
import numpy as np
from pathlib import Path

from config import settings
from config.env import validate_environment
from api.schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse
)
from api.dependencies import (
    get_pipeline,
    validate_batch_size,
    log_request,
    get_request_context,
    check_model_health
)
from pipelines.inference_pipeline import InferencePipeline

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    # Validate environment variables
    validate_environment()
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.API_VERSION,
        debug=settings.DEBUG,
        description="Automated Valuation Model (AVM) API for Nigerian Real Estate",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure based on environment
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize inference pipeline
    model_path = Path(settings.MODEL_ARTIFACTS_PATH).parent if settings.MODEL_ARTIFACTS_PATH else None
    app.state.pipeline = InferencePipeline(model_path=model_path)
    
    logger.info(
        "Application created",
        app_name=settings.APP_NAME,
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG
    )
    
    return app


try:
    app = create_app()
except RuntimeError as e:
    # Handle missing environment variables during import
    logger.warning(
        "Failed to create app with full configuration, using minimal fallback",
        error=str(e)
    )
    # Create minimal FastAPI app fallback
    app = FastAPI(
        title="AVM API (Fallback)",
        description="Minimal fallback application - environment not properly configured",
    )


@app.on_event("startup")
async def startup_event():
    """Load models on application startup."""
    logger.info("Starting up application...")
    
    try:
        # Models are loaded in InferencePipeline __init__
        logger.info("Inference pipeline initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize inference pipeline on startup", error=str(e))
        # Allow app to start but predictions will fail with proper error


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down application...")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": settings.APP_NAME,
        "version": settings.API_VERSION,
        "status": "running",
        "environment": settings.ENVIRONMENT
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(health_data: dict = Depends(check_model_health)):
    """Health check endpoint with detailed component status."""
    return HealthResponse(
        status="healthy" if health_data["healthy"] else "degraded",
        model_loaded=health_data["healthy"],
        version=settings.API_VERSION,
        environment=settings.ENVIRONMENT
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def predict_property(
    request: PredictionRequest,
    pipeline: InferencePipeline = Depends(get_pipeline)
):
    """
    Predict property valuation for a single property.
    
    Args:
        request: Property features for valuation
        pipeline: InferencePipeline instance (injected)
        
    Returns:
        PredictionResponse with estimated price and metadata
    """
    
    try:
        # Convert Pydantic model to dict
        features = request.property.model_dump()
        
        # Generate prediction using InferencePipeline
        predicted_price = pipeline.predict_single(features)
        
        # Create DataFrame for detailed prediction info
        df = pd.DataFrame([features])
        df_processed = pipeline.preprocess_data(df)
        
        # Get ensemble components (if available)
        # Note: This requires modifications to EnsemblePredictor to expose component predictions
        # For now, we'll return the final prediction with placeholder values
        
        logger.info(
            "Prediction generated",
            predicted_price=predicted_price,
        )
        
        return PredictionResponse(
            predicted_price=float(predicted_price),
            log_price=float(np.log(predicted_price)) if predicted_price > 0 else 0.0,
            baseline_price=float(predicted_price * 0.95),  # Placeholder
            residual_correction=float(predicted_price * 0.05),  # Placeholder
            confidence_score=0.85  # Placeholder - implement proper confidence scoring
        )
        
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
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    dependencies=[Depends(validate_batch_size)],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def predict_batch(
    request: BatchPredictionRequest,
    pipeline: InferencePipeline = Depends(get_pipeline)
):
    """
    Predict property valuations for multiple properties.
    
    Args:
        request: List of property features (max 100)
        pipeline: InferencePipeline instance (injected)
        
    Returns:
        BatchPredictionResponse with predictions for all properties
    """
    
    try:
        # Convert Pydantic models to DataFrame
        features_list = [prop.model_dump() for prop in request.properties]
        df = pd.DataFrame(features_list)
        
        # Generate predictions using InferencePipeline
        predictions = pipeline.predict(df, include_preprocessing=True)
        
        # Create response for each prediction
        prediction_responses = []
        for i, pred_price in enumerate(predictions):
            prediction_responses.append(
                PredictionResponse(
                    predicted_price=float(pred_price),
                    log_price=float(np.log(pred_price)) if pred_price > 0 else 0.0,
                    baseline_price=float(pred_price * 0.95),  # Placeholder
                    residual_correction=float(pred_price * 0.05),  # Placeholder
                    confidence_score=0.85  # Placeholder
                )
            )
        
        logger.info(
            "Batch predictions generated",
            total=len(predictions),
            successful=len(prediction_responses)
        )
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            count=len(prediction_responses)
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
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post(
    "/predict/summary",
    tags=["Predictions"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def predict_with_summary(
    request: BatchPredictionRequest,
    pipeline: InferencePipeline = Depends(get_pipeline)
):
    """
    Predict properties and return summary statistics.
    
    Args:
        request: List of property features
        pipeline: InferencePipeline instance (injected)
        
    Returns:
        Predictions with summary statistics
    """
    
    try:
        # Convert to DataFrame
        features_list = [prop.model_dump() for prop in request.properties]
        df = pd.DataFrame(features_list)
        
        # Generate predictions
        predictions = pipeline.predict(df, include_preprocessing=True)
        
        # Get summary statistics
        summary = pipeline.get_prediction_summary(predictions)
        
        # Create individual predictions
        prediction_responses = []
        for pred_price in predictions:
            prediction_responses.append({
                "predicted_price": float(pred_price),
                "log_price": float(pd.np.log(pred_price)) if pred_price > 0 else 0.0
            })
        
        return {
            "predictions": prediction_responses,
            "summary": summary,
            "count": len(predictions)
        }
        
    except Exception as e:
        logger.error("Summary prediction failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.DEBUG else None
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )