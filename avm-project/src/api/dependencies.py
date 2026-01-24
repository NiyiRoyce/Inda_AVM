"""
FastAPI dependency injection utilities.
Provides reusable dependencies for request validation, authentication, and resource management.
"""
from fastapi import HTTPException, status, Request, Depends
from typing import Optional
from pathlib import Path
import structlog

from config import settings
from pipelines.inference_pipeline import InferencePipeline

logger = structlog.get_logger()


# =============================================================================
# Pipeline Dependencies
# =============================================================================

async def get_pipeline(request: Request) -> InferencePipeline:
    """
    Dependency to get the inference pipeline from app state.
    Lazy loads the pipeline on first request if not already loaded.
    
    Args:
        request: FastAPI request object
        
    Returns:
        InferencePipeline instance
        
    Raises:
        HTTPException: If pipeline cannot be initialized
    """
    # Lazy load pipeline on first request
    if not request.app.state.pipeline_loaded:
        try:
            model_path = Path(settings.MODEL_ARTIFACTS_PATH).parent if settings.MODEL_ARTIFACTS_PATH else None
            request.app.state.pipeline = InferencePipeline(model_path=model_path)
            request.app.state.pipeline_loaded = True
            request.app.state.pipeline_error = None
            logger.info("Inference pipeline loaded successfully on first request")
        except FileNotFoundError as e:
            logger.error("Failed to load pipeline: missing model artifacts", error=str(e))
            request.app.state.pipeline_error = f"Model artifacts not found: {str(e)}"
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Prediction service is not ready. Models not found. Please train the models first. Error: {str(e)}"
            )
        except Exception as e:
            logger.error("Failed to load pipeline", error=str(e))
            request.app.state.pipeline_error = str(e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Prediction service failed to initialize. Error: {str(e)}"
            )
    
    pipeline = request.app.state.pipeline
    
    if pipeline is None:
        error_msg = request.app.state.pipeline_error or "Unknown error"
        logger.error("Pipeline not available", error=error_msg)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction service is not ready. {error_msg}"
        )
    
    return pipeline


# =============================================================================
# Validation Dependencies
# =============================================================================

async def validate_batch_size(request: Request) -> None:
    """
    Validate batch prediction request size.
    
    Args:
        request: FastAPI request object
        
    Raises:
        HTTPException: If batch size exceeds limit
    """
    try:
        body = await request.json()
        properties = body.get('properties', [])
        
        if len(properties) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size {len(properties)} exceeds maximum allowed size of {settings.MAX_BATCH_SIZE}"
            )
            
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request body: {str(e)}"
        )


async def validate_coordinates(latitude: float, longitude: float) -> None:
    """
    Validate geographic coordinates are within Nigeria bounds.
    
    Args:
        latitude: Latitude value
        longitude: Longitude value
        
    Raises:
        HTTPException: If coordinates are invalid
    """
    if not (settings.MIN_LATITUDE <= latitude <= settings.MAX_LATITUDE):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Latitude {latitude} is outside Nigeria bounds ({settings.MIN_LATITUDE}-{settings.MAX_LATITUDE})"
        )
    
    if not (settings.MIN_LONGITUDE <= longitude <= settings.MAX_LONGITUDE):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Longitude {longitude} is outside Nigeria bounds ({settings.MIN_LONGITUDE}-{settings.MAX_LONGITUDE})"
        )


# =============================================================================
# Rate Limiting Dependencies (Placeholder)
# =============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter.
    For production, use Redis-based limiter (e.g., slowapi).
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}
    
    async def __call__(self, request: Request) -> None:
        """
        Check rate limit for client.
        
        Args:
            request: FastAPI request object
            
        Raises:
            HTTPException: If rate limit exceeded
        """
        # Get client identifier (IP address)
        client_id = request.client.host
        
        # For production: implement proper rate limiting with Redis
        # This is a placeholder implementation
        
        logger.debug("Rate limit check", client_id=client_id)


# Create rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=60)


# =============================================================================
# Feature Validation Dependencies
# =============================================================================

async def validate_property_features(features: dict) -> dict:
    """
    Validate and sanitize property features.
    
    Args:
        features: Dictionary of property features
        
    Returns:
        Validated and sanitized features
        
    Raises:
        HTTPException: If features are invalid
    """
    # Check required numeric features
    numeric_features = [
        'list_beds', 'list_baths', 'detail_beds', 
        'detail_baths', 'detail_toilets'
    ]
    
    for feature in numeric_features:
        if feature in features:
            value = features[feature]
            
            # Ensure non-negative
            if value < 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"{feature} must be non-negative, got {value}"
                )
            
            # Cap at maximum
            if value > settings.BED_BATH_CAP:
                logger.warning(
                    "Feature value exceeds cap, will be capped",
                    feature=feature,
                    value=value,
                    cap=settings.BED_BATH_CAP
                )
    
    # Validate distance features (if present)
    distance_features = [
        'school_distance_meters', 'hospital_distance_meters',
        'clinic_distance_meters', 'mall_distance_meters',
        'pharmacy_distance_meters', 'police_station_distance_meters',
        'aerodrome_distance_meters'
    ]
    
    for feature in distance_features:
        if feature in features and features[feature] is not None:
            if features[feature] < 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"{feature} must be non-negative"
                )
    
    return features


# =============================================================================
# Logging Dependencies
# =============================================================================

async def log_request(request: Request) -> None:
    """
    Log incoming request details.
    
    Args:
        request: FastAPI request object
    """
    logger.info(
        "Incoming request",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )


# =============================================================================
# Environment Check Dependencies
# =============================================================================

async def check_environment() -> None:
    """
    Verify environment is properly configured.
    
    Raises:
        HTTPException: If environment is misconfigured
    """
    from config.env import validate_environment
    
    try:
        validate_environment()
    except RuntimeError as e:
        logger.error("Environment validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service misconfigured. Contact administrator."
        )


# =============================================================================
# Response Headers Dependencies
# =============================================================================

async def add_response_headers(request: Request):
    """
    Add custom headers to response.
    
    Returns:
        Dictionary of headers to add
    """
    return {
        "X-API-Version": settings.API_VERSION,
        "X-Environment": settings.ENVIRONMENT,
        "X-Request-ID": request.headers.get("X-Request-ID", "unknown")
    }


# =============================================================================
# Health Check Dependencies
# =============================================================================

async def check_model_health(pipeline: InferencePipeline = Depends(get_pipeline)) -> dict:
    """
    Check if models are healthy and ready.
    
    Args:
        pipeline: InferencePipeline instance
        
    Returns:
        Dictionary with health status
    """
    health_status = {
        "ensemble_loaded": pipeline.ensemble is not None,
        "cleaner_loaded": pipeline.cleaner is not None,
        "transformer_loaded": pipeline.transformer is not None,
        "engineer_loaded": pipeline.engineer is not None,
        "extractor_loaded": pipeline.extractor is not None
    }
    
    all_healthy = all(health_status.values())
    
    return {
        "healthy": all_healthy,
        "components": health_status
    }


# =============================================================================
# Authentication Dependencies (Placeholder)
# =============================================================================

async def verify_api_key(api_key: Optional[str] = None) -> None:
    """
    Verify API key for protected endpoints.
    
    For production: Implement proper API key validation.
    
    Args:
        api_key: API key from header
        
    Raises:
        HTTPException: If API key is invalid
    """
    # Placeholder - implement actual API key validation for production
    if settings.ENVIRONMENT == "production":
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        
        # Validate API key against database/config
        # For now, this is a placeholder
        logger.debug("API key validation", masked_key=f"{api_key[:4]}..." if api_key else None)


# =============================================================================
# Request Context Dependencies
# =============================================================================

class RequestContext:
    """Context object to pass request metadata through pipeline."""
    
    def __init__(
        self,
        request_id: str,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        self.request_id = request_id
        self.client_ip = client_ip
        self.user_agent = user_agent


async def get_request_context(request: Request) -> RequestContext:
    """
    Create request context from FastAPI request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        RequestContext object
    """
    import uuid
    
    return RequestContext(
        request_id=request.headers.get("X-Request-ID", str(uuid.uuid4())),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )