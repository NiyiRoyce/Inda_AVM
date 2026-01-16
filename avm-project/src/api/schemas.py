"""
API request/response schemas for AVM predictions.
"""
from typing import Optional, List
from pydantic import BaseModel, Field, validator


class PropertyFeatures(BaseModel):
    """Input features for property valuation."""
    
    # Property configuration
    list_beds: float = Field(..., ge=0, le=10, description="Number of bedrooms (listed)")
    list_baths: float = Field(..., ge=0, le=10, description="Number of bathrooms (listed)")
    detail_beds: float = Field(..., ge=0, le=10, description="Number of bedrooms (detailed)")
    detail_baths: float = Field(..., ge=0, le=10, description="Number of bathrooms (detailed)")
    detail_toilets: float = Field(..., ge=0, le=10, description="Number of toilets")
    
    # Geographic coordinates
    latitude: float = Field(..., ge=4.0, le=14.0, description="Latitude (Nigeria bounds)")
    longitude: float = Field(..., ge=2.0, le=15.0, description="Longitude (Nigeria bounds)")
    
    # Distance features (in meters)
    school_distance_meters: Optional[float] = Field(None, ge=0, description="Distance to nearest school")
    hospital_distance_meters: Optional[float] = Field(None, ge=0, description="Distance to nearest hospital")
    clinic_distance_meters: Optional[float] = Field(None, ge=0, description="Distance to nearest clinic")
    mall_distance_meters: Optional[float] = Field(None, ge=0, description="Distance to nearest mall")
    pharmacy_distance_meters: Optional[float] = Field(None, ge=0, description="Distance to nearest pharmacy")
    police_station_distance_meters: Optional[float] = Field(None, ge=0, description="Distance to nearest police station")
    aerodrome_distance_meters: Optional[float] = Field(None, ge=0, description="Distance to nearest aerodrome")
    
    @validator('latitude', 'longitude')
    def validate_coordinates(cls, v, field):
        if v is None:
            raise ValueError(f"{field.name} cannot be None")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "list_beds": 4.0,
                "list_baths": 4.0,
                "detail_beds": 4.0,
                "detail_baths": 4.0,
                "detail_toilets": 5.0,
                "latitude": 6.5244,
                "longitude": 3.3792,
                "school_distance_meters": 1200.0,
                "hospital_distance_meters": 2500.0,
                "clinic_distance_meters": 1800.0,
                "mall_distance_meters": 3000.0,
                "pharmacy_distance_meters": 1500.0,
                "police_station_distance_meters": 2000.0,
                "aerodrome_distance_meters": 25000.0
            }
        }


class PredictionRequest(BaseModel):
    """Request schema for single property prediction."""
    property: PropertyFeatures


class BatchPredictionRequest(BaseModel):
    """Request schema for batch property predictions."""
    properties: List[PropertyFeatures] = Field(..., max_items=100, description="List of properties (max 100)")


class PredictionResponse(BaseModel):
    """Response schema for property valuation."""
    
    predicted_price: float = Field(..., description="Predicted property price in Naira")
    log_price: float = Field(..., description="Log-transformed predicted price")
    baseline_price: float = Field(..., description="Linear regression baseline prediction")
    residual_correction: float = Field(..., description="Residual model correction")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prediction confidence (0-1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 450000000.0,
                "log_price": 19.924,
                "baseline_price": 430000000.0,
                "residual_correction": 20000000.0,
                "confidence_score": 0.85
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    count: int = Field(..., description="Number of predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether models are loaded")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment (dev/staging/prod)")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "detail": "latitude must be between 4.0 and 14.0"
            }
        }