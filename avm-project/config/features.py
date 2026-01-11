"""
Feature definitions and constants for the AVM model.
"""

# Base numeric features
NUMERIC_FEATURES = [
    "list_beds",
    "list_baths",
    "detail_beds",
    "detail_baths",
    "detail_toilets",
    "latitude",
    "longitude",
]

# Distance features (will be auto-detected by suffix)
DISTANCE_SUFFIX = "_distance_meters"

# Features that need capping
CAPPABLE_FEATURES = [
    "list_beds",
    "detail_beds",
    "list_baths",
    "detail_baths",
    "detail_toilets",
]

# Geographic features
GEOGRAPHIC_FEATURES = ["latitude", "longitude"]

# Target variable
TARGET_VARIABLE = "price_naira"
LOG_TARGET_VARIABLE = "log_price"

# Features to be added during feature engineering
ENGINEERED_FEATURES = [
    "rooms_total",
    "total_bathrooms",
    "diff_beds",
    "diff_baths",
]

# Amenity keywords for binary extraction
AMENITY_KEYWORDS = {
    "amenity_parking": ["parking", "garage", "carport"],
    "amenity_security": ["security", "gated", "guard"],
    "amenity_water": ["water", "borehole", "tank"],
    "amenity_power": ["power", "generator", "solar"],
    "amenity_gym": ["gym", "fitness"],
    "amenity_pool": ["pool", "swimming"],
    "amenity_furnished": ["furnished", "furniture"],
    "amenity_internet": ["internet", "wifi", "broadband"],
}

# Address extraction patterns
ADDRESS_PATTERNS = {
    "state": r"(?:Lagos|Abuja|Rivers|Oyo|Kano)",
    "gated_estate": r"(?:gated|estate)",
}