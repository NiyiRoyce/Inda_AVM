"""
Feature extraction from raw data fields.

This module extracts binary features and categorical information from
text and structured fields in the dataset.
"""
import logging
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from config.features import AMENITY_KEYWORDS, ADDRESS_PATTERNS

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts features from raw data fields."""
    
    def __init__(self):
        """Initialize feature extractor with predefined patterns."""
        self.amenity_keywords = AMENITY_KEYWORDS
        self.address_patterns = ADDRESS_PATTERNS
    
    def extract_amenities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract binary amenity features from description fields.
        
        Searches for amenity keywords in property descriptions and creates
        binary features indicating presence of each amenity.
        
        Args:
            df: Input DataFrame with description fields
            
        Returns:
            DataFrame with new amenity features
        """
        df = df.copy()
        
        # Get description column if available
        description_cols = [
            col for col in df.columns 
            if col.lower() in ['description', 'details', 'property_details']
        ]
        
        if not description_cols:
            logger.warning("No description columns found for amenity extraction")
            return df
        
        description_col = description_cols[0]
        
        # Extract each amenity
        for amenity_name, keywords in self.amenity_keywords.items():
            df[amenity_name] = self._search_keywords(
                df[description_col],
                keywords
            )
            logger.debug(f"Extracted amenity feature: {amenity_name}")
        
        logger.info(
            f"Extracted {len(self.amenity_keywords)} amenity features "
            f"from {description_col}"
        )
        
        return df
    
    @staticmethod
    def _search_keywords(
        series: pd.Series,
        keywords: List[str],
        case_sensitive: bool = False
    ) -> pd.Series:
        """
        Search for keywords in a series of text values.
        
        Args:
            series: Pandas Series with text values
            keywords: List of keywords to search for
            case_sensitive: Whether search is case-sensitive
            
        Returns:
            Binary Series (1 if any keyword found, 0 otherwise)
        """
        result = pd.Series(0, index=series.index, dtype=int)
        
        # Prepare keywords
        search_keywords = keywords if case_sensitive else [k.lower() for k in keywords]
        
        # Create regex pattern
        pattern = '|'.join(re.escape(kw) for kw in search_keywords)
        
        if not pattern:
            return result
        
        # Apply pattern matching
        mask = series.fillna('').str.lower().str.contains(
            pattern,
            case=case_sensitive,
            regex=True,
            na=False
        )
        
        result[mask] = 1
        return result
    
    def extract_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract location-based features from address and description.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new location features
        """
        df = df.copy()
        
        # Extract state from address if available
        if "address" in df.columns:
            df["state_extracted"] = self._extract_state(df["address"])
            logger.debug("Extracted state feature from address")
        
        # Extract gated estate indicator
        description_cols = [
            col for col in df.columns 
            if col.lower() in ['description', 'details', 'property_details']
        ]
        
        if description_cols:
            desc_col = description_cols[0]
            df["is_gated_estate"] = self._search_keywords(
                df[desc_col],
                ["gated", "estate", "secure", "enclosed"]
            )
            logger.debug("Extracted gated estate feature")
        
        return df
    
    @staticmethod
    def _extract_state(address_series: pd.Series) -> pd.Series:
        """
        Extract state information from address strings.
        
        Args:
            address_series: Series containing address strings
            
        Returns:
            Series with extracted state names (or NaN if not found)
        """
        state_patterns = {
            'Lagos': r'\bLagos\b',
            'Abuja': r'\bAbuja\b|FCT',
            'Rivers': r'\bRivers\b',
            'Oyo': r'\bOyo\b',
            'Kano': r'\bKano\b',
        }
        
        result = pd.Series(np.nan, index=address_series.index, dtype=object)
        
        for state, pattern in state_patterns.items():
            mask = address_series.fillna('').str.contains(
                pattern,
                case=False,
                regex=True,
                na=False
            )
            result[mask] = state
        
        return result
    
    def extract_numeric_from_text(
        self,
        df: pd.DataFrame,
        column: str,
        prefix: str = "text"
    ) -> pd.DataFrame:
        """
        Extract numeric values from text fields.
        
        Useful for extracting prices, areas, or other numeric info
        from description or details fields.
        
        Args:
            df: Input DataFrame
            column: Column name to extract from
            prefix: Prefix for new feature names
            
        Returns:
            DataFrame with extracted numeric features
        """
        if column not in df.columns:
            logger.warning(f"Column {column} not found")
            return df
        
        df = df.copy()
        
        # Extract first numeric value found
        def extract_first_number(text):
            if pd.isna(text):
                return np.nan
            numbers = re.findall(r'\d+\.?\d*', str(text))
            return float(numbers[0]) if numbers else np.nan
        
        df[f"{prefix}_first_number"] = df[column].apply(extract_first_number)
        
        # Count numeric values
        def count_numbers(text):
            if pd.isna(text):
                return 0
            return len(re.findall(r'\d+\.?\d*', str(text)))
        
        df[f"{prefix}_number_count"] = df[column].apply(count_numbers)
        
        logger.info(f"Extracted numeric features from {column}")
        
        return df
    
    def extract_text_features(
        self,
        df: pd.DataFrame,
        column: str,
        prefix: str = "text"
    ) -> pd.DataFrame:
        """
        Extract statistical features from text content.
        
        Args:
            df: Input DataFrame
            column: Column name to analyze
            prefix: Prefix for new feature names
            
        Returns:
            DataFrame with text-based statistical features
        """
        if column not in df.columns:
            logger.warning(f"Column {column} not found")
            return df
        
        df = df.copy()
        
        # Length features
        df[f"{prefix}_length"] = df[column].fillna("").str.len()
        df[f"{prefix}_word_count"] = df[column].fillna("").str.split().str.len()
        
        # Capitalization features
        df[f"{prefix}_uppercase_ratio"] = df[column].apply(
            lambda x: (
                sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
                if pd.notna(x) else 0
            )
        )
        
        logger.info(f"Extracted text features from {column}")
        
        return df
    
    def extract_all_features(
        self,
        df: pd.DataFrame,
        extract_amenities: bool = True,
        extract_location: bool = True,
        text_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply all feature extraction methods to DataFrame.
        
        Args:
            df: Input DataFrame
            extract_amenities: Whether to extract amenity features
            extract_location: Whether to extract location features
            text_columns: List of columns for text feature extraction
            
        Returns:
            DataFrame with all extracted features
        """
        df = df.copy()
        
        # Extract amenities
        if extract_amenities:
            df = self.extract_amenities(df)
        
        # Extract location features
        if extract_location:
            df = self.extract_location_features(df)
        
        # Extract text features
        if text_columns:
            for col in text_columns:
                if col in df.columns:
                    df = self.extract_text_features(df, col)
                    df = self.extract_numeric_from_text(df, col)
        
        logger.info(f"Feature extraction complete. DataFrame shape: {df.shape}")
        
        return df
    
    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method for extracting all standard features.
        
        Extracts amenities and location features with defaults.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all extracted features
        """
        return self.extract_all_features(
            df,
            extract_amenities=True,
            extract_location=True,
            text_columns=None
        )
    
    def get_extracted_feature_names(self) -> Dict[str, List[str]]:
        """
        Get names of all features that can be extracted.
        
        Returns:
            Dictionary mapping feature categories to feature names
        """
        return {
            "amenities": list(self.amenity_keywords.keys()),
            "location": ["state_extracted", "is_gated_estate"],
            "text": ["length", "word_count", "uppercase_ratio"],
            "numeric": ["first_number", "number_count"],
        }
