"""
Unit tests for preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np

from src.preprocessing.cleaners import DataCleaner
from src.preprocessing.transformers import DataTransformer
from src.preprocessing.imputers import FeatureImputer


class TestDataCleaner:
    """Tests for DataCleaner class."""
    
    def test_remove_invalid_prices(self):
        """Test removal of invalid prices."""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'price_naira': [100000000, 0, -50000, 200000000],
        })
        
        cleaned = cleaner.remove_invalid_prices(df)
        
        assert len(cleaned) == 2
        assert all(cleaned['price_naira'] > 0)
    
    def test_cap_bed_bath_values(self):
        """Test capping of bed/bath values."""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'list_beds': [3, 50, 4],
            'list_baths': [2, 100, 3],
        })
        
        cleaned = cleaner.cap_bed_bath_values(df)
        
        assert cleaned['list_beds'].max() == cleaner.bed_bath_cap
        assert cleaned['list_baths'].max() == cleaner.bed_bath_cap
    
    def test_clean_geographic_coordinates(self):
        """Test cleaning of invalid coordinates."""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'latitude': [6.5, 20.0, 6.7, -10.0],
            'longitude': [3.4, 3.5, 30.0, 3.6],
        })
        
        cleaned = cleaner.clean_geographic_coordinates(df)
        
        # Check that invalid values are NaN
        assert pd.isna(cleaned.loc[1, 'latitude'])  # 20.0 is invalid
        assert pd.isna(cleaned.loc[2, 'longitude'])  # 30.0 is invalid
        assert pd.isna(cleaned.loc[3, 'latitude'])  # -10.0 is invalid
        
        # Check that valid values remain
        assert cleaned.loc[0, 'latitude'] == 6.5
        assert cleaned.loc[0, 'longitude'] == 3.4


class TestDataTransformer:
    """Tests for DataTransformer class."""
    
    def test_create_log_target(self):
        """Test log target creation."""
        transformer = DataTransformer()
        
        df = pd.DataFrame({
            'price_naira': [100000000, 200000000, 150000000],
        })
        
        transformed = transformer.create_log_target(df)
        
        assert 'log_price' in transformed.columns
        assert np.allclose(
            transformed['log_price'],
            np.log([100000000, 200000000, 150000000])
        )
    
    def test_create_log_target_with_invalid_prices(self):
        """Test that log transform fails with invalid prices."""
        transformer = DataTransformer()
        
        df = pd.DataFrame({
            'price_naira': [100000000, 0, 150000000],
        })
        
        with pytest.raises(ValueError):
            transformer.create_log_target(df)
    
    def test_log_transform_distances(self):
        """Test log transformation of distance features."""
        transformer = DataTransformer()
        
        df = pd.DataFrame({
            'school_distance_meters': [500, 1000, 0],
            'hospital_distance_meters': [1500, 2000, 100],
        })
        
        distance_cols = ['school_distance_meters', 'hospital_distance_meters']
        transformed = transformer.log_transform_distances(df, distance_cols)
        
        assert 'log_school_distance_meters' in transformed.columns
        assert 'log_hospital_distance_meters' in transformed.columns
        
        # log(1 + 0) should be 0
        assert transformed.loc[2, 'log_school_distance_meters'] == 0


class TestFeatureImputer:
    """Tests for FeatureImputer class."""
    
    def test_fit_transform(self):
        """Test fitting and transforming."""
        imputer = FeatureImputer(strategy='median')
        
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4],
            'feature2': [10, np.nan, 30, 40],
        })
        
        features = ['feature1', 'feature2']
        transformed = imputer.fit_transform(df, features)
        
        assert imputer.is_fitted
        assert transformed['feature1'].isna().sum() == 0
        assert transformed['feature2'].isna().sum() == 0
        
        # Median of [1, 2, 4] is 2
        assert transformed.loc[2, 'feature1'] == 2.0
    
    def test_transform_without_fit(self):
        """Test that transform fails without fitting."""
        imputer = FeatureImputer()
        
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan],
        })
        
        with pytest.raises(RuntimeError):
            imputer.transform(df)
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading imputer."""
        imputer = FeatureImputer(strategy='median')
        
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4],
            'feature2': [10, np.nan, 30, 40],
        })
        
        features = ['feature1', 'feature2']
        imputer.fit(df, features)
        
        # Save
        filepath = tmp_path / "imputer.pkl"
        imputer.save(filepath)
        
        # Load
        loaded = FeatureImputer.load(filepath)
        
        assert loaded.is_fitted
        assert loaded.feature_names == features
        assert loaded.strategy == 'median'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])