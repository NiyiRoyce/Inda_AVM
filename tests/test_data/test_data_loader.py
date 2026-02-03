"""
Unit tests for data loading module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.data.loader import DataLoader
from src.data.validator import DataValidator


class TestDataValidator:
    """Tests for DataValidator class."""
    
    def test_validate_valid_data(self):
        """Test validation with valid data."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'price_naira': [100000000, 200000000, 150000000],
            'list_beds': [3, 4, 5],
            'list_baths': [2, 3, 4],
            'detail_beds': [3, 4, 5],
            'detail_baths': [2, 3, 4],
            'detail_toilets': [3, 4, 5],
            'latitude': [6.5, 6.6, 6.7],
            'longitude': [3.4, 3.5, 3.6],
        })
        
        result = validator.validate(df)
        
        assert result['is_valid'] is True
        assert len(result['issues']) == 0
        assert len(result['invalid_indices']) == 0
    
    def test_validate_invalid_prices(self):
        """Test validation with invalid prices."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'price_naira': [100000000, 0, -50000],
            'list_beds': [3, 4, 5],
            'list_baths': [2, 3, 4],
            'detail_beds': [3, 4, 5],
            'detail_baths': [2, 3, 4],
            'detail_toilets': [3, 4, 5],
            'latitude': [6.5, 6.6, 6.7],
            'longitude': [3.4, 3.5, 3.6],
        })
        
        result = validator.validate(df)
        
        assert result['is_valid'] is False
        assert len(result['issues']) > 0
        assert len(result['invalid_indices']) == 2  # Two invalid prices
    
    def test_validate_invalid_coordinates(self):
        """Test validation with invalid coordinates."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'price_naira': [100000000, 200000000, 150000000],
            'list_beds': [3, 4, 5],
            'list_baths': [2, 3, 4],
            'detail_beds': [3, 4, 5],
            'detail_baths': [2, 3, 4],
            'detail_toilets': [3, 4, 5],
            'latitude': [6.5, 20.0, 6.7],  # Invalid latitude
            'longitude': [3.4, 3.5, 30.0],  # Invalid longitude
        })
        
        result = validator.validate(df)
        
        assert result['is_valid'] is False
        assert any('latitude' in issue for issue in result['issues'])
        assert any('longitude' in issue for issue in result['issues'])
    
    def test_get_summary_stats(self):
        """Test summary statistics generation."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'price_naira': [100000000, 200000000, 150000000],
            'list_beds': [3, 4, 5],
        })
        
        stats = validator.get_summary_stats(df)
        
        assert stats['num_rows'] == 3
        assert stats['num_columns'] == 2
        assert 'price_stats' in stats
        assert stats['price_stats']['mean'] == 150000000
        assert stats['price_stats']['median'] == 150000000


class TestDataLoader:
    """Tests for DataLoader class."""
    
    @patch('src.data.loader.BigQueryClient')
    def test_load_and_validate(self, mock_bq_client):
        """Test loading and validation."""
        # Mock BigQuery client
        mock_df = pd.DataFrame({
            'price_naira': [100000000, 200000000],
            'list_beds': [3, 4],
            'list_baths': [2, 3],
            'detail_beds': [3, 4],
            'detail_baths': [2, 3],
            'detail_toilets': [3, 4],
            'latitude': [6.5, 6.6],
            'longitude': [3.4, 3.5],
        })
        
        mock_client_instance = Mock()
        mock_client_instance.load_master_listings.return_value = mock_df
        mock_bq_client.return_value = mock_client_instance
        
        # Load data
        loader = DataLoader()
        df = loader.load_and_validate(validate=True, remove_invalid=False)
        
        assert len(df) == 2
        assert 'price_naira' in df.columns
    
    def test_load_from_csv(self, tmp_path):
        """Test loading from CSV."""
        # Create temporary CSV
        csv_file = tmp_path / "test_data.csv"
        
        df = pd.DataFrame({
            'price_naira': [100000000, 200000000],
            'list_beds': [3, 4],
            'list_baths': [2, 3],
            'detail_beds': [3, 4],
            'detail_baths': [2, 3],
            'detail_toilets': [3, 4],
            'latitude': [6.5, 6.6],
            'longitude': [3.4, 3.5],
        })
        
        df.to_csv(csv_file, index=False)
        
        # Load
        loader = DataLoader()
        loaded_df = loader.load_from_csv(str(csv_file), validate=False)
        
        assert len(loaded_df) == 2
        assert list(loaded_df.columns) == list(df.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])