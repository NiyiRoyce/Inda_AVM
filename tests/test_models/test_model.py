"""
Unit tests for models module.
"""
import pytest
import pandas as pd
import numpy as np

from src.models.linear import LinearAVMModel
from src.models.residual import ResidualAVMModel


class TestLinearAVMModel:
    """Tests for LinearAVMModel class."""
    
    def test_fit_predict(self):
        """Test fitting and prediction."""
        model = LinearAVMModel()
        
        # Create synthetic data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
        })
        
        # Log-space target
        y = pd.Series([18, 19, 20, 21, 22])
        
        # Fit
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.smearing_factor is not None
        assert model.smearing_factor > 0
        
        # Predict
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert all(~np.isnan(predictions))
    
    def test_predict_price(self):
        """Test price-space prediction."""
        model = LinearAVMModel()
        
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
        })
        
        y = pd.Series([18, 19, 20])
        
        model.fit(X, y)
        
        # Predict in price space
        price_predictions = model.predict_price(X)
        
        assert len(price_predictions) == len(X)
        assert all(price_predictions > 0)
    
    def test_get_residuals(self):
        """Test residual computation."""
        model = LinearAVMModel()
        
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
        })
        
        y = pd.Series([18, 19, 20])
        
        model.fit(X, y)
        residuals = model.get_residuals(X, y)
        
        assert len(residuals) == len(X)
        # Mean residual should be close to 0
        assert abs(residuals.mean()) < 1e-10
    
    def test_save_load(self, tmp_path):
        """Test saving and loading."""
        model = LinearAVMModel()
        
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
        })
        
        y = pd.Series([18, 19, 20])
        
        model.fit(X, y)
        
        # Save
        model_path = tmp_path / "linear.pkl"
        smearing_path = tmp_path / "smearing.pkl"
        model.save(model_path, smearing_path)
        
        # Load
        loaded = LinearAVMModel.load(model_path, smearing_path)
        
        assert loaded.is_fitted
        assert loaded.smearing_factor == model.smearing_factor
        
        # Predictions should match
        pred_original = model.predict(X)
        pred_loaded = loaded.predict(X)
        
        assert np.allclose(pred_original, pred_loaded)


class TestResidualAVMModel:
    """Tests for ResidualAVMModel class."""
    
    def test_fit_predict(self):
        """Test fitting and prediction."""
        model = ResidualAVMModel()
        
        # Create synthetic data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5] * 20,  # LightGBM needs more samples
            'feature2': [2, 4, 6, 8, 10] * 20,
        })
        
        # Residuals
        y_residuals = np.array([0.1, -0.2, 0.3, -0.1, 0.2] * 20)
        
        # Fit
        model.fit(X, y_residuals)
        
        assert model.is_fitted
        assert model.best_iteration is not None
        
        # Predict
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert all(~np.isnan(predictions))
    
    def test_save_load(self, tmp_path):
        """Test saving and loading."""
        model = ResidualAVMModel()
        
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5] * 20,
            'feature2': [2, 4, 6, 8, 10] * 20,
        })
        
        y_residuals = np.array([0.1, -0.2, 0.3, -0.1, 0.2] * 20)
        
        model.fit(X, y_residuals)
        
        # Save
        filepath = tmp_path / "residual.pkl"
        model.save(filepath)
        
        # Load
        loaded = ResidualAVMModel.load(filepath)
        
        assert loaded.is_fitted
        assert loaded.best_iteration == model.best_iteration
        
        # Predictions should match
        pred_original = model.predict(X)
        pred_loaded = loaded.predict(X)
        
        assert np.allclose(pred_original, pred_loaded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])