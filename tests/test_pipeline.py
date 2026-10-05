import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.data.validate import validate_data
from src.features.engineer import build_features, build_inference_features
from src.models.pipeline_model import UnifiedServingPipeline


def test_data_validation_valid():
    """Tests that properly formatted hourly time series passes validation."""
    dates = pd.date_range("2023-01-01", periods=48, freq="1h")
    df = pd.DataFrame({"load_kwh": np.random.uniform(50, 150, size=48)}, index=dates)
    assert validate_data(df, target_col="load_kwh") is True


def test_data_validation_catches_negative_values():
    """Tests that physical constraint violations (negative load) raise ValueError."""
    dates = pd.date_range("2023-01-01", periods=48, freq="1h")
    df = pd.DataFrame({"load_kwh": np.random.uniform(50, 150, size=48)}, index=dates)
    df.iloc[5, 0] = -10.0
    with pytest.raises(ValueError, match="negative electricity consumption"):
        validate_data(df, target_col="load_kwh")


def test_data_validation_catches_nans():
    """Tests that missing values raise ValueError."""
    dates = pd.date_range("2023-01-01", periods=48, freq="1h")
    df = pd.DataFrame({"load_kwh": np.random.uniform(50, 150, size=48)}, index=dates)
    df.iloc[10, 0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_data(df, target_col="load_kwh")


def test_feature_engineering_no_leakage():
    """Tests that lag and rolling statistics do not leak future or current target values."""
    dates = pd.date_range("2023-01-01", periods=72, freq="1h")
    # Linear sequence: 0, 1, 2, 3...
    df = pd.DataFrame({"load_kwh": np.arange(72, dtype=float)}, index=dates)
    df_feat = build_features(df, target_col="load_kwh")
    
    # Check that lag_1 at time t equals target at time t-1
    assert df_feat["lag_1"].iloc[0] == df.loc[df_feat.index[0] - pd.Timedelta(hours=1), "load_kwh"]
    
    # Check that rolling mean at time t strictly excludes target at time t
    # For a sequence 0..23, mean is 11.5. Target at t=24 is 24.
    first_row = df_feat.iloc[0]
    expected_rolling = np.mean(np.arange(0, 24))
    assert first_row["rolling_mean_24h"] == pytest.approx(expected_rolling)


def test_unified_serving_pipeline_contract():
    """Tests that the unified serving model returns all required schema fields."""
    mock_m1 = MagicMock()
    mock_m1.predict.return_value = np.array([320.5])
    
    mock_m2 = MagicMock()
    mock_m2.predict.return_value = np.array([1])
    mock_m2.predict_proba.return_value = np.array([[0.15, 0.85]])
    
    pipeline = UnifiedServingPipeline(model_1=mock_m1, model_2=mock_m2, peak_threshold=300.0)
    
    history_24h = [250.0 + i for i in range(24)]
    result = pipeline.predict(history_24h, "2024-07-01 12:00:00")
    
    assert "target_timestamp" in result
    assert "predicted_load_kwh" in result
    assert "is_peak" in result
    assert "peak_probability" in result
    assert "peak_threshold_kwh" in result
    assert result["is_peak"] is True
    assert result["predicted_load_kwh"] == 320.5
    assert result["peak_probability"] == 0.85
