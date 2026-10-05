import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame, target_col: str = "load_kwh") -> pd.DataFrame:
    """
    Creates minimal, leakage-safe features for time-series forecasting.
    Uses calendar attributes, historical lags, and rolling statistics.
    """
    df_feat = df.copy()
    
    # 1. Calendar features
    df_feat["hour"] = df_feat.index.hour
    df_feat["day_of_week"] = df_feat.index.dayofweek
    
    # 2. Historical lags (previous hour and same hour yesterday)
    df_feat["lag_1"] = df_feat[target_col].shift(1)
    df_feat["lag_24"] = df_feat[target_col].shift(24)
    
    # 3. Rolling 24-hour mean (shifted by 1 so current hour's target is never leaked)
    df_feat["rolling_mean_24h"] = df_feat[target_col].shift(1).rolling(24).mean()
    
    # Drop rows at the beginning that have NaNs due to the 24-hour lookback window
    return df_feat.dropna()


def build_inference_features(recent_24h_loads: list[float], timestamp: pd.Timestamp) -> pd.DataFrame:
    """
    Computes identical feature row from a 24-hour historical window at serving time.
    recent_24h_loads must be 24 floats ordered from [t-24, t-23, ..., t-1].
    """
    if len(recent_24h_loads) < 24:
        raise ValueError("Inference requires at least 24 recent hourly load measurements.")
        
    lag_1 = recent_24h_loads[-1]       # previous hour (t-1)
    lag_24 = recent_24h_loads[0]       # same hour yesterday (t-24)
    rolling_mean_24h = float(np.mean(recent_24h_loads[-24:]))
    
    return pd.DataFrame([{
        "hour": timestamp.hour,
        "day_of_week": timestamp.dayofweek,
        "lag_1": lag_1,
        "lag_24": lag_24,
        "rolling_mean_24h": rolling_mean_24h
    }])
