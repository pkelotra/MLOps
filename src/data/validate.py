import pandas as pd


def validate_data(df: pd.DataFrame, target_col: str = "load_kwh") -> bool:
    """
    Validates dataset health before entering the feature & training pipelines.
    Raises ValueError if any critical data quality assumption is violated.
    """
    # 1. Check index type and target column existence
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data validation failed: Index must be a DatetimeIndex.")
    if target_col not in df.columns:
        raise ValueError(f"Data validation failed: Target column '{target_col}' not found.")
        
    # 2. Check for missing / NaN values
    null_count = df[target_col].isna().sum()
    if null_count > 0:
        raise ValueError(f"Data validation failed: Found {null_count} missing values in '{target_col}'.")
        
    # 3. Check physical validity: electricity consumption must be non-negative
    if (df[target_col] < 0).any():
        raise ValueError("Data validation failed: Detected negative electricity consumption values.")
        
    # 4. Check timestamp monotonicity (strictly increasing order)
    if not df.index.is_monotonic_increasing:
        raise ValueError("Data validation failed: Timestamps are not strictly monotonically increasing.")
        
    # 5. Check hourly continuity (no unexpected missing timestamps)
    expected_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600 + 1
    actual_hours = len(df)
    if actual_hours != expected_hours:
        raise ValueError(f"Data validation failed: Time continuity broken. Expected {expected_hours} hours, got {actual_hours}.")
        
    return True
