import pandas as pd


def load_client_data(filepath: str, client_id: str = "MT_200", start_date: str = "2012-01-01") -> pd.DataFrame:
    """
    Loads electricity consumption for a specific client from the raw UCI dataset.
    
    Converts 15-minute kW readings to 1-hour kWh values and filters out
    onboarding zeros before start_date.
    """
    # 1. Identify date column name from header
    header = pd.read_csv(filepath, sep=';', nrows=0)
    date_col = header.columns[0]
    
    # 2. Efficiently read only the timestamp and selected client column
    df = pd.read_csv(
        filepath,
        sep=';',
        decimal=',',
        usecols=[date_col, client_id]
    )
    df.columns = ["timestamp", "load_kw"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    
    # 3. Filter to start_date onwards to avoid initial onboarding zero-values
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
        
    # 4. Handle March daylight saving 0s (1:00 to 2:00 am drop) via forward fill
    df["load_kw"] = df["load_kw"].replace(0, method="ffill")
    
    # 5. Resample to 1-hour intervals: sum four 15-min intervals and divide by 4 to get kWh
    df_hourly = (df.resample("1h").sum() / 4.0).rename(columns={"load_kw": "load_kwh"})
    
    return df_hourly.sort_index()
