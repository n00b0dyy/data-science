import os
import pandas as pd
from src.config import DATA_PATH, ROLLING_WINDOW

def load_eth_data():
    """
    Load ETH candle data from CSV and perform basic preprocessing.
    Returns a pandas DataFrame.
    """
    abs_path = os.path.abspath(DATA_PATH)
    print(f"📂 Loading data from: {abs_path}")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"❌ Data file not found at: {abs_path}")

    df = pd.read_csv(abs_path)
    df["open_time"] = pd.to_datetime(df["open_time"])

    print("✅ Data loaded successfully")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}\n")
    print(df.head())

    return df
