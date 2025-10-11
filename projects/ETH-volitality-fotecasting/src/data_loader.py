import os
import pandas as pd

# === Data Configuration ===
DATA_PATH = os.path.join("..", "candles", "train_sample.csv")
ROLLING_WINDOW = 7 * 24 * 12  # 7 days * 24h * 12 (5-min candles)

def load_eth_data():
    """
    Load ETH candle data from CSV and perform basic preprocessing.
    Returns a pandas DataFrame.
    """
    df = pd.read_csv(DATA_PATH)
    df["open_time"] = pd.to_datetime(df["open_time"])
    
    print("✅ Data loaded successfully")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}\n")
    print(df.head())

    return df
