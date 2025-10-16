from src.config import *
from scipy.stats import kurtosis
import numpy as np
import pandas as pd

def build_features(df, sort=True, compute_stats=True):
    if sort and "open_time" in df.columns:
        df = df.sort_values("open_time").reset_index(drop=True)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1)) * 100
    df["log_return"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["log_return"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if compute_stats:
        mu = df["log_return"].mean()
        sigma = df["log_return"].std(ddof=1)
        kurt = kurtosis(df["log_return"], fisher=True, bias=False)
        print(f"μ={mu:.6f}, σ={sigma:.6f}, kurtosis={kurt:.3f}")
    else:
        mu, sigma, kurt = None, None, None

    return df, mu, sigma, kurt

