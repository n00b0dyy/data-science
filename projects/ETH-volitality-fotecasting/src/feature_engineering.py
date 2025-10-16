from src.config import *
from scipy.stats import kurtosis
import numpy as np
import pandas as pd

def build_features(
    df, 
    sort=True, 
    compute_stats=True, 
    winsorize_pct=0.01, 
    reference_bounds=None, 
    reference_stats=None
):
    if sort and "open_time" in df.columns:
        df = df.sort_values("open_time").reset_index(drop=True)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1)) * 100
    df["log_return"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["log_return"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Winsorization ---
    if reference_bounds is None:
        lower = df["log_return"].quantile(winsorize_pct)
        upper = df["log_return"].quantile(1 - winsorize_pct)
    else:
        lower, upper = reference_bounds

    df["log_return"] = np.clip(df["log_return"], lower, upper)

    # --- Stats ---
    if compute_stats:
        mu = df["log_return"].mean()
        sigma = df["log_return"].std(ddof=1)
        kurt = kurtosis(df["log_return"], fisher=True, bias=False)
    elif reference_stats is not None:
        mu, sigma, kurt = reference_stats
    else:
        mu, sigma, kurt = None, None, None

    return df, mu, sigma, kurt, (lower, upper)

