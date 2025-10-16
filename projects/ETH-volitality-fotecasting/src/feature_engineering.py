from src.config import *
from scipy.stats import kurtosis
import numpy as np
import pandas as pd

def build_features(df, rolling_window=ROLLING_WINDOW, sort=True, compute_stats=True):
    """
    Compute core rolling and exponential features for ETH candle data.
    Ensures temporal causality (no look-ahead bias) and consistent feature generation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'close', 'high', 'low', and 'volume' columns.
    rolling_window : int
        Window size for rolling statistics.
    sort : bool
        Whether to sort by 'open_time' to enforce chronological order.
    compute_stats : bool
        Whether to compute and return μ, σ, kurtosis (only for training set).

    Returns
    -------
    tuple
        (DataFrame, mu, sigma, kurtosis) if compute_stats=True
        (DataFrame, None, None, None) otherwise
    """

    if sort and "open_time" in df.columns:
        df = df.sort_values("open_time").reset_index(drop=True)

    rolling_window = min(rolling_window, len(df))

    # --- Core basic features ---
    df["range"] = df["high"] - df["low"]
    df["log_volume"] = np.log1p(df["volume"])

    # --- Rolling statistics (backward-looking only) ---
    df["rolling_mean"] = (
        df["log_volume"]
        .rolling(window=rolling_window, min_periods=max(3, rolling_window // 2), closed="left")
        .mean()
    )
    df["rolling_std"] = (
        df["log_volume"]
        .rolling(window=rolling_window, min_periods=max(3, rolling_window // 2), closed="left")
        .std()
    )
    df["rolling_median"] = (
        df["log_volume"]
        .rolling(window=rolling_window, min_periods=max(3, rolling_window // 2), closed="left")
        .median()
    )
    df["rolling_mad"] = (
        df["log_volume"]
        .rolling(window=rolling_window, min_periods=max(3, rolling_window // 2), closed="left")
        .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=False)
    )

    # --- Exponentially weighted features (causal form) ---
    ewm_span = max(10, rolling_window // 10)
    df["ewm_volume"] = df["log_volume"].ewm(span=ewm_span, adjust=False).mean()
    df["ewm_mad"] = (
        np.abs(df["log_volume"] - df["ewm_volume"])
        .ewm(span=ewm_span, adjust=False)
        .mean()
    )

    # --- Log returns (stationarized and scaled prices) ---
    df["log_return"] = np.log(df["close"] / df["close"].shift(1)) * 100

    df["log_return"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["log_return"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Optional trimming to avoid forward-looking windows ---
    if len(df) > rolling_window:
        df = df.iloc[rolling_window:].reset_index(drop=True)

    # --- Optional: compute descriptive statistics for training diagnostics ---
    if compute_stats:
        mu = df["log_return"].mean()
        sigma = df["log_return"].std(ddof=1)
        kurt = kurtosis(df["log_return"].to_numpy(dtype=np.float64), fisher=True, bias=False)

        if kurt > 30:
            print(f"⚠️ Extreme leptokurtosis detected (kurtosis={kurt:.2f}) — heavy tails expected.")
        print(f"μ={mu:.6f}, σ={sigma:.6f}, kurtosis={kurt:.3f}")
    else:
        mu, sigma, kurt = None, None, None

    # ✅ Return clean, chronologically correct, leakage-free features
    return df, mu, sigma, kurt
