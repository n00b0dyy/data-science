from src.config import * 

def build_features(df, rolling_window=7 * 24 * 12):
    """
    Compute core rolling and exponential features for ETH candle data.
    Designed for use in notebooks (e.g., 01_EDA_ETH.ipynb).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'close' and 'volume' columns.
    rolling_window : int
        Window size for rolling statistics (default = 7 days of 5-min candles).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new feature columns added.
    """

    rolling_window = min(rolling_window, len(df))

    df["range"] = df["high"] - df["low"]
    
    # --- Log-volume and rolling statistics ---
    df["log_volume"] = np.log1p(df["volume"])

    df["rolling_mean"] = (
        df["log_volume"]
        .rolling(window=rolling_window, min_periods=100, closed="left")
        .mean()
    )
    df["rolling_std"] = (
        df["log_volume"]
        .rolling(window=rolling_window, min_periods=100, closed="left")
        .std()
    )
    df["rolling_median"] = (
        df["log_volume"]
        .rolling(window=rolling_window, min_periods=100, closed="left")
        .median()
    )
    df["rolling_mad"] = (
        df["log_volume"]
        .rolling(window=rolling_window, min_periods=100, closed="left")
        .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=False)
    )

    # --- Exponentially weighted metrics ---
    ewm_span = max(10, rolling_window // 10)
    df["ewm_volume"] = df["log_volume"].ewm(span=ewm_span, adjust=False).mean()
    df["ewm_mad"] = (
        np.abs(df["log_volume"] - df["ewm_volume"])
        .ewm(span=ewm_span, adjust=False)
        .mean()
    )

    # --- Log returns ---
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Drop NaN and infinite values before modeling
    df["log_return"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["log_return"], inplace=True)

    # --- Descriptive statistics ---
    mu = df["log_return"].mean()
    sigma = df["log_return"].std(ddof=1)
    kurt = kurtosis(df["log_return"].to_numpy(dtype=np.float64), fisher=True, bias=False)

    # Print with warning if extreme
    if kurt > 30:
        print(f"⚠️ Extreme leptokurtosis detected (kurtosis={kurt:.2f}) — heavy tails expected.")
    print(f"μ={mu:.6f}, σ={sigma:.6f}, kurtosis={kurt:.3f}")

    return df, mu, sigma, kurt
