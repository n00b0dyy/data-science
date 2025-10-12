import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# --- Local imports (relative to project root) ---
from src.data_loader import load_eth_data
from src.feature_engineering import build_features


def train_egarch(df, arma_lags=1, p=1, o=1, q=1):
    """
    Fit an ARMA–EGARCH model to ETH log returns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least 'close' and 'volume' columns.
    arma_lags : int or tuple
        Lag order for the ARMA mean process (default: AR(1))
    p, o, q : int
        EGARCH model orders:
        p -> GARCH terms, o -> asymmetry (leverage), q -> ARCH terms.

    Returns
    -------
    res : arch.univariate.base.ARCHModelResult
        Fitted model results.
    """

    # --- Prepare Data ---
    df, mu, sigma, kurt = build_features(df)
    returns = df["log_return"].dropna() * 100  # convert to percentage scale

    print("\n=== Model Input Summary ===")
    print(f"Observations: {len(returns)}")
    print(f"Mean: {mu:.6f}, Std: {sigma:.6f}, Kurtosis: {kurt:.3f}")

    # --- Define ARMA–EGARCH Model ---
    model = arch_model(
        returns,
        mean="ARX", lags=arma_lags,
        vol="EGARCH", p=p, o=o, q=q,
        dist="t"
    )

    print("\n=== Training ARMA–EGARCH Model ===")
    res = model.fit(update_freq=10, disp="off")
    print(res.summary())

    return res, df


def plot_conditional_variance(res, df, save_path=None):
    """
    Plot conditional variance (EGARCH output) vs realized volatility.
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    cond_var = res.conditional_volatility
    realized = df["log_return"].rolling(window=288).std() * 100  # 1-day realized volatility

    ax.plot(df["open_time"].iloc[-len(cond_var):], cond_var, label="Predicted Volatility (EGARCH)", color="orange", lw=1.2)
    ax.plot(df["open_time"].iloc[-len(realized):], realized, label="Realized Volatility (Rolling 1D Std)", color="steelblue", alpha=0.7)
    ax.set_title("Conditional Volatility — ARMA–EGARCH(1,1)", fontsize=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Volatility (%)")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    print("=== Loading ETH/USDT Data ===")
    df = load_eth_data()

    print("\n=== Training EGARCH Model ===")
    res, df = train_egarch(df, arma_lags=(1, 1), p=1, o=1, q=1)

    # --- Display & Plot ---
    plot_conditional_variance(
        res, df,
        save_path=os.path.join("plots", "egarch_volatility_comparison.png")
    )
