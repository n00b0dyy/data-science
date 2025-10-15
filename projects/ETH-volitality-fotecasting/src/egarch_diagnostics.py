# === EGARCH Residual Diagnostics ===
# Comprehensive post-estimation diagnostics for EGARCH(1,1) model residuals.
# Tests for autocorrelation, heteroskedasticity, and distributional assumptions.

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import jarque_bera, probplot
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# === Path setup ===
# Ensures imports work when running directly from /src/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# === Local imports ===
# === Local imports ===
from src.config import *                   # global paths and constants
from src.data_loader import load_eth_data  # load CSV data
from src.egarch_model import train_egarch  # model training pipeline


# ============================================================
# Utility
# ============================================================

def print_section(title: str):
    """Pretty section separator for CLI output."""
    line = "═" * 70
    print(f"\n{line}\n{title.center(70)}\n{line}\n")


# ============================================================
# Core Diagnostic Function
# ============================================================
def plot_conditional_variance(res, df, save_path=None):
    """Plot conditional variance (EGARCH output or forecast) vs realized volatility."""
    print_section("Plotting Conditional Variance")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Use forecasted volatility if present
    if "predicted_vol" in df.columns:
        cond_var = df["predicted_vol"]
        label_pred = "Predicted Volatility (Forecast)"
    else:
        cond_var = res.conditional_volatility
        label_pred = "Predicted Volatility (In-Sample EGARCH)"

    realized = df["log_return"].rolling(window=min(len(df), ROLLING_WINDOW)).std() * 100

    # align lengths safely
    n = min(len(df["open_time"]), len(cond_var), len(realized))
    x = df["open_time"].iloc[-n:]
    y1 = cond_var.iloc[-n:]
    y2 = realized.iloc[-n:]

    ax.plot(x, y1, label=label_pred, color="orange", lw=1.2)
    ax.plot(x, y2, label="Realized Volatility (Rolling Std)", color="steelblue", alpha=0.7)

    ax.set_title("Conditional Volatility — EGARCH(1,1)", fontsize=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Volatility (%)")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"📈 Plot saved to {os.path.relpath(save_path, PROJECT_ROOT)}")

    plt.show()


def weekly_volatility_comparison(res, df):
    """
    Compare EGARCH predicted volatility vs realized volatility per week.
    Prints a summary table of their mean differences and correlations.
    """
    print_section("Weekly Volatility Comparison Table")

    cond_var = pd.Series(res.conditional_volatility, index=df.index[-len(res.conditional_volatility):])
    realized = df["log_return"].rolling(window=min(len(df), ROLLING_WINDOW)).std() * 100

    tmp = pd.DataFrame({
        "open_time": df["open_time"].iloc[-len(cond_var):],
        "predicted_vol": cond_var * 100,
        "realized_vol": realized.iloc[-len(cond_var):]
    }).dropna()

    weekly = (
        tmp.set_index("open_time")
        .resample("W")
        .agg({"predicted_vol": "mean", "realized_vol": "mean"})
        .dropna()
    )

 
    weekly["diff_abs"] = (weekly["predicted_vol"] - weekly["realized_vol"]).abs()
    weekly["corr"] = (
        tmp.set_index("open_time")
        .resample("W")[["predicted_vol", "realized_vol"]]
        .corr()
        .iloc[0::2, -1]
        .to_numpy()
    )

 
    print(f"{'Week':<12}{'Predicted':>12}{'Realized':>12}{'AbsDiff':>12}{'Corr':>10}")
    print("-" * 58)
    for i, row in weekly.iterrows():
        print(
            f"{i.strftime('%Y-%m-%d'): <12}"
            f"{row['predicted_vol']: >12.3f}"
            f"{row['realized_vol']: >12.3f}"
            f"{row['diff_abs']: >12.3f}"
            f"{row['corr']: >10.3f}"
        )

    mean_corr = weekly["corr"].mean()
    mean_diff = weekly["diff_abs"].mean()

    print("\nAverages across all weeks:")
    print(f" Mean correlation   : {mean_corr:.3f}")
    print(f" Mean abs. diff (%) : {mean_diff:.3f}")

    return weekly

def egarch_residual_diagnostics(res, lags=20, save_plots=True):
    """
    Perform comprehensive statistical diagnostics on EGARCH residuals.

    Parameters
    ----------
    res : arch.univariate.base.ARCHModelResult
        Fitted EGARCH model result object.
    lags : int
        Number of lags used in ACF and Ljung–Box tests.
    save_plots : bool
        Whether to save diagnostic plots to /plots/.

    Returns
    -------
    dict
        Diagnostic results including p-values and distribution moments.
    """
    print_section("Extracting Standardized Residuals")
    residuals = res.std_resid.dropna()
    cond_vol = res.conditional_volatility
    n = len(residuals)
    print(f"Sample size: {n}")
    print(f"Mean (≈ 0): {residuals.mean():.6f}")
    print(f"Variance (≈ 1): {residuals.var():.6f}")

    # ========================================================
    # 1. Ljung–Box test (autocorrelation)
    # ========================================================
    print_section("Ljung–Box Test for Autocorrelation")
    lb_test = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    print(lb_test)
    lb_p = lb_test['lb_pvalue'].values[-1]

    # ========================================================
    # 2. ARCH-LM test (remaining heteroskedasticity)
    # ========================================================
    print_section("ARCH-LM Test for Conditional Heteroskedasticity")
    arch_stat, arch_p, _, _ = het_arch(residuals)
    print(f"ARCH-LM statistic = {arch_stat:.4f}, p-value = {arch_p:.4f}")

    # ========================================================
    # 3. Jarque–Bera test (normality)
    # ========================================================
    print_section("Jarque–Bera Test for Normality")

    jb_result = jarque_bera(residuals)

    # Handle old/new SciPy return format
    if isinstance(jb_result, tuple) and len(jb_result) == 4:
        jb_stat, jb_p, skew, kurt = jb_result
    else:
        jb_stat, jb_p = jb_result
        skew = residuals.skew()
        kurt = residuals.kurtosis()

    print(f"JB statistic = {jb_stat:.4f}, p-value = {jb_p:.4f}")
    print(f"Skewness = {skew:.4f}, Kurtosis = {kurt:.4f}")

    # ========================================================
    # 4. Plots
    # ========================================================
    print_section("Generating Diagnostic Plots")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # (1) Standardized residuals
    axes[0, 0].plot(residuals, color="slateblue", lw=0.8)
    axes[0, 0].set_title("Standardized Residuals over Time")

    # (2) Conditional volatility
    axes[0, 1].plot(cond_vol, color="darkorange", lw=0.8)
    axes[0, 1].set_title("Conditional Volatility (EGARCH Output)")

    # (3) Histogram
    axes[1, 0].hist(residuals, bins=50, color="steelblue", alpha=0.7)
    axes[1, 0].set_title("Histogram of Standardized Residuals")

    # (4) QQ Plot
    try:
        df_t = res.distribution.df
    except AttributeError:
        df_t = 10  # fallback if t-dist not used
    probplot(residuals, dist="t", sparams=(df_t,), plot=axes[1, 1])
    axes[1, 1].set_title(f"QQ Plot vs Student-t (df={df_t:.1f})")

    # (5) ACF of residuals
    sm.graphics.tsa.plot_acf(residuals, lags=lags, ax=axes[2, 0])
    axes[2, 0].set_title("ACF of Standardized Residuals")

    # (6) ACF of squared residuals
    sm.graphics.tsa.plot_acf(residuals**2, lags=lags, ax=axes[2, 1])
    axes[2, 1].set_title("ACF of Squared Residuals")

    plt.tight_layout()
    if save_plots:
        save_path = os.path.join(PLOTS_DIR, "egarch_diagnostics.png")
        plt.savefig(save_path, dpi=300)
        print(f"📈 Diagnostic plots saved to {save_path}")
    plt.show()

    # ========================================================
    # Summary
    # ========================================================
    results = {
        "Ljung_Box_p": float(lb_p),
        "ARCH_LM_p": float(arch_p),
        "Jarque_Bera_p": float(jb_p),
        "Skewness": float(skew),
        "Kurtosis": float(kurt)
    }

    print_section("Summary of Diagnostic Results")
    for k, v in results.items():
        print(f"{k:<20} : {v:.6f}")

    return results


# ============================================================
# Optional CLI Execution
# ============================================================

if __name__ == "__main__":
    print_section("Loading Data and Training EGARCH Model")
    df = load_eth_data()
    res, df = train_egarch(df, p=EGARCH_P, o=EGARCH_O, q=EGARCH_Q)

    print_section("Running Full Residual Diagnostics")
    diagnostics = egarch_residual_diagnostics(res, lags=20, save_plots=True)
    print("\n✅ Diagnostics completed.")
