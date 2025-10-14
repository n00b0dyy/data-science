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
from config import *                   # global paths and constants
from data_loader import load_eth_data  # load CSV data
from egarch_model import train_egarch  # model training pipeline

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
    jb_stat, jb_p, skew, kurt = jarque_bera(residuals)
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
