# === EGARCH Out-of-Sample Evaluation ===
import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import warnings
from arch.utility.exceptions import DataScaleWarning

# === Suppress ARCH scaling warnings ===
warnings.filterwarnings("ignore", category=DataScaleWarning)

# === Path setup ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# === Local imports ===
from src.config import *
from src.feature_engineering import build_features
from src.egarch_diagnostics import plot_conditional_variance, weekly_volatility_comparison


def print_section(title: str):
    """Pretty section header for CLI output."""
    line = "═" * 70
    print(f"\n{line}\n{title.center(70)}\n{line}\n")


# =====================================================================================
# === Data and Model Loaders
# =====================================================================================

def load_out_of_sample():
    """Load the out-of-sample test data."""
    test_path = os.path.join(PROJECT_ROOT, "candles", "out_of_sample_test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ Out-of-sample CSV not found: {test_path}")
    
    df = pd.read_csv(test_path)
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"✅ Loaded out-of-sample data ({df.shape[0]} rows)")
    return df


def load_trained_model():
    """Load the saved EGARCH model results object."""
    model_path = os.path.join(PROJECT_ROOT, "data", "egarch_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Trained model not found at: {model_path}")
    
    with open(model_path, "rb") as f:
        res = pickle.load(f)
    
    print(f"✅ Loaded trained EGARCH model from {model_path}")
    return res


def load_train_stats():
    """Load training statistics (μ, σ, kurtosis) computed during training."""
    stats_path = os.path.join(PROJECT_ROOT, "data", "train_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"❌ Training statistics file not found: {stats_path}")
    
    with open(stats_path, "r") as f:
        stats = json.load(f)
    print(f"✅ Loaded training statistics from {stats_path}")
    return stats


# =====================================================================================
# === Forecasting Logic
# =====================================================================================

def forecast_egarch_out_of_sample(res, df_test, train_stats, horizon_steps=12):
    """
    Deterministic multi-step-ahead volatility forecasting for EGARCH(1,1).
    Uses recursive propagation of the conditional variance equation.
    Ensures no look-ahead bias by using only training-derived statistics.
    """
    print_section(f"Deterministic {horizon_steps}-Step-Ahead EGARCH Forecasting")

    # 🧱 Build test features WITHOUT recomputing stats (log_return already scaled ×100)
    df_test, _, _, _ = build_features(df_test, rolling_window=ROLLING_WINDOW, compute_stats=False)

    df_test = df_test.sort_values("open_time").reset_index(drop=True)

    # 🧹 Trim incomplete rolling window region if necessary
    initial_len = len(df_test)
    if initial_len > ROLLING_WINDOW:
        df_test = df_test.iloc[ROLLING_WINDOW:].reset_index(drop=True)
        removed = initial_len - len(df_test)
        print(f"🧹 Removed {removed} initial rows due to incomplete rolling windows.")
        print(f"📊 Remaining samples for evaluation: {len(df_test)} rows.\n")
    else:
        print(f"⚠️ Not enough data for a full rolling window (len={initial_len}). Proceeding anyway.\n")

    # --- Use log_return directly from features (already ×100) ---
    test_returns = df_test["log_return"].dropna().to_numpy()
    n_test = len(test_returns)

    # --- Use training statistics ---
    mu = train_stats["mu"]
    sigma = train_stats["sigma"]

    # --- Extract trained EGARCH parameters ---
    params = res.params
    omega = params.get("omega", params[0])
    alpha = params.get("alpha[1]", params.get("alpha", 0))
    gamma = params.get("gamma[1]", params.get("gamma", 0))
    beta  = params.get("beta[1]", params.get("beta", 0))

    # --- Last known volatility and residuals from training ---
    sigma_last = res.conditional_volatility.iloc[-1]
    z_last = res.std_resid.iloc[-1]

    Ez = np.sqrt(2 / np.pi)  # expected |z| for normal dist

    print(f" ω={omega:.6f}, α={alpha:.6f}, γ={gamma:.6f}, β={beta:.6f}")
    print(f" Last σ={sigma_last:.6f}, horizon={horizon_steps} steps\n")

    predicted_vol = []

    # --- Rolling deterministic forecast loop ---
    for i in range(n_test):
        sigma_t = sigma_last
        z_t = z_last

        # recursive forecast for horizon h
        for _ in range(horizon_steps):
            log_sigma2 = omega + beta * np.log(sigma_t ** 2) + alpha * (abs(z_t) - Ez) + gamma * z_t
            sigma_t = np.sqrt(np.exp(log_sigma2))
            z_t = 0  # deterministic path

        predicted_vol.append(sigma_t)

        # update last values for next iteration
        sigma_last = sigma_t
        z_last = (test_returns[i] - mu) / sigma_t

        if (i + 1) % max(1, n_test // 20) == 0 or (i + 1) == n_test:
            percent = (i + 1) / n_test * 100
            print(f" Progress: {i + 1:5d}/{n_test}  ({percent:5.1f}%)")

    df_test = df_test.iloc[-len(predicted_vol):].copy()
    df_test["predicted_vol"] = predicted_vol

    print(f"\n✅ Deterministic {horizon_steps}-step forecasts completed.")
    print(f"📈 Total evaluated samples: {len(df_test)} rows.\n")

    return df_test


# =====================================================================================
# === Evaluation and Reporting
# =====================================================================================

def hourly_volatility_evaluation(res, df):
    """
    Compare EGARCH predicted volatility vs realized volatility every 1 hour.
    """
    print_section("Hourly Volatility Evaluation (Out-of-Sample)")

    cond_var = df["predicted_vol"]
    realized = df["log_return"].rolling(window=min(len(df), ROLLING_WINDOW)).std()

    n = min(len(df["open_time"]), len(cond_var), len(realized))
    tmp = pd.DataFrame({
        "open_time": df["open_time"].iloc[-n:],
        "predicted_vol": cond_var.iloc[-n:],
        "realized_vol": realized.iloc[-n:]
    }).dropna()

    hourly = (
        tmp.set_index("open_time")
        .resample("1H")
        .agg({"predicted_vol": "mean", "realized_vol": "mean"})
        .dropna()
    )

    hourly["bias"] = hourly["predicted_vol"] - hourly["realized_vol"]
    hourly["abs_diff"] = hourly["bias"].abs()
    hourly["corr"] = (
        tmp.set_index("open_time")
        .resample("1H")[["predicted_vol", "realized_vol"]]
        .corr()
        .iloc[0::2, -1]
        .to_numpy()
    )

    print(f"{'Hour':<18}{'Bias':>10}{'AbsDiff':>12}{'Corr':>10}")
    print("-" * 55)
    for i, row in hourly.iterrows():
        print(
            f"{i.strftime('%Y-%m-%d %H:%M'): <18}"
            f"{row['bias']: >10.3f}"
            f"{row['abs_diff']: >12.3f}"
            f"{row['corr']: >10.3f}"
        )

    mean_bias = hourly["bias"].mean()
    mean_absdiff = hourly["abs_diff"].mean()
    mean_corr = hourly["corr"].mean()

    print("\nAverages across 1-hour windows:")
    print(f" Mean bias          : {mean_bias:.3f}")
    print(f" Mean abs. diff (%) : {mean_absdiff:.3f}")
    print(f" Mean correlation   : {mean_corr:.3f}")

    return hourly


def evaluate_egarch_out_of_sample(horizon_steps=12):
    """Evaluate EGARCH model performance on unseen data."""
    print_section("Loading Model, Training Stats and Out-of-Sample Data")

    res = load_trained_model()
    train_stats = load_train_stats()
    df_test = load_out_of_sample()

    print_section(f"Starting Forecasting (horizon={horizon_steps} steps)")
    df_test = forecast_egarch_out_of_sample(res, df_test, train_stats, horizon_steps=horizon_steps)

    print_section("Plotting EGARCH vs Realized Volatility (Out-of-Sample)")
    plot_conditional_variance(
        res,
        df_test,
        save_path=os.path.join(PLOTS_DIR, f"egarch_out_of_sample_{horizon_steps}step.png")
    )

    print_section("Hourly Volatility Evaluation (Out-of-Sample)")
    hourly = hourly_volatility_evaluation(res, df_test)

    print_section("Evaluation Completed")
    print(f"✅ Out-of-sample evaluation finished successfully for horizon={horizon_steps}.\n")
    return hourly


# === CLI Execution ===
if __name__ == "__main__":
    evaluate_egarch_out_of_sample()
