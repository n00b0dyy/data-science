# === EGARCH Out-of-Sample Evaluation ===
import os
import sys
import pickle
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


def load_out_of_sample():
    """Load the out-of-sample test data."""
    test_path = os.path.join(DATA_DIR, "out_of_sample_test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ Out-of-sample CSV not found: {test_path}")
    
    df = pd.read_csv(test_path)
    df["open_time"] = pd.to_datetime(df["open_time"])
    print(f"✅ Loaded out-of-sample data ({df.shape[0]} rows)")
    return df


def load_trained_model():
    """Load the saved EGARCH model results object."""
    model_path = os.path.join(project_root, "data", "egarch_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Trained model not found at: {model_path}")
    
    with open(model_path, "rb") as f:
        res = pickle.load(f)
    
    print(f"✅ Loaded trained EGARCH model from {model_path}")
    return res


def forecast_egarch_out_of_sample(res, df_test, horizon_steps=12):
    """
    Deterministic multi-step-ahead volatility forecasting for EGARCH(1,1).
    Uses recursive propagation of the conditional variance equation.
    """
    print_section(f"Deterministic {horizon_steps}-Step-Ahead EGARCH Forecasting")

    # 1️⃣ Prepare test data
    df_test, mu, sigma, kurt = build_features(df_test, rolling_window=ROLLING_WINDOW)
    test_returns = df_test["log_return"].dropna() * 100
    n_test = len(test_returns)

    # 2️⃣ Extract trained parameters
    params = res.params
    omega = params.get("omega", params[0])
    alpha = params.get("alpha[1]", params.get("alpha", 0))
    gamma = params.get("gamma[1]", params.get("gamma", 0))
    beta  = params.get("beta[1]", params.get("beta", 0))

    # 3️⃣ Last known volatility and residuals from training
    sigma_last = res.conditional_volatility.iloc[-1]
    z_last = res.std_resid.iloc[-1]

    Ez = np.sqrt(2 / np.pi)  # expected |z| for normal dist

    print(f" ω={omega:.6f}, α={alpha:.6f}, γ={gamma:.6f}, β={beta:.6f}")
    print(f" Last σ={sigma_last:.6f}, horizon={horizon_steps} steps\n")

    predicted_vol = []

    # 4️⃣ Rolling deterministic forecast loop
    for i in range(n_test):
        sigma_t = sigma_last
        z_t = z_last

        # recursive forecast for horizon h
        for _ in range(horizon_steps):
            log_sigma2 = omega + beta * np.log(sigma_t ** 2) + alpha * (abs(z_t) - Ez) + gamma * z_t
            sigma_t = np.sqrt(np.exp(log_sigma2))
            # for deterministic path: assume z_t = 0 going forward
            z_t = 0

        predicted_vol.append(sigma_t)

        # update last values for next iteration
        sigma_last = sigma_t
        z_last = (test_returns.iloc[i] - mu) / sigma_t

        # progress print
        if (i + 1) % max(1, n_test // 20) == 0 or (i + 1) == n_test:
            percent = (i + 1) / n_test * 100
            print(f" Progress: {i + 1:5d}/{n_test}  ({percent:5.1f}%)")

    # 5️⃣ Save to DataFrame
    df_test = df_test.iloc[-len(predicted_vol):].copy()
    df_test["predicted_vol"] = predicted_vol
    print(f"\n✅ Deterministic {horizon_steps}-step forecasts completed ({len(predicted_vol)} points).")

    return df_test

def three_day_volatility_comparison(res, df):
    """
    Compare EGARCH predicted volatility vs realized volatility every 3 days
    for out-of-sample evaluation.
    """
    print_section("3-Day Volatility Comparison Table (Out-of-Sample)")

    if "predicted_vol" in df.columns:
        cond_var = df["predicted_vol"]
    else:
        cond_var = pd.Series(res.conditional_volatility, index=df.index[-len(res.conditional_volatility):]) * 100

    realized = df["log_return"].rolling(window=min(len(df), ROLLING_WINDOW)).std() * 100

    n = min(len(df["open_time"]), len(cond_var), len(realized))
    tmp = pd.DataFrame({
        "open_time": df["open_time"].iloc[-n:],
        "predicted_vol": cond_var.iloc[-n:],
        "realized_vol": realized.iloc[-n:]
    }).dropna()

    # --- 3-day aggregation ---
    three_day = (
        tmp.set_index("open_time")
        .resample("3D")
        .agg({"predicted_vol": "mean", "realized_vol": "mean"})
        .dropna()
    )

    three_day["diff_abs"] = (three_day["predicted_vol"] - three_day["realized_vol"]).abs()
    three_day["corr"] = (
        tmp.set_index("open_time")
        .resample("3D")[["predicted_vol", "realized_vol"]]
        .corr()
        .iloc[0::2, -1]
        .to_numpy()
    )

    # --- Print nicely ---
    print(f"{'Period':<14}{'Predicted':>12}{'Realized':>12}{'AbsDiff':>12}{'Corr':>10}")
    print("-" * 60)
    for i, row in three_day.iterrows():
        print(
            f"{i.strftime('%Y-%m-%d'): <14}"
            f"{row['predicted_vol']: >12.3f}"
            f"{row['realized_vol']: >12.3f}"
            f"{row['diff_abs']: >12.3f}"
            f"{row['corr']: >10.3f}"
        )

    mean_corr = three_day["corr"].mean()
    mean_diff = three_day["diff_abs"].mean()

    print("\nAverages across 3-day periods:")
    print(f" Mean correlation   : {mean_corr:.3f}")
    print(f" Mean abs. diff (%) : {mean_diff:.3f}")

    return three_day


def evaluate_egarch_out_of_sample(horizon_steps=12):
    """Evaluate EGARCH model performance on unseen data."""
    print_section("Loading Model and Out-of-Sample Data")
    res = load_trained_model()
    df = load_out_of_sample()

    print_section(f"Starting Forecasting (horizon={horizon_steps} steps)")
    df = forecast_egarch_out_of_sample(res, df, horizon_steps=horizon_steps)

    print_section("Plotting EGARCH vs Realized Volatility (Out-of-Sample)")
    plot_conditional_variance(
        res,
        df,
        save_path=os.path.join(PLOTS_DIR, f"egarch_out_of_sample_{horizon_steps}step.png")
    )

    print_section("3-Day Volatility Comparison (Out-of-Sample)")
    three_day = three_day_volatility_comparison(res, df)


    print_section("Evaluation Completed")
    print(f"✅ Out-of-sample evaluation finished successfully for horizon={horizon_steps}.\n")
    return three_day

# === CLI Execution ===
if __name__ == "__main__":
    evaluate_egarch_out_of_sample()
