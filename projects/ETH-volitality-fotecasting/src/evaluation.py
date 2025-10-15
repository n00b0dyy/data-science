# === EGARCH Out-of-Sample Evaluation ===
import os
import sys
import pickle
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

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


def forecast_egarch_out_of_sample(res, df):
    """
    Generate out-of-sample conditional variance forecasts
    using parameters of a pre-trained EGARCH model.
    """
    print_section("Forecasting Conditional Variance on Out-of-Sample Data")

    # build features for test set
    df, mu, sigma, kurt = build_features(df, rolling_window=ROLLING_WINDOW)

    # create model with same specification
    model = arch_model(
        df["log_return"] * 100,
        mean="Constant",
        vol="EGARCH",
        p=EGARCH_P,
        o=EGARCH_O,
        q=EGARCH_Q,
        dist=DISTRIBUTION,
    )

    # generate forecast using previously estimated parameters
    model_params = res.params
    forecast = model.forecast(params=model_params, reindex=False, horizon=1)

    # Extract last column of variance forecasts
    df["predicted_vol"] = np.sqrt(forecast.variance.values[-len(df):, 0])

    print(f"✅ Forecasting completed — predicted_vol column added ({len(df)} points).")
    return df


def evaluate_egarch_out_of_sample():
    """Evaluate EGARCH model performance on unseen data."""
    print_section("Loading Model and Out-of-Sample Data")
    res = load_trained_model()
    df = load_out_of_sample()

    df = forecast_egarch_out_of_sample(res, df)

    print_section("Plotting EGARCH vs Realized Volatility (Out-of-Sample)")
    plot_conditional_variance(
        res,
        df,
        save_path=os.path.join(PLOTS_DIR, "egarch_out_of_sample_volatility.png")
    )

    print_section("Weekly Volatility Comparison (Out-of-Sample)")
    weekly = weekly_volatility_comparison(res, df)

    print_section("Evaluation Completed")
    print("✅ Out-of-sample evaluation finished successfully.")
    return weekly


# === CLI Execution ===
if __name__ == "__main__":
    evaluate_egarch_out_of_sample()
