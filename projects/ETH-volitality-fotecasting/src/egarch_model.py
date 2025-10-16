# === EGARCH model training and evaluation ===
from src.config import *                     
from src.data_loader import load_eth_data
from src.feature_engineering import build_features
from arch import arch_model
import pickle
import os
import sys
import json

def print_section(title: str):
    """Helper to print section headers nicely."""
    line = "═" * 60
    print(f"\n{line}\n{title.center(60)}\n{line}")

def train_egarch(train_df, p=1, o=1, q=1):
    """
    Fit an EGARCH(p,o,q) model on ETH log returns using only the training dataset.
    Ensures no data leakage, saves both the trained model and training statistics.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset containing 'open_time', 'close', 'high', 'low', 'volume'.
    p, o, q : int
        EGARCH parameters for lag orders.
    """

    print_section("Preparing Data and Features")


    train_df, mu, sigma, kurt = build_features(train_df, compute_stats=True)


    nan_counts = train_df.isna().sum()
    nan_columns = nan_counts[nan_counts > 0]
    if not nan_columns.empty:
        print_section("Missing Data Detected")
        for col, count in nan_columns.items():
            print(f" • {col:<20} : {count:>6} missing values")
        print(f"\n🧹 Automatically dropping NaN rows to ensure data integrity.")
        train_df = train_df.dropna().reset_index(drop=True)

    # --- Verify log_return existence and validity ---
    if "log_return" not in train_df.columns:
        raise KeyError("❌ Column 'log_return' missing — did feature engineering fail?")
    if train_df["log_return"].isna().any():
        missing = train_df["log_return"].isna().sum()
        raise ValueError(f"❌ {missing} NaN values found in 'log_return'. Training aborted.")

    # --- Summary of inputs ---
    returns = train_df["log_return"] * 100
    print_section("Model Input Summary")
    print(f" Observations : {len(returns):>10}")
    print(f" Mean          : {mu:>10.6f}")
    print(f" Std Dev       : {sigma:>10.6f}")
    print(f" Kurtosis      : {kurt:>10.3f}")

    print(f"\nEGARCH configuration: p={p}, o={o}, q={q}, dist='{DISTRIBUTION}'\n")

    # --- Define EGARCH Model ---
    print_section("Initializing EGARCH Model")
    model = arch_model(
        returns,
        mean="Constant",
        vol="EGARCH",
        p=p, o=o, q=q,
        dist=DISTRIBUTION
    )

    # --- Train model ---
    print_section("Training EGARCH Model")
    res = model.fit(update_freq=10, disp="off")

    # --- Display results summary ---
    print_section("Model Training Summary")
    print(res.summary())

    # === Save trained model and stats ===
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    model_path = os.path.join(data_dir, "egarch_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(res, f)


    stats_path = os.path.join(data_dir, "train_stats.json")
    stats = {"mu": float(mu), "sigma": float(sigma), "kurtosis": float(kurt)}
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\n💾 EGARCH model saved to: {os.path.relpath(model_path, PROJECT_ROOT)}")
    print(f"💾 Training stats saved to: {os.path.relpath(stats_path, PROJECT_ROOT)}")
    print("\n🎯 Training completed successfully — model ready for out-of-sample forecasting.")

   
    return res, train_df, stats
