# === EGARCH model training and evaluation ===
from src.config import *                     
from src.data_loader import load_eth_data
from src.feature_engineering import build_features
from arch import arch_model
import pickle

def print_section(title: str):
    """Helper to print section headers nicely."""
    line = "═" * 60
    print(f"\n{line}\n{title.center(60)}\n{line}")

def train_egarch(df, p=1, o=1, q=1):
    """
    Fit a pure EGARCH model to ETH log returns.
    Saves trained model in a portable, platform-independent way.
    """
    print_section("Preparing Data and Features")
    df, mu, sigma, kurt = build_features(df)

    # --- Check for NaN values ---
    nan_counts = df.isna().sum()
    nan_columns = nan_counts[nan_counts > 0]

    if not nan_columns.empty:
        print_section("Missing Data Detected")
        for col, count in nan_columns.items():
            print(f" • {col:<20} : {count:>6} missing values")
        print(f"\nTotal missing values: {nan_counts.sum()}")

        user_input = input("\nRemove NaN values and continue? (y/n): ").strip().lower()
        if user_input == "y":
            print("🧹 Dropping NaN rows and continuing...")
            df = df.dropna()
        else:
            print("❌ Training aborted by user.")
            sys.exit(1)

    # --- Verify log_return ---
    if "log_return" not in df.columns:
        raise KeyError("❌ Column 'log_return' missing — did feature engineering fail?")

    if df["log_return"].isna().any():
        missing = df["log_return"].isna().sum()
        print(f"\n❌ {missing} NaN values found in 'log_return'. Training aborted.\n")
        sys.exit(1)

    # --- Summary of inputs ---
    returns = df["log_return"] * 100
    print_section("Model Input Summary")
    print(f" Observations : {len(returns):>10}")
    print(f" Mean          : {mu:>10.6f}")
    print(f" Std Dev       : {sigma:>10.6f}")
    print(f" Kurtosis      : {kurt:>10.3f}")

    # --- Define EGARCH Model ---
    print_section("Initializing EGARCH Model")
    model = arch_model(
        returns,
        mean="Constant",
        vol="EGARCH", p=p, o=o, q=q,
        dist="t"
    )

    # --- Train model ---
    print_section("Training EGARCH Model")
    res = model.fit(update_freq=10, disp="off")

    # --- Display results summary ---
    print_section("Model Training Summary")
    print(res.summary())

    # === Save trained model ===
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)   # ✅ create if missing

    model_path = os.path.join(data_dir, "egarch_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(res, f)

    print(f"💾 EGARCH model saved to: {os.path.relpath(model_path, PROJECT_ROOT)}")

    return res, df



