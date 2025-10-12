import os
import sys
from src.config import * 

# === Path setup ===
project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# === Local imports ===
from src.data_loader import load_eth_data
from src.egarch_model import train_egarch, plot_conditional_variance

def main():
    print("🚀 Starting EGARCH model training pipeline...\n")

    # 1️⃣ Load ETH/USDT data
    df = load_eth_data()

    # 2️⃣ Train model
    res, df = train_egarch(df, p=1, o=1, q=1)

    # 3️⃣ Plot and save results
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_conditional_variance(
        res,
        df,
        save_path=os.path.join(plots_dir, "egarch_volatility_comparison.png")
    )

    # 4️⃣ Save model summary
    summary_path = os.path.join(project_root, "data", "egarch_summary.txt")
    with open(summary_path, "w") as f:
        f.write(str(res.summary()))
    print(f"\n✅ Model summary saved to {summary_path}")
    print("✅ Conditional variance plot saved in /plots/\n")
    print("🎯 EGARCH pipeline completed successfully.")

if __name__ == "__main__":
    main()

