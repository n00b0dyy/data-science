# === Global Configuration for ETH Volatility Forecasting ===
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, kurtosis, shapiro, normaltest, anderson

# --- Project Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))              # src/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))       # project root

DATA_DIR = os.path.join(PROJECT_ROOT, "candles")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create folders if missing
for path in [PLOTS_DIR, LOG_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

# --- Data Paths ---
DATA_PATH = os.path.join(DATA_DIR, "train_sample.csv")

# --- Rolling Configuration ---
ROLLING_WINDOW = 288  # 1 day (can adjust)

# --- EGARCH Model Defaults ---
EGARCH_P = 1
EGARCH_O = 1
EGARCH_Q = 1
DISTRIBUTION = "t"

# --- Global Print Info ---
print(f"[config] Project root: {PROJECT_ROOT}")
print(f"[config] Data path: {DATA_PATH}")
print(f"[config] Plots directory: {PLOTS_DIR}")
print(f"[config] Logs directory: {LOG_DIR}")
print(f"[config] Default EGARCH params -> p={EGARCH_P}, o={EGARCH_O}, q={EGARCH_Q}, dist={DISTRIBUTION}")
