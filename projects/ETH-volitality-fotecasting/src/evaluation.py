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
from config import *
from feature_engineering import build_features
from egarch_diagnostics import plot_conditional_variance, weekly_volatility_comparison


def print_section(title: str):
    """Pretty section header for CLI output."""
    line = "═" * 70
    print(f"\n{line}\n{title.center(70)}\n{line}\n")