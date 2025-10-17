# ===========================================
# src/egarch_diagnostics.py
# -------------------------------------------
# Prosty moduł diagnostyczny dla modelu EGARCH:
# - wczytuje zapisany model
# - analizuje reszty standaryzowane
# - wykonuje podstawowe testy statystyczne
# - opcjonalnie tworzy proste wykresy
# ===========================================

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from src.config import LOGS_DIR


# ===========================================
# Funkcja: load_model
# -------------------------------------------
# Wczytuje zapisany model EGARCH z pliku .pkl
# (obsługuje zarówno czysty model, jak i słownik {"model":..., "scale":...})
# ===========================================
def load_model(filename: str):
    path = LOGS_DIR / filename
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # jeśli zapisano jako {"model": ..., "scale": ...}, wyciągamy sam model
    model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    print(f"✅ Wczytano model z pliku: {path}")
    return model


# ===========================================
# Funkcja: basic_diagnostics
# -------------------------------------------
# Wyświetla podstawowe statystyki reszt:
# - średnia, wariancja
# - test Ljunga–Boxa (autokorelacja)
# - test Jarque–Bera (normalność)
# ===========================================
def basic_diagnostics(fitted_model):
    resid = fitted_model.std_resid  # standaryzowane reszty

    print("\n📊 Podstawowe statystyki reszt:")
    print(f"Średnia: {np.mean(resid):.6f}")
    print(f"Wariancja: {np.var(resid):.6f}")
    print(f"Skośność: {stats.skew(resid):.4f}")
    print(f"Kurtosis: {stats.kurtosis(resid):.4f}")

    # --- Testy statystyczne ---
    lb_p = acorr_ljungbox(resid, lags=[10], return_df=True)
    jb_stat, jb_p = stats.jarque_bera(resid)

    print("\n🔍 Testy diagnostyczne:")
    print(f"Ljung–Box (lag=10): p-value = {lb_p['lb_pvalue'].iloc[0]:.4f}")
    print(f"Jarque–Bera (normalność): p-value = {jb_p:.4f}")

    return resid


# ===========================================
# Funkcja: plot_residuals
# -------------------------------------------
# Rysuje histogram i ACF reszt
# ===========================================
def plot_residuals(resid):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram reszt
    axes[0].hist(resid, bins=40, color="gray", edgecolor="black", alpha=0.7)
    axes[0].set_title("Rozkład standaryzowanych reszt")

    # Autokorelacja
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(resid, lags=30, ax=axes[1])
    axes[1].set_title("ACF reszt (do 30 lagów)")

    plt.tight_layout()
    plt.show()


# ===========================================
# Szybki test standalone
# ===========================================
if __name__ == "__main__":
    model = load_model("egarch_ETH_5m_p1_o1_q1_t.pkl")
    resid = basic_diagnostics(model)
    plot_residuals(resid)
