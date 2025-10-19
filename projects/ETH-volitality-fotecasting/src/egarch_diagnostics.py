# ===========================================
# src/egarch_diagnostics.py
# -------------------------------------------
# Diagnostyka modelu EGARCH:
# - wczytuje model i skalę
# - analizuje reszty (standaryzowane lub przeskalowane)
# - testy: Ljung–Box, Jarque–Bera, ARCH–LM
# - prosty histogram i ACF
# ===========================================

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf
from src.config import LOGS_DIR


# ===========================================
def load_model(filename: str):
    path = LOGS_DIR / filename
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "model" in obj:
        model, scale = obj["model"], obj.get("scale", 1.0)
    else:
        model, scale = obj, 1.0
    print(f"✅ Wczytano model z pliku: {path.name}  (scale = {scale:.3e})")
    return model, scale


# ===========================================
def basic_diagnostics(fitted_model, scale=1.0):
    resid = fitted_model.std_resid * scale
    print("\n📊 Statystyki reszt:")
    print(f"Średnia:   {np.mean(resid):.6f}")
    print(f"Wariancja: {np.var(resid):.6f}")
    print(f"Skośność:  {stats.skew(resid):.4f}")
    print(f"Kurtosis:  {stats.kurtosis(resid):.4f}")

    lb_p = acorr_ljungbox(resid, lags=[10], return_df=True)['lb_pvalue'].iloc[0]
    jb_stat, jb_p = stats.jarque_bera(resid)
    lm_stat, lm_p, _, _ = het_arch(resid)

    print("\n🔍 Testy diagnostyczne:")
    print(f"Ljung–Box (lag=10): p = {lb_p:.4f}")
    print(f"Jarque–Bera (normalność): p = {jb_p:.4f}")
    print(f"ARCH–LM (heteroskedastyczność): p = {lm_p:.4f}")
    return resid


# ===========================================
def plot_residuals(resid):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(resid, bins=40, color="gray", edgecolor="black", alpha=0.7)
    axes[0].set_title("Rozkład reszt")
    plot_acf(resid, lags=30, ax=axes[1])
    axes[1].set_title("Autokorelacja (ACF)")
    plt.tight_layout()
    plt.show()


# ===========================================
if __name__ == "__main__":
    model, scale = load_model("egarch_ETH_5m_p1_o2_q2_t.pkl")
    resid = basic_diagnostics(model, scale)
    plot_residuals(resid)
