# ===========================================
# src/simulation_live_egarch.py
# -------------------------------------------
# EGARCH "LIVE" – forward simulation (no look-ahead)
# ===========================================

import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pathlib import Path
from datetime import datetime
import warnings

plt.style.use("seaborn-v0_8-darkgrid")

# ===========================================================
# 🔧 Ścieżki projektu i LOGS_DIR
# ===========================================================
ROOT = Path.cwd()
while ROOT.name != "ETH-volitality-fotecasting" and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from src.config import LOGS_DIR
except ModuleNotFoundError:
    LOGS_DIR = ROOT / "logs"
    LOGS_DIR.mkdir(exist_ok=True)

# ===========================================================
# 🔧 Logger
# ===========================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOGS_DIR / f"simulation_console_{timestamp}.log"

class TeeLogger:
    def __init__(self, filename, silent=True):
        self.log = open(filename, "a", encoding="utf-8")
        self.silent = silent
    def write(self, message):
        if not self.silent:
            sys.__stdout__.write(message)
        self.log.write(message)
    def flush(self):
        self.log.flush()

sys.stdout = TeeLogger(log_file, silent=True)
print(f"📜 Logi zapisywane w: {log_file}\n")

# ===========================================================
# 🔁 Streamer świeczek
# ===========================================================
def stream_candles(csv_path, chunk_size=1000, n_obs=None):
    count = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, parse_dates=["open_time"]):
        for _, row in chunk.iterrows():
            yield row
            count += 1
            if n_obs and count >= n_obs:
                return

# ===========================================================
# 🧩 Główna symulacja EGARCH (forward recursion)
# ===========================================================
def simulate_egarch_live_forward(
    csv_path="candles/3000_sample.csv",
    model_filename="egarch_ETH_5m_p1_o1_q2_t.pkl",
    n_obs=1000,
    delay=0.25,
    smoothing=0.3
):
    """
    Pseudo-live symulacja EGARCH:
    - Model trenowany na zeskalowanych log-returnach (r/std(train))
    - Brak look-ahead bias (σ̂_{t+1} obliczane z danych do t)
    - Smoothing działa wyłącznie w przód, na tej samej skali
    - Wyniki korelują σ̂_{t+1} z |r_{t+1}|, jak w realnym rynku
    """

    # --- wczytanie modelu ---
    payload = pickle.load(open(LOGS_DIR / model_filename, "rb"))
    fitted, base_scale = payload["model"], payload["scale"]
    params = fitted.params

    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]
    gamma = params.get("gamma[1]", 0.0)
    mu = params.get("mu", 0.0)

    print(f"📂 Załadowano model: {model_filename}")
    print(f"🔧 Parametry: ω={omega:.4f}, α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}, μ={mu:.4f}")
    print(f"⚖️ Skala treningu (base_scale) = {base_scale:.3e}")

    # --- inicjalizacja stanu ---
    sigma_prev = np.std(fitted.model._y)  # punkt startowy w tej samej skali co fit
    eps_prev = 0.0
    prev_close = None

    timestamps, sigmas_pred, returns_real = [], [], []

    # --- pętla po świeczkach ---
    for i, row in enumerate(stream_candles(csv_path, n_obs=n_obs)):
        close = row["close"]

        if prev_close is None:
            prev_close = close
            continue

        # --- bieżący zwrot (standaryzowany) ---
        log_return = np.log(close / prev_close)
        r_std = log_return / base_scale
        prev_close = close

        # --- EGARCH recursion (forward) ---
        log_sigma2_next = (
            omega
            + beta * np.log(sigma_prev ** 2)
            + alpha * (abs(eps_prev) - np.sqrt(2 / np.pi))
            + gamma * eps_prev
        )
        sigma_next = np.exp(0.5 * log_sigma2_next)

        # --- smoothing (na tej samej skali) ---
        if sigmas_pred:
            sigma_next = smoothing * sigmas_pred[-1] + (1 - smoothing) * sigma_next

        # --- zapis forecastu (σ̂_{t+1}) ---
        sigmas_pred.append(sigma_next)
        returns_real.append(abs(r_std))  # proxy volatility
        timestamps.append(row["open_time"])

        # --- update stanu (dla kolejnej iteracji) ---
        eps_prev = r_std / sigma_prev
        sigma_prev = sigma_next

        # --- rysowanie co jakiś czas ---
        if i % 50 == 0 and i > 0:
            clear_output(wait=True)
            plt.figure(figsize=(12, 5))
            plt.plot(returns_real[-300:], color="black", alpha=0.6, label="|r_t| (real)")
            plt.plot(sigmas_pred[-300:], color="red", label="σ̂_{t+1} (forecast)")
            plt.legend()
            plt.title(f"EGARCH Live Forecast – świeczka {row['open_time']}")
            plt.tight_layout()
            plt.show()

        time.sleep(delay)

    # --- align forecast z realnym r_{t+1} ---
    df_result = pd.DataFrame({
        "open_time": timestamps[1:],
        "sigma_pred": sigmas_pred[:-1],
        "r_abs": returns_real[1:]
    })

    corr = df_result["sigma_pred"].corr(df_result["r_abs"])
    print(f"✅ Symulacja zakończona ({len(df_result)} świeczek).")
    print(f"📈 Korelacja σ̂(t+1) vs |r(t+1)| = {corr:.4f}")

    return df_result
