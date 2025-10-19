
# ===========================================
# src/egarch_model.py
# -------------------------------------------
# Trening modelu EGARCH na wystandaryzowanych
# log-returnach (standaryzacja przez std(train)).
# ===========================================

import pandas as pd
import pickle
import numpy as np
from arch import arch_model
from src.config import LOGS_DIR


def fit_egarch(
    returns: pd.Series,
    p: int = 1,
    o: int = 1,
    q: int = 2,
    dist: str = "t",
    mean: str = "constant"
):
    """
    Trenuje model EGARCH(p, o, q) na wystandaryzowanych log-zwrotach.
    Uwaga:
    - skalowanie przez std(train) poprawia stabilność numeryczną
    - ten sam współczynnik należy stosować przy symulacji live
    """

    # --- Skalowanie (zabezpieczenie przed zbyt małą wariancją) ---
    scale = returns.std(ddof=0)
    if scale < 1e-6:
        raise ValueError("⚠️ Skalowanie niestabilne: std zbyt małe, dane prawdopodobnie błędne.")
    returns_std = returns / scale

    # --- Definicja i trening modelu ---
    model = arch_model(
        returns_std,
        vol="EGARCH",
        p=p, o=o, q=q,
        dist=dist,
        mean=mean
    )

    print(f"🔧 Trenuję model EGARCH(p={p}, o={o}, q={q}, dist={dist})...")
    fitted = model.fit(disp="off", update_freq=0)
    print("✅ Trening zakończony.")
    print("\nParametry modelu:")
    print(fitted.params)

    return fitted, scale


def save_model(fitted_model, scale: float, filename: str = "egarch_model.pkl"):
    """
    Zapisuje wytrenowany model EGARCH oraz współczynnik skalowania do pliku .pkl.
    Dzięki temu evaluation.py nie musi ponownie liczyć std(train).
    """
    path = LOGS_DIR / filename
    payload = {"model": fitted_model, "scale": scale}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"💾 Zapisano model i scale do pliku: {path}")
    return path


# ===========================================
# Szybki test standalone (tylko train)
# ===========================================
if __name__ == "__main__":
    from src.data_loader import load_train_test_data

    # wczytujemy wyłącznie dane treningowe, bez testu
    train_df = load_train_test_data(load_test=False)
    model, scale = fit_egarch(train_df["log_return"])
    save_model(model, scale, "egarch_ETH_5m.pkl")
