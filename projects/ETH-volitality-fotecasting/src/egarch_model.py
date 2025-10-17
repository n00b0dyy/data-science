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
    q: int = 1,
    dist: str = "normal",
    mean: str = "Constant"
):
    """
    Trenuje model EGARCH(p, o, q) na standaryzowanych log-zwrotach.
    Uwaga:
    - skalowanie przez std(train) poprawia stabilność numeryczną
    - przy prognozach out-of-sample należy stosować to samo skalowanie
      (ten sam współczynnik z treningu, bez użycia test.std()).
    """

    # --- Standaryzacja (tylko na train) ---
    scale = returns.std()
    returns_std = returns / scale

    # --- Definicja i trening modelu ---
    model = arch_model(returns_std, vol="EGARCH", p=p, o=o, q=q, dist=dist, mean=mean)
    print("🔧 Trenuję model EGARCH...")
    fitted = model.fit(disp="off")
    print("✅ Trening zakończony.")

    # --- Parametry ---
    print("\nParametry modelu:")
    print(fitted.params)

    # --- Zwracamy dopasowany model + współczynnik skalowania ---
    return fitted, scale




def save_model(fitted_model, filename: str = "egarch_model.pkl"):
    path = LOGS_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(fitted_model, f)
    print(f"💾 Zapisano model do pliku: {path}")
    return path


# ===========================================
# Szybki test standalone
# ===========================================
if __name__ == "__main__":
    from src.data_loader import load_train_test_data

    train_df, test_df = load_train_test_data()
    model, scale = fit_egarch(train_df["log_return"])
    save_model(model, "egarch_ETH_5m.pkl")
