# ===========================================
# src/evaluation.py
# -------------------------------------------
# Ewaluacja modelu EGARCH (out-of-sample)
# - korelacja prognozowanej i rzeczywistej wariancji
# - rolling forecast wykres zmienności
# - dodatkowe miary: RMSE, MAE, QLIKE, Mincer–Zarnowitz
# ===========================================

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import statsmodels.api as sm
from src.data_loader import load_train_test_data
from src.config import LOGS_DIR


def evaluate_egarch(model_filename="egarch_ETH_5m_p1_o1_q1_t.pkl", n_obs: int = 500):
    print(f"📂 Wczytuję model z {model_filename}...")
    payload = pickle.load(open(LOGS_DIR / model_filename, "rb"))
    fitted = payload["model"]
    scale = payload["scale"]

    # --- Wczytanie danych testowych ---
    print("📈 Wczytuję dane testowe...")
    train_df, test_df = load_train_test_data(load_test=True)

    if len(test_df) > n_obs:
        test_df = test_df.head(n_obs).reset_index(drop=True)
        print(f"📏 Używam tylko pierwszych {n_obs} świeczek z testu.")
    else:
        print(f"⚠️ Test ma mniej niż {n_obs} świeczek ({len(test_df)}). Używam całości.")

    # --- Standaryzacja ---
    test_returns_std = test_df["log_return"] / scale

    # --- Rolling forecast ---
    print("🔮 Liczę rolling forecast wariancji...")
    model = fitted.model
    params = fitted.params

    forecasts = []
    vol = fitted.model.volatility
    dist_name = "t"

    for i in range(len(test_returns_std)):
        subseries = np.concatenate([fitted.model._y, test_returns_std.values[:i]])
        temp_model = arch_model(
            subseries,
            vol="EGARCH",
            p=vol.p,
            o=vol.o,
            q=vol.q,
            dist=dist_name,
            mean="Constant",
        )
        res = temp_model.fit(
            disp="off", last_obs=len(subseries) - 1, update_freq=0, starting_values=params
        )
        fcast = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
        forecasts.append(fcast)

    # --- Dopasowanie długości ---
    test_df = test_df.iloc[:len(forecasts)].copy()
    test_df["var_pred"] = np.array(forecasts)
    test_df["var_real"] = (test_df["log_return"] / scale) ** 2

    # --- Wykres ---
    plt.figure(figsize=(12, 5))
    plt.plot(test_df["open_time"], np.sqrt(test_df["var_pred"]), label="Prognozowana σ_t", color="red")
    plt.plot(test_df["open_time"], np.sqrt(test_df["var_real"]), label="Rzeczywista σ_t", color="black", alpha=0.5)
    plt.title(f"Rolling forecast – zmienność warunkowa (ostatnie {n_obs} świeczek)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("✅ Ewaluacja zakończona.")

    # --- Miary błędu prognoz ---
    pred = test_df["var_pred"].values
    real = test_df["var_real"].values

    rmse = np.sqrt(np.mean((pred - real) ** 2))
    mae = np.mean(np.abs(pred - real))
    qlike = np.mean(np.log(pred) + real / pred)  # Patton (2011)
    corr = np.corrcoef(np.sqrt(pred), np.sqrt(real))[0, 1]

    print("\n📊 Miary dokładności prognoz:")
    print(f"   🔹 RMSE  : {rmse:.6e}")
    print(f"   🔹 MAE   : {mae:.6e}")
    print(f"   🔹 QLIKE : {qlike:.6e}")
    print(f"   🔹 Korelacja σ_pred vs σ_real: {corr:.4f}")

    # --- Mincer–Zarnowitz regression ---
    print("\n🧩 Mincer–Zarnowitz test (kalibracja wariancji):")
    X = sm.add_constant(pred)
    model_mz = sm.OLS(real, X).fit()
    alpha, beta = model_mz.params
    r2 = model_mz.rsquared

    # Test hipotezy H0: β = 1
    beta_se = model_mz.bse[1]               # standard error dla beta
    t_stat = (beta - 1) / beta_se           # statystyka t
    df = model_mz.df_resid
    from scipy import stats
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    print(f"   🔸 α (const) = {alpha:.4f}")
    print(f"   🔸 β (slope) = {beta:.4f}")
    print(f"   🔸 R² = {r2:.4f}")
    print(f"   🔸 t-stat β=1 : {t_stat:.4f}")
    print(f"   🔸 p-wartość : {p_val:.4f}")


    # --- Test shuffle ---
    print("\n🧪 Test diagnostyczny: korelacja przed i po przetasowaniu danych")
    shuffled_real = np.random.permutation(np.sqrt(real))
    corr_shuffled = np.corrcoef(np.sqrt(pred), shuffled_real)[0, 1]
    drop = (corr - corr_shuffled) / abs(corr) * 100 if corr != 0 else 0

    print(f"   📉 Korelacja po shuffle σ_real: {corr_shuffled:.4f}")
    print(f"   📉 Spadek korelacji po shuffle: {drop:.2f}%")

    plt.figure(figsize=(8, 5))
    plt.scatter(np.sqrt(pred), shuffled_real, alpha=0.5, s=10, color="purple")
    plt.title(f"σ_pred vs przetasowana σ_real (n_obs={n_obs})")
    plt.xlabel("Prognozowana σ_t")
    plt.ylabel("Rzeczywista σ_t (shuffle)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_egarch(n_obs=300)
