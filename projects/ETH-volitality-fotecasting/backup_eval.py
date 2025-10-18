import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import statsmodels.api as sm
from src.data_loader import load_train_test_data
from src.config import LOGS_DIR


def evaluate_egarch(model_filename="egarch_ETH_5m_p1_o1_q1_t.pkl", n_obs: int = 500, batch_size: int = 20):
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

    # --- Rolling forecast (batch refit co batch_size) ---
    print(f"🔮 Rolling forecast wariancji (batch co {batch_size})...")
    model = fitted.model
    params = fitted.params
    vol = model.volatility
    dist_name = "t"

    forecasts = []
    param_history = []
    res = None  # ostatni dopasowany model (używany między refitami)

    for i in range(len(test_returns_std)):
        if i % batch_size == 0 or res is None:
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
                disp="off",
                last_obs=len(subseries) - 1,
                update_freq=0,
                starting_values=params,
            )
            params = res.params
            param_history.append({"step": i, **params.to_dict()})

        # prognozujemy zawsze na podstawie ostatniego dopasowanego modelu
        fcast = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
        forecasts.append(fcast)

    # --- Dopasowanie długości ---
    test_df = test_df.iloc[:len(forecasts)].copy()
    test_df["var_pred"] = np.array(forecasts)
    test_df["var_real"] = (test_df["log_return"] / scale) ** 2

    # --- Zapis parametrów batchów ---
    param_file = LOGS_DIR / "evaluation_params.json"
    with open(param_file, "w") as f:
        json.dump(param_history, f, indent=2)
    print(f"🧾 Zapisano historię parametrów: {param_file.name}")

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
    qlike = np.mean(np.log(pred) + real / pred)
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

    beta_se = model_mz.bse[1]
    t_stat = (beta - 1) / beta_se
    from scipy import stats
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), model_mz.df_resid))

    print(f"   🔸 α (const) = {alpha:.4f}")
    print(f"   🔸 β (slope) = {beta:.4f}")
    print(f"   🔸 R² = {r2:.4f}")
    print(f"   🔸 t-stat β=1 : {t_stat:.4f}")
    print(f"   🔸 p-wartość : {p_val:.4f}")


if __name__ == "__main__":
    evaluate_egarch(n_obs=300, batch_size=200)
