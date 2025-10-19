# ===========================================
# src/evaluation.py
# -------------------------------------------
# Walidacja modelu EGARCH na danych testowych
# z zachowaniem właściwej skali.
# ===========================================

import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import statsmodels.api as sm
from scipy import stats
from src.data_loader import load_train_test_data
from src.config import LOGS_DIR


def evaluate_egarch(model_filename="egarch_ETH_5m_p1_o2_q2_t.pkl",
                    n_obs=300,
                    batch_size=200,
                    window_size=5000,
                    scale_window=1000):
    """Walidacja EGARCH na out-of-sample z rolling window i poprawnym skalowaniem."""
    # --- Model ---
    payload = pickle.load(open(LOGS_DIR / model_filename, "rb"))
    fitted, base_scale = payload["model"], payload["scale"]
    model, params, vol = fitted.model, fitted.params, fitted.model.volatility
    print(f"✅ Wczytano model: {model_filename} (scale={base_scale:.3e})")

    # --- Dane ---
    _, test_df = load_train_test_data(load_test=True)
    test_df = test_df.head(n_obs).copy()
    test_returns_std = test_df["log_return"] / base_scale

    total_series = np.concatenate([fitted.model._y, test_returns_std])
    offset = len(fitted.model._y)

    forecasts, params_hist = [], []
    for i in range(len(test_returns_std)):
        if i % batch_size == 0:
            end = offset + i
            sub = total_series[max(0, end - window_size):end]

            local_scale = np.std(sub[-scale_window:])
            sub_std = sub / local_scale

            temp_model = arch_model(sub_std, vol="EGARCH",
                                    p=vol.p, o=vol.o, q=vol.q,
                                    dist="t", mean="Constant")
            res = temp_model.fit(disp="off", update_freq=0, starting_values=params)
            params = res.params
            params_hist.append({"step": i, **params.to_dict()})

        fvar = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
        forecasts.append(fvar * (local_scale ** 2) * (base_scale ** 2))

    test_df["var_pred"] = forecasts
    test_df["var_real"] = (test_df["log_return"]) ** 2

    # --- Save params ---
    json.dump(params_hist, open(LOGS_DIR / "evaluation_params.json", "w"), indent=2)

    # --- Plot ---
    plt.figure(figsize=(12, 5))
    plt.plot(test_df["open_time"], np.sqrt(test_df["var_real"]), color="black", alpha=0.6, label="σ_real")
    plt.plot(test_df["open_time"], np.sqrt(test_df["var_pred"]), color="red", label="σ_pred")
    plt.legend(); plt.title("EGARCH – walidacja out-of-sample"); plt.tight_layout(); plt.show()

    # --- Metrics ---
    pred, real = test_df["var_pred"].values, test_df["var_real"].values
    rmse = np.sqrt(np.mean((pred - real) ** 2))
    mae = np.mean(np.abs(pred - real))
    qlike = np.mean(np.log(pred) + real / pred)
    corr = np.corrcoef(np.sqrt(pred), np.sqrt(real))[0, 1]

    print(f"\n📊 Wyniki walidacji:")
    print(f"   RMSE : {rmse:.3e}")
    print(f"   MAE  : {mae:.3e}")
    print(f"   QLIKE: {qlike:.3e}")
    print(f"   Corr σ_pred vs σ_real: {corr:.3f}")

    # --- Mincer–Zarnowitz ---
    X = sm.add_constant(pred)
    mz = sm.OLS(real, X).fit()
    α, β = mz.params
    t = (β - 1) / mz.bse[1]
    p = 2 * (1 - stats.t.cdf(abs(t), mz.df_resid))
    print(f"\n🧩 Mincer–Zarnowitz:")
    print(f"   α={α:.4f}, β={β:.4f}, R²={mz.rsquared:.4f}, p(β=1)={p:.4f}")


if __name__ == "__main__":
    evaluate_egarch()
