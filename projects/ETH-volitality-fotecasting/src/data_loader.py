# ===========================================
# src/data_loader.py
# -------------------------------------------
# Wczytywanie danych ETH (train/test) z kontrolą przecieku.
# Domyślnie ładuje tylko train, test opcjonalnie.
# ===========================================

import pandas as pd
import numpy as np
from src.config import TRAIN_FILE, TEST_FILE


# ===========================================
# Funkcja: compute_log_returns
# -------------------------------------------
# Oblicza logarytmiczne stopy zwrotu (log-returns)
# r_t = ln(P_t) - ln(P_{t-1})
# Działa niezależnie w obrębie jednego zbioru (train lub test).
# ===========================================
def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close"]).diff()
    # usuwamy pierwszy wiersz, gdzie diff() = NaN
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    return df


# ===========================================
# Funkcja: load_data
# -------------------------------------------
# Wczytuje dane CSV, konwertuje kolumnę open_time na datetime
# oraz sortuje dane rosnąco po czasie (na wszelki wypadek).
# ===========================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce")
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


# ===========================================
# Funkcja: load_train_test_data
# -------------------------------------------
# Główna funkcja do wczytania i przygotowania danych.
# - Wczytuje train zawsze
# - Test tylko jeśli load_test=True
# - Każdy zestaw ma osobno policzone log-returny
# ===========================================
def load_train_test_data(load_test: bool = False):
    # wczytanie danych treningowych
    train_df = load_data(TRAIN_FILE)
    train_df = compute_log_returns(train_df)

    if load_test:
        test_df = load_data(TEST_FILE)
        test_df = compute_log_returns(test_df)

        # sanity check — sprawdzenie, czy zakresy czasowe się nie pokrywają
        last_train = train_df["open_time"].max()
        first_test = test_df["open_time"].min()
        if last_train >= first_test:
            print("⚠️  Ostrzeżenie: zakresy train/test się nachodzą!")

        print("✅ Dane załadowane poprawnie (train + test).")
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        print(f"Train range: {train_df['open_time'].min()} → {train_df['open_time'].max()}")
        print(f"Test range:  {test_df['open_time'].min()} → {test_df['open_time'].max()}")

        return train_df, test_df

    # tylko dane treningowe
    print("✅ Dane treningowe załadowane poprawnie.")
    print(f"Train shape: {train_df.shape}")
    print(f"Train range: {train_df['open_time'].min()} → {train_df['open_time'].max()}")

    return train_df


# ===========================================
# Główne wywołanie (dla szybkiego testu)
# ===========================================
if __name__ == "__main__":
    # przykład: tylko train
    train_df = load_train_test_data()

    # przykład: train + test
    # train_df, test_df = load_train_test_data(load_test=True)
