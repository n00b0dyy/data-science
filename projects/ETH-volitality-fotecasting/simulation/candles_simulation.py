import pandas as pd
from pathlib import Path

# Absolutna ścieżka do pliku
data_path = Path(r"C:\GitHubRepo\data-science\projects\ETH-volitality-fotecasting\simulation\candles_5m_test.csv")

candles = pd.read_csv(data_path)
print(f"✅ Wczytano plik: {data_path.name}")
candles.info()

if "open_time" in candles.columns:
    candles["open_time"] = pd.to_datetime(candles["open_time"])
    print(f"\n📅 Początkowa data: {candles['open_time'].min()}")
    print(f"📅 Końcowa data:   {candles['open_time'].max()}")

print(f"🔁 Liczba iteracji (świeczek): {len(candles)}")
