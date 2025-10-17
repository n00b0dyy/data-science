# src/config.py

from pathlib import Path
import os

# === ROOT PATH ===
# automatyczne wykrycie głównego katalogu repozytorium
ROOT_DIR = Path(__file__).resolve().parents[1]

# === FOLDERS ===
DATA_DIR = ROOT_DIR / "candles"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
INTERIM_DATA_DIR = ROOT_DIR / "data" / "interim"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
LOGS_DIR = ROOT_DIR / "logs"
PLOTS_DIR = ROOT_DIR / "plots"
SRC_DIR = ROOT_DIR / "src"

# === FILES ===
TRAIN_FILE = DATA_DIR / "train_sample_clean.csv"
TEST_FILE = DATA_DIR / "out_of_sample_test_clean.csv"

# === RANDOMNESS & SEED ===
RANDOM_SEED = 42

# === LOGGING ===
LOG_FILE = LOGS_DIR / "pipeline.log"

# === RUNTIME FLAGS ===
DEBUG = False

# Upewnij się, że katalogi istnieją
for d in [LOGS_DIR, PLOTS_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR]:
    os.makedirs(d, exist_ok=True)
