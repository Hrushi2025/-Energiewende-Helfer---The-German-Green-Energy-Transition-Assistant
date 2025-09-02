# config.py
from pathlib import Path

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Hrushi@20",  # <-- change
    "database": "energiewende_helfer",
    "port": 3306,
}

# Absolute path to your OPSD CSV (as you shared)
OPSD_CSV_PATH = r"C:\Users\vamtech\PycharmProjects\Chatbot\files\time_series_60min_singleindex.csv"

# How many households to simulate (10–20 as requested)
NUM_HOUSEHOLDS = 15

# Batch size for DB inserts (keep large for speed)
BATCH_SIZE = 5000

# Random seed for reproducibility
RANDOM_SEED = 42

# OPSD column names we need
TIME_COL = "utc_timestamp"
SOLAR_COL = "DE_solar_generation_actual"
LOAD_COL = "DE_load_actual_entsoe_transparency"

# Optional: quick path check when imported
if not Path(OPSD_CSV_PATH).exists():
    raise FileNotFoundError(f"OPSD CSV not found at: {OPSD_CSV_PATH}")
