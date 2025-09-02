# load_weather.py
import numpy as np
import pandas as pd
import mysql.connector as mysql
from pathlib import Path
from config import DB_CONFIG, OPSD_CSV_PATH, BATCH_SIZE, RANDOM_SEED, TIME_COL

np.random.seed(RANDOM_SEED)

def get_conn():
    return mysql.connect(**DB_CONFIG)

def read_all_timestamps(csv_path: str) -> pd.Series:
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path, usecols=[TIME_COL], parse_dates=[TIME_COL])
    df = df.dropna().sort_values(TIME_COL).reset_index(drop=True)
    if df.empty:
        raise ValueError("No timestamps found.")
    return df[TIME_COL]

def get_all_zip_codes(cur):
    cur.execute("SELECT zip_code FROM households")
    return [row[0] for row in cur.fetchall()]

def generate_weather_series(timestamps: pd.Series):
    # Synthetic but plausible weather aligned with timestamps
    doy = timestamps.dt.dayofyear.values
    hour = timestamps.dt.hour.values

    # Daylight factor (0 at night, 1 at noon)
    diurnal = np.clip(np.sin((hour - 6) / 12.0 * np.pi), 0, 1)

    # Cloud cover varies with time + noise (0..100)
    cloud = np.clip(50 + 20 * np.sin(2 * np.pi * hour / 24.0 + 1.2) + np.random.normal(0, 10, size=len(hour)), 0, 100)

    # Sunshine minutes per hour: proportional to daylight and inverse to clouds
    sun_minutes = np.clip(diurnal * (1 - cloud / 100.0) * 60.0 + np.random.normal(0, 2, size=len(hour)), 0, 60)

    # Temperature: base + seasonal + diurnal + noise
    temp = (
        12.0
        + 10.0 * np.sin(2 * np.pi * (doy - 172) / 365.0)  # seasonal
        + 5.0 * np.sin(2 * np.pi * (hour - 8) / 24.0)     # diurnal
        + np.random.normal(0, 1.3, size=len(hour))
    )

    return temp.astype(float), cloud.astype(float), sun_minutes.astype(float)

def insert_weather_batch(cur, rows):
    sql = """
        INSERT INTO weather_data (zip_code, timestamp, temperature_celsius, cloud_cover_percent, sunshine_minutes_per_hour)
        VALUES (%s, %s, %s, %s, %s)
    """
    cur.executemany(sql, rows)

def main():
    conn = get_conn()
    cur = conn.cursor()

    zips = get_all_zip_codes(cur)
    if not zips:
        print("No households found. Run load_energy.py first.")
        cur.close(); conn.close()
        return

    timestamps = read_all_timestamps(OPSD_CSV_PATH)
    print(f"Inserting weather for {len(zips)} zip codes across {len(timestamps)} hours...")

    base_temp, base_cloud, base_sun = generate_weather_series(timestamps)

    for zip_code in zips:
        # Small per-zip variation
        temp = base_temp + np.random.normal(0, 0.7, size=len(base_temp))
        cloud = np.clip(base_cloud + np.random.normal(0, 3, size=len(base_cloud)), 0, 100)
        sun = np.clip(base_sun * np.random.uniform(0.95, 1.05), 0, 60)

        rows = [
            (zip_code, timestamps.iloc[i].to_pydatetime(), float(temp[i]), float(cloud[i]), float(sun[i]))
            for i in range(len(timestamps))
        ]

        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
            insert_weather_batch(cur, batch)
            conn.commit()

        print(f"- Weather inserted for zip {zip_code}")

    cur.close()
    conn.close()
    print("Weather data ingestion complete.")

if __name__ == "__main__":
    main()
