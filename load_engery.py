# load_energy.py
import random
import numpy as np
import pandas as pd
import mysql.connector as mysql
from pathlib import Path
import config as cfg  # use getattr to read optional flags

random.seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)

def get_conn():
    return mysql.connect(**cfg.DB_CONFIG)

def read_opsd_full(csv_path: str) -> pd.DataFrame:
    """Read ALL rows from OPSD CSV but only required columns."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path, usecols=[cfg.TIME_COL, cfg.SOLAR_COL, cfg.LOAD_COL], parse_dates=[cfg.TIME_COL])
    cols = df.columns.tolist()
    needed = [cfg.TIME_COL, cfg.SOLAR_COL, cfg.LOAD_COL]
    missing = [c for c in needed if c not in cols]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {cols[:20]} ...")
    df = (
        df.rename(columns={cfg.SOLAR_COL: "national_solar", cfg.LOAD_COL: "national_consumption"})
          .dropna()
          .sort_values(cfg.TIME_COL)
          .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("No rows after cleaning.")
    return df

def insert_household(cur, zip_code: str, capacity_kw: float, has_battery: bool) -> int:
    sql = "INSERT INTO households (zip_code, solar_panel_capacity_kw, has_battery) VALUES (%s, %s, %s)"
    cur.execute(sql, (zip_code, float(capacity_kw), int(has_battery)))
    return cur.lastrowid

def insert_energy_batch(cur, rows):
    sql = """
        INSERT INTO energy_data (household_id, timestamp, consumption_kwh, solar_generation_kwh)
        VALUES (%s, %s, %s, %s)
    """
    cur.executemany(sql, rows)

def maybe_clear_tables(cur, conn):
    clear = getattr(cfg, "CLEAR_BEFORE_INSERT", False)
    if clear:
        print("CLEAR_BEFORE_INSERT=True: Deleting existing households and energy_data...")
        cur.execute("DELETE FROM energy_data")
        cur.execute("DELETE FROM households")
        conn.commit()
        print("- Tables cleared.")

def main():
    print("Reading full OPSD CSV (all rows)...")
    df = read_opsd_full(cfg.OPSD_CSV_PATH)
    timestamps = df[cfg.TIME_COL]
    national_solar = df["national_solar"].astype(float)
    national_consumption = df["national_consumption"].astype(float)

    # Scalars per your formulas
    solar_max = max(national_solar.max(), 1e-6)
    cons_mean = max(national_consumption.mean(), 1e-6)

    conn = get_conn()
    cur = conn.cursor()

    # Optional clean rerun
    maybe_clear_tables(cur, conn)

    print(f"Creating {cfg.NUM_HOUSEHOLDS} households and inserting personalized energy time series...")

    # Optionally ensure unique zip codes for the generated batch
    # Generate a pool of zip codes and sample without replacement
    zip_pool = list(range(10000, 99999))
    sample_zips = random.sample(zip_pool, cfg.NUM_HOUSEHOLDS)

    for idx in range(cfg.NUM_HOUSEHOLDS):
        # Household metadata
        zip_code = f"{sample_zips[idx]}"
        capacity_kw = round(random.uniform(3.0, 10.0), 2)
        has_battery = random.random() < 0.5  # 50% chance

        household_id = insert_household(cur, zip_code, capacity_kw, has_battery)

        # Solar personalization (exact as requested)
        household_solar = (national_solar / solar_max) * capacity_kw

        # Consumption personalization:
        # Keep national shape but introduce per-household daily patterns so clustering is meaningful.
        hours = timestamps.dt.hour.values

        # Base hourly kWh for this household (around 0.4, with variation)
        base_kwh = np.random.uniform(0.30, 0.55)

        # Morning/evening preference
        morning_boost = np.random.uniform(0.9, 1.3)
        evening_boost = np.random.uniform(0.9, 1.3)

        hour_weights = np.ones(24, dtype=float)
        hour_weights[[6, 7, 8, 9]] *= morning_boost      # morning tilt
        hour_weights[[17, 18, 19, 20]] *= evening_boost  # evening tilt

        # Gentle random hour-to-hour variation and normalize to mean 1
        hour_weights *= (1.0 + np.random.normal(0, 0.05, size=24))
        hour_weights = hour_weights / hour_weights.mean()

        # Map weights to each timestamp by hour-of-day
        w = hour_weights[hours]

        # National shape normalized around ~1, then scaled by base and hour weights
        national_shape = (national_consumption / cons_mean)
        household_consumption = np.clip(national_shape * base_kwh * w, 0.01, None)

        # Build rows
        rows = [
            (household_id, timestamps.iloc[i].to_pydatetime(),
             float(household_consumption.iloc[i]), float(household_solar.iloc[i]))
            for i in range(len(df))
        ]

        # Batch insert
        for start in range(0, len(rows), cfg.BATCH_SIZE):
            batch = rows[start:start + cfg.BATCH_SIZE]
            insert_energy_batch(cur, batch)
            conn.commit()

        print(f"- Household {household_id} inserted: zip={zip_code}, capacity={capacity_kw} kW, has_battery={has_battery}")

    cur.close()
    conn.close()
    print("Energy data ingestion complete.")

if __name__ == "__main__":
    main()
