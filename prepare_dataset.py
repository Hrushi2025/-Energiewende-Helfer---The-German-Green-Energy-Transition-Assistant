# prepare_dataset.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Your MySQL details (filled in)
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Hrushi@20",
    "database": "energiewende_helfer",
    "port": 3306,
}

def get_engine():
    url = URL.create(
        drivername="mysql+mysqlconnector",
        username=DB_CONFIG["user"],
        password=DB_CONFIG["password"],   # safe even with '@'
        host=DB_CONFIG["host"],
        port=DB_CONFIG.get("port", 3306),
        database=DB_CONFIG["database"],
        query={"charset": "utf8mb4"},
    )
    return create_engine(url, pool_pre_ping=True)

def main():
    engine = get_engine()

    # Pick one household (first by id)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT id, zip_code FROM households ORDER BY id LIMIT 1")).fetchone()
        if not row:
            print("No households found. Run load_energy.py first.")
            return
        household_id, zip_code = row

    # Merge energy + weather for that household
    sql = """
        SELECT
            e.timestamp,
            e.solar_generation_kwh,
            w.temperature_celsius,
            w.cloud_cover_percent,
            w.sunshine_minutes_per_hour
        FROM energy_data e
        JOIN households h ON e.household_id = h.id
        JOIN weather_data w ON w.zip_code = h.zip_code AND w.timestamp = e.timestamp
        WHERE e.household_id = :hid
        ORDER BY e.timestamp
    """
    df = pd.read_sql(text(sql), get_engine(), params={"hid": household_id}, parse_dates=["timestamp"])

    if df.empty:
        print("No merged rows found. Ensure weather_data is populated and timestamps align.")
        return

    # Features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["solar_radiation"] = (1.0 - df["cloud_cover_percent"].clip(0, 100) / 100.0) * (df["sunshine_minutes_per_hour"].clip(0, 60) / 60.0)
    df["solar_radiation"] = df["solar_radiation"].clip(0, 1)

    df = df[[
        "timestamp",
        "solar_generation_kwh",               # target
        "solar_radiation",
        "cloud_cover_percent",
        "temperature_celsius",
        "hour_of_day"
    ]].dropna().sort_values("timestamp").reset_index(drop=True)

    # Save to CSV
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"merged_training_household_{household_id}.csv")
    df.to_csv(out_path, index=False)
    print(f"Prepared dataset for household_id={household_id}")
    print(f"Rows: {len(df)}, Time range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Saved CSV: {out_path}")
    print("Next: set CSV_PATH in train_forecaster.py to this path and run it.")

if __name__ == "__main__":
    main()
