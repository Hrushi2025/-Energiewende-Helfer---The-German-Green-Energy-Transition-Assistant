# predict_next_24h_rf.py
import os
import pandas as pd
import numpy as np
import joblib

CSV_PATH = r"./artifacts/merged_training_household_1.csv"  # same CSV used for training
MODEL_PATH = "./models/solar_forecaster_rf.joblib"
META_PATH = "./models/rf_metadata.joblib"
OUT_PATH = "./artifacts/forecast_24h.csv"

def main():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        print("Model or metadata not found. Train first.")
        return
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)

    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Ensure we have enough history for lags
    max_lag = max(meta["lags"])
    if len(df) < max_lag + 1:
        print(f"Need at least {max_lag+1} rows; got {len(df)}.")
        return

    # Start from the last known row
    last_ts = df["timestamp"].iloc[-1]
    last_row = df.iloc[-1].copy()

    # Prepare lag buffer from history: newest at end
    hist_vals = df[meta["target_col"]].tail(max_lag + 1).tolist()  # length max_lag+1

    forecasts = []
    for step in range(1, 25):  # next 24 hours
        next_ts = last_ts + pd.Timedelta(hours=step)

        # Base features: reuse latest weather; roll hour forward
        feats = {
            "solar_radiation": float(last_row["solar_radiation"]),
            "cloud_cover_percent": float(last_row["cloud_cover_percent"]),
            "temperature_celsius": float(last_row["temperature_celsius"]),
            "hour_of_day": int((int(last_row["hour_of_day"]) + step) % 24),
        }

        # Optional realism: set solar_radiation to 0 during night hours
        if feats["hour_of_day"] < 6 or feats["hour_of_day"] > 19:
            feats["solar_radiation"] = 0.0

        # Lag features: use most recent actual/predicted values
        # hist_vals[-1] is the most recent value; use appropriate offsets
        for l in meta["lags"]:
            feats[f"{meta['target_col']}_lag_{l}"] = float(hist_vals[-(l+1)])

        # Ensure correct column order
        X = pd.DataFrame([feats], columns=meta["feature_cols"])
        pred = float(model.predict(X.values)[0])
        pred = max(pred, 0.0)  # clamp negative to zero

        forecasts.append({"timestamp": next_ts, "predicted_kwh": pred})

        # Update history with the new prediction
        hist_vals.append(pred)
        if len(hist_vals) > (max_lag + 1):
            hist_vals = hist_vals[-(max_lag + 1):]

    out_df = pd.DataFrame(forecasts)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"Saved 24h forecast to: {OUT_PATH}")
    print(out_df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
