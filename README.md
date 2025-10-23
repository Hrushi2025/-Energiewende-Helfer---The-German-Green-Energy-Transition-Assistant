🌍 Energiewende-Helfer — The German Green Energy Transition Assistant

🎯 Aim

Help households make smarter, cost-saving energy decisions by forecasting solar generation and advising optimal battery actions.

📌 Objectives

Predict next-hour and 24-hour solar generation from historical energy and weather data.

Learn a control policy that chooses when to charge, discharge, or hold the battery to reduce energy costs.

Provide a simple CLI for users to get forecasts and actionable advice.

📝 Overview

Supervised learning: RandomForest for next-hour and 24-hour solar forecasts.

Reinforcement learning: PPO agent (Stable-Baselines3) for battery control decisions.

Integration: A simple CLI to get forecasts and advice.

📂 Key artifacts

Supervised model: models/solar_forecaster_rf.joblib, models/rf_metadata.joblib

RL agent: models/energy_agent.zip

24h forecast output: artifacts/forecast_24h_cli.csv

📁 Suggested repo structure
files/
    prepare_dataset.py
    train_forecaster_sklearn.py
    predict_next_rf.py
    predict_next_24h_rf.py
    rl_environment.py
    train_agent.py
    test_agent.py
    main.py
artifacts/        # data and forecast outputs
models/           # trained models

⚙️ Requirements

Python: 3.9–3.11

Packages:
pandas, numpy, scikit-learn, joblib, gymnasium>=0.28,<1.0, stable-baselines3==2.3.0, torch

🔧 Install (example)

Create and activate venv:

Windows:

python -m venv .venv
.venv\Scripts\activate


macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate


Install dependencies:

pip install pandas numpy scikit-learn joblib "gymnasium>=0.28,<1.0" "stable-baselines3==2.3.0" torch

📊 Data requirements

CSV with columns:

timestamp (hourly)

solar_generation_kwh

solar_radiation

cloud_cover_percent

temperature_celsius

hour_of_day (0–23; if missing, derived)

➡️ Typical path: artifacts/merged_training_household_1.csv

🛠️ Step-by-step process we used
1. Data preparation

Merge energy and weather into one hourly CSV, sorted by timestamp, with required columns.

2. Feature engineering

Create lag features for solar_generation_kwh (e.g., 1, 2, 3, 6, 12, 24 hours).

Save metadata (feature list, lags, target column).

3. Supervised learning (forecasting)

Train RandomForest forecaster:

python files/train_forecaster_sklearn.py


Next-hour prediction (sanity check):

python files/predict_next_rf.py


24-hour recursive forecast:

python files/predict_next_24h_rf.py


Outputs:

Models → models/solar_forecaster_rf.joblib, models/rf_metadata.joblib

Forecast → artifacts/forecast_24h.csv (or forecast_24h_cli.csv from CLI)

4. Reinforcement learning (battery control)

Custom Gymnasium environment (rl_environment.py):

State: [hour_of_day_norm, battery_soc_norm, gen_now_norm, naive_next_norm]

Actions:

0 = Hold

1 = Charge

2 = Discharge

Reward: price * exports − price * imports − small degradation cost

Battery limits and efficiency enforced; 24-step daily episodes with simple TOU tariff.

Train PPO agent:

python files/train_agent.py


Evaluate:

python files/test_agent.py


Output: models/energy_agent.zip

5. Integration (CLI)

Run:

python files/main.py


Commands:

forecast → generates and prints a 24h forecast (saves artifacts/forecast_24h_cli.csv)

advice → enter battery capacity (kWh) & SOC% → get recommended action

exit → quit

⚡ Quick start (condensed)

Activate venv & install requirements.

Ensure artifacts/merged_training_household_1.csv exists.

Train RF forecaster:

python files/train_forecaster_sklearn.py


(Optional) Next-hour check:

python files/predict_next_rf.py


Train RL agent:

python files/train_agent.py


Test RL agent:

python files/test_agent.py


Run CLI:

python files/main.py


➡️ Use “forecast” or “advice” at the prompt.

🔍 How it works

Forecasting

Predicts next hour using weather, hour_of_day, and lag features.

For 24 hours → uses prior predictions as lags.

At night → solar_radiation = 0 for realism.

RL environment

Simulates PV generation, battery physics, imports/exports, and TOU prices.

Reward = revenue from exports − cost of imports − degradation cost.

Integration

CLI loads both models.

forecast → daily forecast.

advice → charge/hold/discharge suggestion.

🛠️ Common commands

Train RF: python files/train_forecaster_sklearn.py

Next hour: python files/predict_next_rf.py

24 hours: python files/predict_next_24h_rf.py

Train RL: python files/train_agent.py

Test RL: python files/test_agent.py

Run CLI: python files/main.py

🚀 Improvements (optional)

Train PPO longer (total_timesteps=100,000+); use policy_kwargs=dict(net_arch=[64, 64]).

Use RF next-hour forecast in RL observation (replace naive average).

Add live weather API (e.g., Open-Meteo).

Extend RL → multi-day episodes, multiple households, DB/API integration.

🛑 Troubleshooting

Sklearn warning:
“X has feature names…” → use X.values or retrain with named features.

TensorFlow info logs (harmless):
Silence with:

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


File/model not found:
Check CSV_PATH & model paths in scripts.

RL env issues:
Use check_env in train_agent.py to validate shapes/ranges.
