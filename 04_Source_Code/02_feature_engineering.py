"""
02_feature_engineering.py
Builds engineered features requested in the project brief:
  - Sensor stability indicators (rolling volatility per machine)
  - Energy efficiency ratios
  - Error-to-output ratios
  - Network reliability score
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/Thales_Group_Manufacturing.csv"
OUT_PATH = "data/manufacturing_features.parquet"

df = pd.read_csv(DATA_PATH)
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Timestamp"], format="%d-%m-%Y %H:%M:%S")
df = df.sort_values(["Machine_ID", "Datetime"]).reset_index(drop=True)

# ---------------------------------------------------------------
# 1. Sensor stability indicators
#    Rolling (per machine, last 5 readings) coefficient of variation
#    for temperature and vibration -> higher = less stable sensor behaviour
# ---------------------------------------------------------------
def rolling_cv(series, window=5):
    roll_mean = series.rolling(window, min_periods=2).mean()
    roll_std = series.rolling(window, min_periods=2).std()
    return (roll_std / roll_mean.replace(0, np.nan)).fillna(0)

grp = df.groupby("Machine_ID", group_keys=False)
df["Temp_Stability_Index"] = grp["Temperature_C"].apply(lambda s: rolling_cv(s))
df["Vibration_Stability_Index"] = grp["Vibration_Hz"].apply(lambda s: rolling_cv(s))
df["Sensor_Stability_Score"] = 1 - ((df["Temp_Stability_Index"] + df["Vibration_Stability_Index"]) / 2).clip(0, 1)

# ---------------------------------------------------------------
# 2. Energy efficiency ratio: output produced per unit of power drawn
# ---------------------------------------------------------------
df["Energy_Efficiency_Ratio"] = df["Production_Speed_units_per_hr"] / df["Power_Consumption_kW"].replace(0, np.nan)
df["Energy_Efficiency_Ratio"] = df["Energy_Efficiency_Ratio"].fillna(0)

# ---------------------------------------------------------------
# 3. Error-to-output ratio: errors relative to units produced
#    (scaled x1000 for readability) + combined quality-adjusted error
# ---------------------------------------------------------------
df["Error_to_Output_Ratio"] = (df["Error_Rate_%"] / df["Production_Speed_units_per_hr"].replace(0, np.nan)) * 1000
df["Error_to_Output_Ratio"] = df["Error_to_Output_Ratio"].fillna(0)
df["Quality_Adjusted_Error"] = df["Error_Rate_%"] + df["Quality_Control_Defect_Rate_%"]

# ---------------------------------------------------------------
# 4. Network reliability score: composite of latency + packet loss
#    normalised to 0-100 (100 = perfectly reliable)
# ---------------------------------------------------------------
lat_norm = (df["Network_Latency_ms"] - df["Network_Latency_ms"].min()) / (df["Network_Latency_ms"].max() - df["Network_Latency_ms"].min())
loss_norm = (df["Packet_Loss_%"] - df["Packet_Loss_%"].min()) / (df["Packet_Loss_%"].max() - df["Packet_Loss_%"].min())
df["Network_Reliability_Score"] = (1 - (0.5 * lat_norm + 0.5 * loss_norm)) * 100

# ---------------------------------------------------------------
# 5. Time-based features (from Date + Time ordering requirement)
# ---------------------------------------------------------------
df["Hour"] = df["Datetime"].dt.hour
df["DayOfWeek"] = df["Datetime"].dt.dayofweek
df["Is_Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

# Save engineered dataset
df.to_parquet(OUT_PATH, index=False)
df.to_csv("data/manufacturing_features.csv", index=False)

print("Feature engineering complete. Shape:", df.shape)
print(df[["Sensor_Stability_Score", "Energy_Efficiency_Ratio", "Error_to_Output_Ratio",
          "Network_Reliability_Score"]].describe())
