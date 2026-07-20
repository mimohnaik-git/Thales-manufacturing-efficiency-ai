"""
01_eda.py -- Exploratory Data Analysis
AI-Based Manufacturing Efficiency Classification (Thales Group)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 150

DATA_PATH = "data/Thales_Group_Manufacturing.csv"
FIG_DIR = "figures"

df = pd.read_csv(DATA_PATH)
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Timestamp"], format="%d-%m-%Y %H:%M:%S")
df = df.sort_values("Datetime").reset_index(drop=True)

num_cols = ['Temperature_C','Vibration_Hz','Power_Consumption_kW','Network_Latency_ms',
            'Packet_Loss_%','Quality_Control_Defect_Rate_%','Production_Speed_units_per_hr',
            'Predictive_Maintenance_Score','Error_Rate_%']

status_order = ["Low", "Medium", "High"]
status_colors = {"Low": "#d62728", "Medium": "#ff9f1c", "High": "#2ca02c"}

summary = {}
summary["n_rows"] = int(len(df))
summary["n_machines"] = int(df["Machine_ID"].nunique())
summary["date_range"] = [str(df["Datetime"].min()), str(df["Datetime"].max())]
summary["class_counts"] = df["Efficiency_Status"].value_counts().to_dict()
summary["class_pct"] = (df["Efficiency_Status"].value_counts(normalize=True) * 100).round(2).to_dict()
summary["operation_mode_counts"] = df["Operation_Mode"].value_counts().to_dict()
summary["missing_values"] = int(df.isna().sum().sum())
summary["duplicate_rows"] = int(df.duplicated().sum())

# ---------- 1. Class balance ----------
fig, ax = plt.subplots(figsize=(6, 4.5))
counts = df["Efficiency_Status"].value_counts().reindex(status_order)
bars = ax.bar(counts.index, counts.values, color=[status_colors[s] for s in counts.index])
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 500, f"{b.get_height():,}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Class Distribution: Efficiency_Status", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Records")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_class_distribution.png")
plt.close()

# ---------- 2. Correlation heatmap ----------
fig, ax = plt.subplots(figsize=(8, 6.5))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            cbar_kws={"label": "Pearson correlation"})
ax.set_title("Correlation Matrix — Sensor, Production & Network Features", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_correlation_heatmap.png")
plt.close()

# ---------- 3. Feature distributions by efficiency status (key drivers) ----------
key_feats = ["Error_Rate_%", "Production_Speed_units_per_hr"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, feat in zip(axes, key_feats):
    for status in status_order:
        sns.kdeplot(df.loc[df["Efficiency_Status"] == status, feat], ax=ax,
                    label=status, color=status_colors[status], fill=True, alpha=0.25, linewidth=2)
    ax.set_title(f"Distribution of {feat} by Efficiency Status", fontsize=11, fontweight="bold")
    ax.legend(title="Efficiency")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_key_driver_distributions.png")
plt.close()

# ---------- 4. Non-driver sensor features by status (near-identical -> noise) ----------
noise_feats = ["Temperature_C", "Vibration_Hz", "Network_Latency_ms", "Packet_Loss_%"]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.flat, noise_feats):
    sns.boxplot(data=df, x="Efficiency_Status", y=feat, order=status_order, ax=ax,
                palette=status_colors)
    ax.set_title(feat, fontsize=10, fontweight="bold")
plt.suptitle("Sensor/Network Features Show Little Separation Across Efficiency Classes", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_noise_feature_boxplots.png")
plt.close()

# ---------- 5. Efficiency mix by Operation Mode ----------
fig, ax = plt.subplots(figsize=(7, 4.8))
ct = pd.crosstab(df["Operation_Mode"], df["Efficiency_Status"], normalize="index")[status_order] * 100
ct.plot(kind="bar", stacked=True, ax=ax, color=[status_colors[s] for s in status_order])
ax.set_ylabel("% of Records")
ax.set_title("Efficiency Status Mix by Operation Mode", fontsize=12, fontweight="bold")
ax.legend(title="Efficiency", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_efficiency_by_operation_mode.png")
plt.close()

# ---------- 6. Machine-level efficiency profile (top/bottom machines) ----------
machine_eff = df.groupby("Machine_ID")["Efficiency_Status"].apply(
    lambda s: (s == "Low").mean() * 100).sort_values(ascending=False)
summary["machine_low_pct_range"] = [float(machine_eff.min()), float(machine_eff.max())]

fig, ax = plt.subplots(figsize=(10, 5))
machine_eff.plot(kind="bar", ax=ax, color="#4c72b0", width=0.8)
ax.set_ylabel("% of records classified Low")
ax.set_xlabel("Machine ID")
ax.set_title("Share of 'Low' Efficiency Records by Machine", fontsize=12, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=90)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_machine_low_share.png")
plt.close()

# ---------- 7. Daily trend of efficiency mix over the month ----------
daily = df.groupby([df["Datetime"].dt.date, "Efficiency_Status"]).size().unstack(fill_value=0)
daily_pct = daily.div(daily.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(11, 4.8))
for status in status_order:
    ax.plot(daily_pct.index, daily_pct[status], marker="o", markersize=3, label=status, color=status_colors[status])
ax.set_ylabel("% of Records")
ax.set_title("Daily Efficiency Mix Across January 2025", fontsize=12, fontweight="bold")
ax.legend(title="Efficiency")
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_daily_efficiency_trend.png")
plt.close()

# ---------- Rule discovery: what actually separates the classes ----------
err_bin = pd.cut(df["Error_Rate_%"], bins=[-1, 2, 5, 15], labels=["<=2%", "2-5%", ">5%"])
speed_bin = pd.cut(df["Production_Speed_units_per_hr"], bins=[0, 200, 400, 500], labels=["<200", "200-400", "400-500"])
rule_table = pd.crosstab([err_bin, speed_bin], df["Efficiency_Status"])
rule_table.to_csv("outputs/rule_discovery_table.csv")

with open("outputs/eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("EDA complete.")
print(json.dumps(summary, indent=2, default=str))
