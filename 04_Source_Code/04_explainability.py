"""
04_explainability.py
SHAP-based explainability for the XGBoost model: global feature importance
and example local explanations for Low / Medium / High predictions.
"""
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

xgb = joblib.load("models/xgboost_model.joblib")
with open("models/model_meta.json") as f:
    meta = json.load(f)
feature_cols = meta["feature_cols"]
class_names = meta["class_names"]

df = pd.read_parquet("data/manufacturing_features.parquet")
df = df.sort_values("Datetime").reset_index(drop=True)
df_enc = pd.get_dummies(df, columns=["Operation_Mode"], prefix="Mode")

# sample for SHAP speed
sample = df_enc.sample(n=3000, random_state=42)
X_sample = sample[feature_cols]

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_sample)  # list-like per class or array (n, features, classes)

# Handle shap output shape across versions
sv = np.array(shap_values)
if sv.ndim == 3 and sv.shape[0] == len(class_names):
    # shape (classes, n, features)
    shap_per_class = [sv[i] for i in range(len(class_names))]
elif sv.ndim == 3 and sv.shape[-1] == len(class_names):
    # shape (n, features, classes)
    shap_per_class = [sv[:, :, i] for i in range(len(class_names))]
else:
    shap_per_class = [sv]

# ------------------------------------------------------------------
# Global summary plot (mean |SHAP| across classes)
# ------------------------------------------------------------------
mean_abs_shap = np.mean([np.abs(s).mean(axis=0) for s in shap_per_class], axis=0)
importance_series = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)
importance_series.to_csv("outputs/shap_global_importance.csv")

fig, ax = plt.subplots(figsize=(9, 6))
importance_series.head(12)[::-1].plot(kind="barh", ax=ax, color="#8172b2")
ax.set_title("Global SHAP Feature Importance (mean |SHAP value|, all classes)", fontsize=12, fontweight="bold")
ax.set_xlabel("mean(|SHAP value|)")
plt.tight_layout()
plt.savefig("figures/11_shap_global_importance.png")
plt.close()

# ------------------------------------------------------------------
# Per-class SHAP summary (beeswarm) for Low class (index depends on label order)
# ------------------------------------------------------------------
low_idx = class_names.index("Low")
try:
    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_per_class[low_idx], X_sample, feature_names=feature_cols, show=False, max_display=12)
    plt.title("SHAP Summary — Drivers of 'Low' Efficiency Classification", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/12_shap_low_class_summary.png")
    plt.close()
except Exception as e:
    print("Beeswarm plot skipped:", e)

# ------------------------------------------------------------------
# Save a few example local explanations for the app / paper
# ------------------------------------------------------------------
examples = []
for cls in class_names:
    idx = sample[sample[[c for c in feature_cols if c.startswith("Mode_")][0]].notna()].index
    # just grab a row whose true label matches this class
    true_label_col = df.loc[sample.index, "Efficiency_Status"]
    match = sample.index[true_label_col.loc[sample.index] == cls]
    if len(match) > 0:
        ridx = match[0]
        pos = sample.index.get_loc(ridx)
        cls_i = class_names.index(cls)
        row_shap = shap_per_class[cls_i][pos]
        top_feats = pd.Series(row_shap, index=feature_cols).abs().sort_values(ascending=False).head(5)
        examples.append({
            "true_class": cls,
            "row_index": int(ridx),
            "top_contributing_features": {f: float(row_shap[feature_cols.index(f)]) for f in top_feats.index}
        })

with open("outputs/shap_example_explanations.json", "w") as f:
    json.dump(examples, f, indent=2)

print("Explainability analysis complete.")
print(importance_series.head(10))
