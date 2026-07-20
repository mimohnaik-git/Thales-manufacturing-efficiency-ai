"""
03_modeling.py
Trains baseline (Logistic Regression) and advanced models (Random Forest,
XGBoost) to classify Efficiency_Status, with a time-respecting train/test
split, class-imbalance handling, cross-validation stability check, and
persistence of the best model for the Streamlit app.
"""
import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report, roc_auc_score)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

df = pd.read_parquet("data/manufacturing_features.parquet")
df = df.sort_values("Datetime").reset_index(drop=True)

# ------------------------------------------------------------------
# Feature set
# ------------------------------------------------------------------
categorical_features = ["Operation_Mode"]
numeric_features = [
    "Temperature_C", "Vibration_Hz", "Power_Consumption_kW", "Network_Latency_ms",
    "Packet_Loss_%", "Quality_Control_Defect_Rate_%", "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score", "Error_Rate_%",
    "Sensor_Stability_Score", "Energy_Efficiency_Ratio", "Error_to_Output_Ratio",
    "Quality_Adjusted_Error", "Network_Reliability_Score", "Hour", "Is_Weekend"
]

df_enc = pd.get_dummies(df, columns=categorical_features, prefix="Mode")
mode_cols = [c for c in df_enc.columns if c.startswith("Mode_")]
feature_cols = numeric_features + mode_cols

target_col = "Efficiency_Status"
le = LabelEncoder()
df_enc["target"] = le.fit_transform(df_enc[target_col])  # High=0, Low=1, Medium=2 (alphabetical)
class_names = list(le.classes_)

# ------------------------------------------------------------------
# Time-based split: last 20% chronologically held out as test set
# ------------------------------------------------------------------
split_idx = int(len(df_enc) * 0.8)
train_df = df_enc.iloc[:split_idx]
test_df = df_enc.iloc[split_idx:]

X_train, y_train = train_df[feature_cols], train_df["target"]
X_test, y_test = test_df[feature_cols], test_df["target"]

print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Train class balance:\n", y_train.value_counts(normalize=True))
print("Test class balance:\n", y_test.value_counts(normalize=True))

# ------------------------------------------------------------------
# Scale numeric features for Logistic Regression
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# ------------------------------------------------------------------
# Class weights (address imbalance: Low 78% / Medium 19% / High 3%)
# ------------------------------------------------------------------
class_counts = y_train.value_counts()
total = len(y_train)
class_weight_dict = {c: total / (len(class_counts) * n) for c, n in class_counts.items()}
sample_weight_train = y_train.map(class_weight_dict).values

results = {}
models = {}

# ------------------------------------------------------------------
# 1. Baseline: Logistic Regression (multi-class)
# ------------------------------------------------------------------
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)
proba_lr = lr.predict_proba(X_test_scaled)
models["Logistic Regression"] = lr

# ------------------------------------------------------------------
# 2. Random Forest
# ------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                             class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)
models["Random Forest"] = rf

# ------------------------------------------------------------------
# 3. XGBoost (Gradient Boosting)
# ------------------------------------------------------------------
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.9,
    objective="multi:softprob", num_class=3,
    eval_metric="mlogloss", random_state=42, n_jobs=-1
)
xgb.fit(X_train, y_train, sample_weight=sample_weight_train)
pred_xgb = xgb.predict(X_test)
proba_xgb = xgb.predict_proba(X_test)
models["XGBoost"] = xgb

# ------------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------------
def evaluate(name, y_true, y_pred, y_proba, base_model, X_cv, cv_weight=None):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(pd.get_dummies(y_true), y_proba, average="macro", multi_class="ovr")
    except Exception:
        auc = np.nan

    # 5-fold CV stability on training data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(base_model, X_cv, y_train, cv=skf, scoring="accuracy", n_jobs=-1)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    results[name] = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "roc_auc_macro_ovr": float(auc) if not np.isnan(auc) else None,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Weighted F1: {f1_w:.4f} | ROC-AUC(macro,ovr): {auc:.4f}")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}  (stability)")
    return cm

from sklearn.base import clone

cm_lr = evaluate("Logistic Regression", y_test, pred_lr, proba_lr, clone(lr), X_train_scaled)
cm_rf = evaluate("Random Forest", y_test, pred_rf, proba_rf, clone(rf), X_train)
cm_xgb = evaluate("XGBoost", y_test, pred_xgb, proba_xgb, clone(xgb), X_train)

# ------------------------------------------------------------------
# Confusion matrices figure
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, cm) in zip(axes, [("Logistic Regression", cm_lr), ("Random Forest", cm_rf), ("XGBoost", cm_xgb)]):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("figures/08_confusion_matrices.png")
plt.close()

# ------------------------------------------------------------------
# Model comparison bar chart
# ------------------------------------------------------------------
comp_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [results[m]["accuracy"] for m in results],
    "Macro F1": [results[m]["f1_macro"] for m in results],
    "CV Std (lower=more stable)": [results[m]["cv_accuracy_std"] for m in results],
})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
comp_df.plot(x="Model", y=["Accuracy", "Macro F1"], kind="bar", ax=axes[0], color=["#4c72b0", "#dd8452"])
axes[0].set_title("Model Accuracy & Macro F1", fontweight="bold")
axes[0].set_ylim(0, 1.05)
axes[0].legend(loc="lower right")
comp_df.plot(x="Model", y="CV Std (lower=more stable)", kind="bar", ax=axes[1], color="#55a868", legend=False)
axes[1].set_title("Cross-Validation Stability (Std. of Accuracy)", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/09_model_comparison.png")
plt.close()

# ------------------------------------------------------------------
# Feature importance (Random Forest & XGBoost)
# ------------------------------------------------------------------
rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
xgb_imp = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
rf_imp.head(12)[::-1].plot(kind="barh", ax=axes[0], color="#4c72b0")
axes[0].set_title("Random Forest — Top 12 Feature Importances", fontweight="bold")
xgb_imp.head(12)[::-1].plot(kind="barh", ax=axes[1], color="#dd8452")
axes[1].set_title("XGBoost — Top 12 Feature Importances", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/10_feature_importance.png")
plt.close()

# ------------------------------------------------------------------
# Pick best model -> XGBoost typically wins; select by macro F1
# ------------------------------------------------------------------
best_name = max(results, key=lambda m: results[m]["f1_macro"])
print(f"\nBest model by macro F1: {best_name}")

joblib.dump(models["Random Forest"], "models/random_forest_model.joblib")
# XGBoost: use its native serialization (JSON), not pickle/joblib. XGBoost's raw
# booster buffer is pickled as an opaque blob whose format is tied to the exact
# xgboost build that wrote it; loading with a different xgboost version/platform
# raises "XGBoostError: input stream corrupted". The native save_model/load_model
# path is explicitly designed to be stable across xgboost versions and platforms.
models["XGBoost"].save_model("models/xgboost_model.json")
joblib.dump(models["Logistic Regression"], "models/logistic_regression_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(le, "models/label_encoder.joblib")

meta = {
    "feature_cols": feature_cols,
    "numeric_features": numeric_features,
    "mode_cols": mode_cols,
    "categorical_features": categorical_features,
    "class_names": class_names,
    "best_model": best_name,
    "split_index": split_idx,
    "train_size": len(train_df),
    "test_size": len(test_df),
}
with open("models/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

with open("outputs/model_results.json", "w") as f:
    json.dump(results, f, indent=2)

rf_imp.to_csv("outputs/rf_feature_importance.csv")
xgb_imp.to_csv("outputs/xgb_feature_importance.csv")

print("\nModeling complete. Artifacts saved to models/ and outputs/")
