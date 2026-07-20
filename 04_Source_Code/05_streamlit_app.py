"""
Streamlit Web Application — AI-Based Manufacturing Efficiency Classification
Thales Group | Sensor, Production & 6G Network Data
================================================================
Run with:  streamlit run app.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd

# pandas >= 3.0 defaults to an arrow-backed "str" dtype for string columns and
# even for Index objects built from plain Python string lists. PyArrow's
# compute kernels backing that dtype are not safe to call repeatedly from
# Streamlit's background script-runner thread and can segfault on rerun —
# this must be set before any other pandas operation, and reverts pandas to
# its classic numpy/object string storage everywhere in this process.
pd.set_option("future.infer_string", False)

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Thales | Manufacturing Efficiency AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "manufacturing_features.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

STATUS_COLORS = {"High": "#2ca02c", "Medium": "#ff9f1c", "Low": "#d62728"}
STATUS_ORDER = ["Low", "Medium", "High"]

# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Machine_ID"] = df["Machine_ID"].astype("int64")
    for col in ["Operation_Mode", "Efficiency_Status"]:
        df[col] = df[col].astype("object")
    return df

@st.cache_resource
def load_models():
    from xgboost import XGBClassifier
    xgb_model = XGBClassifier()
    xgb_model.load_model(os.path.join(MODEL_DIR, "xgboost_model.json"))
    models = {
        "XGBoost": xgb_model,
        "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest_model.joblib")),
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.joblib")),
    }
    # Force single-threaded prediction — safer in constrained/sandboxed hosting environments
    try:
        models["XGBoost"].set_params(n_jobs=1)
    except Exception:
        pass
    try:
        models["Random Forest"].n_jobs = 1
    except Exception:
        pass
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    return models, scaler, le, meta

@st.cache_data
def load_importance():
    rf_imp = pd.read_csv(os.path.join(OUTPUTS_DIR, "rf_feature_importance.csv"), index_col=0).iloc[:, 0]
    xgb_imp = pd.read_csv(os.path.join(OUTPUTS_DIR, "xgb_feature_importance.csv"), index_col=0).iloc[:, 0]
    shap_imp = pd.read_csv(os.path.join(OUTPUTS_DIR, "shap_global_importance.csv"), index_col=0).iloc[:, 0]
    return rf_imp, xgb_imp, shap_imp

@st.cache_data
def load_model_results():
    with open(os.path.join(OUTPUTS_DIR, "model_results.json")) as f:
        return json.load(f)

df = load_data()
try:
    models, scaler, le, meta = load_models()
except Exception as e:
    st.error(
        "**Could not load the trained models.**\n\n"
        "This usually means the installed package versions don't match the ones "
        "used to train the models. Please install the exact pinned versions:\n\n"
        "```\npip install -r requirements.txt\n```\n\n"
        f"Underlying error: `{type(e).__name__}: {e}`"
    )
    st.stop()
rf_imp, xgb_imp, shap_imp = load_importance()
model_results = load_model_results()
feature_cols = meta["feature_cols"]
numeric_features = meta["numeric_features"]
class_names = meta["class_names"]

FEATURE_GROUPS = {
    "Network": ["Network_Latency_ms", "Packet_Loss_%", "Network_Reliability_Score"],
    "Sensor": ["Temperature_C", "Vibration_Hz", "Sensor_Stability_Score", "Predictive_Maintenance_Score"],
    "Production/Quality": ["Production_Speed_units_per_hr", "Error_Rate_%", "Quality_Control_Defect_Rate_%",
                            "Error_to_Output_Ratio", "Quality_Adjusted_Error", "Power_Consumption_kW",
                            "Energy_Efficiency_Ratio"],
}

def predict_row(model_name, row_df):
    """row_df: single-row DataFrame with raw feature columns (pre-encoding done outside)."""
    model = models[model_name]
    X = row_df[feature_cols]
    if model_name == "Logistic Regression":
        X = X.copy()
        X[numeric_features] = scaler.transform(X[numeric_features])
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = le.inverse_transform([pred_idx])[0]
    proba_dict = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}
    return pred_label, proba_dict

# ==================================================================
# SIDEBAR — Global Filters & Live Predictor
# ==================================================================
st.sidebar.markdown("## ⚙️ Thales Smart Factory")
st.sidebar.caption("AI-Based Manufacturing Efficiency Classification")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🔍 Filters")

machine_list = sorted(df["Machine_ID"].unique().tolist())
selected_machines = st.sidebar.multiselect("Machine selector", options=machine_list,
                                            default=machine_list, help="Choose one or more machines")

mode_list = sorted(df["Operation_Mode"].unique().tolist())
selected_modes = st.sidebar.multiselect("Operation mode", options=mode_list, default=mode_list)

min_date, max_date = df["Datetime"].min(), df["Datetime"].max()
date_range = st.sidebar.slider(
    "Time window filter",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="DD-MM-YY HH:mm",
)

network_range = st.sidebar.slider(
    "Network quality filter (Reliability Score)",
    min_value=0.0, max_value=100.0, value=(0.0, 100.0), step=1.0,
    help="0 = worst network reliability, 100 = best"
)

st.sidebar.markdown("### 🎚️ Metric Sensitivity (Alert Thresholds)")
error_sensitivity = st.sidebar.slider("Error Rate alert threshold (%)", 0.0, 15.0, 5.0, 0.5)
speed_sensitivity = st.sidebar.slider("Low production-speed alert threshold (units/hr)", 0.0, 500.0, 200.0, 10.0)

st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox("Prediction model", options=list(models.keys()),
                                     index=list(models.keys()).index("XGBoost"))

# Apply filters
if not selected_machines:
    selected_machines = machine_list
if not selected_modes:
    selected_modes = mode_list

mask = (
    df["Machine_ID"].isin(selected_machines)
    & df["Operation_Mode"].isin(selected_modes)
    & (df["Datetime"] >= pd.Timestamp(date_range[0]))
    & (df["Datetime"] <= pd.Timestamp(date_range[1]))
    & (df["Network_Reliability_Score"] >= network_range[0])
    & (df["Network_Reliability_Score"] <= network_range[1])
)
fdf = df.loc[mask].copy()

st.sidebar.markdown(f"**Filtered records:** {len(fdf):,} / {len(df):,}")

# ==================================================================
# HEADER
# ==================================================================
st.title("⚙️ AI-Based Manufacturing Efficiency Classification")
st.caption("Sensor, Production & 6G Network Data · Thales Group Smart Factory Initiative")

if fdf.empty:
    st.warning("No records match the current filters. Please broaden your selection.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Efficiency Prediction",
    "🏭 Machine-Level Insights",
    "🔬 Explainability Panel",
    "🌐 Operational Monitoring",
])

# ==================================================================
# TAB 1 — Efficiency Prediction Dashboard
# ==================================================================
with tab1:
    st.subheader("Real-Time Efficiency Classification")

    kc1, kc2, kc3, kc4 = st.columns(4)
    class_counts = fdf["Efficiency_Status"].value_counts()
    kc1.metric("Records in view", f"{len(fdf):,}")
    kc2.metric("% High Efficiency", f"{class_counts.get('High', 0) / len(fdf) * 100:.1f}%")
    kc3.metric("% Medium Efficiency", f"{class_counts.get('Medium', 0) / len(fdf) * 100:.1f}%")
    kc4.metric("% Low Efficiency", f"{class_counts.get('Low', 0) / len(fdf) * 100:.1f}%",
               delta=None, delta_color="inverse")

    st.markdown("---")
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("#### 🎛️ Simulate a Live Machine Reading")
        st.caption("Adjust sensor and production values to get an instant AI efficiency classification.")

        c1, c2 = st.columns(2)
        with c1:
            temp = st.slider("Temperature (°C)", 30.0, 90.0, float(fdf["Temperature_C"].mean()))
            vib = st.slider("Vibration (Hz)", 0.0, 5.0, float(fdf["Vibration_Hz"].mean()))
            power = st.slider("Power Consumption (kW)", 0.0, 10.0, float(fdf["Power_Consumption_kW"].mean()))
            latency = st.slider("Network Latency (ms)", 0.0, 50.0, float(fdf["Network_Latency_ms"].mean()))
            packet_loss = st.slider("Packet Loss (%)", 0.0, 5.0, float(fdf["Packet_Loss_%"].mean()))
        with c2:
            defect = st.slider("QC Defect Rate (%)", 0.0, 10.0, float(fdf["Quality_Control_Defect_Rate_%"].mean()))
            speed = st.slider("Production Speed (units/hr)", 0.0, 500.0, float(fdf["Production_Speed_units_per_hr"].mean()))
            maint = st.slider("Predictive Maintenance Score", 0.0, 1.0, float(fdf["Predictive_Maintenance_Score"].mean()))
            error = st.slider("Error Rate (%)", 0.0, 15.0, float(fdf["Error_Rate_%"].mean()))
            op_mode = st.selectbox("Operation Mode", options=mode_list)

        if st.button("🔮 Classify Efficiency", type="primary", width='stretch'):
            row = {c: 0 for c in feature_cols}
            row.update({
                "Temperature_C": temp, "Vibration_Hz": vib, "Power_Consumption_kW": power,
                "Network_Latency_ms": latency, "Packet_Loss_%": packet_loss,
                "Quality_Control_Defect_Rate_%": defect, "Production_Speed_units_per_hr": speed,
                "Predictive_Maintenance_Score": maint, "Error_Rate_%": error,
                "Hour": 12, "Is_Weekend": 0,
            })
            row["Sensor_Stability_Score"] = float(fdf["Sensor_Stability_Score"].mean())
            row["Energy_Efficiency_Ratio"] = speed / power if power > 0 else 0
            row["Error_to_Output_Ratio"] = (error / speed * 1000) if speed > 0 else 0
            row["Quality_Adjusted_Error"] = error + defect
            lat_min, lat_max = df["Network_Latency_ms"].min(), df["Network_Latency_ms"].max()
            loss_min, loss_max = df["Packet_Loss_%"].min(), df["Packet_Loss_%"].max()
            lat_norm = (latency - lat_min) / (lat_max - lat_min)
            loss_norm = (packet_loss - loss_min) / (loss_max - loss_min)
            row["Network_Reliability_Score"] = (1 - (0.5 * lat_norm + 0.5 * loss_norm)) * 100
            row[f"Mode_{op_mode}"] = 1

            row_df = pd.DataFrame([row])
            pred_label, proba_dict = predict_row(model_choice, row_df)

            st.session_state["last_pred"] = (pred_label, proba_dict)

        if "last_pred" in st.session_state:
            pred_label, proba_dict = st.session_state["last_pred"]
            st.markdown(
                f"### Predicted Efficiency: "
                f"<span style='color:{STATUS_COLORS[pred_label]}; font-size:1.5em;'>●</span> **{pred_label}**",
                unsafe_allow_html=True,
            )
            conf_fig = go.Figure(go.Bar(
                x=[proba_dict.get(s, 0) * 100 for s in STATUS_ORDER],
                y=STATUS_ORDER, orientation="h",
                marker_color=[STATUS_COLORS[s] for s in STATUS_ORDER],
                text=[f"{proba_dict.get(s, 0) * 100:.1f}%" for s in STATUS_ORDER],
                textposition="outside",
            ))
            conf_fig.update_layout(title="Prediction Confidence by Class", xaxis_title="Confidence (%)",
                                    height=280, margin=dict(l=10, r=10, t=40, b=10), xaxis_range=[0, 110])
            st.plotly_chart(conf_fig, width='stretch')

    with right:
        st.markdown("#### Confidence Distribution — Filtered Batch")
        st.caption(f"Model **{model_choice}** applied to a sample of the filtered dataset")

        batch = fdf.sample(n=min(1500, len(fdf)), random_state=1).copy()
        batch_enc = pd.get_dummies(batch, columns=["Operation_Mode"], prefix="Mode")
        for c in meta["mode_cols"]:
            if c not in batch_enc.columns:
                batch_enc[c] = 0
        Xb = batch_enc[feature_cols]
        model = models[model_choice]
        if model_choice == "Logistic Regression":
            Xb = Xb.copy()
            Xb[numeric_features] = scaler.transform(Xb[numeric_features])
        proba_b = model.predict_proba(Xb)
        pred_b = le.inverse_transform(np.argmax(proba_b, axis=1))
        batch["Predicted"] = pred_b
        batch["Confidence"] = proba_b.max(axis=1) * 100

        fig = px.histogram(batch, x="Confidence", color="Predicted", nbins=25,
                            color_discrete_map=STATUS_COLORS,
                            title="Prediction Confidence Histogram")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width='stretch')

        agree = (batch["Predicted"] == batch["Efficiency_Status"]).mean() * 100
        st.metric("Model agreement with logged status (sample)", f"{agree:.1f}%")

        st.markdown("##### Recent Classifications")
        show_cols = ["Datetime", "Machine_ID", "Operation_Mode", "Efficiency_Status", "Predicted", "Confidence"]
        st.dataframe(
            batch.sort_values("Datetime", ascending=False)[show_cols].head(15).style.format({"Confidence": "{:.1f}%"}),
            width='stretch', height=320
        )

# ==================================================================
# TAB 2 — Machine-Level Insights
# ==================================================================
with tab2:
    st.subheader("Machine-Level Efficiency Trends")

    machine_focus = st.multiselect("Focus machines (leave empty = all filtered machines)",
                                    options=selected_machines, default=selected_machines[:5]
                                    if len(selected_machines) > 5 else selected_machines)
    focus_df = fdf[fdf["Machine_ID"].isin(machine_focus)] if machine_focus else fdf

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown("##### Daily Efficiency Trend (per machine)")
        trend = focus_df.groupby([focus_df["Datetime"].dt.date, "Machine_ID"])["Efficiency_Status"].apply(
            lambda s: (s == "Low").mean() * 100).reset_index(name="Pct_Low")
        trend.columns = ["Date", "Machine_ID", "Pct_Low"]
        trend["Machine_ID"] = trend["Machine_ID"].astype(str)
        fig = px.line(trend, x="Date", y="Pct_Low", color="Machine_ID",
                       title="% Low-Efficiency Records per Day, by Machine")
        fig.update_layout(height=420)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("##### Machine Efficiency Ranking")
        rank = fdf.groupby("Machine_ID")["Efficiency_Status"].apply(
            lambda s: (s == "Low").mean() * 100).sort_values(ascending=False)
        rank_df = rank.reset_index()
        rank_df.columns = ["Machine_ID", "Pct_Low"]
        rank_df["Machine_ID"] = rank_df["Machine_ID"].astype(str)
        fig2 = px.bar(rank_df, x="Pct_Low", y="Machine_ID", orientation="h",
                       title="Share of 'Low' Records by Machine", height=420,
                       color="Pct_Low", color_continuous_scale="Reds")
        fig2.update_layout(yaxis=dict(tickfont=dict(size=8)))
        st.plotly_chart(fig2, width='stretch')

    st.markdown("##### Historical Classification Pattern (Machine × Efficiency Status)")
    hist_pattern = pd.crosstab(fdf["Machine_ID"], fdf["Efficiency_Status"])
    for s in STATUS_ORDER:
        if s not in hist_pattern.columns:
            hist_pattern[s] = 0
    hist_pattern = hist_pattern[STATUS_ORDER]
    hist_pattern_pct = hist_pattern.div(hist_pattern.sum(axis=1), axis=0) * 100
    fig3 = px.imshow(hist_pattern_pct.T, aspect="auto", color_continuous_scale="RdYlGn_r",
                      labels=dict(x="Machine ID", y="Efficiency Status", color="% of records"),
                      title="Historical Efficiency Classification Heatmap")
    st.plotly_chart(fig3, width='stretch')

    st.markdown("##### Machine Summary Table")
    summary_tbl = fdf.groupby("Machine_ID").agg(
        Records=("Efficiency_Status", "count"),
        Pct_Low=("Efficiency_Status", lambda s: (s == "Low").mean() * 100),
        Pct_Medium=("Efficiency_Status", lambda s: (s == "Medium").mean() * 100),
        Pct_High=("Efficiency_Status", lambda s: (s == "High").mean() * 100),
        Avg_Error_Rate=("Error_Rate_%", "mean"),
        Avg_Production_Speed=("Production_Speed_units_per_hr", "mean"),
        Avg_Maintenance_Score=("Predictive_Maintenance_Score", "mean"),
    ).round(2).sort_values("Pct_Low", ascending=False)
    st.dataframe(summary_tbl, width='stretch', height=350)

# ==================================================================
# TAB 3 — Explainability Panel
# ==================================================================
with tab3:
    st.subheader("Model Explainability")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Global Feature Importance (SHAP, mean |value|)")
        shap_top = shap_imp.sort_values(ascending=False).head(12)
        fig = px.bar(shap_top[::-1], orientation="h", title="Top Drivers of Efficiency Classification",
                     labels={"value": "mean |SHAP value|", "index": "Feature"})
        fig.update_layout(showlegend=False, height=430)
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.markdown("##### Model Comparison — Feature Importance")
        model_imp_choice = st.radio("Importance source", ["Random Forest", "XGBoost"], horizontal=True)
        imp_series = rf_imp if model_imp_choice == "Random Forest" else xgb_imp
        top = imp_series.sort_values(ascending=False).head(12)
        fig2 = px.bar(top[::-1], orientation="h", title=f"{model_imp_choice} Feature Importance",
                      labels={"value": "Importance", "index": "Feature"})
        fig2.update_layout(showlegend=False, height=430)
        st.plotly_chart(fig2, width='stretch')

    st.markdown("---")
    st.markdown("##### 🔎 Why did efficiency drop or improve? (Local explanation)")
    st.caption("Pick a record from the filtered dataset to see which features pushed the prediction toward its class.")

    sample_for_explain = fdf.sample(n=min(300, len(fdf)), random_state=7).reset_index()
    sample_for_explain["label"] = (
        sample_for_explain["Datetime"].astype(str) + " | Machine " + sample_for_explain["Machine_ID"].astype(str)
        + " | " + sample_for_explain["Efficiency_Status"]
    )
    choice = st.selectbox("Select a record", options=sample_for_explain["label"])
    row = sample_for_explain[sample_for_explain["label"] == choice].iloc[0]

    row_enc = pd.get_dummies(pd.DataFrame([row]), columns=["Operation_Mode"], prefix="Mode")
    for c in meta["mode_cols"]:
        if c not in row_enc.columns:
            row_enc[c] = 0
    pred_label, proba_dict = predict_row("XGBoost", row_enc)

    m1, m2, m3 = st.columns(3)
    m1.metric("Logged Efficiency Status", row["Efficiency_Status"])
    m2.metric("Model Prediction", pred_label)
    m3.metric("Confidence", f"{max(proba_dict.values()) * 100:.1f}%")

    explain_cols = ["Error_Rate_%", "Production_Speed_units_per_hr", "Error_to_Output_Ratio",
                     "Quality_Adjusted_Error", "Network_Reliability_Score", "Sensor_Stability_Score"]
    st.markdown("**Key metric values for this record vs. dataset average:**")
    comp = pd.DataFrame({
        "Metric": explain_cols,
        "This Record": [row[c] for c in explain_cols],
        "Dataset Average": [df[c].mean() for c in explain_cols],
    })
    comp["This Record"] = comp["This Record"].round(2)
    comp["Dataset Average"] = comp["Dataset Average"].round(2)
    st.dataframe(comp, width='stretch', hide_index=True)

    if row["Error_Rate_%"] > error_sensitivity:
        st.error(f"⚠️ Error Rate ({row['Error_Rate_%']:.2f}%) exceeds your alert threshold of {error_sensitivity}% — a primary driver of reduced efficiency.")
    if row["Production_Speed_units_per_hr"] < speed_sensitivity:
        st.error(f"⚠️ Production Speed ({row['Production_Speed_units_per_hr']:.1f} units/hr) is below your alert threshold of {speed_sensitivity} units/hr — a primary driver of reduced efficiency.")
    if row["Error_Rate_%"] <= error_sensitivity and row["Production_Speed_units_per_hr"] >= speed_sensitivity:
        st.success("✅ Both key drivers (error rate, production speed) are within healthy sensitivity thresholds.")

# ==================================================================
# TAB 4 — Operational Monitoring View
# ==================================================================
with tab4:
    st.subheader("Operational Monitoring")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Efficiency Status by Operation Mode")
        ct = pd.crosstab(fdf["Operation_Mode"], fdf["Efficiency_Status"], normalize="index") * 100
        for s in STATUS_ORDER:
            if s not in ct.columns:
                ct[s] = 0
        ct = ct[STATUS_ORDER].reset_index().melt(id_vars="Operation_Mode", var_name="Efficiency_Status", value_name="Pct")
        fig = px.bar(ct, x="Operation_Mode", y="Pct", color="Efficiency_Status", barmode="stack",
                     color_discrete_map=STATUS_COLORS, title="Efficiency Mix by Operation Mode (%)")
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("##### Network vs. Sensor vs. Production Impact")
        group_scores = {}
        for grp, feats in FEATURE_GROUPS.items():
            valid = [f for f in feats if f in shap_imp.index]
            group_scores[grp] = shap_imp[valid].sum()
        gdf = pd.DataFrame({"Group": list(group_scores.keys()), "Total |SHAP| Impact": list(group_scores.values())})
        fig2 = px.pie(gdf, names="Group", values="Total |SHAP| Impact", hole=0.45,
                      title="Relative Impact on Efficiency Classification",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2, width='stretch')
        st.caption("Production/quality signals (error rate, throughput) dominate the classification; "
                   "raw network and sensor telemetry contribute comparatively little in this dataset — "
                   "see Explainability Panel and the Research Paper for details.")

    st.markdown("---")
    st.markdown("##### Network Reliability vs. Efficiency Status")
    fig3 = px.box(fdf, x="Efficiency_Status", y="Network_Reliability_Score", color="Efficiency_Status",
                  category_orders={"Efficiency_Status": STATUS_ORDER}, color_discrete_map=STATUS_COLORS,
                  title="Network Reliability Score Distribution by Efficiency Class")
    st.plotly_chart(fig3, width='stretch')

    st.markdown("##### At-Risk Records (based on your sensitivity thresholds)")
    at_risk = fdf[(fdf["Error_Rate_%"] > error_sensitivity) | (fdf["Production_Speed_units_per_hr"] < speed_sensitivity)]
    st.metric("Records flagged at-risk", f"{len(at_risk):,} ({len(at_risk) / len(fdf) * 100:.1f}% of filtered data)")
    st.dataframe(
        at_risk.sort_values("Datetime", ascending=False)[
            ["Datetime", "Machine_ID", "Operation_Mode", "Error_Rate_%",
             "Production_Speed_units_per_hr", "Efficiency_Status"]
        ].head(20),
        width='stretch'
    )

    st.markdown("---")
    st.subheader("Model Performance Summary")
    perf_rows = []
    for name, r in model_results.items():
        perf_rows.append({
            "Model": name, "Accuracy": r["accuracy"], "Macro F1": r["f1_macro"],
            "Weighted F1": r["f1_weighted"], "CV Std (stability)": r["cv_accuracy_std"],
        })
    perf_df = pd.DataFrame(perf_rows).round(4)
    st.dataframe(perf_df, width='stretch', hide_index=True)

st.markdown("---")
st.caption("Prototype dashboard built for the Thales Group Smart Manufacturing AI project · "
            "Unified Mentor Program · For demonstration and evaluation purposes.")
