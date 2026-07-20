# AI-Based Manufacturing Efficiency Classification
### Thales Group — Sensor, Production & 6G Network Data | Unified Mentor Program

This package contains the complete deliverable set for the project: an AI system
that classifies real-time manufacturing efficiency (High / Medium / Low) from
sensor, production, quality, and 6G network telemetry, plus an interactive
Streamlit dashboard for plant engineers.

---

## 📁 Folder Guide

| Folder | Contents |
|---|---|
| **01_Reports/** | The two written deliverables: the full **Research Paper** (.docx) and the **Executive Summary** (.docx) for government/leadership stakeholders. |
| **02_Streamlit_Dashboard/** | The **ready-to-run** dashboard app, with its own `data/`, `models/`, and `outputs/` folders already in place. See "Running the Dashboard" below. |
| **03_Data/** | The original dataset (`Thales_Group_Manufacturing.csv`) and the engineered dataset (`manufacturing_features.parquet`) used for modeling. |
| **04_Source_Code/** | The full, numbered analysis pipeline as standalone Python scripts (EDA → feature engineering → modeling → explainability), plus a copy of the dashboard source. Re-running these regenerates everything in this package from scratch. |
| **05_Figures/** | All 12 charts generated during the analysis, in PNG format (also embedded in the Research Paper). |

---

## 🚀 Running the Dashboard

```bash
cd 02_Streamlit_Dashboard
pip install -r requirements.txt
streamlit run app/app.py
```

The app opens in your browser (default `http://localhost:8501`) and includes four modules:

1. **Efficiency Prediction** — live "what-if" classification with confidence scores
2. **Machine-Level Insights** — per-machine efficiency trends and rankings
3. **Explainability Panel** — global feature importance (SHAP) and per-record local explanations
4. **Operational Monitoring** — fleet-wide health, network-vs-sensor impact, at-risk record alerts

Sidebar controls: machine selector, operation-mode dropdown, time-window filter,
network-quality filter, and metric-sensitivity sliders for the error-rate and
production-speed alert thresholds.

### ⚠️ Important: install the exact pinned versions

`requirements.txt` pins **exact** package versions (`==`), not minimums. This
matters more than usual here because the trained models are serialized files:

- The Random Forest and Logistic Regression models are `scikit-learn` pickles
  (`.joblib`) — these are only guaranteed to load correctly with the same
  scikit-learn version they were trained with.
- The XGBoost model is saved in XGBoost's **native JSON format**
  (`xgboost_model.json`), which is designed to be portable across XGBoost
  versions — but the `xgboost` Python package itself still needs to be
  installed to load and run it.

If you install these packages with unpinned/mismatched versions instead of
running `pip install -r requirements.txt` as-is, you may see errors such as
`XGBoostError: input stream corrupted` (this specifically happens if a model
was pickled with `joblib`/`pickle` rather than loaded via `load_model()` —
already fixed in this package) or a `scikit-learn` unpickling error. If you
ever see a serialization error, the fix is always the same: create a fresh
virtual environment and run `pip install -r requirements.txt` exactly as
provided, then re-launch.

```bash
python3 -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
streamlit run app/app.py

---

## 🔁 Reproducing the Full Analysis

The scripts in `04_Source_Code/` are numbered in run order:

```bash
cd 04_Source_Code
pip install -r requirements.txt
python 01_eda.py               # exploratory analysis + figures 1-7
python 02_feature_engineering.py   # builds engineered features
python 03_modeling.py          # trains LR / Random Forest / XGBoost + figures 8-10
python 04_explainability.py    # SHAP analysis + figures 11-12
streamlit run 05_streamlit_app.py  # launch the dashboard
```

Each script reads from / writes to `data/`, `models/`, `outputs/`, and `figures/`
folders that it creates relative to wherever it's run from — point it at
`03_Data/Thales_Group_Manufacturing.csv` as the source file if re-running
outside the original project layout.

---

## 📊 Headline Results

| Model | Accuracy | Macro F1 | 5-fold CV Stability (std) |
|---|---|---|---|
| Logistic Regression (baseline) | 88.15% | 82.01% | ± 0.36% |
| Random Forest | 100.00% | 100.00% | ± 0.01% |
| **XGBoost (deployed in dashboard)** | 99.76% | 99.68% | ± 0.04% |

**Key insight:** Efficiency status is driven almost entirely by two signals —
`Error_Rate_%` and `Production_Speed_units_per_hr` — while ambient sensor and
6G network telemetry contribute negligibly to the classification. Full detail,
methodology, and business recommendations are in the Research Paper and
Executive Summary in `01_Reports/`.

---

## 🛠 Tech Stack

Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit · Plotly · matplotlib/seaborn
