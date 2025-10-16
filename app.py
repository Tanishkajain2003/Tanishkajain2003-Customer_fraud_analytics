# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime, timedelta

# ---------- Config ----------
st.set_page_config(page_title="Customer Transaction Data Quality & Fraud Analytics",
                   page_icon="💳", layout="wide")

# ---------- Robust path resolving ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
REPORT_PATH = os.path.join(BASE_DIR, "data", "reports", "fraud_predictions.csv")

# ---------- Helper: create small dummy dataset if file missing ----------
def create_dummy(path, n=10000, fraud_rate=0.07):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(42)
    timestamps = [datetime.now() - timedelta(days=int(x)) for x in rng.integers(0, 180, n)]
    merchant_types = ["Grocery", "Electronics", "Travel", "Dining", "Fuel", "Healthcare", "Utilities", "Entertainment"]
    channels = ["POS", "Online", "ATM", "Mobile"]
    locations = list(range(8))  # numeric location codes 0–7
    df = pd.DataFrame({
        "transaction_id": [f"TXN_{i+1}" for i in range(n)],
        "customer_id": [f"CUST{rng.integers(1000,9999)}" for _ in range(n)],
        "timestamp": timestamps,
        "amount": np.round(np.clip(rng.exponential(100.0, n), 0.5, 20000), 2),
        "merchant_type": rng.choice(merchant_types, n),
        "location": rng.choice(locations, n),
        "device_id": [f"DEV{rng.integers(10000,99999)}" for _ in range(n)],
        "channel": rng.choice(channels, n),
        "is_international": rng.choice([0,1], n, p=[0.95,0.05]),
        "previous_transactions": rng.poisson(2, n),
        "avg_transaction_value": np.round(np.clip(rng.normal(50, 20, n), 1.0, None), 2),
    })
    # label fraud intelligently for demo
    score = (df["amount"] / (df["avg_transaction_value"] + 1)) + df["is_international"]*2
    probs = (score - score.min()) / (score.max() - score.min() + 1e-9)
    probs = probs + rng.random(n)*0.3
    probs = probs / probs.sum()
    n_fraud = int(n * fraud_rate)
    fraud_idx = rng.choice(df.index, size=n_fraud, replace=False, p=probs)
    df["is_fraud"] = 0
    df.loc[fraud_idx, "is_fraud"] = 1
    df["fraud_score"] = (0.6 * (df["is_fraud"]) + 0.4 * (probs / probs.max())).clip(0,1)
    df.to_csv(path, index=False)
    return df

# ---------- Load data (robust) ----------
if os.path.exists(REPORT_PATH):
    try:
        df_all = pd.read_csv(REPORT_PATH, parse_dates=["timestamp"], low_memory=False)
    except Exception as e:
        st.warning("Could not read fraud_predictions.csv (will recreate dummy). Error: " + str(e))
        df_all = create_dummy(REPORT_PATH, n=100000, fraud_rate=0.07)
else:
    st.info("No fraud_predictions.csv found — creating a realistic dummy dataset for demo.")
    df_all = create_dummy(REPORT_PATH, n=100000, fraud_rate=0.07)

# ---------- ✅ Mapping for location, channel, merchant ----------
location_map = {
    0: "Mumbai",
    1: "Delhi",
    2: "Bangalore",
    3: "Hyderabad",
    4: "Chennai",
    5: "Kolkata",
    6: "Pune",
    7: "Ahmedabad"
}
channel_map = {
    0: "POS",
    1: "Online",
    2: "ATM",
    3: "Mobile"
}
merchant_map = {
    0: "Grocery",
    1: "Electronics",
    2: "Travel",
    3: "Dining",
    4: "Fuel",
    5: "Healthcare",
    6: "Utilities",
    7: "Entertainment"
}

# If numeric codes exist, map them back
if df_all["location"].dtype != "O":
    df_all["location"] = df_all["location"].map(location_map).fillna(df_all["location"])
if df_all["channel"].dtype != "O":
    df_all["channel"] = df_all["channel"].map(channel_map).fillna(df_all["channel"])
if df_all["merchant_type"].dtype != "O":
    df_all["merchant_type"] = df_all["merchant_type"].map(merchant_map).fillna(df_all["merchant_type"])

# ---------- Ensure required columns ----------
for c in ["fraud_score", "is_fraud", "amount", "merchant_type", "channel", "location", "timestamp"]:
    if c not in df_all.columns:
        st.error(f"Required column missing from data: {c}")
        st.stop()

# ---------- Normalize fields ----------
df_all["merchant_type"] = df_all["merchant_type"].astype(str)
df_all["channel"] = df_all["channel"].astype(str)
df_all["location"] = df_all["location"].astype(str)
df_all["is_fraud"] = df_all["is_fraud"].astype(int)
df_all["fraud_score"] = pd.to_numeric(df_all["fraud_score"], errors="coerce").fillna(0.0)
df_all["amount"] = pd.to_numeric(df_all["amount"], errors="coerce").fillna(0.0)
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")

# ---------- Streamlit UI styling ----------
st.markdown("""
    <style>
    .stButton>button, .stTextInput>div>input, .stDateInput>div>div>input, .stSlider>div input {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    #MainMenu, header, footer { visibility: visible !important; display:block !important; }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.image(r"C:\Users\TANISHKA\OneDrive\Documents\customer_fraud_analytics\Icon\3D_data_analysis_illustration-removebg-preview.png", width=180)
st.sidebar.markdown("### 🎛 Control Panel")

min_score, max_score = st.sidebar.slider("Select Fraud Score Range", 0.0, 1.0, (0.0, 1.0), 0.01)
min_ts = df_all["timestamp"].min().date() if not df_all["timestamp"].isna().all() else datetime.now().date()
max_ts = df_all["timestamp"].max().date() if not df_all["timestamp"].isna().all() else datetime.now().date()
start_date = st.sidebar.date_input("Start date", min_ts)
end_date = st.sidebar.date_input("End date", max_ts)

# ---------- Filter data ----------
mask = (df_all["fraud_score"] >= min_score) & (df_all["fraud_score"] <= max_score)
if "timestamp" in df_all.columns:
    mask = mask & (df_all["timestamp"].dt.date >= start_date) & (df_all["timestamp"].dt.date <= end_date)
filtered = df_all[mask].copy()
if filtered.shape[0] == 0:
    st.warning("No data matches your filters. Showing sample instead.")
    filtered = df_all.sample(min(500, len(df_all)), random_state=42)

# ---------- Header ----------
st.markdown("""
<div style="
  background: linear-gradient(135deg, #1E1E2F, #2B2D42, #3A0CA3);
  padding:30px;
  border-radius:16px;
  color:white;
  text-align:center;
  font-size:34px;
  font-weight:800;
  box-shadow: 0 6px 25px rgba(0,0,0,0.25);
">
💳 Customer Transaction Data Quality & Fraud Analytics
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align:center;color:#AEE1F9;font-weight:600;font-size:18px;">
Empowering Financial Trust with Data, Insights, and AI ⚡
</p>
""", unsafe_allow_html=True)

st.write("---")

# ---------- KPIs ----------
total_txns = len(filtered)
fraud_txns = filtered["is_fraud"].sum()
fraud_rate = (fraud_txns / total_txns * 100) if total_txns > 0 else 0.0
avg_amount = filtered["amount"].mean()

c1, c2, c3 = st.columns(3)
c1.metric("💰 Total Transactions", f"{total_txns:,}")
c2.metric("🚨 Fraudulent Transactions", f"{fraud_txns:,}", delta=f"{fraud_rate:.2f}%")
c3.metric("📈 Average Amount", f"₹{avg_amount:,.2f}")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Fraud Detection", "🧹 Data Quality", "🌍 Insights", "📑 Reports"])

# ---------- Tab 1 ----------
with tab1:
    st.header("🚨 Fraud Detection Insights")
    fig = px.histogram(filtered, x="fraud_score", color="is_fraud", nbins=40,
                       color_discrete_map={0: "#4895EF", 1: "#F72585"},
                       title="Fraud Score Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 🔎 Top flagged transactions (by fraud_score)")
    top = filtered.sort_values("fraud_score", ascending=False).head(20)
    st.dataframe(top[["transaction_id", "customer_id", "timestamp", "amount", "location", "fraud_score", "is_fraud"]],
                 use_container_width=True)

# ---------- Tab 2 ----------
with tab2:
    st.header("🧹 Data Quality Checks")
    missing_total = filtered.isnull().sum().sum()
    dup_count = filtered.duplicated(subset=["transaction_id"]).sum()
    invalid_amounts = (filtered["amount"] <= 0).sum()

    dq_df = pd.DataFrame({
        "Quality Check": ["Missing Values", "Duplicate Rows", "Invalid Amounts"],
        "Count": [int(missing_total), int(dup_count), int(invalid_amounts)]
    })

    # Always show bars even if all counts are zero
    fig2 = px.bar(
        dq_df,
        x="Quality Check",
        y="Count",
        text="Count",
        title="Data Quality Checks"
    )
    fig2.update_traces(texttemplate='%{text}', textposition='outside')
    fig2.update_layout(yaxis_range=[0, max(dq_df["Count"].max(), 1)])
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Data sample (cleaned / filtered)")
    st.dataframe(filtered.head(200), use_container_width=True)

# ---------- Tab 3 ----------
with tab3:
    st.header("🌍 Transaction Insights")
    fig3 = px.box(filtered, x="merchant_type", y="amount", color="is_fraud")
    st.plotly_chart(fig3, use_container_width=True)
    fig4 = px.pie(filtered, names="channel", title="Transaction Channel Distribution")
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("### 📍 Geographic Trends")
    loc_counts = filtered.groupby("location").size().reset_index(name="count")
    fig5 = px.bar(loc_counts, x="location", y="count", title="Fraud Cases by Location")
    st.plotly_chart(fig5, use_container_width=True)

# ---------- Tab 4 ----------
with tab4:
    st.header("📑 Data Lineage & Pipeline")
    st.markdown("**Data Flow:**  🗂️ Raw → 🧹 Cleaned → 🤖 Model Scoring → 📊 Reports → 🌐 Dashboard")

    st.markdown("### 🔄 End-to-End Data Flow")
    st.markdown("""
    **1️⃣ Raw Data (data/raw/)** → Collected from transaction systems, APIs, or CSVs  
    **2️⃣ Data Cleaning (data/processed/)** → Missing values, duplicates, invalid formats fixed  
    **3️⃣ Feature Engineering & Fraud Model (src/fraud_detection.py)** →  
       Encodes categorical data, trains Random Forest, predicts `fraud_score`  
    **4️⃣ Reports (data/reports/)** → Final predictions, risk scores, fraud probabilities  
    **5️⃣ Dashboard (dashboard/app.py)** → Interactive visual insights built with Streamlit  
    """)

    st.markdown("### 📊 Summary Statistics")
    st.dataframe(filtered.describe(include='all').T, use_container_width=True)

    st.markdown("### 🧾 Model Performance (From Training Report)")
    st.info("""
    - **Algorithm Used:** Random Forest Classifier  
    - **Evaluation Metrics:** Precision, Recall, F1-score, Accuracy  
    - **Model Goal:** Identify fraudulent transactions while minimizing false positives  
    - **Feature Inputs:** Transaction Amount, Channel, Merchant Type, and Location  
    """)

    st.markdown("### 🧠 Model Insights")
    st.markdown("""
    ✅ High fraud likelihood often corresponds to:  
    - Unusually large transaction amounts  
    - New merchant types not seen in normal history  
    - High-value transactions from international locations  
    - Repeated small test transactions (fraud testing behavior)  
    """)

    st.markdown("### 💡 Next Steps & Recommendations")
    st.success("""
    - Integrate **real-time fraud alerts** via APIs or webhooks  
    - Add **transaction velocity features** (e.g., number of transactions per minute)  
    - Enhance model using **Gradient Boosting (XGBoost, LightGBM)**  
    - Deploy automated data validation with **Great Expectations / Pandera**  
    - Combine with **Tableau or Power BI** for enterprise-ready dashboards  
    """)
