import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Seed Demand Forecasting Demo",
    page_icon="🌱",
    layout="centered",
)

# ---------- load model artifacts ----------
@st.cache_resource
def load_artifacts():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("product_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("product_list.pkl", "rb") as f:
        products = pickle.load(f)
    history = pd.read_csv("synthetic_demo_data.csv")
    history["Month"] = pd.to_datetime(history["Month"])
    return model, encoder, products, history

model, encoder, products, history = load_artifacts()

# ---------- header / business context ----------
st.title("🌱 Seed Demand Forecasting — Interactive Demo")

st.markdown(
    """
This demo recreates, on **synthetic data**, the core interface my team proposed for a
real seed-company client during my graduate Business Analytics capstone.

**The business problem:** the client's packaging and procurement teams were over-ordering
seed stock, especially after demand swings during the pandemic. Overbuying was far costlier
than underbuying — excess inventory tied up warehouse space and went unsold — but the team had
no simple way to see, by product, what demand was likely to look like next season. The client
asked specifically for forecasts broken down **by product and package size** (small packets for
home gardeners vs. bulk sizes for professional growers), delivered through **a simple,
user-friendly interface** that non-technical staff could use directly — not a notebook or a
static report.

This app is that interface, rebuilt end-to-end on made-up data so it's safe to share publicly:
pick a product, pick a month, and see the model's predicted order quantity alongside its
historical pattern.

*Note: all product names and quantities below are synthetic and do not represent any real
company's data.*
"""
)

st.divider()

# ---------- controls ----------
col1, col2 = st.columns(2)
with col1:
    product = st.selectbox("Product", products)
with col2:
    month_label = st.selectbox(
        "Forecast month",
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
    )
    month_num = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ].index(month_label) + 1

year = st.slider("Forecast year", min_value=2024, max_value=2027, value=2027)

# ---------- predict ----------
product_enc = encoder.transform([product])[0]
month_sin = np.sin(2 * np.pi * month_num / 12)
month_cos = np.cos(2 * np.pi * month_num / 12)

X_pred = pd.DataFrame(
    [[product_enc, month_num, month_sin, month_cos, year]],
    columns=["product_enc", "month_num", "month_sin", "month_cos", "year"],
)
pred = model.predict(X_pred)[0]
pred = max(pred, 0)

st.subheader("Predicted order quantity")
st.metric(label=f"{product} — {month_label} {year}", value=f"{pred:,.0f} units")

# ---------- chart: historical + predicted point ----------
hist_series = history[["Month", product]].rename(columns={product: "quantity"})

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=hist_series["Month"], y=hist_series["quantity"],
        mode="lines+markers", name="Historical (synthetic)",
        line=dict(color="#2E7D32"),
    )
)
fig.add_trace(
    go.Scatter(
        x=[pd.Timestamp(year=year, month=month_num, day=1)], y=[pred],
        mode="markers", name="Predicted",
        marker=dict(color="#E65100", size=14, symbol="star"),
    )
)
fig.update_layout(
    xaxis_title="Month", yaxis_title="Quantity",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("How this model works"):
    st.markdown(
        """
- **Model:** XGBoost regressor trained on seasonal month-level features (month, sine/cosine
  of month for cyclical seasonality, year, and product).
- **Why XGBoost:** it handled the seasonal curve well without overfitting on a short synthetic
  history, and it's fast enough to retrain live in a free-tier app.
- **Evaluation:** on a held-out 20% test split, this model's mean absolute error is
  **~4.5 units** against a mean monthly order quantity of ~23 units — roughly 20% relative error.
- **Data:** this app uses fully synthetic data (2024–2026) generated to mimic realistic seasonal
  shapes, not any real company's historical records.
"""
    )

st.caption(
    "Built by Rongjiali Ding as a public, privacy-safe recreation of a graduate capstone "
    "forecasting project. All data shown is synthetic."
)
