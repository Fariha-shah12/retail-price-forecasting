import streamlit as st
import pandas as pd
import boto3
import io
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Retail Price Forecast",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; }
    .metric-card {
        background: #111827;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #00d4ff;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff, #7b2fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

S3_BUCKET       = os.environ.get("S3_BUCKET", "walmart-scraper-data")
FORECAST_PREFIX = "forecasts/"

# ── Load forecast from S3 ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_latest_forecast() -> pd.DataFrame:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=FORECAST_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".csv"):
                keys.append(obj["Key"])
    if not keys:
        return pd.DataFrame()
    latest_key = sorted(keys)[-1]
    body = s3.get_object(Bucket=S3_BUCKET, Key=latest_key)["Body"].read()
    df   = pd.read_csv(io.BytesIO(body))
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_sample_data() -> pd.DataFrame:
    """Sample data for demo mode."""
    import numpy as np
    np.random.seed(42)
    dates = pd.date_range("2026-02-17", periods=85, freq="D")
    rows  = []
    base  = {"eggs": 4.5, "olive_oil": 8.2, "paper_towels": 12.5}
    for retailer in ["walmart", "kroger"]:
        for product, bp in base.items():
            if retailer == "walmart" and dates[0] < pd.Timestamp("2026-03-01"):
                start_idx = 12
            else:
                start_idx = 0
            for i, d in enumerate(dates[start_idx:]):
                actual = bp + np.random.normal(0, 0.15) + 0.002 * i
                forecast = actual + np.random.normal(0, 0.1)
                is_future = i >= (len(dates) - start_idx - 14)
                rows.append({
                    "date": d, "retailer": retailer, "product": product,
                    "actual":        np.nan if is_future else round(actual, 2),
                    "forecast":      round(forecast, 2),
                    "forecast_low":  round(forecast * 0.97, 2),
                    "forecast_high": round(forecast * 1.03, 2),
                    "mae":           0.08, "mape": 1.9
                })
    return pd.DataFrame(rows)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📈 Retail Price Forecast</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#6b7a99">14-day price forecasts for Walmart & Kroger · Powered by Facebook Prophet + AWS</p>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filters")
    use_demo = st.toggle("Demo mode", value=True)

    if use_demo:
        df = load_sample_data()
    else:
        df = load_latest_forecast()
        if df.empty:
            st.error("No forecast found in S3!")
            st.stop()

    products  = st.multiselect("Products",
                    options=sorted(df["product"].unique()),
                    default=sorted(df["product"].unique())[:2])
    retailers = st.multiselect("Retailers",
                    options=sorted(df["retailer"].unique()),
                    default=sorted(df["retailer"].unique()))
    st.divider()
    st.markdown("**Stack**")
    for s in ["AWS Lambda", "Amazon S3", "EventBridge",
              "Facebook Prophet", "Plotly", "Streamlit"]:
        st.markdown(f"- {s}")

if not products or not retailers:
    st.warning("Select at least one product and retailer.")
    st.stop()

filtered = df[df["product"].isin(products) & df["retailer"].isin(retailers)]
today = str((filtered["date"].max() - timedelta(days=14)).date())

# ── Summary metrics ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
hist = filtered[filtered["actual"].notna()]
future = filtered[filtered["actual"].isna()]

col1.metric("Training Days",   hist["date"].nunique())
col2.metric("Forecast Days",   future["date"].nunique())
col3.metric("Avg MAE",         f"${filtered['mae'].mean():.3f}")
col4.metric("Avg MAPE",        f"{filtered['mape'].mean():.1f}%")

st.divider()

# ── Price forecast charts ─────────────────────────────────────────────────────
st.subheader("📊 Price Forecasts")

for product in products:
    st.markdown(f"#### {product.replace('_', ' ').title()}")
    cols = st.columns(len(retailers))

    for ci, retailer in enumerate(retailers):
        subset = filtered[(filtered["product"] == product) &
                          (filtered["retailer"] == retailer)]
        if subset.empty:
            cols[ci].info(f"No data for {retailer}")
            continue

        fig = go.Figure()

        # Confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([subset["date"], subset["date"][::-1]]),
            y=pd.concat([subset["forecast_high"], subset["forecast_low"][::-1]]),
            fill="toself",
            fillcolor="rgba(0,212,255,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% CI",
            showlegend=True
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=subset["date"], y=subset["forecast"],
            mode="lines",
            name="Forecast",
            line=dict(color="#00d4ff", width=2, dash="dot")
        ))

        # Actual line
        hist_subset = subset[subset["actual"].notna()]
        fig.add_trace(go.Scatter(
            x=hist_subset["date"], y=hist_subset["actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#7b2fff", width=2),
            marker=dict(size=4)
        ))

        # Vertical line for today
        
        # Vertical line for today
        fig.add_trace(go.Scatter(
            x=[today, today],
            y=[subset["forecast_low"].min(), subset["forecast_high"].max()],
            mode="lines",
            name="Today",
            line=dict(color="#ff6b6b", width=1, dash="dash"),
            showlegend=False
        ))

        fig.update_layout(
            title=f"{retailer.title()} — {product.replace('_',' ').title()}",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="#c8d6f0"),
            xaxis=dict(gridcolor="#1e2a45"),
            yaxis=dict(gridcolor="#1e2a45",
                       tickprefix="$",
                       title="Price"),
            legend=dict(bgcolor="#111827"),
            height=300,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        cols[ci].plotly_chart(fig, use_container_width=True)

# ── Price gap comparison ──────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Walmart vs Kroger Price Gap")

if len(retailers) == 2:
    for product in products:
        w = filtered[(filtered["product"] == product) &
                     (filtered["retailer"] == "walmart")][["date","forecast"]].rename(
                         columns={"forecast": "walmart"})
        k = filtered[(filtered["product"] == product) &
                     (filtered["retailer"] == "kroger")][["date","forecast"]].rename(
                         columns={"forecast": "kroger"})
        gap = w.merge(k, on="date")
        gap["gap"] = gap["walmart"] - gap["kroger"]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=gap["date"], y=gap["gap"],
            marker_color=["#ff6b6b" if g > 0 else "#00d4ff" for g in gap["gap"]],
            name="Walmart - Kroger"
        ))
        fig2.add_hline(y=0, line_color="#ffffff", opacity=0.3)
        fig2.update_layout(
            title=f"{product.replace('_',' ').title()} — Price Gap (Walmart minus Kroger)",
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font=dict(color="#c8d6f0"),
            xaxis=dict(gridcolor="#1e2a45"),
            yaxis=dict(gridcolor="#1e2a45", tickprefix="$"),
            height=250,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Select both Walmart and Kroger to see price gap chart.")

# ── Model accuracy table ──────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Model Accuracy")
accuracy = (
    filtered.groupby(["retailer", "product"])
    [["mae", "mape"]].first().reset_index()
)
accuracy.columns = ["Retailer", "Product", "MAE ($)", "MAPE (%)"]
accuracy["MAE ($)"]  = accuracy["MAE ($)"].round(3)
accuracy["MAPE (%)"] = accuracy["MAPE (%)"].round(1)
st.dataframe(accuracy, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("AWS Lambda · S3 · EventBridge · Facebook Prophet · Plotly · Streamlit")
