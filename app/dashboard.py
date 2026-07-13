import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import joblib

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Bitcoin GenAI Predictor",
    page_icon="₿",
    layout="wide"
)

st.title("₿ Bitcoin Price Predictor Dashboard")
st.markdown("Live BTC price data with AI-powered price predictions")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Settings")
period = st.sidebar.selectbox(
    "Historical Data Range",
    ["7d", "1mo", "3mo", "6mo", "1y", "2y"],
    index=2
)
interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1h", "15m"],
    index=0
)

# ---------------------------
# Load BTC Data
# ---------------------------
@st.cache_data(ttl=300)
def load_data(period, interval):
    df = yf.download("BTC-USD", period=period, interval=interval)
    df.reset_index(inplace=True)
    return df

with st.spinner("Fetching Bitcoin data..."):
    data = load_data(period, interval)

if data.empty:
    st.error("Could not fetch data. Try a different range/interval.")
    st.stop()

date_col = "Date" if "Date" in data.columns else "Datetime"

# ---------------------------
# Key Metrics
# ---------------------------
latest_price = float(data["Close"].iloc[-1])
prev_price = float(data["Close"].iloc[-2])
change = latest_price - prev_price
pct_change = (change / prev_price) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${latest_price:,.2f}", f"{pct_change:.2f}%")
col2.metric("24h High", f"${data['High'].iloc[-1]:,.2f}")
col3.metric("24h Low", f"${data['Low'].iloc[-1]:,.2f}")

# ---------------------------
# Price Chart
# ---------------------------
st.subheader("Price History")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data[date_col],
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="BTC-USD"
))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_dark",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Model Prediction Section
# ---------------------------
st.subheader("🤖 AI Prediction")

MODEL_PATH = os.path.join("models", "btc_model.pkl")

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.warning(
        f"No trained model found at `{MODEL_PATH}`. "
        "Train and save a model there, then reload this dashboard."
    )
else:
    # Example feature engineering — adjust to match your model's training features
    features = pd.DataFrame({
        "close": [latest_price],
        "high": [data["High"].iloc[-1]],
        "low": [data["Low"].iloc[-1]],
        "volume": [data["Volume"].iloc[-1]] if "Volume" in data.columns else [0],
    })

    prediction = model.predict(features)[0]
    direction = "📈 Up" if prediction > latest_price else "📉 Down"

    pcol1, pcol2 = st.columns(2)
    pcol1.metric("Predicted Next Price", f"${prediction:,.2f}", direction)
    pcol2.metric("Expected Change", f"{((prediction - latest_price)/latest_price)*100:.2f}%")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------
# Raw Data Table
# ---------------------------
with st.expander("View Raw Data"):
    st.dataframe(data.tail(50), use_container_width=True)
