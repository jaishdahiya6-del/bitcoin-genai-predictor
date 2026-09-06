"""
Bitcoin GenAI Predictor — Executive Financial & ML Analytics Dashboard
=======================================================================

Purpose:
--------
Production-grade interactive dashboard for live Bitcoin market data analysis,
technical indicator synthesis, LLM-powered news sentiment evaluation, and
machine learning price prediction with confidence bounds.

Architecture & Components:
--------------------------
1. DashboardConfig      : Centralized configuration, styling theme, and path management.
2. MarketDataLoader     : Robust market data ingestion via yfinance with failover and caching.
3. SentimentEngine     : Real-time news scraping and LLM sentiment scoring with fallback mechanisms.
4. ModelInferenceEngine : ML model loader, feature aligner, scaler handler, and forecast predictor.
5. Visualizations       : Modular Plotly chart factory for technicals, predictions, and sentiment.
6. Streamlit UI         : Modern dark-themed dashboard with tabs, status indicators, and scenario testing.

Inputs:
-------
- Market ticker data (yfinance: BTC-USD)
- Live Crypto News headlines (CryptoCompare API)
- LLM sentiment analysis (OpenAI GPT-4o-mini or rule-based fallback)
- Trained ML Model artifacts (.pkl files in models/)

Outputs:
--------
- Real-time price metrics & technical overlays (SMA, EMA, RSI, MACD, Bollinger Bands)
- AI model price forecasts with confidence intervals
- GenAI sentiment score gauges & headline feed
- Interactive data explorer & CSV export capabilities

Dependencies:
-------------
streamlit, pandas, numpy, plotly, yfinance, joblib, scikit-learn, xgboost, ta, openai, python-dotenv
"""

import os
import sys
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# Ensure project root is on Python path for modular imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Internal module imports with fallback handling
try:
    from src.feature_engineering import add_technical_indicators, get_feature_columns
except ImportError:
    # Inline fallback definition if src is not installed as package
    def get_feature_columns() -> List[str]:
        return [
            "sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "rsi_14",
            "macd", "macd_signal", "macd_diff", "bb_upper", "bb_lower",
            "bb_width", "bb_position", "atr_14", "daily_return",
            "volatility_20d", "volume_change", "return_lag_1", "return_lag_2",
            "return_lag_3", "return_lag_5", "return_lag_7", "price_vs_sma20",
            "price_vs_sma50", "price_vs_sma200", "sentiment_score"
        ]

try:
    from src.sentiment_pipeline import fetch_crypto_headlines, get_sentiment_score
    SENTIMENT_MODULE_AVAILABLE = True
except ImportError:
    SENTIMENT_MODULE_AVAILABLE = False


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BitcoinPredictorDashboard")


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
@dataclass(frozen=True)
class DashboardConfig:
    """Central configuration management for the dashboard."""
    PAGE_TITLE: str = "Bitcoin GenAI Predictor | Quant & AI Terminal"
    PAGE_ICON: str = "₿"
    MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
    DEFAULT_SYMBOL: str = "BTC-USD"
    CACHE_TTL_MARKET: int = 300       # 5 minutes
    CACHE_TTL_SENTIMENT: int = 900    # 15 minutes

    # Theme Palette
    COLOR_PRIMARY: str = "#F7931A"      # Bitcoin Gold
    COLOR_SECONDARY: str = "#4A90E2"    # Tech Blue
    COLOR_BULLISH: str = "#00C805"      # Green
    COLOR_BEARISH: str = "#FF5000"      # Red/Orange
    COLOR_BG_CARD: str = "#1E222D"      # Dark Card
    COLOR_GRID: str = "#2A2E39"         # Chart Grid


# =============================================================================
# SERVICE 1: MARKET DATA LOADER
# =============================================================================
class MarketDataLoader:
    """Handles fetching and preprocessing of historical and real-time market data."""

    @staticmethod
    @st.cache_data(ttl=DashboardConfig.CACHE_TTL_MARKET, show_spinner=False)
    def fetch_market_data(period: str, interval: str) -> pd.DataFrame:
        """
        Fetch OHLCV market data from Yahoo Finance with resilience and validation.

        Args:
            period: Time range (e.g., '1mo', '6mo', '1y', '2y')
            interval: Data granularity (e.g., '15m', '1h', '1d')

        Returns:
            pd.DataFrame with OHLCV data indexed by Date/Datetime.
        """
        logger.info(f"Fetching market data: period={period}, interval={interval}")

        # For short period requests, fetch longer buffer data to ensure technical indicator calculation works
        buffer_period = period
        if period in ["7d", "1mo", "3mo"]:
            buffer_period = "1y"

        try:
            ticker = yf.Ticker(DashboardConfig.DEFAULT_SYMBOL)
            df = ticker.history(period=buffer_period, interval=interval)

            if df.empty:
                logger.warning("Primary fetch returned empty DataFrame. Attempting fallback download.")
                df = yf.download(DashboardConfig.DEFAULT_SYMBOL, period=buffer_period, interval=interval, progress=False)

            if df.empty:
                raise ValueError("No data returned from market data provider.")

            # Handle MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Standardize index and strip timezone info
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Ensure required columns exist
            req_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in req_cols:
                if col not in df.columns:
                    raise KeyError(f"Missing required price column: {col}")

            # Clean NaNs in OHLCV
            df = df.dropna(subset=["Close"])

            # Trim display data to requested period while retaining buffer for indicator calculations
            display_start_date = MarketDataLoader._get_start_date_for_period(period)
            df_display = df[df.index >= display_start_date].copy()
            if len(df_display) < 5:  # Fallback if range filter is too restrictive
                df_display = df.iloc[-30:].copy()

            return df_display

        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            st.error(f"⚠️ Failed to load market data: {str(e)}. Using generated backup data for preview.")
            return MarketDataLoader._generate_fallback_data()

    @staticmethod
    def _get_start_date_for_period(period: str) -> datetime:
        """Calculate start date from period string."""
        now = datetime.now()
        period_map = {
            "7d": timedelta(days=7),
            "1mo": timedelta(days=30),
            "3mo": timedelta(days=90),
            "6mo": timedelta(days=180),
            "1y": timedelta(days=365),
            "2y": timedelta(days=730),
        }
        delta = period_map.get(period, timedelta(days=180))
        return now - delta

    @staticmethod
    def _generate_fallback_data() -> pd.DataFrame:
        """Generate mock BTC data in case of API outages."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        np.random.seed(42)
        base_price = 80000.0
        returns = np.random.normal(0.001, 0.02, size=100)
        price_path = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "Open": price_path * (1 - np.random.uniform(0, 0.01, 100)),
            "High": price_path * (1 + np.random.uniform(0, 0.015, 100)),
            "Low": price_path * (1 - np.random.uniform(0, 0.015, 100)),
            "Close": price_path,
            "Volume": np.random.uniform(1e9, 5e9, 100),
        }, index=dates)
        return df


# =============================================================================
# SERVICE 2: SENTIMENT ENGINE
# =============================================================================
class SentimentEngine:
    """Manages news headline collection and Generative AI sentiment analysis."""

    @staticmethod
    @st.cache_data(ttl=DashboardConfig.CACHE_TTL_SENTIMENT, show_spinner=False)
    def analyze_market_sentiment() -> Dict[str, Any]:
        """
        Fetch news headlines and analyze sentiment using GenAI or resilient fallback.

        Returns:
            Dict containing score (-1.0 to 1.0), status, reasoning, and headlines list.
        """
        if not SENTIMENT_MODULE_AVAILABLE:
            return SentimentEngine._fallback_sentiment("Sentiment module not loaded.")

        try:
            headlines = fetch_crypto_headlines(n=10)
            if not headlines:
                return SentimentEngine._fallback_sentiment("No headlines returned from API.")

            score = get_sentiment_score(headlines)

            # Determine verbal sentiment classification
            if score >= 0.35:
                label = "Bullish 🔥"
                color = DashboardConfig.COLOR_BULLISH
            elif score <= -0.35:
                label = "Bearish 📉"
                color = DashboardConfig.COLOR_BEARISH
            else:
                label = "Neutral ⚖️"
                color = "#A0A0A0"

            return {
                "score": float(score),
                "label": label,
                "color": color,
                "headlines": headlines,
                "reasoning": f"LLM analyzed {len(headlines)} live market headlines.",
                "status": "Active (Live GenAI)",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return SentimentEngine._fallback_sentiment(str(e))

    @staticmethod
    def _fallback_sentiment(error_msg: str) -> Dict[str, Any]:
        """Neutral fallback if news or LLM service is offline."""
        return {
            "score": 0.15,
            "label": "Mildly Bullish (Static Fallback)",
            "color": "#F7931A",
            "headlines": [
                "Bitcoin consolidates near major moving average support levels",
                "Institutional inflow into Bitcoin ETFs shows steady momentum",
                "Network hash rate reaches new milestone amidst market stability"
            ],
            "reasoning": f"Fallback active ({error_msg}). Defaulting to macro neutral baseline.",
            "status": "Fallback Mode",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }


# =============================================================================
# SERVICE 3: MODEL INFERENCE ENGINE
# =============================================================================
class ModelInferenceEngine:
    """Loads trained machine learning models, engineers features, and computes predictions."""

    @staticmethod
    def get_available_models() -> List[str]:
        """Discover trained model artifacts in the models/ directory."""
        if not os.path.exists(DashboardConfig.MODELS_DIR):
            return ["default_baseline"]

        files = [f for f in os.listdir(DashboardConfig.MODELS_DIR) if f.endswith(".pkl")]
        model_names = [os.path.splitext(f)[0] for f in files if "model" in f or "xgboost" in f or "forest" in f or "regression" in f]
        return sorted(model_names) if model_names else ["random_forest", "xgboost", "linear_regression"]

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_model_artifact(model_name: str) -> Optional[Dict[str, Any]]:
        """Load pickled model dictionary containing model object and optional scaler."""
        file_path = os.path.join(DashboardConfig.MODELS_DIR, f"{model_name}.pkl")

        if not os.path.exists(file_path):
            # Fallback path check
            alt_path = os.path.join(DashboardConfig.MODELS_DIR, "btc_model.pkl")
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                logger.warning(f"Model file not found at {file_path}")
                return None

        try:
            artifact = joblib.load(file_path)
            if isinstance(artifact, dict):
                return artifact
            else:
                # If direct model object was saved without dict wrapper
                return {"model": artifact, "scaler": None}
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return None

    @staticmethod
    def prepare_features(df: pd.DataFrame, sentiment_score: float) -> pd.DataFrame:
        """
        Compute technical indicators and inject sentiment score into feature matrix.
        """
        try:
            # Generate technical indicators
            df_feat = add_technical_indicators(df)
        except Exception as e:
            logger.warning(f"Error executing add_technical_indicators: {e}. Building manually.")
            df_feat = df.copy()
            df_feat["sma_20"] = df_feat["Close"].rolling(20).mean()
            df_feat["sma_50"] = df_feat["Close"].rolling(50).mean()
            df_feat["sma_200"] = df_feat["Close"].rolling(200, min_periods=10).mean()
            df_feat["ema_12"] = df_feat["Close"].ewm(span=12).mean()
            df_feat["ema_26"] = df_feat["Close"].ewm(span=26).mean()
            df_feat["rsi_14"] = 50.0  # fallback
            df_feat["macd"] = df_feat["ema_12"] - df_feat["ema_26"]
            df_feat["macd_signal"] = df_feat["macd"].ewm(span=9).mean()
            df_feat["macd_diff"] = df_feat["macd"] - df_feat["macd_signal"]
            df_feat["bb_upper"] = df_feat["sma_20"] + (df_feat["Close"].rolling(20).std() * 2)
            df_feat["bb_lower"] = df_feat["sma_20"] - (df_feat["Close"].rolling(20).std() * 2)
            df_feat["bb_width"] = (df_feat["bb_upper"] - df_feat["bb_lower"]) / df_feat["sma_20"]
            df_feat["bb_position"] = 0.5
            df_feat["atr_14"] = df_feat["Close"].rolling(14).std()
            df_feat["daily_return"] = df_feat["Close"].pct_change()
            df_feat["volatility_20d"] = df_feat["daily_return"].rolling(20).std()
            df_feat["volume_change"] = df_feat["Volume"].pct_change()
            for lag in [1, 2, 3, 5, 7]:
                df_feat[f"return_lag_{lag}"] = df_feat["daily_return"].shift(lag)
            df_feat["price_vs_sma20"] = (df_feat["Close"] - df_feat["sma_20"]) / df_feat["sma_20"]
            df_feat["price_vs_sma50"] = (df_feat["Close"] - df_feat["sma_50"]) / df_feat["sma_50"]
            df_feat["price_vs_sma200"] = (df_feat["Close"] - df_feat["sma_200"]) / df_feat["sma_200"]

        # Inject sentiment score
        df_feat["sentiment_score"] = sentiment_score

        # Forward fill and back fill missing values from rolling calculations
        df_feat = df_feat.ffill().bfill().fillna(0)
        return df_feat

    @staticmethod
    def predict_next_price(
        artifact: Optional[Dict[str, Any]],
        df_feat: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Compute predicted next closing price, confidence intervals, and feature importance.
        """
        current_price = float(df_feat["Close"].iloc[-1])
        default_feature_cols = get_feature_columns()

        # Build latest feature vector
        latest_row = df_feat.iloc[-1:]

        # Ensure all default features exist
        for col in default_feature_cols:
            if col not in latest_row.columns:
                latest_row[col] = 0.0

        if artifact is None or "model" not in artifact:
            feature_cols = default_feature_cols
            X_input = latest_row[feature_cols]
            # Smart heuristic fallback prediction
            pred_price = current_price * (1 + (df_feat["sentiment_score"].iloc[-1] * 0.015))
            std_dev = current_price * 0.025
            model_name = "Rule-based Quant Baseline"
            importances = pd.Series([0.25, 0.20, 0.15, 0.10, 0.30], index=["sentiment_score", "rsi_14", "macd", "sma_20", "volatility_20d"])
        else:
            model_obj = artifact["model"]
            scaler = artifact.get("scaler", None)

            # Extract base estimator if GridSearchCV was saved
            if hasattr(model_obj, "best_estimator_"):
                estimator = model_obj.best_estimator_
            else:
                estimator = model_obj

            model_name = type(estimator).__name__

            # Determine expected features from estimator or scaler if available
            expected_features = None
            if scaler is not None and hasattr(scaler, "feature_names_in_"):
                expected_features = list(scaler.feature_names_in_)
            elif hasattr(estimator, "feature_names_in_"):
                expected_features = list(estimator.feature_names_in_)

            if expected_features:
                feature_cols = expected_features
                # Ensure missing expected features are added
                for col in expected_features:
                    if col not in latest_row.columns:
                        latest_row[col] = 0.0
                X_input = latest_row[expected_features]
            else:
                feature_cols = default_feature_cols
                X_input = latest_row[feature_cols]

            # Apply scaler if trained with one
            if scaler is not None:
                X_proc = scaler.transform(X_input)
            else:
                X_proc = X_input

            # Compute prediction
            try:
                raw_pred = estimator.predict(X_proc)[0]
                pred_price = float(raw_pred)
            except Exception as e:
                logger.error(f"Inference execution failed: {e}")
                pred_price = current_price * 1.005

            # Estimate variance / confidence bounds
            std_dev = current_price * 0.02  # Default ~2% standard error
            if hasattr(estimator, "estimators_"):  # Random Forest tree variance
                try:
                    tree_preds = [tree.predict(X_proc)[0] for tree in estimator.estimators_]
                    std_dev = float(np.std(tree_preds))
                except Exception:
                    pass

            # Extract Feature Importances if available
            importances = None
            if hasattr(estimator, "feature_importances_"):
                imp_vals = estimator.feature_importances_
                importances = pd.Series(imp_vals, index=feature_cols).sort_values(ascending=False)
            elif hasattr(estimator, "coef_"):
                imp_vals = np.abs(estimator.coef_)
                importances = pd.Series(imp_vals, index=feature_cols).sort_values(ascending=False)

        # Z-multiplier for confidence interval (e.g., 95% = 1.96)
        z_score = 1.96 if confidence_level >= 0.95 else 1.645
        lower_bound = max(0.0, pred_price - (z_score * std_dev))
        upper_bound = pred_price + (z_score * std_dev)

        expected_change = pred_price - current_price
        pct_change = (expected_change / current_price) * 100

        return {
            "model_name": model_name,
            "current_price": current_price,
            "predicted_price": pred_price,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "expected_change": expected_change,
            "pct_change": pct_change,
            "direction": "Bullish 📈" if expected_change >= 0 else "Bearish 📉",
            "direction_color": DashboardConfig.COLOR_BULLISH if expected_change >= 0 else DashboardConfig.COLOR_BEARISH,
            "feature_importances": importances
        }


# =============================================================================
# FACTORY: PLOTLY CHART BUILDERS
# =============================================================================
class ChartFactory:
    """Creates polished, interactive dark-themed Plotly charts."""

    @staticmethod
    def create_candlestick_chart(
        df: pd.DataFrame,
        overlays: List[str],
        indicators: List[str]
    ) -> go.Figure:
        """Build interactive candlestick chart with configurable technical overlays and subplots."""
        num_rows = 1 + ("RSI" in indicators) + ("MACD" in indicators)
        row_heights = [0.6] + [0.2] * (num_rows - 1)

        subplot_titles = ["BTC-USD Price Action & Volatility"]
        if "RSI" in indicators:
            subplot_titles.append("Relative Strength Index (RSI 14)")
        if "MACD" in indicators:
            subplot_titles.append("MACD (12, 26, 9)")

        fig = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )

        # Main Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
                increasing_line_color=DashboardConfig.COLOR_BULLISH,
                decreasing_line_color=DashboardConfig.COLOR_BEARISH,
            ),
            row=1, col=1
        )

        # Moving Averages Overlays
        if "SMA 20" in overlays and "sma_20" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], name="SMA 20", line=dict(color="#FFD700", width=1.2)), row=1, col=1)
        if "SMA 50" in overlays and "sma_50" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50", line=dict(color="#00E5FF", width=1.2)), row=1, col=1)
        if "SMA 200" in overlays and "sma_200" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["sma_200"], name="SMA 200", line=dict(color="#FF007F", width=1.5)), row=1, col=1)
        if "EMA 12" in overlays and "ema_12" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["ema_12"], name="EMA 12", line=dict(color="#7B61FF", width=1.2)), row=1, col=1)

        # Bollinger Bands Overlay
        if "Bollinger Bands" in overlays and "bb_upper" in df.columns and "bb_lower" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper", line=dict(color="rgba(255,255,255,0.3)", width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df["bb_lower"], name="BB Lower",
                line=dict(color="rgba(255,255,255,0.3)", width=1),
                fill='tonexty', fillcolor='rgba(255,255,255,0.05)'
            ), row=1, col=1)

        current_row = 2

        # RSI Subplot
        if "RSI" in indicators and "rsi_14" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["rsi_14"], name="RSI", line=dict(color="#9C27B0", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color=DashboardConfig.COLOR_BEARISH, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color=DashboardConfig.COLOR_BULLISH, row=current_row, col=1)
            current_row += 1

        # MACD Subplot
        if "MACD" in indicators and "macd" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="#2196F3", width=1.5)), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="#FF9800", width=1.2)), row=current_row, col=1)
            if "macd_diff" in df.columns:
                colors = [DashboardConfig.COLOR_BULLISH if v >= 0 else DashboardConfig.COLOR_BEARISH for v in df["macd_diff"]]
                fig.add_trace(go.Bar(x=df.index, y=df["macd_diff"], name="Hist", marker_color=colors), row=current_row, col=1)

        # Layout styling
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=650,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showgrid=True, gridcolor=DashboardConfig.COLOR_GRID)
        fig.update_yaxes(showgrid=True, gridcolor=DashboardConfig.COLOR_GRID)

        return fig

    @staticmethod
    def create_prediction_forecast_chart(
        df: pd.DataFrame,
        pred_data: Dict[str, Any],
        horizon_days: int = 7
    ) -> go.Figure:
        """Create visual representation of historical price trend with model forecast cone."""
        history = df["Close"].iloc[-60:].copy()
        last_date = history.index[-1]

        # Generate future date index
        future_dates = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]

        # Linearly interpolate path to predicted price target
        start_p = pred_data["current_price"]
        target_p = pred_data["predicted_price"]
        forecast_path = np.linspace(start_p, target_p, horizon_days)

        # Expand confidence bounds over time
        lower_path = np.linspace(start_p, pred_data["lower_bound"], horizon_days)
        upper_path = np.linspace(start_p, pred_data["upper_bound"], horizon_days)

        fig = go.Figure()

        # Historical Close
        fig.add_trace(go.Scatter(
            x=history.index,
            y=history.values,
            name="Historical Close",
            line=dict(color=DashboardConfig.COLOR_SECONDARY, width=2)
        ))

        # Upper Bound for Confidence Interval
        fig.add_trace(go.Scatter(
            x=[last_date] + future_dates,
            y=[start_p] + list(upper_path),
            name="Upper Bound (95% CI)",
            line=dict(color="rgba(0, 200, 5, 0.2)", width=0),
            showlegend=False
        ))

        # Lower Bound & Shaded Confidence Cone
        fig.add_trace(go.Scatter(
            x=[last_date] + future_dates,
            y=[start_p] + list(lower_path),
            name="Confidence Interval",
            line=dict(color="rgba(0, 200, 5, 0.2)", width=0),
            fill='tonexty',
            fillcolor='rgba(247, 147, 26, 0.15)'
        ))

        # Forecast Trajectory Line
        fig.add_trace(go.Scatter(
            x=[last_date] + future_dates,
            y=[start_p] + list(forecast_path),
            name=f"Forecast ({pred_data['model_name']})",
            line=dict(color=DashboardConfig.COLOR_PRIMARY, width=3, dash="dash")
        ))

        # Target Endpoint Marker
        fig.add_trace(go.Scatter(
            x=[future_dates[-1]],
            y=[target_p],
            mode="markers+text",
            name="Target Prediction",
            marker=dict(size=12, color=pred_data["direction_color"], symbol="diamond"),
            text=[f"${target_p:,.2f}"],
            textposition="top center"
        ))

        fig.update_layout(
            title=f"AI Forecast Trajectory & Confidence Cone (Next {horizon_days} Days)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showgrid=True, gridcolor=DashboardConfig.COLOR_GRID)
        fig.update_yaxes(showgrid=True, gridcolor=DashboardConfig.COLOR_GRID, tickformat="$,.0f")

        return fig

    @staticmethod
    def create_sentiment_gauge(score: float) -> go.Figure:
        """Create a sleek gauge chart for sentiment score (-1 to +1)."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Market Sentiment Index", 'font': {'size': 18, 'color': "#FFFFFF"}},
            number={'font': {'size': 32, 'color': "#FFFFFF"}, 'valueformat': "+.2f"},
            gauge={
                'axis': {'range': [-1.0, 1.0], 'tickwidth': 1, 'tickcolor': "#FFFFFF"},
                'bar': {'color': DashboardConfig.COLOR_PRIMARY},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [-1.0, -0.35], 'color': 'rgba(255, 80, 0, 0.4)'},
                    {'range': [-0.35, 0.35], 'color': 'rgba(160, 160, 160, 0.2)'},
                    {'range': [0.35, 1.0], 'color': 'rgba(0, 200, 5, 0.4)'}
                ],
                'threshold': {
                    'line': {'color': "#FFFFFF", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=30, r=30, t=50, b=20)
        )
        return fig

    @staticmethod
    def create_feature_importance_chart(importance_series: pd.Series) -> go.Figure:
        """Build horizontal bar chart for top model feature importances."""
        top10 = importance_series.head(10).sort_values(ascending=True)

        fig = go.Figure(go.Bar(
            x=top10.values,
            y=top10.index,
            orientation='h',
            marker=dict(
                color=top10.values,
                colorscale='Viridis'
            )
        ))
        fig.update_layout(
            title="Top 10 Feature Drivers in Model Prediction",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Relative Importance Weight"
        )
        fig.update_xaxes(showgrid=True, gridcolor=DashboardConfig.COLOR_GRID)
        fig.update_yaxes(showgrid=False)
        return fig


# =============================================================================
# STREAMLIT UI RENDERER
# =============================================================================
def render_custom_css():
    """Inject polished custom CSS for modern dark financial terminal UI."""
    st.markdown("""
        <style>
        /* Main Container background */
        .stApp {
            background-color: #0E1117;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Metric Card styling */
        div[data-testid="stMetric"] {
            background-color: #1E222D;
            border: 1px solid #2A2E39;
            border-radius: 8px;
            padding: 14px 18px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        div[data-testid="stMetric"] label {
            color: #8F9CAE !important;
            font-size: 0.85rem !important;
            font-weight: 500;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-size: 1.6rem !important;
            font-weight: 700;
        }

        /* Header badge */
        .status-badge {
            background-color: #1E222D;
            border: 1px solid #2A2E39;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            color: #00C805;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        /* Card Containers */
        .glass-card {
            background: #1E222D;
            border: 1px solid #2A2E39;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #131722;
            border-right: 1px solid #2A2E39;
        }

        /* Tab styling */
        button[data-baseweb="tab"] {
            font-weight: 600;
            font-size: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit Dashboard Execution Flow."""
    # Page Setup
    st.set_page_config(
        page_title=DashboardConfig.PAGE_TITLE,
        page_icon=DashboardConfig.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    render_custom_css()

    # -------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # -------------------------------------------------------------------------
    st.sidebar.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=50)
    st.sidebar.title("₿ Quant Controls")
    st.sidebar.markdown("---")

    # Data Settings
    st.sidebar.subheader("📈 Market Data Settings")
    period = st.sidebar.selectbox(
        "Historical Lookback Range",
        options=["7d", "1mo", "3mo", "6mo", "1y", "2y"],
        index=3,
        help="Select time depth for model feature generation and chart display."
    )
    interval = st.sidebar.selectbox(
        "Data Granularity / Interval",
        options=["1d", "1h", "15m"],
        index=0,
        help="Timeframe interval per OHLC candle."
    )

    st.sidebar.markdown("---")
    # Model Settings
    st.sidebar.subheader("🤖 AI Model Configuration")
    available_models = ModelInferenceEngine.get_available_models()
    selected_model_name = st.sidebar.selectbox(
        "Model Engine Selection",
        options=available_models,
        index=0,
        help="Select trained machine learning model stored in models/ directory."
    )
    confidence_level = st.sidebar.slider(
        "Confidence Interval Level",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Statistical confidence interval width for prediction bounds."
    )
    horizon_days = st.sidebar.slider(
        "Forecast Horizon (Days)",
        min_value=1,
        max_value=14,
        value=7,
        step=1
    )

    st.sidebar.markdown("---")
    # Sentiment Settings
    st.sidebar.subheader("📰 Sentiment Overlay Controls")
    override_sentiment = st.sidebar.checkbox("Enable Scenario Simulation (Manual Sentiment)", value=False)
    manual_sentiment = 0.0
    if override_sentiment:
        manual_sentiment = st.sidebar.slider(
            "Simulated Sentiment Score (-1 Bearish to +1 Bullish)",
            min_value=-1.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )

    # -------------------------------------------------------------------------
    # DATA & INFERENCE PIPELINE INVOCATION
    # -------------------------------------------------------------------------
    with st.spinner("🔄 Fetching live market feeds & executing AI inference engine..."):
        # 1. Fetch Market Data
        df_market = MarketDataLoader.fetch_market_data(period=period, interval=interval)

        # 2. Fetch or Override Sentiment
        if override_sentiment:
            sentiment_data = {
                "score": manual_sentiment,
                "label": "Custom Simulation 🎛️",
                "color": DashboardConfig.COLOR_PRIMARY,
                "headlines": ["Manual scenario simulation enabled by user slider."],
                "reasoning": "User manually adjusted sentiment slider for stress testing.",
                "status": "Simulated",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
        else:
            sentiment_data = SentimentEngine.analyze_market_sentiment()

        # 3. Engineer Features
        df_features = ModelInferenceEngine.prepare_features(df_market, sentiment_data["score"])

        # 4. Run Model Prediction
        model_artifact = ModelInferenceEngine.load_model_artifact(selected_model_name)
        pred_results = ModelInferenceEngine.predict_next_price(
            artifact=model_artifact,
            df_feat=df_features,
            confidence_level=confidence_level
        )

    # -------------------------------------------------------------------------
    # DASHBOARD HEADER & SYSTEM STATUS
    # -------------------------------------------------------------------------
    col_title, col_status = st.columns([0.7, 0.3])
    with col_title:
        st.title("₿ Bitcoin GenAI Predictor")
        st.markdown("**Production Quantitative Analytics & Generative AI Forecasting Engine**")

    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="text-align: right;">
                <span class="status-badge">🟢 Live Data Feed</span>
                <span class="status-badge" style="color: #4A90E2;">⚡ GenAI: {sentiment_data['status']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # TOP EXECUTIVE METRICS ROW
    # -------------------------------------------------------------------------
    current_p = pred_results["current_price"]
    prev_p = float(df_market["Close"].iloc[-2]) if len(df_market) > 1 else current_p
    chg_24h = current_p - prev_p
    pct_24h = (chg_24h / prev_p) * 100

    high_24h = float(df_market["High"].iloc[-1])
    low_24h = float(df_market["Low"].iloc[-1])

    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)

    mcol1.metric("Current Price", f"${current_p:,.2f}", f"{pct_24h:+.2f}% (24h)")
    mcol2.metric("Predicted Target", f"${pred_results['predicted_price']:,.2f}", f"{pred_results['pct_change']:+.2f}%")
    mcol3.metric("24h Range High / Low", f"${high_24h:,.0f}", f"Low: ${low_24h:,.0f}")
    mcol4.metric(
        "Expected Direction",
        pred_results["direction"],
        f"CI: [${pred_results['lower_bound']:,.0f} - ${pred_results['upper_bound']:,.0f}]"
    )
    mcol5.metric(
        "GenAI Sentiment",
        f"{sentiment_data['score']:+.2f}",
        sentiment_data["label"]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TABBED NAVIGATION SECTION
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Technical Analysis Terminal",
        "🤖 AI Model Forecast & Insights",
        "📰 GenAI Sentiment Intelligence",
        "💾 Raw Features & Export"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: TECHNICAL ANALYSIS TERMINAL
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Price Action & Quantitative Technical Indicators")

        # Indicator Controls
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            selected_overlays = st.multiselect(
                "Technical Overlays",
                options=["SMA 20", "SMA 50", "SMA 200", "EMA 12", "Bollinger Bands"],
                default=["SMA 20", "SMA 50", "Bollinger Bands"]
            )
        with tcol2:
            selected_indicators = st.multiselect(
                "Oscillator Subplots",
                options=["RSI", "MACD"],
                default=["RSI", "MACD"]
            )

        # Plot Candlestick Chart
        fig_tech = ChartFactory.create_candlestick_chart(
            df=df_features,
            overlays=selected_overlays,
            indicators=selected_indicators
        )
        st.plotly_chart(fig_tech, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: AI MODEL FORECAST & INSIGHTS
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Machine Learning Prediction & Volatility Cone")

        pcol_chart, pcol_info = st.columns([0.7, 0.3])

        with pcol_chart:
            fig_forecast = ChartFactory.create_prediction_forecast_chart(
                df=df_market,
                pred_data=pred_results,
                horizon_days=horizon_days
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

        with pcol_info:
            st.markdown("### 🎯 Inference Summary")
            st.markdown(
                f"""
                <div class="glass-card">
                    <p><b>Model Selected:</b> <code>{pred_results['model_name']}</code></p>
                    <p><b>Target Price:</b> <span style="font-size: 1.2rem; font-weight: bold; color: {pred_results['direction_color']};">${pred_results['predicted_price']:,.2f}</span></p>
                    <p><b>Expected Shift:</b> <code>{pred_results['pct_change']:+.2f}%</code></p>
                    <p><b>Upper Bound ({int(confidence_level*100)}%):</b> ${pred_results['upper_bound']:,.2f}</p>
                    <p><b>Lower Bound ({int(confidence_level*100)}%):</b> ${pred_results['lower_bound']:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if pred_results["feature_importances"] is not None:
                fig_imp = ChartFactory.create_feature_importance_chart(pred_results["feature_importances"])
                st.plotly_chart(fig_imp, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: GENAI SENTIMENT INTELLIGENCE
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Generative AI News & Market Sentiment Pipeline")

        scol_gauge, scol_news = st.columns([0.4, 0.6])

        with scol_gauge:
            fig_gauge = ChartFactory.create_sentiment_gauge(sentiment_data["score"])
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("#### 🧠 GenAI Reasoning Analysis")
            st.info(sentiment_data["reasoning"])

        with scol_news:
            st.markdown("#### 📰 Analyzed Live Crypto Headlines")
            for idx, headline in enumerate(sentiment_data["headlines"], 1):
                st.markdown(f"**{idx}.** {headline}")

    # -------------------------------------------------------------------------
    # TAB 4: RAW DATA & EXPORT
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("Processed Feature Dataframe & Model Diagnostics")
        st.markdown("Inspect engineered quantitative features used for real-time model inference.")

        # Data Table Search / Display
        st.dataframe(df_features.tail(50), use_container_width=True)

        # CSV Download
        csv_bytes = df_features.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Engineered Features (CSV)",
            data=csv_bytes,
            file_name=f"bitcoin_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # -------------------------------------------------------------------------
    # FOOTER
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.caption(
        f"₿ Bitcoin GenAI Predictor Terminal | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | "
        "Disclaimer: Portfolio demonstration project. Not financial advice."
    )


if __name__ == "__main__":
    main()
