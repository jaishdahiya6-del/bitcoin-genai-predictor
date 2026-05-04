"""
feature_engineering.py
-----------------------
WHY THIS EXISTS:
  Raw price data (open, close, volume) is weak for prediction.
  We create derived features that capture trend, momentum, and risk.
  This is exactly what quant funds do before feeding data to models.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import SMAIndicator, EMAIndicator, MACD # type: ignore
import ta.momentum # type: ignore
from ta.volatility import BollingerBands, AverageTrueRange # type: ignore


def fetch_bitcoin_data(period: str = "2y") -> pd.DataFrame:
    """
    Fetch Bitcoin OHLCV data from Yahoo Finance.
    
    Args:
        period: How far back to fetch. "2y" = 2 years of daily data.
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    
    Business context:
        2 years gives ~730 data points. Enough to train a model
        AND have a meaningful test set without look-ahead bias.
    """
    print(f"[INFO] Fetching Bitcoin data for period: {period}")
    
    ticker = yf.Ticker("BTC-USD")
    df = ticker.history(period=period)
    
    # yfinance returns a MultiIndex on some versions — flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Remove timezone info to avoid downstream issues
    df.index = df.index.tz_localize(None)
    
    print(f"[INFO] Fetched {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators as new columns.
    
    WHY EACH INDICATOR:
    
    Moving Averages (SMA/EMA):
      Capture trend direction. When price > SMA_50, bullish signal.
      EMA reacts faster to recent price changes than SMA.
      Used by every professional trading desk.
    
    RSI (Relative Strength Index):
      Measures momentum. RSI > 70 = overbought (sell signal).
      RSI < 30 = oversold (buy signal). Range: 0 to 100.
    
    MACD (Moving Average Convergence Divergence):
      Captures momentum shifts. When MACD line crosses signal line
      upward, that's a bullish crossover — classic entry signal.
    
    Bollinger Bands:
      Measure volatility. When price touches upper band, often reverts.
      Band width tells you if the market is calm or explosive.
    
    ATR (Average True Range):
      Pure volatility measure. Traders use ATR to set stop-losses.
      High ATR = wide stop, Low ATR = tight stop.
    
    Lag Features:
      Yesterday's return predicts today's return (momentum effect).
      Classic feature in all time-series ML models.
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # ── Trend indicators ──────────────────────────────────────────
    df["sma_20"]  = SMAIndicator(close=close, window=20).sma_indicator()
    df["sma_50"]  = SMAIndicator(close=close, window=50).sma_indicator()
    df["sma_200"] = SMAIndicator(close=close, window=200).sma_indicator()
    df["ema_12"]  = EMAIndicator(close=close, window=12).ema_indicator()
    df["ema_26"]  = EMAIndicator(close=close, window=26).ema_indicator()

    # ── Momentum indicators ───────────────────────────────────────
    df["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd_obj = MACD(close=close)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"]   = macd_obj.macd_diff()   # histogram: positive = bullish

    # ── Volatility indicators ─────────────────────────────────────
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_width"]    = bb.bollinger_wband()   # (upper-lower)/middle
    df["bb_position"] = bb.bollinger_pband()   # where price sits in band (0-1)

    df["atr_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # ── Price-derived features ────────────────────────────────────
    df["daily_return"]    = close.pct_change()            # today's % change
    df["volatility_20d"]  = df["daily_return"].rolling(20).std()   # 20-day rolling vol
    df["volume_change"]   = df["Volume"].pct_change()

    # ── Lag features (momentum) ───────────────────────────────────
    for lag in [1, 2, 3, 5, 7]:
        df[f"return_lag_{lag}"] = df["daily_return"].shift(lag)

    # ── Target variable ───────────────────────────────────────────
    # Predict next day's closing price
    df["target"] = df["Close"].shift(-1)

    # ── Price relative to moving averages ─────────────────────────
    df["price_vs_sma20"]  = (close - df["sma_20"])  / df["sma_20"]
    df["price_vs_sma50"]  = (close - df["sma_50"])  / df["sma_50"]
    df["price_vs_sma200"] = (close - df["sma_200"]) / df["sma_200"]

    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove NaN rows created by rolling window calculations.
    
    WHY: SMA_200 needs 200 rows before producing a value.
    The first 200 rows will have NaN — we drop them.
    This is not data loss, it's data quality enforcement.
    """
    original_len = len(df)
    df = df.dropna()
    dropped = original_len - len(df)
    print(f"[INFO] Dropped {dropped} NaN rows. Remaining: {len(df)} rows.")
    return df


def get_feature_columns() -> list:
    """
    Returns the exact list of feature columns used for model training.
    Centralising this here means model_training.py and dashboard.py
    both use identical features — no mismatch bugs.
    """
    return [
        "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26",
        "rsi_14",
        "macd", "macd_signal", "macd_diff",
        "bb_upper", "bb_lower", "bb_width", "bb_position",
        "atr_14",
        "daily_return", "volatility_20d", "volume_change",
        "return_lag_1", "return_lag_2", "return_lag_3",
        "return_lag_5", "return_lag_7",
        "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
        "sentiment_score",   # added in Phase 3 by sentiment_pipeline.py
    ]


if __name__ == "__main__":
    # Quick test — run this file directly to verify everything works
    df_raw  = fetch_bitcoin_data(period="1y")
    df_feat = add_technical_indicators(df_raw)
    df_feat = clean_features(df_feat)
    
    print("\n── Feature columns created ──")
    for col in df_feat.columns:
        print(f"  {col}")
    
    print(f"\n── Sample (last 3 rows) ──")
    print(df_feat[["Close", "rsi_14", "macd", "bb_width", "target"]].tail(3))
    
    # Save processed data
    df_feat.to_csv("data/processed/bitcoin_features.csv")
    print("\n[INFO] Saved to data/processed/bitcoin_features.csv")