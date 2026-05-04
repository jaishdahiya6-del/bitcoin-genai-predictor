"""
evaluate.py
-----------
Load saved models and generate evaluation charts.
Run this after model_training.py to visualise performance.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from feature_engineering import get_feature_columns


def load_model(name: str):
    path = f"models/{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_predictions(df: pd.DataFrame, model_name: str = "xgboost"):
    """
    Plot actual vs predicted prices for the test period.
    This is the chart that goes in your README and portfolio.
    """
    obj = load_model(model_name)
    model = obj["model"]
    scaler = obj.get("scaler", None)
    
    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    split_idx = int(len(df) * 0.8)
    
    X_test = df[feature_cols].iloc[split_idx:]
    y_test = df["target"].iloc[split_idx:]
    dates  = df.index[split_idx:]
    
    X_input = scaler.transform(X_test) if scaler else X_test
    preds = model.predict(X_input)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f"Bitcoin Price Prediction — {model_name.replace('_', ' ').title()}", fontsize=15, fontweight="bold")
    
    # ── Plot 1: Actual vs Predicted ──────────────────────────────
    axes[0].plot(dates, y_test.values, label="Actual Price",    color="#2563EB", linewidth=1.8)
    axes[0].plot(dates, preds,         label="Predicted Price", color="#DC2626", linewidth=1.5, linestyle="--")
    axes[0].fill_between(dates, y_test.values, preds, alpha=0.08, color="#DC2626")
    axes[0].set_title("Actual vs Predicted Next-Day Close Price")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(alpha=0.3)

    # ── Plot 2: Prediction Error ─────────────────────────────────
    errors = y_test.values - preds
    axes[1].bar(dates, errors, color=np.where(errors >= 0, "#16A34A", "#DC2626"), alpha=0.7, width=1)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Prediction Error (Actual − Predicted)")
    axes[1].set_ylabel("Error (USD)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("models/evaluation_chart.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[INFO] Chart saved → models/evaluation_chart.png")


if __name__ == "__main__":
    df = pd.read_csv("data/processed/bitcoin_features.csv", index_col=0, parse_dates=True)
    plot_predictions(df, model_name="xgboost")