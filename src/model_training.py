"""
model_training.py
------------------
WHY THIS EXISTS:
  Moves model training OUT of notebooks and into reusable code.
  GridSearchCV systematically finds the best hyperparameters
  instead of manual guessing — reduces RMSE by 15-30% typically.
  
  Models trained:
    1. XGBoost  — best for tabular data with non-linear patterns
    2. Random Forest — robust, resistant to overfitting
    3. Linear Regression — baseline to beat
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor # type: ignore

from feature_engineering import get_feature_columns


def load_processed_data(filepath: str = "data/processed/bitcoin_features.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"[INFO] Loaded {len(df)} rows from {filepath}")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    WHY WE DON'T USE random_state SHUFFLE for time series:
      If we randomly split, test data can be BEFORE training data.
      The model would have "seen the future" — that's data leakage.
      TimeSeriesSplit respects time order: train on past, test on future.
    """
    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    
    X = df[feature_cols]
    y = df["target"]
    
    # Keep chronological order — last 20% is our "future" test set
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"[INFO] Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test, feature_cols


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """Calculate and print model performance metrics."""
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    
    metrics = {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}
    
    print(f"\n── {name} Results ──")
    print(f"  RMSE : ${rmse:,.0f}  (avg dollar error per prediction)")
    print(f"  MAE  : ${mae:,.0f}")
    print(f"  R²   : {r2:.4f}  (1.0 = perfect, 0 = no better than mean)")
    print(f"  MAPE : {mape:.2f}% (avg % error)")
    
    return metrics


def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train 3 models. Use GridSearchCV for XGBoost (most important).
    
    WHY GridSearchCV with TimeSeriesSplit:
      Instead of one train/test split, we use 5 rolling windows.
      Each window trains on older data, validates on newer data.
      This gives a more reliable estimate of real-world performance.
    """
    results = []
    trained_models = {}
    
    # ── Time-series cross-validation ────────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)

    # ── 1. Linear Regression (baseline) ─────────────────────────
    print("\n[1/3] Training Linear Regression (baseline)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    results.append(evaluate_model(lr, X_test_scaled, y_test, "Linear Regression"))
    trained_models["linear_regression"] = {"model": lr, "scaler": scaler}

    # ── 2. Random Forest ─────────────────────────────────────────
    print("\n[2/3] Training Random Forest...")
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth":    [10, 20, None],
        "min_samples_split": [2, 5],
    }
    rf = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_params,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        verbose=1,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print(f"  Best RF params: {rf.best_params_}")
    results.append(evaluate_model(rf, X_test, y_test, "Random Forest"))
    trained_models["random_forest"] = {"model": rf}

    # ── 3. XGBoost (usually wins) ────────────────────────────────
    print("\n[3/3] Training XGBoost with GridSearchCV...")
    xgb_params = {
        "n_estimators":    [100, 300, 500],
        "max_depth":       [3, 5, 7],
        "learning_rate":   [0.01, 0.05, 0.1],
        "subsample":       [0.8, 1.0],
        "colsample_bytree":[0.8, 1.0],
    }
    xgb = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
        xgb_params,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        verbose=1,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    print(f"  Best XGB params: {xgb.best_params_}")
    results.append(evaluate_model(xgb, X_test, y_test, "XGBoost (tuned)"))
    trained_models["xgboost"] = {"model": xgb}

    return results, trained_models


def save_models(trained_models: dict, output_dir: str = "models"):
    """Save all trained models to disk as .pkl files."""
    os.makedirs(output_dir, exist_ok=True)
    for name, obj in trained_models.items():
        path = os.path.join(output_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"[INFO] Saved {name} → {path}")


def print_comparison_table(results: list):
    """Print a clean comparison table of all models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    df = pd.DataFrame(results).set_index("model")
    df["RMSE"] = df["RMSE"].map("${:,.0f}".format)
    df["MAE"]  = df["MAE"].map("${:,.0f}".format)
    df["R2"]   = df["R2"].map("{:.4f}".format)
    df["MAPE"] = df["MAPE"].map("{:.2f}%".format)
    print(df.to_string())
    print("="*60)


if __name__ == "__main__":
    df        = load_processed_data()
    X_tr, X_te, y_tr, y_te, feat_cols = split_data(df)
    results, models = train_all_models(X_tr, X_te, y_tr, y_te)
    print_comparison_table(results)
    save_models(models)
    print("\n[DONE] All models saved to models/ folder.")