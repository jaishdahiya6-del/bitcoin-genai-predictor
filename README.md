# 🪙 Bitcoin GenAI Predictor
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20/%20PyTorch-orange)](https://tensorflow.org)

An advanced predictive modeling system that leverages **Generative AI** and **Deep Learning** to forecast Bitcoin price volatility and market trends. This project moves beyond standard regression by incorporating sentiment-driven generative insights.

---

## 📊 Project Overview
Predicting cryptocurrency is notoriously difficult due to high volatility. This project solves that by:
*   **Time-Series Analysis:** Utilizing LSTM/GRUs for historical price patterns.
*   **GenAI Integration:** Using LLMs to analyze real-time news sentiment and social media "hype".
*   **End-to-End Pipeline:** Automated data collection, cleaning, modeling, and visualization.

## 🏗️ Technical Architecture
We follow the **CRISP-DM** framework for a structured data science workflow:
1. **Data Acquisition:** Scrapping market data via Yahoo Finance or Binance API.
2. **Feature Engineering:** Technical indicators (RSI, Moving Averages) combined with GenAI sentiment scores.
3. **Modeling:** A hybrid architecture combining recurrent neural networks (RNNs) with attention mechanisms.
4. **Evaluation:** Testing against RMSE, MAE, and R² metrics.

## 📂 Repository Structure
```text
├── app/                # Streamlit/Dash web dashboard
├── models/             # Trained .h5 or .pkl model files
│   └── evaluation.png  # Performance charts
├── notebooks/          # Exploratory Data Analysis (EDA)
├── src/                # Production-grade Python scripts
└── requirements.txt    # Project dependencies
