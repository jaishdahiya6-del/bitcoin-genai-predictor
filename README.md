# 🪙 Bitcoin GenAI Predictor: Hybrid Forecasting System
![Banner](https://img.shields.io/badge/Status-In--Development-green?style=for-the-badge)
![Tech](https://img.shields.io/badge/AI-Generative--Deep--Learning-A97CF8?style=for-the-badge)

An advanced predictive modeling system that leverages **Generative AI** and **Deep Learning** to forecast Bitcoin price volatility. This project moves beyond standard regression by incorporating sentiment-driven generative insights from real-time news.

---

## 🏗️ Technical Architecture
We utilize a hybrid approach combining quantitative data with qualitative sentiment analysis.



| Layer | Technoly | Purpose |
| :--- | :--- | :--- |
| **Data Engine** | `yfinance` & `NewsAPI` | Multi-source data ingestion |
| **Sentiment Layer** | `Generative AI (LLMs)` | Real-time "fear & greed" index analysis |
| **Model Core** | `LSTM / GRU` | Capturing long-term temporal dependencies |
| **UI / Dashboard** | `Streamlit` | Interactive price forecasting visualizations |

---

## 🚀 Key Features
- **GenAI Sentiment Analysis:** Uses LLMs to process tech news and social media trends into numerical "Sentiment Scores."
- **Time-Series Forecasting:** Implements Recurrent Neural Networks (RNNs) for high-accuracy price trend prediction.
- **Automated Pipeline:** Full end-to-end workflow from data scraping to model evaluation.
- **Risk Metrics:** Calculates RSI, Bollinger Bands, and Moving Averages automatically.

---

## 📂 Repository Structure
```text
Bitcoin-GenAI-Predictor/
│
├── app/
│   ├── app.py
│   ├── dashboard.py
│   └── pages/
│
├── src/
│   ├── data_loader.py
│   ├── news_scraper.py
│   ├── sentiment.py
│   ├── indicators.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluation.py
│   └── utils.py
│
├── models/
│
├── notebooks/
│
├── data/
│
├── requirements.txt
└── README.md
