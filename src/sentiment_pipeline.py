"""
sentiment_pipeline.py
----------------------
WHY THIS EXISTS:
  Bitcoin price is driven by news as much as technicals.
  In 2021, a single Elon Musk tweet moved BTC by 15% in hours.
  This module fetches crypto headlines and asks an LLM to score
  the sentiment, which we inject as a feature into our ML model.

  Business context:
    Professional crypto trading firms (Pantera, Galaxy Digital)
    use NLP sentiment as a core signal in their alpha models.
    We're replicating that here with a free API call.
"""

import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads OPENAI_API_KEY from .env file


def fetch_crypto_headlines(n: int = 10) -> list[str]:
    """
    Fetch latest Bitcoin headlines from CryptoCompare's free news API.
    No API key required for basic usage.
    
    Returns:
        List of headline strings, most recent first.
    """
    url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories=BTC"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get("Data", [])
        headlines = [article["title"] for article in articles[:n]]
        print(f"[INFO] Fetched {len(headlines)} headlines")
        return headlines
    
    except requests.RequestException as e:
        print(f"[WARNING] Could not fetch headlines: {e}")
        # Fallback so the pipeline doesn't crash
        return ["Bitcoin holds steady amid market uncertainty"]


def get_sentiment_score(headlines: list[str]) -> float:
    """
    Send headlines to GPT-4o-mini and get a sentiment score.
    
    WHY GPT-4o-mini:
        It's cheap (~$0.00015 per 1K tokens), fast, and accurate
        for classification tasks. Perfect for this use case.
    
    Returns:
        Float between -1.0 (very bearish) and +1.0 (very bullish).
        0.0 = neutral market sentiment.
    
    Business context:
        A score of +0.7 with a bullish technical setup = strong buy signal.
        A score of -0.8 despite price rising = divergence warning.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    headlines_text = "\n".join([f"- {h}" for h in headlines])
    
    prompt = f"""You are a professional crypto market analyst.

Analyse these Bitcoin news headlines and return a single JSON object only.
No explanation, no markdown, just raw JSON.

Headlines:
{headlines_text}

Return this exact format:
{{"sentiment_score": <float between -1.0 and 1.0>, "reasoning": "<one sentence>"}}

Where:
  -1.0 = extremely bearish (crash, ban, hack, collapse)
   0.0 = neutral
  +1.0 = extremely bullish (ETF approval, adoption, all-time high)"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,   # low temp = consistent, factual outputs
            max_tokens=150,
        )
        
        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        score = float(result["sentiment_score"])
        score = max(-1.0, min(1.0, score))  # clamp to valid range
        
        print(f"[INFO] Sentiment score: {score:+.3f} | {result.get('reasoning', '')}")
        return score
    
    except Exception as e:
        print(f"[WARNING] LLM call failed: {e}. Returning neutral score 0.0")
        return 0.0


def get_daily_sentiment() -> dict:
    """
    Master function: fetch headlines + score them.
    Returns a dict with score and metadata for logging.
    
    Call this once per day before running predictions.
    """
    headlines = fetch_crypto_headlines(n=10)
    score = get_sentiment_score(headlines)
    
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "sentiment_score": score,
        "headlines_used": len(headlines),
        "sample_headline": headlines[0] if headlines else "N/A",
    }


if __name__ == "__main__":
    print("── Running sentiment pipeline ──")
    result = get_daily_sentiment()
    print(f"\nDate:            {result['date']}")
    print(f"Sentiment score: {result['sentiment_score']:+.3f}")
    print(f"Headlines used:  {result['headlines_used']}")
    print(f"Sample:          {result['sample_headline']}")