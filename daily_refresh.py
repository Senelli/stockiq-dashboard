"""
daily_refresh.py
================
Run once per day to pull fresh Alpha Vantage data for all 16 stocks,
apply the keyword classifier, and update live_predictions.csv.

Usage:
    python daily_refresh.py --api_key YOUR_AV_KEY

Schedule automatically with GitHub Actions (see .github/workflows/daily_refresh.yml)
or run manually each morning.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = ['AAPL','AMD','AMZN','GOOGL','IBM','INTC','META','MSFT',
           'NVDA','ORCL','PLTR','TSLA','ADBE','AI','CRM','TSM']

AV_BASE = "https://www.alphavantage.co/query"
LIMIT   = 50      # articles per ticker per call
SLEEP   = 13      # seconds between calls (5/min free tier = 12s min, use 13 to be safe)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Keyword classifier (same logic as your v2 notebook) ──────────────────────
CATEGORY_KEYWORDS = {
    'AI Innovation': {
        'artificial intelligence': 3, 'generative ai': 3, 'large language model': 3,
        'machine learning': 3, 'genai': 3, 'llm': 3, 'foundation model': 3,
        'ai': 1, 'chip': 1, 'chips': 1, 'semiconductor': 1, 'data center': 1,
        'gpu': 1, 'openai': 1, 'claude': 1, 'gemini': 1, 'gpt': 1,
        'autonomous': 1, 'robotaxi': 1, 'self-driving': 1, 'fsd': 1,
        'quantum': 1, 'aws': 1, 'azure': 1, 'copilot': 1, 'blackwell': 1,
        'agentforce': 1, 'watsonx': 1, 'robotics': 1,
    },
    'Collaboration & Acquisitions': {
        'acquisition': 3, 'acquire': 3, 'merger': 3, 'joint venture': 3,
        'partnership': 1, 'partner': 1, 'collaborate': 1, 'deal': 1,
        'agreement': 1, 'invest': 1, 'funding': 1, 'stake': 1,
        'contract': 1, 'license': 1, 'backed': 1,
    },
    'AI-related Policy & Regulation': {
        'antitrust': 3, 'executive order': 3, 'export controls': 3, 'export ban': 3,
        'regulation': 1, 'regulatory': 1, 'government': 1, 'ftc': 1, 'doj': 1,
        'congress': 1, 'china': 1, 'us-china': 1, 'tariff': 1, 'sanctions': 1,
        'pentagon': 1, 'defense contract': 1, 'national security': 1,
    },
    'Leadership & Governance': {
        'steps down': 3, 'resignation': 3, 'appointed': 3, 'new ceo': 3,
        'ceo': 1, 'cfo': 1, 'board': 1, 'layoffs': 1, 'job cuts': 1,
        'elon musk': 1, 'musk': 1, 'compensation': 1, 'buyback': 1,
        'dividend': 1, 'shareholder vote': 1, 'earnings call': 1,
    },
    'Litigation & Security': {
        'lawsuit': 3, 'sued': 3, 'class action': 3, 'securities fraud': 3,
        'data breach': 3, 'cyberattack': 3, 'fraud': 3,
        'court': 1, 'settlement': 1, 'patent': 1, 'cybersecurity': 1,
        'hack': 1, 'vulnerability': 1, 'fine': 1, 'penalty': 1,
    },
}

_WORD_CACHE = {}

def _get_word_re(kw):
    if kw not in _WORD_CACHE:
        import re as _re
        _WORD_CACHE[kw] = _re.compile(r'\b' + _re.escape(kw) + r'\b')
    return _WORD_CACHE[kw]

def classify_text(text):
    """Return (category, confidence_score) for a piece of text."""
    text = text.lower()
    scores = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        score = 0
        for kw, wt in kws.items():
            if ' ' in kw:
                if kw in text:
                    score += wt
            else:
                if _get_word_re(kw).search(text):
                    score += wt
        scores[cat] = score

    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]

    # Override rules
    if scores['Litigation & Security'] >= 3 and scores['Litigation & Security'] >= best_score:
        best_cat = 'Litigation & Security'
        best_score = scores['Litigation & Security']
    elif scores['AI-related Policy & Regulation'] >= 3 and best_cat == 'AI Innovation':
        best_cat = 'AI-related Policy & Regulation'
        best_score = scores['AI-related Policy & Regulation']

    if best_score < 1:
        return 'Other / Unclear', 0

    return best_cat, best_score


# ── Alpha Vantage fetch ───────────────────────────────────────────────────────
def fetch_av_news(ticker, api_key, limit=50):
    """Fetch latest news for a ticker from Alpha Vantage."""
    params = {
        'function':   'NEWS_SENTIMENT',
        'tickers':    ticker,
        'limit':      limit,
        'sort':       'LATEST',
        'apikey':     api_key,
    }
    try:
        r = requests.get(AV_BASE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if 'feed' not in data:
            print(f"  WARNING: No 'feed' in response for {ticker}: {list(data.keys())}")
            return []
        return data['feed']
    except Exception as e:
        print(f"  ERROR fetching {ticker}: {e}")
        return []


def parse_articles(articles, ticker):
    """Parse AV news feed into a DataFrame."""
    rows = []
    for art in articles:
        title   = art.get('title', '')
        summary = art.get('summary', '')
        source  = art.get('source', '')
        time_pub = art.get('time_published', '')
        overall_score = float(art.get('overall_sentiment_score', 0))

        # Find ticker-specific sentiment
        tkr_score = overall_score
        for ts in art.get('ticker_sentiment', []):
            if ts.get('ticker') == ticker:
                tkr_score = float(ts.get('ticker_sentiment_score', overall_score))
                break

        # Classify
        combined = f"{ticker} {title} {summary} {source}"
        category, confidence = classify_text(combined)

        rows.append({
            'ticker':                  ticker,
            'title':                   title,
            'summary':                 summary,
            'source':                  source,
            'time_published':          time_pub,
            'overall_sentiment_score': overall_score,
            'ticker_sentiment_score':  tkr_score,
            'final_news_category':     category,
            'category_confidence':     confidence,
            'fetch_date':              datetime.today().strftime('%Y-%m-%d'),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Aggregate daily features ──────────────────────────────────────────────────
def aggregate_daily(df):
    """Collapse article rows to daily aggregates per ticker."""
    if len(df) == 0:
        return pd.DataFrame()

    df['overall_sentiment_score'] = pd.to_numeric(
        df['overall_sentiment_score'], errors='coerce').fillna(0)

    agg = df.groupby('ticker').agg(
        sentiment_mean       =('overall_sentiment_score', 'mean'),
        ticker_sent_mean     =('ticker_sentiment_score',  'mean'),
        article_count        =('title',                   'count'),
        ai_innovation        =('final_news_category',
                               lambda x: (x == 'AI Innovation').sum()),
        collaboration        =('final_news_category',
                               lambda x: (x == 'Collaboration & Acquisitions').sum()),
        leadership           =('final_news_category',
                               lambda x: (x == 'Leadership & Governance').sum()),
        regulation           =('final_news_category',
                               lambda x: (x == 'AI-related Policy & Regulation').sum()),
        litigation           =('final_news_category',
                               lambda x: (x == 'Litigation & Security').sum()),
    ).reset_index()

    agg['date']             = datetime.today().strftime('%Y-%m-%d')
    agg['sentiment_score']  = ((agg['sentiment_mean'] + 1) / 2 * 100).round(1)

    # Simple directional prediction:
    # positive sentiment mean + AI Innovation > Litigation -> predict UP
    agg['prediction']  = (agg['sentiment_mean'] > 0).astype(int)
    agg['probability'] = (0.50 + agg['sentiment_mean'].clip(-0.4, 0.4) * 0.5).round(3)

    return agg


# ── Fetch price from yfinance ─────────────────────────────────────────────────
def fetch_prices(tickers):
    """Get latest closing prices using yfinance."""
    try:
        import yfinance as yf
        data = yf.download(tickers, period='5d', progress=False, auto_adjust=True)
        closes = data['Close'].iloc[-1]
        prev   = data['Close'].iloc[-2]
        result = {}
        for t in tickers:
            if t in closes.index:
                c = float(closes[t])
                p = float(prev[t])
                result[t] = {
                    'close':        round(c, 2),
                    'daily_return': round((c - p) / p, 5) if p else 0.0,
                }
        return result
    except Exception as e:
        print(f"  yfinance error: {e} — prices skipped")
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', required=True, help='Alpha Vantage API key')
    parser.add_argument('--tickers', nargs='+', default=TICKERS,
                        help='Tickers to fetch (default: all 16)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  StockIQ Daily Refresh — {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Tickers: {', '.join(args.tickers)}")
    print(f"{'='*60}\n")

    # 1. Fetch articles
    all_articles = []
    for i, ticker in enumerate(args.tickers):
        print(f"  [{i+1}/{len(args.tickers)}] Fetching {ticker}...")
        articles = fetch_av_news(ticker, args.api_key)
        df = parse_articles(articles, ticker)
        if len(df):
            all_articles.append(df)
            cats = df['final_news_category'].value_counts().to_dict()
            print(f"    {len(df)} articles | sentiment={df['overall_sentiment_score'].mean():.3f} | {cats}")
        else:
            print(f"    No articles returned")

        # Rate limit: 5 calls/min on free tier
        if i < len(args.tickers) - 1:
            time.sleep(SLEEP)

    if not all_articles:
        print("\nERROR: No articles fetched. Check your API key.")
        return

    combined = pd.concat(all_articles, ignore_index=True)

    # 2. Save raw articles
    raw_path = DATA_DIR / f"articles_{datetime.today().strftime('%Y%m%d')}.csv"
    combined.to_csv(raw_path, index=False)
    print(f"\n  Raw articles saved: {raw_path}")

    # 3. Aggregate daily
    daily = aggregate_daily(combined)

    # 4. Fetch prices
    print("\n  Fetching prices...")
    prices = fetch_prices(args.tickers)
    if prices:
        daily['close'] = daily['ticker'].map(
            lambda t: prices.get(t, {}).get('close', None))
        daily['daily_return'] = daily['ticker'].map(
            lambda t: prices.get(t, {}).get('daily_return', None))

    # 5. Append to live predictions history
    live_path = DATA_DIR / 'live_predictions.csv'
    if live_path.exists():
        existing = pd.read_csv(live_path)
        # Remove today's rows if re-running
        existing = existing[existing['date'] != datetime.today().strftime('%Y-%m-%d')]
        updated = pd.concat([existing, daily], ignore_index=True)
    else:
        updated = daily

    updated.to_csv(live_path, index=False)

    print(f"\n  Live predictions updated: {live_path}")
    print(f"  Tickers updated: {', '.join(daily['ticker'].tolist())}")
    print(f"\n{'='*60}")
    print(f"  Refresh complete. Streamlit will reload automatically.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
