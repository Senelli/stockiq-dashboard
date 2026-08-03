# StockIQ — AI Stock Sentiment Monitor

**Cal Poly Pomona MSBA Capstone · GBA Capstone Group**  
Live research prototype. Not financial advice.

🌐 **Live dashboard:** https://stockiq-dashboard.streamlit.app  
📁 **GitHub repo:** https://github.com/Senelli/stockiq-dashboard

---

## Contributors

This dashboard and the underlying research study were produced as part of the Cal Poly Pomona Master of Science in Business Analytics (MSBA) capstone project.

| Name | Role |
|---|---|
| **Aneesh Aryal** | Phase 2 sentiment model development and evaluation · 8 model versions (2a–2h) · 19 models per ticker · Performance evaluation via MASE, Accuracy, and AUC |
| **Aditi Bhatnagar** | Phase 1 price-only baseline models · 19 models per ticker (12 regression, 7 classification) · R², MASE, directional accuracy, F1, ROC-AUC evaluation · Baseline performance floor establishment |
| **Gilbert Garcia** | Data collection and organization · Historical price data and ~45,000 AI news articles via Alpha Vantage API · Dataset alignment by company, timestamp, and trading date · Event-study methodology for abnormal returns analysis |
| **Anna He** | Market impact analysis across six sector groups · Phase 1 vs Phase 2 consolidation pipeline · Group-level rollup logic · Sector normalization and coverage analysis |
| **Senelli Jinadasa** | Dashboard creator · Keyword classification pipeline · BERTopic topic modeling · Phase 2 versions 2g and 2h (topic feature extensions) · Analytical Objective 2 |
| **Ishmam Kamal** | FinBERT sentiment scoring · Ticker-specific and general market news sentiment · Model evaluation using lagged returns and sentiment features · Comparison of ticker-specific vs general news sentiment effectiveness |

**Committee Chair:** Dr. Xuesong (Sonya) Zhang, Cal Poly Pomona  
**Committee Member:** Dr. Hyounae (Kelly) Min, Cal Poly Pomona

---

## What this is

A fully interactive dashboard connected to actual project data, model results, and live market data. It is not a mock-up.

**What the dashboard shows:**

- **Sentiment score** — daily sentiment from Alpha Vantage (aggregates 50+ sources including Reuters, Bloomberg, CNBC, MarketWatch). Labeled as Strongly Bullish / Bullish / Neutral / Bearish / Strongly Bearish
- **Live closing price** — pulled daily via yfinance, shown with day-over-day change
- **Next-day directional prediction** — UP or DOWN with confidence percentage, based on best Phase 2 model per ticker
- **5-day sentiment trend** — Improving or Declining
- **Today's topic mix** — keyword classifier counts per category: AI Innovation, Collaboration & Acquisitions, Leadership & Governance, AI-related Policy & Regulation, Litigation & Security
- **Sentiment vs Price chart** — study period Jan 2022–Dec 2025, configurable window (14 / 30 / 60 / 90 trading days)
- **Article history** — daily sentiment and topic breakdown, same configurable window
- **Model accuracy tab** — Phase 2 classification accuracy across all 8 versions (2a–2h) per stock
- **All 16 stocks overview** — live sentiment scores, prices, predictions in one table

**Data:**
- 17,761 news articles · 16 AI/tech stocks · Jan 2022–Dec 2025
- Stocks: AAPL, ADBE, AI, AMD, AMZN, CRM, GOOGL, IBM, INTC, META, MSFT, NVDA, ORCL, PLTR, TSLA, TSM
- Grouped into 6 sectors: AI Cloud Powerhouses, AI Hardware Enablers, Enterprise AI Integrators, Pure-Play & Specialized AI, Consumer AI Ecosystem, AI Mobility & Robotics

---

## How the daily refresh works

Every weekday at **9:45 AM ET**, GitHub Actions automatically runs `daily_refresh.py` which:

1. Pulls the latest 50 articles per ticker from Alpha Vantage News Sentiment API (16 API calls total — free tier allows 25/day)
2. Runs the keyword classifier on all article summaries to assign topic categories (AI Innovation, Collaboration, Leadership, Regulation, Litigation)
3. Aggregates Alpha Vantage sentiment scores per ticker per day
4. Pulls latest closing price and daily return via yfinance
5. Computes next-day directional prediction based on sentiment direction
6. Writes results to `data/live_predictions.csv`
7. GitHub Actions commits and pushes the updated CSV to the repo
8. Streamlit reads the updated file and refreshes automatically

No manual intervention required after setup.

---

## Sentiment scoring

Sentiment scores come from the **Alpha Vantage overall_sentiment_score** field, which is returned directly in the API response for each article. Alpha Vantage aggregates news from 50+ sources including Reuters, Bloomberg, CNBC, MarketWatch, and others. The raw score is in [-1, 1] and is rescaled to 0–100 for display.

FinBERT (ProsusAI/finbert) was used during the Phase 2 research study for historical scoring and is included in `daily_refresh.py` as an optional scoring layer — it runs in GitHub Actions if the `--skip_finbert` flag is not passed.

---

## Prediction method

The next-day directional prediction uses the mean sentiment score from today's articles:

- Positive mean sentiment → predict UP
- Negative mean sentiment → predict DOWN
- Confidence is derived from the magnitude of the sentiment signal

The best Phase 2 classification model per ticker (from `best_models.csv`) is shown for reference alongside its accuracy and ROC-AUC from the research study.

---

## Phase 2 model versions

| Version | Sentiment Source | Topic Features | Overall Rank |
|---|---|---|---|
| 2a | AV General Sentiment | None | 3 |
| 2b | AV Ticker Sentiment | None | 7 |
| 2c | FinBERT Ticker-Linked | None | 5 |
| 2d | AV + FinBERT Combined | None | 2 |
| 2e | FinBERT Ticker-Focused | None | 4 |
| 2f | FinBERT General News | None | **1 (best)** |
| 2g | FinBERT General News | 7 keyword topic flags | 5 |
| 2h | FinBERT General News | 28 BERTopic cluster flags | 7 |

Best regression version: **2f** — Mean MASE 0.5168  
Best classification result: **TSLA Gradient Boosting (2f)** — 69.70% accuracy

---

## File structure

```
stockiq_dashboard/
├── app.py                          # Main Streamlit dashboard
├── daily_refresh.py                # Daily data pull and scoring script
├── requirements.txt                # Python dependencies (Streamlit Cloud)
├── daily_data.csv                  # Historical sentiment + price (Jan 2022–Dec 2025)
├── best_models.csv                 # Best Phase 2 model per ticker
├── data/
│   └── live_predictions.csv        # Updated daily by GitHub Actions
└── .github/
    └── workflows/
        └── daily_refresh.yml       # Weekday schedule: 9:45 AM ET Mon–Fri
```

---

## Setup — Deploy to Streamlit Cloud

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial deploy"
git remote add origin https://github.com/YOUR_USERNAME/stockiq-dashboard.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **Create app** → **Deploy a public app from GitHub**
4. Repository: `YOUR_USERNAME/stockiq-dashboard`
5. Branch: `main` · Main file: `app.py`
6. App URL: `stockiq-dashboard`
7. Click **Deploy**

### Step 3 — Add Alpha Vantage API key

**Streamlit Secrets** (for the dashboard):
1. Manage app → Settings → Secrets
2. Add: `AV_API_KEY = "your_key_here"`

**GitHub Secrets** (for daily refresh):
1. Repo → Settings → Secrets and variables → Actions
2. New secret: Name `AV_API_KEY`, Value: your key

Get a free key at: https://www.alphavantage.co/support/#api-key

---

## Run locally

```bash
pip install -r requirements.txt

# Run daily refresh manually
python daily_refresh.py --api_key YOUR_AV_KEY

# Run dashboard locally
streamlit run app.py
```

---

## Free tier limits

| Service | Limit | This app uses |
|---|---|---|
| Alpha Vantage | 25 requests/day, 5/min | 16/day (one per ticker) |
| GitHub Actions | 2,000 min/month | ~4 min/day = ~80 min/month |
| Streamlit Cloud | 1 app free | 1 app |
| yfinance | Unlimited (unofficial) | 1 batch call/day |

---

## Research context

This dashboard is the deployment layer for the Cal Poly Pomona MSBA capstone study:

- **Analytical Objective 2**: Keyword classification of 17,761 news articles into 5 topic categories (AI Innovation, Collaboration & Acquisitions, Leadership & Governance, AI-related Policy & Regulation, Litigation & Security) using a weighted keyword-mapping classifier achieving 89.8% classification rate. BERTopic with FinBERT embeddings applied for exploratory semantic clustering.
- **Phase 2 models**: 8 versions (2a–2h) testing Alpha Vantage sentiment, ticker-linked FinBERT, general news FinBERT, keyword topic flags, and BERTopic cluster flags as predictive features for next-day stock returns across 16 AI/tech stocks
- **Best regression version**: 2f — FinBERT General News — Mean MASE 0.5168
- **Best classification result**: TSLA Gradient Boosting Classifier in Version 2f — 69.70% accuracy, AUC 0.6682
- **Study period**: January 2022 – December 2025
