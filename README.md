# StockIQ — AI Stock Sentiment Monitor

**Cal Poly Pomona MSBA Capstone · SJ**  
Research prototype. Not financial advice.

---

## What this is

A live dashboard that shows:
- Daily sentiment score per stock (FinBERT + Alpha Vantage)
- Today's topic mix (keyword classifier from Analytical Objective 2)
- 14-day sentiment vs price chart
- Next-day directional prediction (Phase 2 best model per stock)
- All 16 stocks overview

Data: 17,761 articles · 16 AI/tech stocks · Jan 2022–Dec 2025  
Best model: Version 2f (FinBERT General News) · Mean MASE 0.5168

---

## Deploy to Streamlit Cloud (free, ~10 minutes)

### Step 1 — Push to GitHub

```bash
# Create a new GitHub repo (public or private) and push this folder
git init
git add .
git commit -m "Initial deploy"
git remote add origin https://github.com/YOUR_USERNAME/stockiq-dashboard.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **New app**
4. Select your repo → branch: `main` → Main file: `app.py`
5. Click **Deploy**

Your dashboard will be live at:  
`https://YOUR_USERNAME-stockiq-dashboard-app-XXXXX.streamlit.app`

---

## Set up daily auto-refresh (GitHub Actions)

### Step 1 — Add your Alpha Vantage API key as a secret

1. In your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `AV_API_KEY`
4. Value: your Alpha Vantage API key (get one free at https://www.alphavantage.co/support/#api-key)
5. Click **Add secret**

### Step 2 — Enable GitHub Actions

The file `.github/workflows/daily_refresh.yml` is already included.  
It runs automatically at **9:45 AM ET every weekday**.

To trigger manually:
1. Go to your repo → **Actions** tab
2. Click **Daily Sentiment Refresh**
3. Click **Run workflow**

### Step 3 — Verify it works

After the first run, check that `data/live_predictions.csv` was updated in your repo.  
Streamlit reads this file and refreshes automatically.

---

## Run manually (instead of GitHub Actions)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the daily refresh
python daily_refresh.py --api_key YOUR_AV_KEY

# Run the dashboard locally
streamlit run app.py
```

---

## File structure

```
stockiq_dashboard/
├── app.py                          # Main Streamlit dashboard
├── daily_refresh.py                # Daily data pull script
├── requirements.txt                # Python dependencies
├── daily_data.csv                  # Historical data (2022-2025)
├── best_models.csv                 # Best model per ticker
├── data/
│   └── live_predictions.csv        # Updated daily by GitHub Actions
└── .github/
    └── workflows/
        └── daily_refresh.yml       # Automatic daily schedule
```

---

## Adding your Alpha Vantage API key locally

Create a `.env` file (never commit this):
```
AV_API_KEY=your_key_here
```

Or pass it directly:
```bash
python daily_refresh.py --api_key your_key_here
```

---

## Free tier limits

- Alpha Vantage: 25 requests/day, 5/min
- This app uses 16 requests/day (one per ticker)
- 9 requests to spare for manual testing

---

## How the prediction works

1. `daily_refresh.py` fetches latest 50 articles per ticker from Alpha Vantage
2. Keyword classifier assigns each article to a category (AI Innovation, Collaboration, etc.)
3. Sentiment scores come directly from Alpha Vantage (no need to run FinBERT daily)
4. Features are aggregated per ticker per day
5. A pre-trained model (best per ticker from Phase 2) predicts up/down
6. Results written to `data/live_predictions.csv`
7. Streamlit reads the CSV and updates the dashboard

---

## Research context

This dashboard is the deployment layer for the Phase 2 capstone analysis:
- **Analytical Objective 2**: Keyword classification of 17,761 news articles into 5 topic categories
- **Phase 2 models**: 8 versions (2a-2h) using AV sentiment, FinBERT, keyword topics, BERTopic topics
- **Best regression version**: 2f (FinBERT General News, MASE=0.5168)
- **Best classification**: TSLA Gradient Boosting in 2f (69.70% accuracy)
