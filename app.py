"""
AI Stock Sentiment Monitor
==========================
Streamlit dashboard — SJ Capstone, Cal Poly Pomona MSBA
Data: news_keyword_bertopic_classified_v2 | Phase 2 Models (2a-2h)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os, json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Sentiment Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme colors ──────────────────────────────────────────────────────────────
DARK_BG    = "#0D1B2A"
CARD_BG    = "#1A2940"
ACCENT     = "#00C9A7"
ACCENT2    = "#F4B942"
TEXT_MAIN  = "#E8F0FE"
TEXT_DIM   = "#8A9BB5"
POSITIVE   = "#00C9A7"
NEGATIVE   = "#F45B69"
BORDER     = "#2A3F5F"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    /* Main background */
    header[data-testid="stHeader"] {{
        background-color: {DARK_BG} !important;
        border-bottom: 1px solid {BORDER};
    }}
    header[data-testid="stHeader"] * {{
        color: {TEXT_MAIN} !important;
    }}
    .stApp {{ margin-top: 0px; }}
    .stApp {{ background-color: {DARK_BG}; color: {TEXT_MAIN}; }}
    .block-container {{ padding: 3rem 2rem 2rem 2rem; max-width: 1400px; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{ background-color: {CARD_BG}; border-right: 1px solid {BORDER}; }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{ color: {TEXT_MAIN}; }}

    /* Cards */
    .metric-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
    }}
    .card-label {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: {TEXT_DIM};
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }}
    .card-value {{
        font-size: 2.4rem;
        font-weight: 800;
        color: {TEXT_MAIN};
        line-height: 1;
    }}
    .card-sub {{
        font-size: 0.82rem;
        color: {TEXT_DIM};
        margin-top: 0.4rem;
    }}
    .positive {{ color: {POSITIVE}; }}
    .negative {{ color: {NEGATIVE}; }}

    /* Section headers */
    .section-header {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: {TEXT_DIM};
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid {BORDER};
    }}

    /* Prediction badge */
    .pred-up {{
        background: rgba(0,201,167,0.15);
        border: 1px solid {POSITIVE};
        border-radius: 8px;
        padding: 0.6rem 1rem;
        color: {POSITIVE};
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
    }}
    .pred-down {{
        background: rgba(244,91,105,0.15);
        border: 1px solid {NEGATIVE};
        border-radius: 8px;
        padding: 0.6rem 1rem;
        color: {NEGATIVE};
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
    }}

    /* Page title */
    .page-title {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {TEXT_MAIN};
        margin-bottom: 0.1rem;
    }}
    .page-sub {{
        font-size: 0.85rem;
        color: {TEXT_DIM};
        margin-bottom: 1.5rem;
    }}

    /* Streamlit overrides */
    .stSelectbox > div > div {{
        background-color: {CARD_BG};
        border-color: {BORDER};
        color: {TEXT_MAIN};
    }}
    .stSelectbox > div > div > div {{
        color: {TEXT_MAIN} !important;
    }}
    div[data-testid="metric-container"] {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.8rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {CARD_BG};
        border-radius: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {TEXT_DIM};
    }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT};
        border-bottom-color: {ACCENT};
    }}
    [data-baseweb="select"] input, [data-baseweb="select"] [aria-selected] {{
        color: white !important;
    }}
    [class*="ValueContainer"] *, [class*="singleValue"] {{
        color: white !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] * {{
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    base = os.path.dirname(__file__)
    daily = pd.read_csv(os.path.join(base, 'daily_data.csv'))
    daily['matched_market_date'] = pd.to_datetime(daily['matched_market_date'])
    daily['sentiment_mean'] = pd.to_numeric(daily['sentiment_mean'], errors='coerce')
    daily['close'] = pd.to_numeric(daily['close'], errors='coerce')
    daily['daily_return'] = pd.to_numeric(daily['daily_return'], errors='coerce')

    best_models = pd.read_csv(os.path.join(base, 'best_models.csv'))

    # Load live predictions if available
    live_path = os.path.join(base, 'data', 'live_predictions.csv')
    if os.path.exists(live_path):
        live = pd.read_csv(live_path)
        live['date'] = pd.to_datetime(live['date'])
    else:
        live = None

    return daily, best_models, live

daily, best_models, live_preds = load_data()

# ── Sector map ────────────────────────────────────────────────────────────────
SECTOR_MAP = {
    'MSFT':  'AI Cloud Powerhouses',  'AMZN': 'AI Cloud Powerhouses',
    'GOOGL': 'AI Cloud Powerhouses',  'ORCL': 'AI Cloud Powerhouses',
    'NVDA':  'AI Hardware Enablers',  'AMD':  'AI Hardware Enablers',
    'INTC':  'AI Hardware Enablers',  'TSM':  'AI Hardware Enablers',
    'CRM':   'Enterprise AI Integrators', 'ADBE': 'Enterprise AI Integrators',
    'IBM':   'Enterprise AI Integrators',
    'AI':    'Pure-Play & Specialized AI', 'PLTR': 'Pure-Play & Specialized AI',
    'AAPL':  'Consumer AI Ecosystem', 'META': 'Consumer AI Ecosystem',
    'TSLA':  'AI Mobility & Robotics',
}

COMPANY_NAMES = {
    'MSFT':'Microsoft','AMZN':'Amazon','GOOGL':'Alphabet','ORCL':'Oracle',
    'NVDA':'NVIDIA','AMD':'AMD','INTC':'Intel','TSM':'TSMC',
    'CRM':'Salesforce','ADBE':'Adobe','IBM':'IBM',
    'AI':'C3.ai','PLTR':'Palantir',
    'AAPL':'Apple','META':'Meta',
    'TSLA':'Tesla',
}

ALL_TICKERS = sorted(daily['ticker'].unique())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
        <div style="font-size:1.6rem; font-weight:900; color:{ACCENT};">📊 StockIQ</div>
        <div style="font-size:0.75rem; color:{TEXT_DIM}; margin-top:0.2rem;">
            AI Sentiment Monitor
        </div>
    </div>
    """, unsafe_allow_html=True)

    selected_ticker = st.selectbox(
        "Select Stock",
        ALL_TICKERS,
        index=ALL_TICKERS.index('NVDA') if 'NVDA' in ALL_TICKERS else 0,
        format_func=lambda t: f"{t} — {COMPANY_NAMES.get(t, t)}"
    )

    lookback = st.selectbox(
        "Chart Window",
        [14, 30, 60, 90],
        format_func=lambda x: f"Last {x} trading days",
        index=0
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.75rem; color:{TEXT_DIM};">
        <b style="color:{TEXT_MAIN};">Research Prototype</b><br>
        Cal Poly Pomona — MSBA Capstone<br>
        Phase 2 Models: 2a through 2h<br>
        17,761 articles · 16 stocks<br>
        Jan 2022 – Dec 2025<br><br>
        <b style="color:{ACCENT};">Best version:</b> 2f (FinBERT General News)<br>
        <b style="color:{ACCENT};">Avg MASE:</b> 0.5168
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Data freshness
    tkr_data = daily[daily['ticker'] == selected_ticker]
    last_date = tkr_data['matched_market_date'].max()
    st.markdown(f"""
    <div style="font-size:0.72rem; color:{TEXT_DIM}; text-align:center;">
        Last data: <b style="color:{TEXT_MAIN};">{last_date.strftime('%b %d, %Y')}</b>
    </div>
    """, unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────
company_name = COMPANY_NAMES.get(selected_ticker, selected_ticker)
sector = SECTOR_MAP.get(selected_ticker, '')

st.markdown(f"""
<div style="font-size:0.85rem; font-weight:700; letter-spacing:0.12em;
            color:#00C9A7; text-transform:uppercase; margin-bottom:0.3rem;">
    Currently Viewing
</div>
<div style="font-size:2.8rem; font-weight:900; color:#E8F0FE; line-height:1.1;
            margin-bottom:0.2rem;">
    {company_name}
</div>
<div style="font-size:1.2rem; font-weight:600; color:#8A9BB5; margin-bottom:0.2rem;">
    {selected_ticker} &nbsp;·&nbsp; {sector}
</div>
<div class="page-sub">AI Sentiment & Prediction Dashboard</div>
""", unsafe_allow_html=True)

# ── Get ticker data ───────────────────────────────────────────────────────────
tkr = daily[daily['ticker'] == selected_ticker].sort_values('matched_market_date').copy()
tkr_recent = tkr.tail(lookback)
tkr_last = tkr.iloc[-1]
tkr_prev = tkr.iloc[-2] if len(tkr) > 1 else tkr.iloc[-1]

# Sentiment score: rescale from [-1,1] to [0,100]
sent_score  = int((tkr_last['sentiment_mean'] + 1) / 2 * 100)
sent_prev   = int((tkr_prev['sentiment_mean'] + 1) / 2 * 100)
sent_delta  = sent_score - sent_prev
sent_label  = 'Positive' if sent_score >= 60 else 'Neutral' if sent_score >= 40 else 'Negative'
sent_color  = POSITIVE if sent_score >= 60 else ACCENT2 if sent_score >= 40 else NEGATIVE

# Price info
# Price info — prefer live_predictions.csv if available
price_now  = tkr_last['close']
price_prev = tkr_prev['close']
price_date = last_date

if live_preds is not None:
    lp_tkr = live_preds[live_preds['ticker'] == selected_ticker]
    if len(lp_tkr) and pd.notna(lp_tkr.iloc[-1].get('close', None)):
        live_row   = lp_tkr.sort_values('date').iloc[-1]
        price_now  = live_row['close']
        price_date = pd.to_datetime(live_row['date'])
        if pd.notna(live_row.get('daily_return', None)):
            price_chg   = live_row['daily_return'] * 100
            price_color = POSITIVE if price_chg >= 0 else NEGATIVE
        else:
            price_chg   = ((price_now - price_prev) / price_prev * 100) if price_prev else 0
            price_color = POSITIVE if price_chg >= 0 else NEGATIVE
    else:
        price_chg   = ((price_now - price_prev) / price_prev * 100) if price_prev else 0
        price_color = POSITIVE if price_chg >= 0 else NEGATIVE
else:
    price_chg   = ((price_now - price_prev) / price_prev * 100) if price_prev else 0
    price_color = POSITIVE if price_chg >= 0 else NEGATIVE

# Topic mix today
topic_cols  = ['ai_innovation','collaboration','leadership','regulation','litigation']
topic_labels= ['AI Innovation','Collaboration','Leadership','Regulation','Litigation']
topic_vals  = [int(tkr_last.get(c, 0)) for c in topic_cols]
total_arts  = int(tkr_last['article_count'])

# Best model for this ticker
bm_row = best_models[best_models['Entity'] == selected_ticker]
best_model_name = bm_row['Model'].values[0] if len(bm_row) else 'Logistic Regression'
best_model_acc  = bm_row['Accuracy (%)'].values[0] if len(bm_row) else 50.0
best_model_auc  = bm_row['ROC-AUC'].values[0] if len(bm_row) else 0.5

# Prediction: use sentiment + recent return to compute a simple rule
# (In production this would call your saved sklearn model)
avg_sent_5d = tkr.tail(5)['sentiment_mean'].mean()
avg_ret_5d  = tkr.tail(5)['daily_return'].mean()
pred_score  = avg_sent_5d * 0.6 + avg_ret_5d * 0.4
pred_up     = pred_score > 0
pred_prob   = min(0.95, max(0.50, 0.50 + abs(pred_score) * 5))

# Live prediction override if available
if live_preds is not None:
    lp = live_preds[live_preds['ticker'] == selected_ticker]
    if len(lp):
        row = lp.iloc[-1]
        pred_up   = row.get('prediction', 1) == 1
        pred_prob = row.get('probability', pred_prob)

# ── TOP ROW: 4 metric cards ───────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    arrow = "↑" if sent_delta >= 0 else "↓"
    delta_color = POSITIVE if sent_delta >= 0 else NEGATIVE
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">Sentiment Score</div>
        <div class="card-value" style="color:{sent_color};">{sent_score}</div>
        <div class="card-sub">/ 100 · {sent_label} &nbsp;
            <span style="color:{delta_color};">{arrow} {abs(sent_delta)} pts vs prev day</span>
        </div>
        <div class="card-sub" style="margin-top:0.3rem;">
            Based on {total_arts} articles on last trading day
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    p_arrow = "▲" if price_chg >= 0 else "▼"
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">Last Close Price</div>
        <div class="card-value">${price_now:,.2f}</div>
        <div class="card-sub">
            <span style="color:{price_color};">{p_arrow} {abs(price_chg):.2f}% vs prev day</span>
        </div>
        <div class="card-sub" style="margin-top:0.3rem;">
            As of {price_date.strftime('%b %d, %Y')} {'· 🟢 Live' if live_preds is not None and len(live_preds[live_preds['ticker']==selected_ticker]) > 0 else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    pred_text  = "▲ UP" if pred_up else "▼ DOWN"
    pred_color = POSITIVE if pred_up else NEGATIVE
    pred_pct   = f"{pred_prob*100:.0f}%"
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">Next-Day Prediction</div>
        <div class="card-value" style="color:{pred_color}; font-size:1.8rem;">{pred_text}</div>
        <div class="card-sub">
            Confidence: <b style="color:{TEXT_MAIN};">{pred_pct}</b>
        </div>
        <div class="card-sub" style="margin-top:0.3rem;">
            Best model: {best_model_name.replace(' Classifier','').replace(' Regression','')}
            · Acc {best_model_acc:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    # 5-day sentiment trend
    sent_5d = tkr.tail(5)['sentiment_mean'].values
    trend = "Improving" if sent_5d[-1] > sent_5d[0] else "Declining"
    trend_color = POSITIVE if trend == "Improving" else NEGATIVE
    avg_arts = int(tkr.tail(30)['article_count'].mean())
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">5-Day Sentiment Trend</div>
        <div class="card-value" style="color:{trend_color}; font-size:1.5rem;">{trend}</div>
        <div class="card-sub">
            ROC-AUC: <b style="color:{TEXT_MAIN};">{best_model_auc:.4f}</b>
        </div>
        <div class="card-sub" style="margin-top:0.3rem;">
            Avg {avg_arts} articles/day (30d)
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── MIDDLE ROW: Topic mix + Sentiment vs Price ────────────────────────────────
left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown(f'<div class="section-header">Today\'s Topic Mix</div>', unsafe_allow_html=True)

    # Horizontal bar chart
    fig_topic = go.Figure()
    colors_topic = [ACCENT, '#4ECDC4', '#45B7D1', ACCENT2, NEGATIVE]
    sorted_pairs = sorted(zip(topic_vals, topic_labels, colors_topic),
                          key=lambda x: x[0])
    s_vals, s_labels, s_colors = zip(*sorted_pairs) if sorted_pairs else ([],[],[])

    fig_topic.add_trace(go.Bar(
        x=list(s_vals),
        y=list(s_labels),
        orientation='h',
        marker=dict(color=list(s_colors), opacity=0.85),
        text=[str(v) for v in s_vals],
        textposition='auto',
        textfont=dict(color=TEXT_MAIN, size=13, family='Arial Black'),
    ))
    fig_topic.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=40, t=10, b=10),
        height=260,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                   color=TEXT_DIM),
        yaxis=dict(showgrid=False, color=TEXT_MAIN, tickfont=dict(size=12)),
        showlegend=False,
        font=dict(color=TEXT_MAIN),
    )
    st.plotly_chart(fig_topic, use_container_width=True, config={'displayModeBar': False})

    # Article count note
    other_arts = max(0, total_arts - sum(topic_vals))
    st.markdown(f"""
    <div style="font-size:0.78rem; color:{TEXT_DIM}; margin-top:-1rem; padding: 0 0.5rem;">
        {total_arts} articles on last trading day · {other_arts} uncategorized
    </div>
    """, unsafe_allow_html=True)

with right_col:
    st.markdown(f'<div class="section-header">Sentiment vs Price ({lookback}D)</div>',
                unsafe_allow_html=True)

    fig_main = make_subplots(specs=[[{"secondary_y": True}]])

    # Sentiment line (rescaled to 0-100)
    sent_series = ((tkr_recent['sentiment_mean'] + 1) / 2 * 100).round(1)
    fig_main.add_trace(
        go.Scatter(
            x=tkr_recent['matched_market_date'],
            y=sent_series,
            name='Sentiment (0-100)',
            line=dict(color=ACCENT, width=2.5),
            mode='lines+markers',
            marker=dict(size=5, color=ACCENT),
            fill='tozeroy',
            fillcolor='rgba(0,201,167,0.06)',
        ),
        secondary_y=True,
    )

    # Price line
    fig_main.add_trace(
        go.Scatter(
            x=tkr_recent['matched_market_date'],
            y=tkr_recent['close'],
            name='Price ($)',
            line=dict(color=ACCENT2, width=2.5),
            mode='lines+markers',
            marker=dict(size=5, color=ACCENT2),
        ),
        secondary_y=False,
    )

    fig_main.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=40),
        height=280,
        legend=dict(
            orientation='h', yanchor='bottom', y=-0.25,
            xanchor='center', x=0.5,
            font=dict(color=TEXT_MAIN, size=12),
            bgcolor='rgba(0,0,0,0)',
        ),
        xaxis=dict(
            showgrid=True, gridcolor=BORDER, color=TEXT_DIM,
            tickformat='%b %d', tickfont=dict(size=11),
        ),
        font=dict(color=TEXT_MAIN),
        hovermode='x unified',
    )
    fig_main.update_yaxes(
        title_text="Price ($)", secondary_y=False,
        showgrid=True, gridcolor=BORDER,
        color=ACCENT2, tickfont=dict(color=ACCENT2, size=11),
    )
    fig_main.update_yaxes(
        title_text="Sentiment (0-100)", secondary_y=True,
        showgrid=False, color=ACCENT,
        tickfont=dict(color=ACCENT, size=11),
        range=[0, 100],
    )

    st.plotly_chart(fig_main, use_container_width=True, config={'displayModeBar': False})

# ── BOTTOM ROW: Model performance + Recent articles ───────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📈 Model Performance", "📰 Article History", "🏆 All Stocks"])

with tab1:
    st.markdown(f'<div class="section-header">Phase 2 Model Accuracy — {selected_ticker}</div>',
                unsafe_allow_html=True)

    MODEL_ACCS = {
        'AAPL': {'2a':60.0,'2b':57.1,'2c':58.3,'2d':61.5,'2e':55.6,'2f':60.0,'2g':54.2,'2h':54.0},
        'AMD':  {'2a':48.6,'2b':47.1,'2c':50.0,'2d':51.5,'2e':50.0,'2f':50.0,'2g':41.3,'2h':48.3},
        'AMZN': {'2a':47.3,'2b':48.8,'2c':52.9,'2d':50.0,'2e':52.9,'2f':61.3,'2g':55.5,'2h':50.4},
        'GOOGL':{'2a':52.5,'2b':50.7,'2c':64.6,'2d':64.6,'2e':64.6,'2f':63.6,'2g':51.3,'2h':54.0},
        'IBM':  {'2a':51.4,'2b':50.9,'2c':50.4,'2d':49.8,'2e':49.8,'2f':56.3,'2g':48.9,'2h':48.4},
        'INTC': {'2a':52.5,'2b':53.4,'2c':52.5,'2d':51.5,'2e':51.5,'2f':60.5,'2g':53.4,'2h':55.3},
        'META': {'2a':44.6,'2b':49.5,'2c':52.0,'2d':52.6,'2e':50.7,'2f':64.0,'2g':50.3,'2h':51.3},
        'MSFT': {'2a':53.5,'2b':51.5,'2c':52.9,'2d':53.8,'2e':53.8,'2f':55.6,'2g':47.6,'2h':50.3},
        'NVDA': {'2a':52.7,'2b':51.9,'2c':52.0,'2d':52.0,'2e':52.0,'2f':55.8,'2g':50.5,'2h':50.7},
        'ORCL': {'2a':50.2,'2b':45.6,'2c':59.6,'2d':51.1,'2e':51.1,'2f':59.6,'2g':47.7,'2h':48.5},
        'PLTR': {'2a':53.4,'2b':50.6,'2c':47.6,'2d':53.5,'2e':48.7,'2f':54.7,'2g':47.4,'2h':48.9},
        'TSLA': {'2a':49.1,'2b':51.7,'2c':51.0,'2d':47.1,'2e':50.8,'2f':69.7,'2g':51.6,'2h':51.9},
    }

    accs = MODEL_ACCS.get(selected_ticker,
                           {v: 50.0 for v in ['2a','2b','2c','2d','2e','2f','2g','2h']})
    versions = list(accs.keys())
    accuracies = list(accs.values())
    bar_colors = [ACCENT if v == '2f' else
                  ACCENT2 if v in ('2g','2h') else
                  '#3A5F8A' for v in versions]

    fig_acc = go.Figure(go.Bar(
        x=versions, y=accuracies,
        marker=dict(color=bar_colors, opacity=0.9),
        text=[f'{a:.1f}%' for a in accuracies],
        textposition='outside',
        textfont=dict(color=TEXT_MAIN, size=11),
    ))
    fig_acc.add_hline(y=50, line_dash='dash', line_color=TEXT_DIM,
                      annotation_text='50% baseline',
                      annotation_font_color=TEXT_DIM,
                      annotation_position='bottom right')
    fig_acc.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=30),
        height=280,
        xaxis=dict(showgrid=False, color=TEXT_MAIN,
                   tickfont=dict(size=12, color=TEXT_MAIN)),
        yaxis=dict(showgrid=True, gridcolor=BORDER, color=TEXT_DIM,
                   range=[30, max(accuracies)+10],
                   ticksuffix='%', tickfont=dict(size=11)),
        showlegend=False,
        font=dict(color=TEXT_MAIN),
    )
    st.plotly_chart(fig_acc, use_container_width=True, config={'displayModeBar': False})

    col_a, col_b, col_c = st.columns(3)
    col_a.markdown(f"""
    <div style="font-size:0.8rem; color:{TEXT_DIM};">
        <b style="color:{ACCENT};">■</b> Best version (2f)<br>
        <b style="color:{ACCENT2};">■</b> Topic extensions (2g, 2h)<br>
        <b style="color:#3A5F8A;">■</b> Other versions
    </div>""", unsafe_allow_html=True)
    col_b.markdown(f"""
    <div style="font-size:0.8rem; color:{TEXT_DIM};">
        Best accuracy: <b style="color:{TEXT_MAIN};">{max(accuracies):.1f}%</b><br>
        Versions above 50%: <b style="color:{TEXT_MAIN};">
        {sum(1 for a in accuracies if a > 50)}/8</b>
    </div>""", unsafe_allow_html=True)
    col_c.markdown(f"""
    <div style="font-size:0.8rem; color:{TEXT_DIM};">
        Models: 12 regression + 7 classification<br>
        Split: chronological 80/20
    </div>""", unsafe_allow_html=True)

with tab2:
    st.markdown(f'<div class="section-header">Daily Sentiment History — {selected_ticker} (Last {lookback} days)</div>',
                unsafe_allow_html=True)

    hist = tkr.tail(lookback)[['matched_market_date','sentiment_mean',
                                'close','article_count','ai_innovation',
                                'collaboration','leadership','regulation','litigation',
                                'daily_return']].copy()
    hist['sentiment_score'] = ((hist['sentiment_mean'] + 1) / 2 * 100).round(1)
    hist['daily_return_pct'] = (hist['daily_return'] * 100).round(2)
    hist['matched_market_date'] = hist['matched_market_date'].dt.strftime('%b %d, %Y')

    display = hist[['matched_market_date','sentiment_score','close',
                    'daily_return_pct','article_count',
                    'ai_innovation','collaboration','leadership',
                    'regulation','litigation']].copy()
    display.columns = ['Date','Sentiment (0-100)','Close ($)',
                       'Daily Return (%)','# Articles',
                       'AI Innovation','Collaboration','Leadership',
                       'Regulation','Litigation']
    display = display.sort_values('Date', ascending=False)

    st.dataframe(
        display,
        use_container_width=True,
        height=350,
        hide_index=True,
    )

with tab3:
    st.markdown(f'<div class="section-header">All 16 Stocks — Current Sentiment Overview</div>',
                unsafe_allow_html=True)

    # Build summary for all tickers
    rows = []
    for t in ALL_TICKERS:
        t_data = daily[daily['ticker'] == t].sort_values('matched_market_date')
        if len(t_data) == 0:
            continue
        last = t_data.iloc[-1]
        prev = t_data.iloc[-2] if len(t_data) > 1 else last
        sent = int((last['sentiment_mean'] + 1) / 2 * 100)
        sent_d = sent - int((prev['sentiment_mean'] + 1) / 2 * 100)
        price = last['close']
        ret = last['daily_return'] * 100 if pd.notna(last['daily_return']) else 0.0
        bm = best_models[best_models['Entity'] == t]
        acc = bm['Accuracy (%)'].values[0] if len(bm) else 50.0

        # 5-day trend
        s5 = t_data.tail(5)['sentiment_mean'].values
        trend = '↑' if len(s5) >= 2 and s5[-1] > s5[0] else '↓'

        rows.append({
            'Ticker': t,
            'Company': COMPANY_NAMES.get(t, t),
            'Sector': SECTOR_MAP.get(t, ''),
            'Sentiment': sent,
            '5D Trend': trend,
            'Close ($)': f'{price:,.2f}',
            'Day Ret%': f'{ret:+.2f}%',
            'Best Model Acc': f'{acc:.1f}%',
            '# Articles': int(last['article_count']),
        })

    summary_df = pd.DataFrame(rows)

    # Color sentiment column
    def color_sent(val):
        if val >= 60:
            return f'color: {POSITIVE}'
        elif val < 40:
            return f'color: {NEGATIVE}'
        return f'color: {ACCENT2}'

    st.dataframe(
        summary_df,
        use_container_width=True,
        height=500,
        hide_index=True,
        column_config={
            'Sentiment': st.column_config.ProgressColumn(
                'Sentiment (0-100)',
                min_value=0, max_value=100,
                format='%d',
            ),
        }
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; font-size:0.75rem; color:{TEXT_DIM};
            border-top: 1px solid {BORDER}; padding-top: 1rem;">
    AI Stock Sentiment Monitor · Cal Poly Pomona MSBA Capstone ·
    Research prototype using historical data (2022–2025) ·
    Not financial advice
</div>
""", unsafe_allow_html=True)
