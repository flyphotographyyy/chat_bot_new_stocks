# -*- coding: utf-8 -*-
# Stock Signals PRO – Enhanced Multi-Source Analysis (Streamlit-ready)

import math
import re
import json
import datetime as dt
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
import pytz
import requests
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional extras (safe fallbacks)
try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    import pandas_market_calendars as mcal
except Exception:
    mcal = None

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

APP_TITLE = "Stock Signals PRO – Enhanced Multi-Source Analysis"

# ---------- CONFIG ----------
CACHE_TTL = 15 * 60         # 15 minutes – controls st.cache_data TTL
EARNINGS_WINDOW_DAYS = 7    # show earnings due within X days
MAX_WORKERS = 8             # parallel requests

WATCHLIST_FILE = Path.home() / "stock_signals_watchlist.json"
SETTINGS_FILE  = Path.home() / "stock_signals_settings.json"

MARKETS = {
    "US – NYSE/Nasdaq (09:30–16:00 ET)": {"tz": "America/New_York", "open": (9,30), "close": (16,0), "cal": "XNYS"},
    "Germany – XETRA (09:00–17:30 DE)":  {"tz": "Europe/Berlin",    "open": (9,0),  "close": (17,30), "cal": "XETR"},
    "UK – LSE (08:00–16:30 UK)":         {"tz": "Europe/London",    "open": (8,0),  "close": (16,30), "cal": "XLON"},
    "France – Paris (09:00–17:30 FR)":   {"tz": "Europe/Paris",     "open": (9,0),  "close": (17,30), "cal": "XPAR"},
    "Japan – TSE (09:00–15:00 JST)":     {"tz": "Asia/Tokyo",       "open": (9,0),  "close": (15,0),  "cal": "XTKS"},
    "Australia – ASX (10:00–16:00 AEST)":{"tz": "Australia/Sydney", "open": (10,0), "close": (16,0),  "cal": "XASX"},
}

# ---------- PERSISTENCE ----------
def _ensure_files():
    WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

def load_watchlist() -> List[str]:
    _ensure_files()
    try:
        if WATCHLIST_FILE.exists():
            data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                out = []
                for t in data:
                    if isinstance(t, str) and t.strip():
                        t2 = t.strip().upper()
                        if 1 <= len(t2) <= 12:
                            out.append(t2)
                return sorted(set(out))
    except Exception:
        pass
    default = ["MSFT","AMZN","GOOGL","NVDA","META","AAPL","TSLA"]
    save_watchlist(default)
    return default

def save_watchlist(tickers: List[str]) -> None:
    _ensure_files()
    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    WATCHLIST_FILE.write_text(json.dumps(sorted(set(tickers)), indent=2), encoding="utf-8")

def load_settings() -> Dict:
    _ensure_files()
    try:
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    default = {
        "risk_profile": "balanced",
        "lookback_days": 120,
        "interval": "30m",
        "use_news": True,
        "news_days": 7,
        "indicators": ["RSI","MACD","Bollinger"],
        "auto_refresh": True
    }
    SETTINGS_FILE.write_text(json.dumps(default, indent=2), encoding="utf-8")
    return default

def save_settings(obj: Dict) -> None:
    _ensure_files()
    SETTINGS_FILE.write_text(json.dumps(obj, indent=2), encoding="utf-8")

# ---------- MARKET HOURS ----------
def is_market_open(profile_key: str) -> bool:
    prof = MARKETS.get(profile_key)
    if not prof:
        return False
    tz = pytz.timezone(prof["tz"])
    now = dt.datetime.now(tz)
    if mcal and prof.get("cal"):
        try:
            cal = mcal.get_calendar(prof["cal"])
            sched = cal.schedule(start_date=now.date(), end_date=now.date())
            if sched.empty:
                return False
            o = sched.iloc[0]["market_open"].tz_convert(tz)
            c = sched.iloc[0]["market_close"].tz_convert(tz)
            return o <= now < c
        except Exception:
            pass
    if now.weekday() > 4:
        return False
    o_h,o_m = prof["open"]; c_h,c_m = prof["close"]
    o = now.replace(hour=o_h, minute=o_m, second=0, microsecond=0)
    c = now.replace(hour=c_h, minute=c_m, second=0, microsecond=0)
    return o <= now < c

# ---------- CACHED FETCHERS ----------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_price_history(ticker: str, days: int, interval: str = "1d") -> pd.DataFrame:
    if interval == "30m":
        period = "60d"  # Yahoo limit for 30m
        interv = "30m"
    else:
        period = f"{days}d"
        interv = "1d"
    df = yf.download(ticker, period=period, interval=interv, auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]
    df = df.rename(columns=lambda x: str(x).strip())
    return df.dropna(subset=["Close"])

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_info_fast(ticker: str) -> Dict:
    out = {}
    try:
        tk = yf.Ticker(ticker)
        # fast_info where possible
        try:
            fi = getattr(tk, "fast_info", None) or {}
        except Exception:
            fi = {}
        out["beta"] = fi.get("beta", None)
        # legacy info (can fail)
        try:
            inf = tk.info or {}
        except Exception:
            inf = {}
        out["marketCap"] = inf.get("marketCap")
        out["trailingPE"] = inf.get("trailingPE")
        out["forwardPE"] = inf.get("forwardPE")
        out["dividendYield"] = inf.get("dividendYield") * 100 if inf.get("dividendYield") else None
        out["sector"] = inf.get("sector")
        out["industry"] = inf.get("industry")
    except Exception:
        pass
    return out

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_spy_close(days: int, interval: str) -> pd.Series:
    spy = get_price_history("SPY", days, interval)
    return spy["Close"] if not spy.empty else pd.Series(dtype=float)

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_multi_source_news(ticker: str, days: int = 7) -> List[Dict]:
    items = []
    try:
        google_url = f"https://news.google.com/rss/search?q={ticker}+stock+when:{days}d&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(google_url)
        for e in feed.entries[:25]:
            d = dt.datetime(*e.published_parsed[:6]) if e.get("published_parsed") else dt.datetime.utcnow()
            items.append({"title": e.get("title",""), "source":"Google News", "published": d, "url": e.get("link","")})
    except Exception:
        pass
    try:
        yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(yahoo_url)
        for e in feed.entries[:20]:
            d = dt.datetime(*e.published_parsed[:6]) if e.get("published_parsed") else dt.datetime.utcnow()
            items.append({"title": e.get("title",""), "source":"Yahoo Finance", "published": d, "url": e.get("link","")})
    except Exception:
        pass
    return items

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_upcoming_earnings(symbol: str) -> Optional[dt.date]:
    """Fetch next earnings date via Yahoo quoteSummary (more reliable)."""
    try:
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        params = {"modules": "calendarEvents"}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        js = r.json()
        node = js["quoteSummary"]["result"][0]["calendarEvents"]["earnings"]["earningsDate"]
        if not node:
            return None
        raw = node[0]["raw"] if isinstance(node, list) else node["raw"]
        dt_utc = dt.datetime.fromtimestamp(int(raw), tz=pytz.UTC)
        return dt_utc.date()
    except Exception:
        return None

# ---------- INDICATORS ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0))
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    avg_gain = avg_gain.shift(1).ewm(alpha=1/period, adjust=False).mean()
    avg_loss = avg_loss.shift(1).ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal = ema(macd_line, sig)
    hist = macd_line - signal
    return macd_line, signal, hist

def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series,pd.Series,pd.Series]:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    c = df["Close"]
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["RSI14"] = rsi(c, 14)
    macd_line, macd_sig, macd_hist = macd(c, 12, 26, 9)
    df["MACD"] = macd_line; df["MACD_SIG"] = macd_sig; df["MACD_HIST"] = macd_hist
    u,m,l = bollinger_bands(c, 20, 2)
    df["BB_Upper"]=u; df["BB_Middle"]=m; df["BB_Lower"]=l
    df["BB_Position"] = np.where((u-l)!=0, (c - l) / (u - l) * 100, np.nan)
    # 52w hi/low
    w = min(len(df), 252)
    df["HI52"] = c.rolling(w).max()
    df["LO52"] = c.rolling(w).min()
    # momentum & volume
    df["Return_5d"]  = c.pct_change(5)*100
    df["Return_20d"] = c.pct_change(20)*100
    if "Volume" in df.columns:
        vma = df["Volume"].rolling(20).mean()
        df["Volume_Ratio"] = np.where(vma>0, df["Volume"]/vma, np.nan)
    else:
        df["Volume_Ratio"] = np.nan
    return df

def compute_relative_strength20(stock_close: pd.Series, spy_close: pd.Series) -> Optional[float]:
    if stock_close is None or spy_close is None or len(stock_close)<21 or len(spy_close)<21:
        return None
    try:
        s = (stock_close.iloc[-1] / stock_close.iloc[-21]) - 1.0
        b = (spy_close.iloc[-1]   / spy_close.iloc[-21])   - 1.0
        return float((s - b) * 100.0)  # percentage points vs SPY
    except Exception:
        return None

# ---------- SENTIMENT ----------
def analyze_sentiment(headlines: List[Dict]) -> Dict[str,float]:
    if not headlines:
        return {"compound": 0.0, "n": 0, "confidence": 0.0}
    vader = SentimentIntensityAnalyzer()
    vs = []
    tb = []
    for n in headlines:
        t = n.get("title","")
        s = vader.polarity_scores(t)["compound"]
        vs.append(s)
        if TextBlob:
            try:
                tb.append(TextBlob(t).sentiment.polarity)
            except Exception:
                pass
    vmean = np.mean(vs) if vs else 0.0
    tmean = np.mean(tb) if tb else None
    comb = (0.7*vmean + 0.3*tmean) if tmean is not None else vmean
    std = np.std(vs) if len(vs)>1 else 0.0
    conf = max(0.0, 1 - std/2)
    return {"compound": float(comb), "n": len(headlines), "confidence": float(conf)}

# ---------- CLASSIFIER ----------
def enhanced_signal_classification(ticker: str, df: pd.DataFrame, news: Optional[Dict],
                                   risk_profile: str, rs20: Optional[float], info: Dict,
                                   earn_date: Optional[dt.date]) -> Dict:
    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df)>=2 else row
    price = float(row["Close"])
    sma20 = row.get("SMA20", np.nan); sma50 = row.get("SMA50", np.nan); sma200 = row.get("SMA200", np.nan)
    rsi_now = row.get("RSI14", np.nan); rsi_prev = prev.get("RSI14", np.nan)
    bb_pos = row.get("BB_Position", np.nan)

    signals = {"technical":0, "momentum":0, "volume":0, "sentiment":0, "relative":0, "fundamental":0}
    reasons = []
    confs = []

    # Moving averages
    if not any(pd.isna([sma20,sma50,sma200])):
        if price > sma20 > sma50 > sma200:
            signals["technical"] += 20; reasons.append("Uptrend: price > SMA20 > SMA50 > SMA200"); confs.append(0.9)
        elif price < sma20 < sma50 < sma200:
            signals["technical"] -= 20; reasons.append("Downtrend: price < SMA20 < SMA50 < SMA200"); confs.append(0.9)
        elif price > sma50:
            signals["technical"] += 8; reasons.append("Above SMA50"); confs.append(0.6)

    # RSI
    if not pd.isna(rsi_now) and not pd.isna(rsi_prev):
        if rsi_now < 30:
            signals["technical"] += 12; reasons.append(f"RSI oversold ({rsi_now:.1f})"); confs.append(0.8)
        elif rsi_now > 70:
            signals["technical"] -= 12; reasons.append(f"RSI overbought ({rsi_now:.1f})"); confs.append(0.8)
        elif rsi_prev < 50 <= rsi_now:
            signals["technical"] += 6; reasons.append("RSI crossed above 50"); confs.append(0.6)
        elif rsi_prev > 50 >= rsi_now:
            signals["technical"] -= 6; reasons.append("RSI crossed below 50"); confs.append(0.6)

    # MACD cross
    macd_now, macd_sig_now = row.get("MACD", np.nan), row.get("MACD_SIG", np.nan)
    macd_prev, macd_sig_prev = prev.get("MACD", np.nan), prev.get("MACD_SIG", np.nan)
    if not any(pd.isna([macd_now, macd_sig_now, macd_prev, macd_sig_prev])):
        if macd_prev < macd_sig_prev and macd_now > macd_sig_now:
            signals["momentum"] += 10; reasons.append("MACD bullish crossover"); confs.append(0.7)
        elif macd_prev > macd_sig_prev and macd_now < macd_sig_now:
            signals["momentum"] -= 10; reasons.append("MACD bearish crossover"); confs.append(0.7)

    # Bollinger position
    if not pd.isna(bb_pos):
        if bb_pos < 10:
            signals["technical"] += 6; reasons.append("Near Bollinger lower band"); confs.append(0.5)
        elif bb_pos > 90:
            signals["technical"] -= 6; reasons.append("Near Bollinger upper band"); confs.append(0.5)

    # Volume
    vr = row.get("Volume_Ratio", np.nan)
    if not pd.isna(vr):
        if vr > 1.5:
            signals["volume"] += 6; reasons.append(f"High volume ({vr:.1f}× avg)"); confs.append(0.4)
        elif vr < 0.5:
            signals["volume"] -= 4; reasons.append("Low volume"); confs.append(0.3)

    # Momentum
    r5 = row.get("Return_5d", np.nan); r20 = row.get("Return_20d", np.nan)
    if not any(pd.isna([r5,r20])):
        if r5 > 5 and r20 > 10:
            signals["momentum"] += 10; reasons.append("Strong positive momentum"); confs.append(0.6)
        elif r5 < -5 and r20 < -10:
            signals["momentum"] -= 10; reasons.append("Strong negative momentum"); confs.append(0.6)

    # Relative strength vs SPY (20d)
    if rs20 is not None:
        if rs20 > 0:
            signals["relative"] += 6; reasons.append(f"Outperforming SPY by {rs20:+.1f}pp (20d)")
        elif rs20 < 0:
            signals["relative"] -= 6; reasons.append(f"Underperforming SPY by {rs20:+.1f}pp (20d)")

    # Sentiment
    if news and news.get("n",0) > 0:
        sc = news.get("compound", 0.0); conf = news.get("confidence", 0.5)
        if sc > 0.3:
            signals["sentiment"] += int(10*conf); reasons.append(f"Very positive news ({sc:+.2f})"); confs.append(conf)
        elif sc > 0.1:
            signals["sentiment"] += int(5*conf); reasons.append(f"Positive news ({sc:+.2f})"); confs.append(conf*0.7)
        elif sc < -0.3:
            signals["sentiment"] -= int(10*conf); reasons.append(f"Very negative news ({sc:+.2f})"); confs.append(conf)
        elif sc < -0.1:
            signals["sentiment"] -= int(5*conf); reasons.append(f"Negative news ({sc:+.2f})"); confs.append(conf*0.7)

    # Fundamentals (very soft)
    pe = info.get("forwardPE") or info.get("trailingPE")
    if pe:
        try:
            if pe < 15:
                signals["fundamental"] += 6; reasons.append(f"Low P/E ({pe:.1f})"); confs.append(0.5)
            elif pe > 30:
                signals["fundamental"] -= 4; reasons.append(f"High P/E ({pe:.1f})"); confs.append(0.4)
        except Exception:
            pass

    # Score & risk
    total = sum(signals.values())
    mult = {"conservative":0.7, "balanced":1.0, "aggressive":1.3}.get(risk_profile,1.0)
    adj = total * mult

    avg_conf = float(np.mean(confs)) if confs else 0.5

    if adj >= 25: label = "STRONG BUY"
    elif adj >= 15: label = "BUY"
    elif adj >= 5:  label = "WEAK BUY"
    elif adj <= -25: label = "STRONG SELL"
    elif adj <= -15: label = "SELL"
    elif adj <= -5:  label = "WEAK SELL"
    else: label = "HOLD"

    return {
        "ticker": ticker,
        "signal": label,
        "score": int(adj),
        "confidence": min(100, int(avg_conf*100)),
        "price": price,
        "reasons": reasons[:8],
    }

# ---------- VISUALS ----------
def plot_ticker(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxis=True,
                        vertical_spacing=0.07, row_heights=[0.6,0.2,0.2],
                        subplot_titles=[f"{ticker} Price", "RSI(14)", "MACD"])
    # price
    if all(col in df.columns for col in ["Open","High","Low","Close"]):
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                                     close=df["Close"], name="Price"), row=1,col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"), row=1,col=1)
    # MAs
    for p in [20,50]:
        col = f"SMA{p}"
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)), row=1,col=1)
    # BB
    for col, nm in [("BB_Upper","BB Upper"),("BB_Lower","BB Lower")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=nm, line=dict(width=1, dash="dash")), row=1,col=1)

    # RSI
    if "RSI14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14"), row=2,col=1)
        fig.add_hline(y=70, line_dash="dash", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", row=2, col=1)

    # MACD
    if {"MACD","MACD_SIG","MACD_HIST"}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"), row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIG"], name="Signal"), row=3,col=1)
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Hist", opacity=0.5), row=3,col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white", showlegend=True)
    return fig

# ---------- PIPELINE ----------
def process_one(ticker: str, cfg: Dict, spy_close: pd.Series) -> Optional[Tuple[Dict, Dict]]:
    try:
        df = get_price_history(ticker, cfg["lookback_days"], cfg["interval"])
        if df.empty:
            return None
        df = compute_indicators(df)

        # Relative strength vs SPY (20d)
        rs20 = compute_relative_strength20(df["Close"], spy_close)

        # Info & earnings
        info = get_info_fast(ticker)
        earn_date = get_upcoming_earnings(ticker)

        # News & sentiment
        news_sent = None
        if cfg["use_news"]:
            news = get_multi_source_news(ticker, cfg["news_days"])
            news_sent = analyze_sentiment(news)

        # Classification
        res = enhanced_signal_classification(
            ticker, df, news_sent, cfg["risk_profile"], rs20, info, earn_date
        )

        # Table row
        r = df.iloc[-1]
        earn_col = "—"
        if earn_date:
            delta = (earn_date - dt.date.today()).days
            if 0 <= delta <= EARNINGS_WINDOW_DAYS:
                earn_col = earn_date.strftime("%Y-%m-%d")

        row = {
            "Ticker": ticker,
            "Signal": res["signal"],
            "Score": res["score"],
            "Confidence": f"{res['confidence']}%",
            "Price": f"${res['price']:.2f}",
            "RSI": f"{r.get('RSI14', np.nan):.1f}" if not pd.isna(r.get('RSI14', np.nan)) else "N/A",
            "Volume Ratio": f"{r.get('Volume_Ratio', np.nan):.1f}x" if not pd.isna(r.get('Volume_Ratio', np.nan)) else "N/A",
            "5D Return": f"{r.get('Return_5d', np.nan):+.1f}%" if not pd.isna(r.get('Return_5d', np.nan)) else "N/A",
            "Sentiment": f"{news_sent.get('compound',0):+.2f}" if news_sent else "N/A",
            "News Count": news_sent.get("n",0) if news_sent else 0,
            "P/E Ratio": f"{(info.get('forwardPE') or info.get('trailingPE')):.1f}" if (info.get('forwardPE') or info.get('trailingPE')) else "N/A",
            "RS vs SPY (20d)": f"{rs20:+.1f}pp" if rs20 is not None else "N/A",
            "Earnings ≤7d": earn_col,
        }
        return res, row
    except Exception:
        return None

def scan_tickers(tickers: List[str], cfg: Dict) -> Tuple[List[Dict], List[Dict]]:
    results, rows = [], []
    spy_close = get_spy_close(cfg["lookback_days"], cfg["interval"])
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(tickers)))) as ex:
        futs = {ex.submit(process_one, t, cfg, spy_close): t for t in tickers}
        for fut in as_completed(futs):
            out = fut.result()
            if out:
                res, row = out
                results.append(res); rows.append(row)
    return results, rows

# ---------- STREAMLIT UI ----------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Advanced multi-source financial analysis with caching, sentiment, earnings awareness. Not financial advice.")

    settings = load_settings()

    with st.sidebar:
        st.header("Configuration")
        market_key = st.selectbox("Market Profile:", list(MARKETS.keys()), index=0)
        mi = MARKETS[market_key]
        tz = pytz.timezone(mi["tz"])
        open_now = is_market_open(market_key)
        st.markdown(f"**Market Status:** {'OPEN' if open_now else 'CLOSED'}")
        st.markdown(f"**Local Time:** {dt.datetime.now(tz).strftime('%H:%M:%S %Z')}")

        st.subheader("Analysis")
        risk_profile = st.selectbox("Risk Profile:", ["conservative","balanced","aggressive"],
                                    index=["conservative","balanced","aggressive"].index(settings.get("risk_profile","balanced")))
        lookback_days = st.slider("Historical Data (days):", 30, 365, settings.get("lookback_days",120), step=5)
        interval = st.selectbox("Data Interval:", ["1d","30m"], index=1 if settings.get("interval","30m")=="30m" else 0)

        st.subheader("News")
        use_news = st.checkbox("Enable News Sentiment", value=settings.get("use_news", True))
        news_days = st.slider("News Lookback (days):", 1, 30, settings.get("news_days",7))

        st.subheader("Indicators")
        st.multiselect("Active Indicators (info)", ["RSI","MACD","Bollinger"],
                       default=settings.get("indicators",["RSI","MACD","Bollinger"]))

        st.subheader("Extras")
        auto_refresh = st.checkbox("Auto refresh every 15 min", value=settings.get("auto_refresh", True))
        if auto_refresh and st_autorefresh:
            st_autorefresh(interval=CACHE_TTL*1000, key="auto15m")

        # Watchlist controls
        st.subheader("Watchlist (persistent)")
        wl = load_watchlist()
        st.caption(f"Saved: {len(wl)}  →  {', '.join(wl[:8]) + (' ...' if len(wl)>8 else '')}")
        colA, colB = st.columns([2,1])
        with colA:
            add_t = st.text_input("Add ticker", placeholder="e.g., AAPL").strip().upper()
        with colB:
            if st.button("Add"):
                if add_t and add_t not in wl:
                    wl.append(add_t); save_watchlist(wl); st.experimental_rerun()
        rem = st.multiselect("Remove selected", wl, [])
        if st.button("Remove"):
            wl2 = [x for x in wl if x not in rem]; save_watchlist(wl2); st.experimental_rerun()

    # Save settings snapshot
    save_settings({
        "risk_profile": risk_profile, "lookback_days": lookback_days,
        "interval": interval, "use_news": use_news, "news_days": news_days,
        "indicators": ["RSI","MACD","Bollinger"], "auto_refresh": auto_refresh
    })

    if not wl:
        st.warning("No tickers in watchlist. Add some from the sidebar.")
        return

    cfg = {
        "risk_profile": risk_profile, "lookback_days": lookback_days,
        "interval": interval, "use_news": use_news, "news_days": news_days
    }

    if st.button("Run Enhanced Analysis", type="primary"):
        with st.spinner("Scanning..."):
            results, rows = scan_tickers(wl, cfg)

        # Dashboard
        st.header("Analysis Dashboard")
        strong_buy = len([r for r in results if r["signal"]=="STRONG BUY"])
        buys = len([r for r in results if "BUY" in r["signal"]])
        sells = len([r for r in results if "SELL" in r["signal"]])
        avg_conf = np.mean([r["confidence"] for r in results]) if results else 0
        total = len(results)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Strong Buy", strong_buy, f"{strong_buy}/{total}")
        c2.metric("Buy Signals", buys, f"{buys}/{total}")
        c3.metric("Sell Signals", sells, f"{sells}/{total}")
        c4.metric("Avg Confidence", f"{avg_conf:.0f}%")
        c5.metric("Stocks Analyzed", total)

        st.subheader("Detailed Results")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name=f"signals_{dt.date.today().strftime('%Y%m%d')}.csv", mime="text/csv")

        st.subheader("Individual Stock Analysis")
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            with st.expander(f"{r['ticker']} — {r['signal']} (Score {r['score']}, Conf {r['confidence']}%)"):
                st.markdown("**Key reasons:**")
                for i, reason in enumerate(r.get("reasons", []), 1):
                    st.write(f"{i}. {reason}")
                # chart
                try:
                    df_ch = get_price_history(r["ticker"], lookback_days, interval)
                    df_ch = compute_indicators(df_ch)
                    fig = plot_ticker(df_ch, r["ticker"])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Chart unavailable: {e}")

    st.caption(f"Data: Yahoo Finance (delayed). Earnings via Yahoo quoteSummary API. Cache TTL: {CACHE_TTL//60} min.")

if __name__ == "__main__":
    main()
