# Enhanced Stock Signals PRO - Multi-Source Analysis Platform with Persistent Watchlist
import math, re, json, datetime as dt
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os

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

# Enhanced sentiment analysis
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

# Statistical analysis
try:
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
except ImportError:
    stats = None
    StandardScaler = None
    KMeans = None

# Web scraping for additional sources
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# Optional: precise exchange calendars
try:
    import pandas_market_calendars as mcal
except Exception:
    mcal = None

# Optional: auto refresh
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

APP_TITLE = "📈 Stock Signals PRO - Enhanced Multi-Source Analysis"

# Persistent file storage - uses user's home directory for persistence across sessions
WATCHLIST_FILE = Path.home() / "stock_signals_watchlist.json"
SETTINGS_FILE = Path.home() / "stock_signals_settings.json"

# Make sure directories exist
WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

# -------------------- Enhanced Market Data Sources --------------------
MARKETS = {
    "US — NYSE/Nasdaq (09:30—16:00 ET)": {"tz": "America/New_York", "open": (9,30), "close": (16,0), "cal": "XNYS"},
    "Germany — XETRA (09:00—17:30 DE)": {"tz": "Europe/Berlin", "open": (9,0),  "close": (17,30), "cal": "XETR"},
    "UK — LSE (08:00—16:30 UK)": {"tz": "Europe/London", "open": (8,0),  "close": (16,30), "cal": "XLON"},
    "France — Euronext Paris (09:00—17:30 FR)": {"tz": "Europe/Paris", "open": (9,0), "close": (17,30), "cal": "XPAR"},
    "Japan — TSE (09:00—15:00 JST)": {"tz": "Asia/Tokyo", "open": (9,0), "close": (15,0), "cal": "XTKS"},
    "Australia — ASX (10:00—16:00 AEST)": {"tz": "Australia/Sydney", "open": (10,0), "close": (16,0), "cal": "XASX"},
}

# News sources for enhanced sentiment analysis
NEWS_SOURCES = {
    "Google Finance": "https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en",
    "Yahoo Finance": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={query}&region=US&lang=en-US",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
    "Seeking Alpha": "https://seekingalpha.com/api/sa/combined/{query}.xml",
}

# Technical indicator configurations
INDICATOR_CONFIGS = {
    "RSI": {"periods": [14, 21, 30], "overbought": 70, "oversold": 30},
    "MACD": {"fast": [12, 9], "slow": [26, 21], "signal": [9, 7]},
    "Bollinger": {"period": 20, "std_dev": 2},
    "Stochastic": {"k_period": 14, "d_period": 3},
    "Williams_R": {"period": 14},
    "CCI": {"period": 20},
    "MFI": {"period": 14},
}

def load_watchlist() -> List[str]:
    """Load persistent watchlist from file"""
    try:
        if WATCHLIST_FILE.exists():
            with open(WATCHLIST_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Clean and validate tickers
                    valid_tickers = []
                    for ticker in data:
                        if isinstance(ticker, str) and ticker.strip():
                            clean_ticker = ticker.strip().upper()
                            if len(clean_ticker) >= 1 and len(clean_ticker) <= 10:  # Valid ticker length
                                valid_tickers.append(clean_ticker)
                    return sorted(list(set(valid_tickers)))  # Remove duplicates and sort
    except Exception as e:
        st.error(f"Error loading watchlist: {str(e)}")
    
    # Default watchlist if file doesn't exist or has issues
    default_watchlist = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META"]
    save_watchlist(default_watchlist)  # Save default list
    return default_watchlist

def save_watchlist(tickers: List[str]) -> bool:
    """Save watchlist to persistent file"""
    try:
        # Clean and validate tickers
        clean_tickers = []
        for ticker in tickers:
            if isinstance(ticker, str) and ticker.strip():
                clean_ticker = ticker.strip().upper()
                if len(clean_ticker) >= 1 and len(clean_ticker) <= 10:
                    clean_tickers.append(clean_ticker)
        
        # Remove duplicates and sort
        unique_tickers = sorted(list(set(clean_tickers)))
        
        # Save to file
        with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(unique_tickers, f, indent=2, ensure_ascii=False)
        
        # Show success message
        st.sidebar.success(f"✅ Watchlist saved ({len(unique_tickers)} stocks)")
        return True
        
    except Exception as e:
        st.sidebar.error(f"❌ Error saving watchlist: {str(e)}")
        return False

def load_settings() -> Dict:
    """Load user settings from persistent file"""
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    
    # Default settings
    default_settings = {
        "risk_profile": "balanced",
        "news_sources": ["Google Finance", "Yahoo Finance"],
        "indicators": ["RSI", "MACD", "Bollinger"],
        "lookback_days": 90,
        "news_days": 7,
        "show_charts": True,
        "auto_refresh": True,
    }
    save_settings(default_settings)
    return default_settings

def save_settings(settings: Dict) -> bool:
    """Save user settings to persistent file"""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving settings: {str(e)}")
        return False

# -------------------- Enhanced Technical Analysis --------------------

def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Williams %R"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * ((highest_high - close) / (highest_high - lowest_low))

def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index (CCI)"""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (typical_price - sma_tp) / (0.015 * mean_deviation)

def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index (MFI)"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = pd.Series(index=close.index, dtype=float)
    negative_flow = pd.Series(index=close.index, dtype=float)
    
    for i in range(1, len(close)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.iloc[i] = money_flow.iloc[i]
            negative_flow.iloc[i] = 0
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            positive_flow.iloc[i] = 0
            negative_flow.iloc[i] = money_flow.iloc[i]
        else:
            positive_flow.iloc[i] = 0
            negative_flow.iloc[i] = 0
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    return 100 - (100 / (1 + (positive_mf / negative_mf)))

def ema(series: pd.Series, span: int) -> pd.Series:
    """Enhanced Exponential Moving Average"""
    result = series.ewm(span=span, adjust=False).mean()
    return result if isinstance(result, pd.Series) else pd.Series(result)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Enhanced RSI calculation with better error handling"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).abs()
    loss = (-delta.where(delta < 0, 0.0)).abs()
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Ensure Series type
    if not isinstance(avg_gain, pd.Series):
        avg_gain = pd.Series(avg_gain)
    if not isinstance(avg_loss, pd.Series):
        avg_loss = pd.Series(avg_loss)
    
    avg_gain = avg_gain.fillna(0)
    avg_loss = avg_loss.fillna(0)
    
    # Use Wilder's smoothing
    for i in range(period, len(series)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi_values = 100 - (100 / (1 + rs))
    return rsi_values.fillna(50)  # Fill NaN with neutral RSI

def macd(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    """Enhanced MACD calculation"""
    macd_line = ema(series, fast) - ema(series, slow)
    signal = ema(macd_line, sig)
    histogram = macd_line - signal
    return macd_line, signal, histogram

# -------------------- Enhanced Data Sources --------------------

def get_enhanced_price_data(ticker: str, days: int, interval: str = "1d") -> pd.DataFrame:
    """Get enhanced price data with additional metrics"""
    try:
        # Get basic price data
        stock = yf.Ticker(ticker)
        
        if interval == "30m":
            df = stock.history(period="60d", interval="30m")
        else:
            df = stock.history(period=f"{days}d", interval="1d")
        
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Get additional info
        try:
            info = stock.info
            # Add fundamental data safely
            df['MarketCap'] = info.get('marketCap', np.nan)
            df['PE_Ratio'] = info.get('forwardPE', info.get('trailingPE', np.nan))
            df['Beta'] = info.get('beta', np.nan)
            df['DividendYield'] = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else np.nan
            df['Sector'] = info.get('sector', 'Unknown')
            df['Industry'] = info.get('industry', 'Unknown')
        except Exception:
            # If info fails, add default values
            df['MarketCap'] = np.nan
            df['PE_Ratio'] = np.nan
            df['Beta'] = np.nan
            df['DividendYield'] = np.nan
            df['Sector'] = 'Unknown'
            df['Industry'] = 'Unknown'
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def get_multi_source_news(ticker: str, days: int = 7) -> List[dict]:
    """Collect news from multiple sources"""
    news_items = []
    
    # Google News
    try:
        google_url = f"https://news.google.com/rss/search?q={ticker}+stock+when:{days}d&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(google_url)
        
        for entry in feed.entries[:20]:
            try:
                pub_date = dt.datetime(*entry.published_parsed[:6]) if entry.get('published_parsed') else dt.datetime.now()
                news_items.append({
                    "title": entry.title,
                    "source": "Google News",
                    "published": pub_date,
                    "sentiment": 0,  # Will be calculated later
                    "url": entry.get('link', '')
                })
            except Exception:
                continue
                
    except Exception as e:
        st.warning(f"Could not fetch Google News: {str(e)}")
    
    # Yahoo Finance RSS (if available)
    try:
        yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(yahoo_url)
        
        for entry in feed.entries[:15]:
            try:
                pub_date = dt.datetime(*entry.published_parsed[:6]) if entry.get('published_parsed') else dt.datetime.now()
                news_items.append({
                    "title": entry.title,
                    "source": "Yahoo Finance",
                    "published": pub_date,
                    "sentiment": 0,
                    "url": entry.get('link', '')
                })
            except Exception:
                continue
                
    except Exception:
        pass  # Yahoo RSS might not be available
    
    return news_items

def analyze_sentiment_enhanced(news_items: List[dict]) -> Dict[str, float]:
    """Enhanced sentiment analysis using multiple methods"""
    if not news_items:
        return {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0, "n": 0, "confidence": 0.0}
    
    # VADER sentiment analyzer
    vader = SentimentIntensityAnalyzer()
    
    # TextBlob for additional sentiment
    textblob_scores = []
    vader_scores = []
    
    for news in news_items:
        # VADER analysis
        vader_score = vader.polarity_scores(news["title"])
        vader_scores.append(vader_score['compound'])
        news["sentiment"] = vader_score['compound']
        
        # TextBlob analysis (if available)
        if TextBlob:
            try:
                blob = TextBlob(news["title"])
                textblob_scores.append(blob.sentiment.polarity)
            except Exception:
                textblob_scores.append(0)
    
    # Calculate combined metrics
    vader_mean = np.mean(vader_scores) if vader_scores else 0
    textblob_mean = np.mean(textblob_scores) if textblob_scores else 0
    
    # Weighted average (VADER has more weight for financial news)
    combined_score = (vader_mean * 0.7 + textblob_mean * 0.3) if textblob_scores else vader_mean
    
    # Calculate confidence based on consistency
    score_std = np.std(vader_scores) if len(vader_scores) > 1 else 0
    confidence = max(0, 1 - (score_std / 2))  # Lower std = higher confidence
    
    # Count sentiment categories
    positive = sum(1 for score in vader_scores if score > 0.1)
    negative = sum(1 for score in vader_scores if score < -0.1)
    neutral = len(vader_scores) - positive - negative
    
    return {
        "compound": combined_score,
        "pos": positive / len(vader_scores) if vader_scores else 0,
        "neu": neutral / len(vader_scores) if vader_scores else 1,
        "neg": negative / len(vader_scores) if vader_scores else 0,
        "n": len(news_items),
        "confidence": confidence,
        "recent_trend": np.mean(vader_scores[-5:]) if len(vader_scores) >= 5 else combined_score
    }

def compute_enhanced_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Compute comprehensive technical indicators"""
    if df.empty:
        return df
    
    # Basic price series
    close = df['Close']
    high = df['High'] if 'High' in df.columns else close
    low = df['Low'] if 'Low' in df.columns else close
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series([1000000] * len(df), index=df.index)
    
    # Moving averages (multiple periods)
    for period in [10, 20, 50, 100, 200]:
        df[f'SMA{period}'] = close.rolling(period).mean()
        df[f'EMA{period}'] = ema(close, period)
    
    # RSI (multiple periods)
    for period in config.get("RSI", {}).get("periods", [14]):
        df[f'RSI{period}'] = rsi(close, period)
    
    # MACD variations
    macd_configs = config.get("MACD", {})
    fast_periods = macd_configs.get("fast", [12])
    slow_periods = macd_configs.get("slow", [26])
    signal_periods = macd_configs.get("signal", [9])
    
    for i, (fast, slow, sig) in enumerate(zip(fast_periods, slow_periods, signal_periods)):
        suffix = f"_{i+1}" if i > 0 else ""
        macd_line, macd_signal, macd_hist = macd(close, fast, slow, sig)
        df[f'MACD{suffix}'] = macd_line
        df[f'MACD_SIG{suffix}'] = macd_signal
        df[f'MACD_HIST{suffix}'] = macd_hist
    
    # Bollinger Bands
    if "Bollinger" in config.get("indicators", []):
        bb_config = config.get("Bollinger", {})
        period = bb_config.get("period", 20)
        std_dev = bb_config.get("std_dev", 2)
        
        bb_upper, bb_middle, bb_lower = bollinger_bands(close, period, std_dev)
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        df['BB_Width'] = (bb_upper - bb_lower) / bb_middle * 100
        df['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower) * 100
    
    # Stochastic Oscillator
    if "Stochastic" in config.get("indicators", []):
        stoch_config = config.get("Stochastic", {})
        k_period = stoch_config.get("k_period", 14)
        d_period = stoch_config.get("d_period", 3)
        
        stoch_k, stoch_d = stochastic_oscillator(high, low, close, k_period, d_period)
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_d
    
    # Williams %R
    if "Williams_R" in config.get("indicators", []):
        williams_config = config.get("Williams_R", {})
        period = williams_config.get("period", 14)
        df['Williams_R'] = williams_r(high, low, close, period)
    
    # Commodity Channel Index
    if "CCI" in config.get("indicators", []):
        cci_config = config.get("CCI", {})
        period = cci_config.get("period", 20)
        df['CCI'] = commodity_channel_index(high, low, close, period)
    
    # Money Flow Index
    if "MFI" in config.get("indicators", []):
        mfi_config = config.get("MFI", {})
        period = mfi_config.get("period", 14)
        df['MFI'] = money_flow_index(high, low, close, volume, period)
    
    # Volume indicators
    df['Volume_SMA20'] = volume.rolling(20).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA20']
    
    # Volatility indicators
    df['ATR'] = (high - low).rolling(14).mean()  # Simplified ATR
    df['Volatility'] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    
    # Price momentum
    for period in [1, 5, 10, 20]:
        df[f'Return_{period}d'] = close.pct_change(period) * 100
    
    # Support and Resistance levels
    df['Resistance'] = high.rolling(20).max()
    df['Support'] = low.rolling(20).min()
    
    # 52-week high/low (or available data period)
    window_52w = min(len(df), 252)
    df['HI52'] = close.rolling(window_52w).max()
    df['LO52'] = close.rolling(window_52w).min()
    
    return df

def create_enhanced_visualizations(df: pd.DataFrame, ticker: str, indicators: List[str]) -> go.Figure:
    """Create comprehensive interactive charts"""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxis=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f'{ticker} Price Analysis', 'RSI & Technical Indicators', 'MACD & Momentum']
    )
    
    # Main price chart with candlesticks
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=f'{ticker} Price'
            ),
            row=1, col=1
        )
    else:
        # Fallback to line chart
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name=f'{ticker} Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    
    # Add moving averages
    for period in [20, 50]:
        if f'SMA{period}' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[f'SMA{period}'],
                    mode='lines',
                    name=f'SMA{period}',
                    line=dict(width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
    
    # Bollinger Bands
    if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.5,
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                mode='lines',
                name='Bollinger Bands',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                opacity=0.5
            ),
            row=1, col=1
        )
    
    # RSI
    if 'RSI14' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI14'],
                mode='lines',
                name='RSI(14)',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
    
    # MACD
    if all(col in df.columns for col in ['MACD', 'MACD_SIG', 'MACD_HIST']):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_SIG'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=2)
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACD_HIST'],
                name='Histogram',
                opacity=0.6
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} - Enhanced Technical Analysis',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

def enhanced_signal_classification(ticker: str, df: pd.DataFrame, news_sentiment: Dict, 
                                  risk_profile: str = "balanced", config: Dict = None) -> Dict:
    """Enhanced multi-factor signal classification"""
    if df.empty:
        return {"error": "No data available"}
    
    config = config or INDICATOR_CONFIGS
    
    # Get latest data
    current = df.iloc[-1]
    previous = df.iloc[-2] if len(df) >= 2 else current
    
    # Initialize scoring system
    signals = {
        "technical": 0,
        "momentum": 0, 
        "volume": 0,
        "sentiment": 0,
        "fundamental": 0
    }
    
    reasons = []
    confidence_factors = []
    
    # Technical Analysis Signals
    price = current['Close']
    
    # Moving average signals
    sma20 = current.get('SMA20', price)
    sma50 = current.get('SMA50', price)
    sma200 = current.get('SMA200', price)
    
    if sma20 and sma50 and sma200 and not any(pd.isna([sma20, sma50, sma200])):
        if price > sma20 > sma50 > sma200:
            signals["technical"] += 20
            reasons.append("Strong uptrend - price above all MAs")
            confidence_factors.append(0.9)
        elif price < sma20 < sma50 < sma200:
            signals["technical"] -= 20
            reasons.append("Strong downtrend - price below all MAs")
            confidence_factors.append(0.9)
        elif price > sma50:
            signals["technical"] += 10
            reasons.append("Above medium-term trend")
            confidence_factors.append(0.6)
    
    # RSI signals
    rsi14 = current.get('RSI14', 50)
    rsi_prev = previous.get('RSI14', 50)
    
    if rsi14 and rsi_prev and not pd.isna(rsi14) and not pd.isna(rsi_prev):
        if rsi14 < 30:
            signals["technical"] += 15
            reasons.append(f"RSI oversold ({rsi14:.1f})")
            confidence_factors.append(0.8)
        elif rsi14 > 70:
            signals["technical"] -= 15
            reasons.append(f"RSI overbought ({rsi14:.1f})")
            confidence_factors.append(0.8)
        elif rsi_prev < 50 <= rsi14:
            signals["technical"] += 8
            reasons.append("RSI crossed above 50")
            confidence_factors.append(0.6)
        elif rsi_prev > 50 >= rsi14:
            signals["technical"] -= 8
            reasons.append("RSI crossed below 50")
            confidence_factors.append(0.6)
    
    # MACD signals
    macd_line = current.get('MACD', 0)
    macd_signal_line = current.get('MACD_SIG', 0)
    macd_prev = previous.get('MACD', 0)
    macd_sig_prev = previous.get('MACD_SIG', 0)
    
    if all(not pd.isna(x) for x in [macd_line, macd_signal_line, macd_prev, macd_sig_prev]):
        if macd_prev < macd_sig_prev and macd_line > macd_signal_line:
            signals["momentum"] += 12
            reasons.append("MACD bullish crossover")
            confidence_factors.append(0.7)
        elif macd_prev > macd_sig_prev and macd_line < macd_signal_line:
            signals["momentum"] -= 12
            reasons.append("MACD bearish crossover")
            confidence_factors.append(0.7)
    
    # Bollinger Bands signals
    bb_position = current.get('BB_Position')
    if bb_position is not None and not pd.isna(bb_position):
        if bb_position < 10:
            signals["technical"] += 10
            reasons.append("Near Bollinger lower band")
            confidence_factors.append(0.6)
        elif bb_position > 90:
            signals["technical"] -= 10
            reasons.append("Near Bollinger upper band")
            confidence_factors.append(0.6)
    
    # Volume analysis
    volume_ratio = current.get('Volume_Ratio', 1)
    if volume_ratio and not pd.isna(volume_ratio):
        if volume_ratio > 1.5:
            signals["volume"] += 8
            reasons.append(f"High volume ({volume_ratio:.1f}x avg)")
            confidence_factors.append(0.5)
        elif volume_ratio < 0.5:
            signals["volume"] -= 5
            reasons.append("Low volume")
            confidence_factors.append(0.3)
    
    # Momentum signals
    return_5d = current.get('Return_5d', 0)
    return_20d = current.get('Return_20d', 0)
    
    if return_5d and return_20d and not any(pd.isna([return_5d, return_20d])):
        if return_5d > 5 and return_20d > 10:
            signals["momentum"] += 15
            reasons.append("Strong positive momentum")
            confidence_factors.append(0.7)
        elif return_5d < -5 and return_20d < -10:
            signals["momentum"] -= 15
            reasons.append("Strong negative momentum")
            confidence_factors.append(0.7)
    
    # Sentiment analysis
    if news_sentiment and news_sentiment.get('n', 0) > 0:
        sentiment_score = news_sentiment.get('compound', 0)
        confidence = news_sentiment.get('confidence', 0.5)
        
        if sentiment_score > 0.3:
            signals["sentiment"] += int(10 * confidence)
            reasons.append(f"Very positive news sentiment ({sentiment_score:+.2f})")
            confidence_factors.append(confidence)
        elif sentiment_score > 0.1:
            signals["sentiment"] += int(5 * confidence)
            reasons.append(f"Positive news sentiment ({sentiment_score:+.2f})")
            confidence_factors.append(confidence * 0.7)
        elif sentiment_score < -0.3:
            signals["sentiment"] -= int(10 * confidence)
            reasons.append(f"Very negative news sentiment ({sentiment_score:+.2f})")
            confidence_factors.append(confidence)
        elif sentiment_score < -0.1:
            signals["sentiment"] -= int(5 * confidence)
            reasons.append(f"Negative news sentiment ({sentiment_score:+.2f})")
            confidence_factors.append(confidence * 0.7)
    
    # Fundamental signals (if available)
    pe_ratio = current.get('PE_Ratio')
    if pe_ratio and not pd.isna(pe_ratio):
        if pe_ratio < 15:
            signals["fundamental"] += 8
            reasons.append(f"Low P/E ratio ({pe_ratio:.1f})")
            confidence_factors.append(0.6)
        elif pe_ratio > 30:
            signals["fundamental"] -= 5
            reasons.append(f"High P/E ratio ({pe_ratio:.1f})")
            confidence_factors.append(0.4)
    
    # Calculate total score with risk adjustment
    total_score = sum(signals.values())
    
    # Risk profile adjustments
    risk_multiplier = {
        "conservative": 0.7,
        "balanced": 1.0,
        "aggressive": 1.3
    }.get(risk_profile, 1.0)
    
    adjusted_score = total_score * risk_multiplier
    
    # Calculate overall confidence
    avg_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
    
    # Determine signal classification
    if adjusted_score >= 25:
        signal = "STRONG BUY"
    elif adjusted_score >= 15:
        signal = "BUY"
    elif adjusted_score >= 5:
        signal = "WEAK BUY"
    elif adjusted_score <= -25:
        signal = "STRONG SELL"
    elif adjusted_score <= -15:
        signal = "SELL"
    elif adjusted_score <= -5:
        signal = "WEAK SELL"
    else:
        signal = "HOLD"
    
    # Prepare result
    result = {
        "ticker": ticker,
        "signal": signal,
        "score": int(adjusted_score),
        "confidence": min(100, int(avg_confidence * 100)),
        "price": float(price),
        "signals_breakdown": signals,
        "reasons": reasons[:8],  # Limit to top 8 reasons
        "sentiment": news_sentiment,
        "risk_profile": risk_profile,
        "fundamental_data": {
            "pe_ratio": float(pe_ratio) if pe_ratio and not pd.isna(pe_ratio) else None,
            "beta": float(current.get('Beta', np.nan)) if not pd.isna(current.get('Beta', np.nan)) else None,
            "market_cap": current.get('MarketCap'),
            "dividend_yield": float(current.get('DividendYield', np.nan)) if not pd.isna(current.get('DividendYield', np.nan)) else None,
            "sector": current.get('Sector', 'Unknown'),
            "industry": current.get('Industry', 'Unknown')
        }
    }
    
    return result

def is_market_open(profile_key: str) -> bool:
    """Enhanced market open check with more markets"""
    prof = MARKETS.get(profile_key)
    if not prof:
        return False
    tz = pytz.timezone(prof["tz"])
    now = dt.datetime.now(tz)

    # Use market calendar if available
    if mcal is not None and prof.get("cal"):
        try:
            cal = mcal.get_calendar(prof["cal"])
            sched = cal.schedule(start_date=now.date(), end_date=now.date())
            if sched.empty:
                return False
            open_ts = sched.iloc[0]["market_open"].tz_convert(tz)
            close_ts = sched.iloc[0]["market_close"].tz_convert(tz)
            return open_ts <= now < close_ts
        except Exception:
            pass

    # Fallback to weekday check
    if now.weekday() > 4:  # Saturday or Sunday
        return False
    
    o_h, o_m = prof["open"]
    c_h, c_m = prof["close"]
    open_time = now.replace(hour=o_h, minute=o_m, second=0, microsecond=0)
    close_time = now.replace(hour=c_h, minute=c_m, second=0, microsecond=0)
    
    return open_time <= now < close_time

def scan_enhanced_tickers(tickers: List[str], config: Dict, progress_callback=None) -> Tuple[List[Dict], List[Dict]]:
    """Enhanced ticker scanning with parallel processing"""
    results = []
    table_rows = []
    
    total_tickers = len(tickers)
    
    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback((i + 1) / total_tickers)
        
        try:
            # Get enhanced price data
            df = get_enhanced_price_data(ticker, 
                                       config.get("lookback_days", 90), 
                                       config.get("interval", "1d"))
            
            if df.empty:
                st.warning(f"No data available for {ticker}")
                continue
            
            # Compute enhanced indicators
            df = compute_enhanced_indicators(df, INDICATOR_CONFIGS)
            
            # Get news sentiment if enabled
            sentiment = {}
            if config.get("use_news", True):
                news_items = get_multi_source_news(ticker, config.get("news_days", 7))
                sentiment = analyze_sentiment_enhanced(news_items)
            
            # Generate enhanced signals
            analysis = enhanced_signal_classification(
                ticker, df, sentiment, 
                config.get("risk_profile", "balanced"),
                INDICATOR_CONFIGS
            )
            
            results.append(analysis)
            
            # Prepare table row
            current = df.iloc[-1]
            table_rows.append({
                "Ticker": ticker,
                "Signal": analysis["signal"],
                "Score": analysis["score"],
                "Confidence": f"{analysis['confidence']}%",
                "Price": f"${analysis['price']:.2f}",
                "RSI": f"{current.get('RSI14', 0):.1f}" if current.get('RSI14') and not pd.isna(current.get('RSI14')) else "N/A",
                "Volume Ratio": f"{current.get('Volume_Ratio', 1):.1f}x" if current.get('Volume_Ratio') and not pd.isna(current.get('Volume_Ratio')) else "N/A",
                "5D Return": f"{current.get('Return_5d', 0):+.1f}%" if current.get('Return_5d') and not pd.isna(current.get('Return_5d')) else "N/A",
                "Sentiment": f"{sentiment.get('compound', 0):+.2f}" if sentiment.get('compound') else "N/A",
                "News Count": sentiment.get('n', 0) if sentiment else 0,
                "P/E Ratio": f"{current.get('PE_Ratio', 0):.1f}" if current.get('PE_Ratio') and not pd.isna(current.get('PE_Ratio', np.nan)) else "N/A"
            })
            
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
            continue
    
    return results, table_rows

def main():
    """Enhanced main application with persistent watchlist"""
    st.set_page_config(
        page_title=APP_TITLE, 
        page_icon="📈", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title(APP_TITLE)
    st.markdown("*Advanced multi-source financial analysis platform with persistent watchlist*")
    
    # Auto-refresh setup
    if st_autorefresh:
        count = st_autorefresh(interval=30*60*1000, key="enhanced_refresh")
    
    # Load persistent settings
    settings = load_settings()
    
    # Enhanced Sidebar Configuration
    st.sidebar.header("⚙️ Enhanced Configuration")
    
    # Market selection with more options
    market_key = st.sidebar.selectbox("Market Profile:", list(MARKETS.keys()), index=0)
    market_info = MARKETS[market_key]
    
    # Market status display
    is_open = is_market_open(market_key)
    tz = pytz.timezone(market_info["tz"])
    local_time = dt.datetime.now(tz)
    
    status_emoji = "🟢" if is_open else "🔴"
    status_text = "OPEN" if is_open else "CLOSED"
    st.sidebar.markdown(f"**Market Status:** {status_emoji} {status_text}")
    st.sidebar.markdown(f"**Local Time:** {local_time.strftime('%H:%M:%S %Z')}")
    
    # Enhanced Analysis Settings
    st.sidebar.subheader("📊 Analysis Configuration")
    
    # Risk profile with descriptions
    risk_descriptions = {
        "conservative": "Lower risk, higher confidence thresholds",
        "balanced": "Balanced risk-reward approach", 
        "aggressive": "Higher risk, more sensitive signals"
    }
    
    risk_profile = st.sidebar.selectbox(
        "Risk Profile:", 
        list(risk_descriptions.keys()),
        index=list(risk_descriptions.keys()).index(settings.get("risk_profile", "balanced")),
        help="Choose your risk tolerance level"
    )
    st.sidebar.caption(risk_descriptions[risk_profile])
    
    # Enhanced parameters
    lookback_days = st.sidebar.slider("Historical Data (days):", 30, 365, settings.get("lookback_days", 90))
    interval = st.sidebar.selectbox("Data Interval:", ["1d", "30m"], index=0)
    
    # News analysis settings
    st.sidebar.subheader("📰 News Analysis")
    use_news = st.sidebar.checkbox("Enable News Sentiment", value=True)
    
    if use_news:
        news_days = st.sidebar.slider("News Lookback (days):", 1, 30, settings.get("news_days", 7))
        news_sources = st.sidebar.multiselect(
            "News Sources:",
            list(NEWS_SOURCES.keys()),
            default=settings.get("news_sources", ["Google Finance", "Yahoo Finance"])
        )
    else:
        news_days = 7
        news_sources = []
    
    # Technical indicator selection
    st.sidebar.subheader("📈 Technical Indicators")
    
    available_indicators = list(INDICATOR_CONFIGS.keys())
    selected_indicators = st.sidebar.multiselect(
        "Active Indicators:",
        available_indicators,
        default=settings.get("indicators", ["RSI", "MACD", "Bollinger"])
    )
    
    # Advanced features
    st.sidebar.subheader("🔬 Advanced Features")
    show_charts = st.sidebar.checkbox("Interactive Charts", value=settings.get("show_charts", True))
    show_correlations = st.sidebar.checkbox("Market Correlations", value=False)
    enable_alerts = st.sidebar.checkbox("Price Alerts", value=False)
    
    # Enhanced Persistent Watchlist Management
    st.sidebar.subheader("📋 Persistent Watchlist")
    st.sidebar.markdown("*Your watchlist is automatically saved and restored between sessions*")
    
    # Load current watchlist
    current_watchlist = load_watchlist()
    
    # Watchlist display with metrics
    if current_watchlist:
        st.sidebar.markdown(f"**Saved Stocks ({len(current_watchlist)}):**")
        watchlist_display = ", ".join(current_watchlist[:8])  # Show first 8
        if len(current_watchlist) > 8:
            watchlist_display += f" ... (+{len(current_watchlist)-8} more)"
        st.sidebar.markdown(f"`{watchlist_display}`")
    else:
        st.sidebar.info("No stocks in watchlist yet. Add some below!")
    
    # Add/remove tickers with immediate persistence
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        new_ticker = st.text_input("Add Stock:", placeholder="AAPL", key="add_ticker").strip().upper()
        if st.button("➕ Add", key="add_btn") and new_ticker:
            if len(new_ticker) >= 1 and len(new_ticker) <= 10:  # Validate ticker length
                if new_ticker not in current_watchlist:
                    current_watchlist.append(new_ticker)
                    if save_watchlist(current_watchlist):
                        st.rerun()  # Refresh to show updated list
                else:
                    st.sidebar.warning(f"{new_ticker} already in watchlist")
            else:
                st.sidebar.error("Invalid ticker symbol")
    
    with col2:
        if current_watchlist:
            ticker_to_remove = st.selectbox("Remove:", ["Select..."] + current_watchlist, key="remove_select")
            if st.button("➖ Remove", key="remove_btn") and ticker_to_remove != "Select...":
                current_watchlist.remove(ticker_to_remove)
                if save_watchlist(current_watchlist):
                    st.rerun()  # Refresh to show updated list
    
    # Bulk operations
    if st.sidebar.button("📂 Load Popular Stocks", help="Add popular tech and blue-chip stocks"):
        popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC", "SPY", "QQQ"]
        original_count = len(current_watchlist)
        new_watchlist = sorted(set(current_watchlist + popular_stocks))
        if save_watchlist(new_watchlist):
            added_count = len(new_watchlist) - original_count
            if added_count > 0:
                st.sidebar.success(f"Added {added_count} new stocks")
            else:
                st.sidebar.info("All popular stocks already in watchlist")
            st.rerun()
    
    if current_watchlist and st.sidebar.button("🗑️ Clear Watchlist", help="Remove all stocks from watchlist"):
        if save_watchlist([]):
            st.sidebar.success("Watchlist cleared")
            st.rerun()
    
    # Save current settings
    current_settings = {
        "risk_profile": risk_profile,
        "news_sources": news_sources,
        "indicators": selected_indicators,
        "lookback_days": lookback_days,
        "news_days": news_days,
        "show_charts": show_charts,
        "auto_refresh": True,
    }
    save_settings(current_settings)
    
    # Main Analysis Section
    if not current_watchlist:
        st.warning("🚨 No stocks in watchlist. Please add some tickers to analyze.")
        st.info("💡 Use the sidebar to add stocks or click 'Load Popular Stocks' to get started!")
        st.info("🔄 Your watchlist is automatically saved and will be restored when you restart the app.")
        return
    
    # Configuration summary
    config = {
        "lookback_days": lookback_days,
        "interval": interval,
        "use_news": use_news,
        "news_days": news_days,
        "news_sources": news_sources,
        "risk_profile": risk_profile,
        "indicators": selected_indicators
    }
    
    # Analysis button
    if st.button("🚀 Run Enhanced Analysis", type="primary", key="enhanced_scan"):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(pct):
            progress_bar.progress(pct)
            status_text.text(f"Analyzing {len(current_watchlist)} stocks... {int(pct*100)}% complete")
        
        # Run enhanced analysis
        with st.spinner("Running comprehensive market analysis..."):
            try:
                results, table_data = scan_enhanced_tickers(current_watchlist, config, update_progress)
                
                progress_bar.empty()
                status_text.empty()
                
                if not results:
                    st.error("❌ No analysis results generated. Please check your watchlist.")
                    return
                
                # Summary Dashboard
                st.header("📊 Analysis Dashboard")
                
                # Key metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                strong_buy = len([r for r in results if r["signal"] == "STRONG BUY"])
                buy_signals = len([r for r in results if "BUY" in r["signal"]])
                sell_signals = len([r for r in results if "SELL" in r["signal"]])
                avg_confidence = np.mean([r["confidence"] for r in results])
                total_analyzed = len(results)
                
                col1.metric("Strong Buy", strong_buy, delta=f"{strong_buy}/{total_analyzed}")
                col2.metric("Buy Signals", buy_signals, delta=f"{buy_signals}/{total_analyzed}")
                col3.metric("Sell Signals", sell_signals, delta=f"{sell_signals}/{total_analyzed}")
                col4.metric("Avg Confidence", f"{avg_confidence:.0f}%")
                col5.metric("Stocks Analyzed", total_analyzed)
                
                # Enhanced Results Table
                st.subheader("📈 Detailed Analysis Results")
                
                if table_data:
                    df_results = pd.DataFrame(table_data)
                    
                    # Color code signals
                    def color_signals(val):
                        if "STRONG BUY" in val:
                            return "background-color: #00ff00; color: black; font-weight: bold"
                        elif "BUY" in val:
                            return "background-color: #90EE90; color: black; font-weight: bold"
                        elif "STRONG SELL" in val:
                            return "background-color: #ff0000; color: white; font-weight: bold"
                        elif "SELL" in val:
                            return "background-color: #FFB6C1; color: black; font-weight: bold"
                        else:
                            return "background-color: #f0f0f0; color: black"
                    
                    # Apply styling
                    styled_df = df_results.style.applymap(color_signals, subset=['Signal'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Download button for results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"stock_analysis_{dt.date.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                # Individual Stock Analysis
                st.subheader("🎯 Individual Stock Analysis")
                
                # Sort results by signal strength
                results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
                
                for analysis in results_sorted:
                    ticker = analysis["ticker"]
                    signal = analysis["signal"]
                    score = analysis["score"]
                    confidence = analysis["confidence"]
                    
                    # Color code the header
                    if "STRONG BUY" in signal:
                        header_color = "🟢"
                    elif "BUY" in signal:
                        header_color = "🟡"
                    elif "SELL" in signal:
                        header_color = "🔴"
                    else:
                        header_color = "⚪"
                    
                    with st.expander(f"{header_color} {ticker} - {signal} (Score: {score}, Confidence: {confidence}%)"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Analysis details
                            st.markdown("**Key Signals:**")
                            for i, reason in enumerate(analysis.get("reasons", [])[:6], 1):
                                st.markdown(f"{i}. {reason}")
                            
                            # Signal breakdown
                            if analysis.get("signals_breakdown"):
                                st.markdown("**Signal Components:**")
                                breakdown = analysis["signals_breakdown"]
                                for component, value in breakdown.items():
                                    if value != 0:
                                        emoji = "➕" if value > 0 else "➖"
                                        st.markdown(f"  {emoji} {component.title()}: {value:+d}")
                        
                        with col2:
                            # Price and fundamental data
                            st.markdown("**Current Data:**")
                            st.markdown(f"Price: **${analysis['price']:.2f}**")
                            
                            fund_data = analysis.get("fundamental_data", {})
                            if fund_data.get("pe_ratio"):
                                st.markdown(f"P/E Ratio: **{fund_data['pe_ratio']:.1f}**")
                            if fund_data.get("beta"):
                                st.markdown(f"Beta: **{fund_data['beta']:.2f}**")
                            if fund_data.get("dividend_yield"):
                                st.markdown(f"Dividend Yield: **{fund_data['dividend_yield']:.2f}%**")
                            
                            # Sector information
                            if fund_data.get("sector", "Unknown") != "Unknown":
                                st.markdown(f"Sector: **{fund_data['sector']}**")
                            
                            # Sentiment data
                            sentiment = analysis.get("sentiment", {})
                            if sentiment and sentiment.get("n", 0) > 0:
                                st.markdown("**News Sentiment:**")
                                compound = sentiment.get("compound", 0)
                                emoji = "😊" if compound > 0.1 else "😐" if compound > -0.1 else "😟"
                                st.markdown(f"{emoji} Score: **{compound:+.2f}**")
                                st.markdown(f"📰 Articles: **{sentiment['n']}**")
                                st.markdown(f"🎯 Confidence: **{sentiment.get('confidence', 0)*100:.0f}%**")
                        
                        # Interactive chart for individual stock
                        if show_charts:
                            try:
                                # Get fresh data for charting
                                chart_df = get_enhanced_price_data(ticker, lookback_days, interval)
                                if not chart_df.empty:
                                    chart_df = compute_enhanced_indicators(chart_df, INDICATOR_CONFIGS)
                                    
                                    fig = create_enhanced_visualizations(chart_df, ticker, selected_indicators)
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not generate chart for {ticker}: {str(e)}")
                
                # Market Overview Section
                if len(results) > 1:
                    st.subheader("🌍 Market Overview")
                    
                    # Signal distribution chart
                    signal_counts = {}
                    for result in results:
                        signal = result["signal"]
                        signal_counts[signal] = signal_counts.get(signal, 0) + 1
                    
                    if signal_counts:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_pie = px.pie(
                                values=list(signal_counts.values()),
                                names=list(signal_counts.keys()),
                                title="Signal Distribution",
                                color_discrete_map={
                                    "STRONG BUY": "#00ff00",
                                    "BUY": "#90EE90", 
                                    "WEAK BUY": "#FFFFE0",
                                    "HOLD": "#f0f0f0",
                                    "WEAK SELL": "#FFB6C1",
                                    "SELL": "#FFA07A",
                                    "STRONG SELL": "#ff0000"
                                }
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            # Score distribution
                            scores = [r["score"] for r in results]
                            fig_hist = px.histogram(
                                x=scores,
                                nbins=15,
                                title="Score Distribution",
                                labels={"x": "Signal Score", "y": "Count"},
                                color_discrete_sequence=["#1f77b4"]
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                
                # Success message
                st.success(f"✅ Analysis complete! Processed {total_analyzed} stocks from your saved watchlist.")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("Please try again or contact support if the issue persists.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    **📈 Enhanced Stock Signals PRO** - *Comprehensive multi-source financial analysis*
    
    🔄 **Persistent Watchlist:** Your {len(current_watchlist)} stocks are automatically saved
    
    ⚠️ **Disclaimer:** This application provides educational analysis only. Not financial advice. 
    Always consult with qualified financial professionals before making investment decisions.
    """)

if __name__ == "__main__":
    main()