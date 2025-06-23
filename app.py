import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import feedparser

# === Helper Functions ===
def fetch_google_news_rss(query="India stock market"):
    rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '%20')}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    if not feed.entries:
        return ["✅ No recent major headlines found."]
    headlines = [f"📰 {entry.title}" for entry in feed.entries[:5]]
    return headlines

def check_event_warnings(latest_vix, nifty_change_pct):
    warnings = []
    if latest_vix >= 22:
        warnings.append(f"⚠️ High VIX detected ({latest_vix:.2f}) — market may be volatile.")
    if abs(nifty_change_pct) >= 1.5:
        warnings.append(f"⚠️ Large Nifty move ({nifty_change_pct:.2f}%) — possible event impact.")
    if not warnings:
        warnings.append("✅ No major risk signals detected.")
    return warnings

def suggest_option_strategy(signal, latest_price, vix_level):
    atm = round(latest_price / 10) * 10
    spread_width = 20
    if 'Ultra Strong Bullish' in signal or 'Strong Bullish' in signal:
        return (f"💡 **Bull Call Spread**\n"
                f"👉 Buy {atm} CE, Sell {atm + spread_width} CE\n"
                f"👉 Target swing move in 2-3 days")
    elif 'Ultra Strong Bearish' in signal or 'Strong Bearish' in signal:
        return (f"💡 **Bear Put Spread**\n"
                f"👉 Buy {atm} PE, Sell {atm - spread_width} PE\n"
                f"👉 Target swing move in 2-3 days")
    elif 'Mixed' in signal or 'Neutral' in signal:
        if vix_level >= 20:
            return (f"💡 **Iron Condor / Short Strangle**\n"
                    f"👉 Mixed signal + high volatility (VIX {vix_level:.2f})\n"
                    f"👉 Sell OTM Call & Put to collect premium")
        else:
            return "💡 Market indecisive + low VIX — safer to wait for clearer signal."
    else:
        return "💡 No clear strategy. Monitor further."

def compute_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    df['RSI'] = RSIIndicator(close).rsi()
    macd = MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['EMA20'] = EMAIndicator(close, 20).ema_indicator()
    df['EMA50'] = EMAIndicator(close, 50).ema_indicator()
    df['EMA200'] = EMAIndicator(close, 200).ema_indicator()
    df['OBV'] = OnBalanceVolumeIndicator(close, df['Volume']).on_balance_volume()
    bb = BollingerBands(close)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['ATR'] = AverageTrueRange(high, low, close).average_true_range()
    df['ADX'] = ADXIndicator(high, low, close).adx()
    return df

def clean_yf_data(df):
    if df.empty:
        return None
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    if 'Close' not in df.columns:
        return None
    df.dropna(subset=['Close'], inplace=True)
    return df if not df.empty else None

def analyze_timeframe(df, name):
    df = compute_indicators(df)
    latest = df.iloc[-1]
    clues = []
    if latest['RSI'] > 60:
        clues.append('RSI Bullish')
    elif latest['RSI'] < 40:
        clues.append('RSI Bearish')
    if latest['MACD'] > latest['MACD_Signal']:
        clues.append('MACD Bullish')
    elif latest['MACD'] < latest['MACD_Signal']:
        clues.append('MACD Bearish')
    if latest['Close'] > latest['EMA20'] > latest['EMA50']:
        clues.append('EMA Bullish alignment')
    elif latest['Close'] < latest['EMA20'] < latest['EMA50']:
        clues.append('EMA Bearish alignment')
    if latest['ADX'] > 25:
        clues.append('Strong Trend')
    else:
        clues.append('Weak/Moderate Trend')
    if latest['OBV'] > df['OBV'].iloc[-5]:
        clues.append('OBV Up')
    else:
        clues.append('OBV Down')

    bull = sum('Bullish' in c or 'Up' in c for c in clues)
    bear = sum('Bearish' in c or 'Down' in c for c in clues)
    if bull > bear:
        signal = f"Bullish ({name})"
    elif bear > bull:
        signal = f"Bearish ({name})"
    else:
        signal = f"Neutral ({name})"
    return clues, signal, latest['Close']

def stock_analyzer(symbol, vix, nifty_change):
    df_1h = clean_yf_data(yf.download(symbol, period='3mo', interval='1h'))
    df_4h = clean_yf_data(yf.download(symbol, period='6mo', interval='4h'))
    df_1d = clean_yf_data(yf.download(symbol, period='6mo', interval='1d'))
    if not df_1h or not df_4h or not df_1d:
        st.warning(f"⚠️ Data issue with {symbol}")
        return None, None
    clues1h, sig1h, price1h = analyze_timeframe(df_1h, '1H')
    clues4h, sig4h, price4h = analyze_timeframe(df_4h, '4H')
    clues1d, sig1d, price1d = analyze_timeframe(df_1d, '1D')

    # Display details
    st.subheader(f"{symbol} - 1H")
    for c in clues1h:
        st.write(f"🔹 {c}")
    st.write(f"➡ Signal: {sig1h}")

    st.subheader(f"{symbol} - 4H")
    for c in clues4h:
        st.write(f"🔹 {c}")
    st.write(f"➡ Signal: {sig4h}")

    st.subheader(f"{symbol} - 1D")
    for c in clues1d:
        st.write(f"🔹 {c}")
    st.write(f"➡ Signal: {sig1d}")

    # Combine signals
    signals = [sig1h, sig4h, sig1d]
    if all('Bullish' in s for s in signals):
        final = '💹 Ultra Strong Bullish'
    elif all('Bearish' in s for s in signals):
        final = '🔻 Ultra Strong Bearish'
    elif sum('Bullish' in s for s in signals) >= 2:
        final = '📈 Strong Bullish'
    elif sum('Bearish' in s for s in signals) >= 2:
        final = '📉 Strong Bearish'
    else:
        final = '⚖️ Mixed / Neutral'

    st.success(f"Final Signal: {final}")
    st.info(f"VIX: {vix:.2f}, Nifty % Change: {nifty_change:.2f}%")

    # Return final signal + latest price
    return final, price1d

# === Streamlit App ===
st.title("📊 Multi-Timeframe Stock Analyzer + Option Suggestion")

symbols = st.text_input("Enter NSE symbols (comma-separated):", "RECLTD.NS, INFY.NS").split(",")

if st.button("Run Analysis"):
    vix_df = clean_yf_data(yf.download('^INDIAVIX', period='1d', interval='1m'))
    nifty_df = clean_yf_data(yf.download('^NSEI', period='7d'))
    vix_val = vix_df['Close'].iloc[-1] if vix_df is not None else 0
    nifty_change = ((nifty_df['Close'].iloc[-1] - nifty_df['Close'].iloc[0]) / nifty_df['Close'].iloc[0] * 100) if nifty_df is not None else 0

    for sym in symbols:
        final_sig, last_price = stock_analyzer(sym.strip(), vix_val, nifty_change)
        if final_sig and last_price:
            st.markdown(suggest_option_strategy(final_sig, last_price, vix_val))

    # Show news
    st.subheader("📰 Top Market Headlines")
    for news in fetch_google_news_rss("Nifty OR Sensex OR RBI OR India stock market"):
        st.write(news)

    # Event warnings
    st.subheader("⚠️ Market Risk Check")
    for warn in check_event_warnings(vix_val, nifty_change):
        st.write(warn)
