import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import streamlit as st
import feedparser

def stock_analyzer(symbols):
    def detect_divergence(price, indicator):
        if len(price) < 3 or len(indicator) < 3:
            return None
        if price.iloc[-1] < price.iloc[-2] and indicator.iloc[-1] > indicator.iloc[-2]:
            return 'Bullish Divergence'
        if price.iloc[-1] > price.iloc[-2] and indicator.iloc[-1] < indicator.iloc[-2]:
            return 'Bearish Divergence'
        return None
    def check_event_warnings(latest_vix, nifty_trend):
        warnings = []
    
        # VIX warning
        if latest_vix >= 22:
            warnings.append(f"⚠️ High VIX detected ({latest_vix:.2f}) — market may be very volatile.")
        
        # Nifty sudden reversal alert
        if abs(nifty_trend) >= 1.5:
            warnings.append("⚠️ Large Nifty move in recent days — possible reaction to event/news.")
        
        # TODO: You could expand this to scrape a calendar of events (Fed meeting, RBI policy, etc)
        
        if warnings:
            warning_text = "\n".join(warnings)
        else:
            warning_text = "✅ No major risk signals detected. Proceed with caution but no red flags."
    
        return warning_text

    def calc_support_resistance(close_series):
        window = 20
        support = close_series.rolling(window).min().iloc[-1]
        resistance = close_series.rolling(window).max().iloc[-1]
        return support, resistance
        
    def compute_indicators(df):
        close = df['Close']
        high = df['High']
        low = df['Low']
        df['RSI'] = RSIIndicator(close).rsi()
        macd_obj = MACD(close)
        df['MACD'] = macd_obj.macd()
        df['MACD_Signal'] = macd_obj.macd_signal()
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
    def fetch_google_news_rss(query="India stock market"):
        # Build the RSS feed URL
        rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '%20')}&hl=en-IN&gl=IN&ceid=IN:en"
        
        feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            return "✅ No recent major headlines found."
        
        headlines = []
        for entry in feed.entries[:5]:  # Get top 5 headlines
            headlines.append(f"📰 {entry.title} ({entry.published})")
        
        return "\n".join(headlines)
    def detect_trend_reversal(df):
        rsi = df['RSI']
        obv = df['OBV']
    
        # Ensure enough data points
        if len(rsi) < 5 or len(obv) < 5:
            return "Not enough data"
    
        recent_rsi = rsi.tail(5)
        recent_obv = obv.tail(5)
    
        # RSI conditions
        rsi_bull_cond = (recent_rsi.iloc[-1] > recent_rsi.iloc[-2]) and (recent_rsi.min() < 35)
        rsi_bear_cond = (recent_rsi.iloc[-1] < recent_rsi.iloc[-2]) and (recent_rsi.max() > 65)
    
        # OBV conditions
        obv_bull_cond = (recent_obv.iloc[-1] > recent_obv.iloc[-2])
        obv_bear_cond = (recent_obv.iloc[-1] < recent_obv.iloc[-2])
    
        # Combine conditions
        if rsi_bull_cond and obv_bull_cond:
            return "📈 Possible Bullish Reversal"
        elif rsi_bear_cond and obv_bear_cond:
            return "📉 Possible Bearish Reversal"
        else:
            return "⚖️ No clear reversal signal"
    def suggest_option_strategy(signal, latest_price, vix_level):
        """
        Suggest detailed option strategy based on final signal, price, and VIX.
        Returns a strategy description string.
        """
        atm = round(latest_price / 10) * 10  # Nearest 10 strike
        spread_width = 20  # You can make this configurable
        suggestion = ""
    
        if 'Strong Bullish' in signal:
            suggestion = (f"💡 Option Strategy: **Bull Call Spread**\n"
                          f"👉 Reason: Strong bullish signal with technical alignment.\n"
                          f"👉 Action: Buy {atm} CE, Sell {atm + spread_width} CE.\n"
                          f"👉 Target: Swing move over 2-3 days.\n"
                          f"👉 Benefit: Limited risk with defined reward.\n")
    
        elif 'Strong Bearish' in signal:
            suggestion = (f"💡 Option Strategy: **Bear Put Spread**\n"
                          f"👉 Reason: Strong bearish signal detected.\n"
                          f"👉 Action: Buy {atm} PE, Sell {atm - spread_width} PE.\n"
                          f"👉 Target: Swing move over 2-3 days.\n"
                          f"👉 Benefit: Limited risk, cost-effective bearish play.\n")
    
        elif 'Mixed' in signal or 'Neutral' in signal:
            if vix_level >= 20:
                suggestion = (f"💡 Option Strategy: **Iron Condor / Short Strangle**\n"
                              f"👉 Reason: Mixed/neutral signal + high volatility (VIX {vix_level}).\n"
                              f"👉 Action: Sell OTM Call & Put to collect premium.\n"
                              f"👉 Benefit: Profit from time decay & volatility crush.\n")
            else:
                suggestion = (f"💡 Option Strategy: **Wait / Small Range Bound Trade**\n"
                              f"👉 Reason: Market indecisive, VIX low (VIX {vix_level}).\n"
                              f"👉 Action: Safer to wait for clearer direction.\n")
    
        else:
            suggestion = (f"💡 Option Strategy: **No clear edge**\n"
                          f"👉 Reason: Signal unclear; better to avoid entry.\n"
                          f"👉 Action: Observe or look for better setups.\n")
    
        return suggestion

    def clean_yf_data(df):
        if df.empty:
            return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        if 'Close' not in df.columns:
            return None
        df.dropna(subset=['Close'], inplace=True)
        return df if not df.empty else None

    # Download VIX + Nifty trend
    try:
        df_vix = clean_yf_data(yf.download('^INDIAVIX', period='1d', interval='1m'))
        latest_vix = df_vix['Close'].iloc[-1] if df_vix is not None else None
        vix_comment = (
            "High volatility" if latest_vix > 20 else
            "Low volatility" if latest_vix < 14 else
            "Moderate"
        ) if latest_vix is not None else "N/A"
    except:
        latest_vix = None
        vix_comment = "N/A"

    try:
        df_nifty = clean_yf_data(yf.download('^NSEI', period='7d'))
        nifty_trend = (
            "up" if df_nifty['Close'].iloc[-1] > df_nifty['Close'].iloc[0] else "down"
        ) if df_nifty is not None else "N/A"
    except:
        nifty_trend = "N/A"

    for symbol in symbols:
        st.header(f"🔍 Analyzing {symbol}")

        df_4h = clean_yf_data(yf.download(symbol, period='6mo', interval='4h'))
        df_1d = clean_yf_data(yf.download(symbol, period='6mo', interval='1d'))
        df_1h = clean_yf_data(yf.download(symbol, period='3mo', interval='1h'))

        if df_4h is None or df_1d is None or df_1h is None:
            st.warning(f"⚠️ Insufficient or invalid data for {symbol}. Skipping...")
            continue

        df_4h = compute_indicators(df_4h)
        df_1d = compute_indicators(df_1d)
        df_1h = compute_indicators(df_1h)

        def analyze_df(df, tf_name):
            latest = df.iloc[-1]
            close = df['Close']
            clues = []
            reversal_signal = detect_trend_reversal(df)
            clues.append(reversal_signal)
            atr_pct = (latest['ATR'] / latest['Close']) * 100
            rsi_bull = 65 if atr_pct > 3 else 60
            rsi_bear = 35 if atr_pct > 3 else 40

            if latest['RSI'] > rsi_bull:
                clues.append(f'RSI Bullish (>{rsi_bull})')
            elif latest['RSI'] < rsi_bear:
                clues.append(f'RSI Bearish (<{rsi_bear})')
            else:
                clues.append('RSI Neutral')

            if latest['MACD'] > latest['MACD_Signal']:
                clues.append('MACD Bullish')
            elif latest['MACD'] < latest['MACD_Signal']:
                clues.append('MACD Bearish')
                
            macd_cross_bars = np.where((df['MACD'] > df['MACD_Signal']) != (df['MACD'].shift(1) > df['MACD_Signal'].shift(1)))[0]
            if len(macd_cross_bars) > 0:
                bars_since_cross = len(df) - macd_cross_bars[-1]
                clues.append(f'MACD crossover {bars_since_cross} bars ago')
                
            atr_mean = df['ATR'].tail(10).mean()
            if latest['ATR'] > 1.2 * atr_mean:
                clues.append('High Volatility')
            elif latest['ATR'] < 0.8 * atr_mean:
                clues.append('Low Volatility')
            else:
                clues.append('Normal Volatility')    
                
          
            if latest['Close'] > df['Close'].iloc[-2]:
                clues.append('Last candle bullish close')
            else:
                clues.append('Last candle bearish close')

           
            recent_range = close.tail(10).max() - close.tail(10).min()
            if recent_range / latest['Close'] < 0.02:
                clues.append('Consolidation zone (<2% range)')
               
            if latest['ADX'] > 25:
                clues.append('Strong Trend (ADX > 25)')
            elif latest['ADX'] < 20:
                clues.append('Weak Trend (ADX < 20)')
            else:
                clues.append('Moderate Trend (ADX 20-25)')

            if latest['Close'] > latest['EMA20'] > latest['EMA50']:
                clues.append('EMA Bullish alignment')
            elif latest['Close'] < latest['EMA20'] < latest['EMA50']:
                clues.append('EMA Bearish alignment')

            bb_width = latest['BB_High'] - latest['BB_Low']
            bb_mean = (df['BB_High'] - df['BB_Low']).tail(10).mean()
            if bb_width < 0.7 * bb_mean:
                clues.append('BB Squeeze (Potential breakout)')

            if latest['OBV'] > df['OBV'].iloc[-5]:
                clues.append('OBV Up')
            else:
                clues.append('OBV Down')

            div = detect_divergence(close.tail(5), df['RSI'].tail(5))
            if div:
                clues.append(div)

            support, resistance = calc_support_resistance(close)
            clues.append(f"Support ~{support:.2f}, Resistance ~{resistance:.2f}")

            bull = sum('Bullish' in c or 'Up' in c for c in clues)
            bear = sum('Bearish' in c or 'Down' in c for c in clues)
            if bull > bear:
                signal = f"Bullish (hold ~{3 if tf_name == '4H' else 7} bars)"
            elif bear > bull:
                signal = f"Bearish (hold ~{3 if tf_name == '4H' else 7} bars)"
            else:
                signal = f"Neutral (hold ~{3 if tf_name == '4H' else 7} bars)"

            return clues, signal

        clues_4h, signal_4h = analyze_df(df_4h, '4H')
        clues_1d, signal_1d = analyze_df(df_1d, '1D')
        clues_1h, signal_1h = analyze_df(df_1h, '1H')
        if 'Bullish' in signal_1h and 'Bullish' in signal_4h and 'Bullish' in signal_1d:
            final = '💹 Ultra Strong Bullish (1H + 4H + 1D agree)'
        elif 'Bearish' in signal_1h and 'Bearish' in signal_4h and 'Bearish' in signal_1d:
            final = '🔻 Ultra Strong Bearish (1H + 4H + 1D agree)'
        elif ('Bullish' in signal_1h and 'Bullish' in signal_4h) or ('Bullish' in signal_4h and 'Bullish' in signal_1d):
            final = '📈 Strong Bullish (2 TF agree)'
        elif ('Bearish' in signal_1h and 'Bearish' in signal_4h) or ('Bearish' in signal_4h and 'Bearish' in signal_1d):
            final = '📉 Strong Bearish (2 TF agree)'
        else:
            final = '⚖️ Mixed / Neutral'
        

        st.subheader(f"{symbol} 1H")
        for c in clues_1h:
            st.write(f"🔹 {c}")
        st.write(f"➡ 1H Signal: {signal_1h}")
        st.subheader(f"{symbol} 4H")
        
        for c in clues_4h:
            st.write(f"🔹 {c}")
        st.write(f"➡ 4H Signal: {signal_4h}")

        st.subheader(f"{symbol} 1D")
        for c in clues_1d:
            st.write(f"🔹 {c}")
        st.write(f"➡ 1D Signal: {signal_1d}")

        st.success(f"Final Combined Signal: {final}")
        st.info(f"VIX: {latest_vix:.2f} ({vix_comment}), Nifty Trend: {nifty_trend}")
        
        news_report = fetch_google_news_rss("Nifty OR Sensex OR RBI OR India stock market")
        print("\n=== 📰 Market Headlines ===")
        print(news_report)
        
        nifty_change_pct = (df_nifty['Close'].iloc[-1] - df_nifty['Close'].iloc[0]) / df_nifty['Close'].iloc[0] * 100
        event_warning = check_event_warnings(latest_vix, nifty_change_pct)
        print("\n=== ⚠️ Market Risk Check ===")
        print(event_warning)
        latest_price = df_1d['Close'].iloc[-1]
        option_suggestion = suggest_option_strategy(final, latest_price, latest_vix)
        print(option_suggestion)

# === Streamlit app code ===
st.title("📈 Stock Analyzer")

# User input for stock symbols
symbols = st.text_input("Enter stock symbols (comma-separated):", "RECLTD.NS, INFY.NS").split(",")

# Run analysis
if st.button("Run Analysis"):
    stock_analyzer([s.strip() for s in symbols])
    
