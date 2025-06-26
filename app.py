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
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NEWS_API_KEY = "fe8fa3ba495e480dbcc76feabad630b0"  
analyzer = SentimentIntensityAnalyzer()
def fetch_sentiment_from_newsapi(query, max_articles=10):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize={max_articles}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return None

    data = response.json()
    sentiments = []
    for article in data.get("articles", []):
        score = analyzer.polarity_scores(article['title'])['compound']
        sentiments.append(score)
    
    if not sentiments:
        return None
    
    avg_sentiment = sum(sentiments) / len(sentiments)
    return avg_sentiment
def display_sentiment_summary(symbols):
    st.markdown("### 🧠 Market Sentiment Summary (NewsAPI + VADER)")

    # Nifty/market-wide sentiment
    market_sentiment = fetch_sentiment_from_newsapi("Nifty OR Sensex OR Indian Stock Market")
    if market_sentiment is not None:
        market_summary = (
            "📈 Bullish" if market_sentiment > 0.05 else
            "📉 Bearish" if market_sentiment < -0.05 else
            "⚖️ Neutral"
        )
        st.markdown(f"**🗞️ Nifty Market Sentiment**: `{market_sentiment:+.2f}` → {market_summary}")
    else:
        st.warning("Could not fetch market sentiment.")

    # Per-stock sentiment
    for symbol in symbols:
        sentiment = fetch_sentiment_from_newsapi(symbol)
        if sentiment is not None:
            label = (
                "📈 Bullish" if sentiment > 0.05 else
                "📉 Bearish" if sentiment < -0.05 else
                "⚖️ Neutral"
            )
            st.markdown(f"**🧾 {symbol} Sentiment**: `{sentiment:+.2f}` → {label}")
        else:
            st.warning(f"No sentiment found for {symbol}")

def compute_supertrend(df, period=10, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=period).average_true_range()
    supertrend = pd.Series(index=df.index, dtype='float64')
    direction = pd.Series(index=df.index, dtype='int')

    supertrend.iloc[0] = hl2.iloc[0] + multiplier * atr.iloc[0]
    direction.iloc[0] = 1  

    for i in range(1, len(df)):
        if direction.iloc[i-1] == 1:
            supertrend.iloc[i] = min(hl2.iloc[i] + multiplier * atr.iloc[i], supertrend.iloc[i-1])
        else:
            supertrend.iloc[i] = max(hl2.iloc[i] - multiplier * atr.iloc[i], supertrend.iloc[i-1])

        if df['Close'].iloc[i] > supertrend.iloc[i]:
            direction.iloc[i] = 1
        else:
            direction.iloc[i] = -1

    df['Supertrend'] = supertrend
    df['Supertrend_dir'] = direction
    return df

def compute_stoch_rsi(df, window=14, smooth1=3, smooth2=3):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    stoch_rsi = (rsi - rsi.rolling(window).min()) / (rsi.rolling(window).max() - rsi.rolling(window).min())
    stoch_rsi_k = stoch_rsi.rolling(smooth1).mean() * 100
    stoch_rsi_d = stoch_rsi_k.rolling(smooth2).mean()

    df['StochRSI_K'] = stoch_rsi_k
    df['StochRSI_D'] = stoch_rsi_d
    return df

def compute_vwap(df):
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    return df

def stock_analyzer(symbols):
    def detect_divergence(price, indicator):
        if len(price) < 3 or len(indicator) < 3:
            return None
        if price.iloc[-1] < price.iloc[-2] and indicator.iloc[-1] > indicator.iloc[-2]:
            return 'Bullish Divergence'
        if price.iloc[-1] > price.iloc[-2] and indicator.iloc[-1] < indicator.iloc[-2]:
            return 'Bearish Divergence'
        return None
    def generate_market_warnings(latest_vix, nifty_change_pct):
        """
        Generate risk warnings based on VIX level and Nifty % change over period.
        """
        warnings = []
    
        # VIX warning
        if latest_vix is not None:
            if latest_vix >= 22:
                warnings.append(f"⚠️ High VIX ({latest_vix:.2f}) → Market may be very volatile!")
            elif latest_vix < 12:
                warnings.append(f"⚠️ Very low VIX ({latest_vix:.2f}) → Risk of complacency in market!")
    
        # Nifty sudden change
        if nifty_change_pct is not None:
            if abs(nifty_change_pct) >= 1.5:
                warnings.append(f"⚠️ Nifty moved {nifty_change_pct:.2f}% in recent days → Possible event/news impact!")
    
        # Combine warnings
        if warnings:
            return "\n".join(warnings)
        else:
            return "✅ No major risk signals detected. Market seems stable at the moment."

    def calc_support_resistance(close_series):
        window = 20
        support = close_series.rolling(window).min().iloc[-1]
        resistance = close_series.rolling(window).max().iloc[-1]
        return support, resistance
    def support_resistance_alert(latest_price, support, resistance):
        """
        Generate alert if price is too close to support or resistance.
        """
        support_gap_pct = (latest_price - support) / latest_price * 100
        resistance_gap_pct = (resistance - latest_price) / latest_price * 100
        
        alerts = []
        
        if support_gap_pct >= 0 and support_gap_pct < 2:
            alerts.append(f"⚠️ Price is within {support_gap_pct:.2f}% of support — risk of breakdown if breached.")
        
        if resistance_gap_pct >= 0 and resistance_gap_pct < 2:
            alerts.append(f"⚠️ Price is within {resistance_gap_pct:.2f}% of resistance — possible reversal zone.")
        
        if not alerts:
            return "✅ No immediate support/resistance barrier risk."
        else:
            return "\n".join(alerts)
    def suggest_option_strategy(final_signal, latest_price, vix_level, confidence_percent, support, resistance):
        strike_step = 10
        atm = round(latest_price / strike_step) * strike_step
        spread_width = 20
        vix_display = f"{vix_level:.2f}" if vix_level is not None else "N/A"
    
        support_gap_pct = (latest_price - support) / latest_price * 100
        resistance_gap_pct = (resistance - latest_price) / latest_price * 100
    
        near_support = 0 <= support_gap_pct < 2
        near_resistance = 0 <= resistance_gap_pct < 2
    
        strong_conf = confidence_percent >= 70
        moderate_conf = 50 <= confidence_percent < 70
        low_conf = confidence_percent < 50
    
        suggestion = ""
        strategy_type = "Wait"
    
        if 'Bullish' in final_signal:
            if near_resistance:
                suggestion = (
                    f"⚠️ Price is near resistance ({resistance_gap_pct:.2f}%) — risky to go long.\n"
                    f"👉 Wait for breakout or use tight SL.\n"
                    f"📈 Confidence: {confidence_percent}%"
                )
                strategy_type = "Wait"
            elif strong_conf:
                suggestion = (
                    f"💡 **Bull Call Spread** (High Confidence)\n"
                    f"👉 Buy {atm} CE, Sell {atm + spread_width} CE\n"
                    f"🎯 Target: 2–5 day swing\n"
                    f"📈 Confidence: {confidence_percent}%"
                )
                strategy_type = "Swing"
            elif moderate_conf:
                suggestion = (
                    f"💡 **Moderate Bull Spread**\n"
                    f"👉 Buy {atm + 10} CE, Sell {atm + 30} CE\n"
                    f"⚠️ Medium Confidence: {confidence_percent}%, use tight SL"
                )
                strategy_type = "Intraday/Swing"
            else:
                suggestion = (
                    f"⚠️ Weak Bullish Bias ({confidence_percent}%)\n"
                    f"👉 Suggest: Intraday scalps or wait"
                )
                strategy_type = "Scalp"
    
        elif 'Bearish' in final_signal:
            if near_support:
                suggestion = (
                    f"⚠️ Price is near support ({support_gap_pct:.2f}%) — risky to short.\n"
                    f"👉 Wait for breakdown confirmation.\n"
                    f"📉 Confidence: {confidence_percent}%"
                )
                strategy_type = "Wait"
            elif strong_conf:
                suggestion = (
                    f"💡 **Bear Put Spread** (High Confidence)\n"
                    f"👉 Buy {atm} PE, Sell {atm - spread_width} PE\n"
                    f"📉 Confidence: {confidence_percent}%"
                )
                strategy_type = "Swing"
            elif moderate_conf:
                suggestion = (
                    f"💡 **Moderate Bear Spread**\n"
                    f"👉 Buy {atm - 10} PE, Sell {atm - 30} PE\n"
                    f"⚠️ Confidence: {confidence_percent}%, use SL"
                )
                strategy_type = "Intraday/Swing"
            else:
                suggestion = (
                    f"⚠️ Weak Bearish Bias ({confidence_percent}%)\n"
                    f"👉 Suggest: Avoid fresh shorts, wait for breakdown"
                )
                strategy_type = "Scalp"
    
        elif 'Neutral' in final_signal:
            if vix_level and vix_level >= 20 and strong_conf:
                suggestion = (
                    f"💡 **Iron Condor** (High Volatility)\n"
                    f"👉 Sell {atm + 50} CE & {atm - 50} PE\n"
                    f"🎯 Benefit: Theta decay\n"
                    f"📊 Confidence: {confidence_percent}%"
                )
                strategy_type = "Range Positional"
            else:
                suggestion = (
                    f"🔍 Neutral bias — {confidence_percent}%\n"
                    f"👉 Suggest: Wait for breakout or use range-bound strategy"
                )
                strategy_type = "Wait"
    
        return suggestion, strategy_type
    
    def detect_candlestick_patterns(df):
        df = df.copy()
        body = abs(df['Close'] - df['Open'])
        range_ = df['High'] - df['Low']
        upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
        lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
        avg_volume = df['Volume'].rolling(window=5).mean()
        high_volume = df['Volume'] > avg_volume
    
        df['Doji'] = (body <= 0.1 * range_)
        df['Hammer'] = (
            (body <= 0.3 * range_) &
            (lower_shadow >= 2 * body) &
            (upper_shadow <= 0.1 * range_) &
            high_volume
        )
        df['Inverted_Hammer'] = (
            (body <= 0.3 * range_) &
            (upper_shadow >= 2 * body) &
            (lower_shadow <= 0.1 * range_) &
            high_volume
        )
        df['Hanging_Man'] = df['Hammer'] & (df['Close'] < df['Open'])
        df['Shooting_Star'] = (
            (body <= 0.3 * range_) &
            (upper_shadow >= 2 * body) &
            (lower_shadow <= 0.1 * range_) &
            high_volume
        )
        df['Bullish_Engulfing'] = (
            (df['Close'] > df['Open']) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Open'] <= df['Close'].shift(1)) &
            (df['Close'] >= df['Open'].shift(1)) &
            high_volume
        )
        df['Bearish_Engulfing'] = (
            (df['Close'] < df['Open']) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Open'] >= df['Close'].shift(1)) &
            (df['Close'] <= df['Open'].shift(1)) &
            high_volume
        )
        df['Piercing_Line'] = (
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Close'] > df['Open']) &
            (df['Close'] > (df['Open'].shift(1) + df['Close'].shift(1)) / 2) &
            (df['Open'] < df['Close'].shift(1)) &
            high_volume
        )
        df['Dark_Cloud_Cover'] = (
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Close'] < df['Open']) &
            (df['Close'] < (df['Open'].shift(1) + df['Close'].shift(1)) / 2) &
            (df['Open'] > df['Close'].shift(1)) &
            high_volume
        )
        df['Three_White_Soldiers'] = (
            (df['Close'] > df['Open']) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Close'].shift(2) > df['Open'].shift(2)) &
            high_volume
        )
        df['Three_Black_Crows'] = (
            (df['Close'] < df['Open']) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            high_volume
        )
        df['Morning_Star'] = (
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            (abs(df['Close'].shift(1) - df['Open'].shift(1)) <= 0.1 * (df['High'].shift(1) - df['Low'].shift(1))) &
            (df['Close'] > df['Open']) &
            (df['Close'] > (df['Open'].shift(2) + df['Close'].shift(2)) / 2) &
            high_volume
        )
        df['Evening_Star'] = (
            (df['Close'].shift(2) > df['Open'].shift(2)) &
            (abs(df['Close'].shift(1) - df['Open'].shift(1)) <= 0.1 * (df['High'].shift(1) - df['Low'].shift(1))) &
            (df['Close'] < df['Open']) &
            (df['Close'] < (df['Open'].shift(2) + df['Close'].shift(2)) / 2) &
            high_volume
        )
        return df

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
        df = compute_supertrend(df)
        df = compute_stoch_rsi(df)
        df = compute_vwap(df)
        df = detect_candlestick_patterns(df)
        return df
    
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
    def detect_gap(df_1d):
        if len(df_1d) < 2:
            return "Not enough data"
        prev_close = df_1d['Close'].iloc[-2]
        today_open = df_1d['Open'].iloc[-1]
        gap_pct = (today_open - prev_close) / prev_close * 100
        if abs(gap_pct) > 1:
            return f"⚠️ {gap_pct:.2f}% gap at open — exercise caution!"
        else:
            return "No significant gap."
   
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
        
        sentiment_score = fetch_sentiment_from_newsapi(symbol)
        if sentiment_score is None:
            sentiment_score = 0.0  # default neutral

        if df_4h is None or df_1d is None or df_1h is None:
            st.warning(f"⚠️ Insufficient or invalid data for {symbol}. Skipping...")
            continue

        df_4h = compute_indicators(df_4h)
        df_1d = compute_indicators(df_1d)
        gap_info = detect_gap(df_1d)
        st.info(gap_info)
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
            # Swing / Positional signals
            swing_msg = ""
            positional_msg = ""
            if latest['Supertrend_dir'] == 1 and latest['StochRSI_K'] > 50:
                swing_msg = "🚀 Swing Bullish Signal (Supertrend + StochRSI)"
            elif latest['Supertrend_dir'] == -1 and latest['StochRSI_K'] < 50:
                swing_msg = "⚠️ Swing Bearish Signal (Supertrend + StochRSI)"
        
            if latest['Close'] > latest['VWAP']:
                positional_msg = "✅ Positional Bullish Bias (Price above VWAP)"
            else:
                positional_msg = "🔻 Positional Bearish Bias (Price below VWAP)"
            # Price Action Confirmation Logic
            if latest['Close'] > resistance * 1.002:  # Breakout with at least 0.2% margin
                clues.append("✅ Price action confirms breakout above resistance")
            elif latest['Bullish_Engulfing'] or latest['Three_White_Soldiers'] or latest['Piercing_Line']:
                clues.append("✅ Bullish candlestick pattern confirms breakout intent")
            else:
                clues.append("⚠️ No strong price action confirmation above resistance")
            recent_vol = df['Volume'].tail(5).mean()
            breakout_vol_confirmed = latest['Volume'] > 1.5 * recent_vol
            
            if breakout_vol_confirmed:
                clues.append("📊 Volume supports move — strong breakout potential")
            else:
                clues.append("⚠️ Weak volume — move may not sustain")

            clues.append(swing_msg)
            clues.append(positional_msg)

            return clues, signal, support, resistance
        clues_4h, signal_4h, support_4h, resistance_4h = analyze_df(df_4h, '4H')
        clues_1d, signal_1d, support_1d, resistance_1d = analyze_df(df_1d, '1D')
        clues_1h, signal_1h, support_1h, resistance_1h = analyze_df(df_1h, '1H')
        # === Compute weighted final signal ===
        
        bull_clues = sum('Bullish' in c or 'Up' in c for c in clues_1h + clues_4h + clues_1d)
        bear_clues = sum('Bearish' in c or 'Down' in c for c in clues_1h + clues_4h + clues_1d)
        
        total_clues = bull_clues + bear_clues
        confidence = (abs(bull_clues - bear_clues) / total_clues) if total_clues else 0
        confidence_percent = round(confidence * 100)
        
        # === Weighted Signal Logic ===
        score = 0
        if 'Bullish' in signal_1d:
            score += 0.6
        if 'Bullish' in signal_4h:
            score += 0.3
        if 'Bullish' in signal_1h:
            score += 0.1
        if 'Bearish' in signal_1d:
            score -= 0.6
        if 'Bearish' in signal_4h:
            score -= 0.3
        if 'Bearish' in signal_1h:
            score -= 0.1
        
        # Determine bias and final signal
        confidence = round(abs(score) * 100)
        bias = 'Bullish' if score > 0 else 'Bearish' if score < 0 else 'Neutral'
        # === Adjust confidence using support/resistance proximity ===
        resistance_gap_pct = (resistance_1d - latest_price) / latest_price * 100
        support_gap_pct = (latest_price - support_1d) / latest_price * 100
        
        if 0 <= resistance_gap_pct <= 1.5:
            confidence -= 10  # Near resistance — risky
        
        if 0 <= support_gap_pct <= 1.5:
            confidence += 10  # Near support — could bounce
        
        # === OBV confirmation adjustment ===
        obv_trend = df_1d['OBV'].iloc[-1] - df_1d['OBV'].iloc[-5]
        if obv_trend > 0 and latest_price > resistance_1d:
            confidence += 10  # Volume supports breakout
        elif obv_trend < 0 and latest_price >= resistance_1d:
            confidence -= 10  # Price at top, OBV falling = caution
        
        # Clamp confidence between 0 and 100
        confidence = max(0, min(100, confidence))

        if confidence >= 70:
            final = f"💹 Ultra Strong {bias} (Confidence: {confidence}%)"
        elif confidence >= 40:
            final = f"📈 Moderate {bias} Bias (Confidence: {confidence}%)"
        else:
            final = f"⚖️ Mixed/Neutral (Confidence: {confidence}%)"
        
        # === Trade Type Suggestion ===
        def suggest_trade_timing(signal_1h, signal_4h, signal_1d):
            if 'Bullish' in signal_1h and 'Bullish' in signal_4h and 'Bullish' in signal_1d:
                return "🧭 Positional Buy Setup (>5 days)"
            elif 'Bullish' in signal_1h and 'Bullish' in signal_4h:
                return "🔁 Swing Trade Opportunity (2–5 days)"
            elif 'Bullish' in signal_1h:
                return "🕐 Intraday Long Bias"
            elif 'Bearish' in signal_1h and 'Bearish' in signal_4h and 'Bearish' in signal_1d:
                return "🧭 Positional Short Setup (>5 days)"
            elif 'Bearish' in signal_1h and 'Bearish' in signal_4h:
                return "🔁 Swing Short Opportunity (2–5 days)"
            elif 'Bearish' in signal_1h:
                return "🕐 Intraday Short Bias"
            else:
                return "⚠️ Unclear — Better to Wait"
        
        trade_type = suggest_trade_timing(signal_1h, signal_4h, signal_1d)
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

        st.info(f"VIX: {latest_vix:.2f} ({vix_comment}), Nifty Trend: {nifty_trend}")
        st.markdown(f"**🧮 Clue Breakdown**: Bullish clues = {bull_clues}, Bearish clues = {bear_clues}")
        st.progress(confidence)  # Confidence as a visual progress bar
        st.subheader("📢 Final Signal")
        st.success(final)
        st.markdown(f"🔁 Trade Type: **{trade_type}**")
        if latest_vix and latest_vix > 20:
            st.warning(f"⚠️ VIX {latest_vix:.2f} is high — prefer non-directional strategies (Iron Condor etc).")
        if 'Bullish' in final and nifty_trend == 'down':
            st.warning(f"⚠️ {final} but Nifty down — caution advised!")
        elif 'Bearish' in final and nifty_trend == 'up':
            st.warning(f"⚠️ {final} but Nifty up — caution advised!")
        st.subheader("📊 Candlestick Patterns (1H)")
        st.markdown(candlestick_summary(df_1h))
        st.subheader("📊 Candlestick Patterns (4H)")
        st.markdown(candlestick_summary(df_4h))
        st.subheader("📊 Candlestick Patterns (1D)")
        st.markdown(candlestick_summary(df_1d))
        latest_price = df_1d['Close'].iloc[-1]
        vix_for_strategy = latest_vix if latest_vix is not None else 0
        nifty_change_pct = None
        if df_nifty is not None and not df_nifty.empty:
            nifty_change_pct = (df_nifty['Close'].iloc[-1] - df_nifty['Close'].iloc[0]) / df_nifty['Close'].iloc[0] * 100
        warnings_text = generate_market_warnings(latest_vix, nifty_change_pct)     
        st.subheader("⚠️ Market Risk Warnings")
        st.markdown(warnings_text)
        strategy_suggestion, strat_type = suggest_option_strategy(final, latest_price, vix_for_strategy, confidence_percent, support_1d, resistance_1d)
        st.subheader("💡 Option Strategy Suggestion")
        st.markdown(strategy_suggestion)
        st.subheader("📏 Support/Resistance Alert")
        sr_alert = support_resistance_alert(latest_price, support_1d, resistance_1d)
        st.markdown(sr_alert)

def candlestick_summary(df):
    recent = df.iloc[-1]
    msgs = []

    if recent['Doji']:
        msgs.append("⚠️ Doji: Market indecision or reversal risk.")

    if recent['Hammer']:
        msgs.append("🔨 Hammer (volume confirmed): Potential bullish reversal.")

    if recent['Inverted_Hammer']:
        msgs.append("🔄 Inverted Hammer (volume confirmed): Possible bullish reversal.")

    if recent['Hanging_Man']:
        msgs.append("📉 Hanging Man (volume confirmed): Bearish reversal risk at top.")

    if recent['Shooting_Star']:
        msgs.append("🌠 Shooting Star (volume confirmed): Potential bearish reversal.")

    if recent['Bullish_Engulfing']:
        msgs.append("🚀 Bullish Engulfing (volume confirmed): Strong bullish signal.")

    if recent['Bearish_Engulfing']:
        msgs.append("⚠️ Bearish Engulfing (volume confirmed): Strong bearish signal.")

    if recent['Piercing_Line']:
        msgs.append("💡 Piercing Line (volume confirmed): Bullish reversal hint.")

    if recent['Dark_Cloud_Cover']:
        msgs.append("🌩️ Dark Cloud Cover (volume confirmed): Bearish reversal hint.")

    if recent['Three_White_Soldiers']:
        msgs.append("🏹 Three White Soldiers (volume confirmed): Strong bullish momentum.")

    if recent['Three_Black_Crows']:
        msgs.append("🐦 Three Black Crows (volume confirmed): Strong bearish momentum.")

    if recent['Morning_Star']:
        msgs.append("🌅 Morning Star (volume confirmed): Bullish 3-bar reversal.")

    if recent['Evening_Star']:
        msgs.append("🌇 Evening Star (volume confirmed): Bearish 3-bar reversal.")

    if not msgs:
        msgs.append("No strong candlestick pattern in last bar.")

    return "\n".join(msgs)

    
# === Streamlit app code ===
st.title("📈 Stock Analyzer")

# User input for stock symbols
symbols = st.text_input("Enter stock symbols (comma-separated):", "INFY.NS").split(",")

# Run analysis
if st.button("Run Analysis"):
    clean_symbols = [s.strip() for s in symbols]
    stock_analyzer(clean_symbols)
    display_sentiment_summary(clean_symbols) 
