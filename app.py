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
    def suggest_option_strategy(final_signal, latest_price, vix_level):
        """
        Suggest a suitable option strategy.
        """
        strike_step = 10  # You could adjust this based on the stock
        atm = round(latest_price / strike_step) * strike_step
        spread_width = 20
        vix_display = f"{vix_level:.2f}" if vix_level is not None else "N/A"
        
        suggestion = ""
        
        if 'Ultra Strong Bullish' in final_signal or 'Strong Bullish' in final_signal:
            suggestion = (
                f"💡 **Bull Call Spread**\n"
                f"👉 Buy {atm} CE, Sell {atm + spread_width} CE\n"
                f"👉 Reason: Strong bullish signal with technical alignment\n"
                f"👉 Target: Swing move 2-5 days\n"
                f"👉 Risk: Limited risk, defined reward\n"
            )
        elif 'Ultra Strong Bearish' in final_signal or 'Strong Bearish' in final_signal:
            suggestion = (
                f"💡 **Bear Put Spread**\n"
                f"👉 Buy {atm} PE, Sell {atm - spread_width} PE\n"
                f"👉 Reason: Strong bearish signal detected\n"
                f"👉 Target: Swing move 2-5 days\n"
                f"👉 Risk: Limited risk, cost-effective bearish play\n"
            )
        elif 'Mixed' in final_signal or 'Neutral' in final_signal:
            if vix_level is not None and vix_level >= 20:
                suggestion = (
                    f"💡 **Iron Condor / Short Strangle**\n"
                    f"👉 Sell OTM Call & Put, e.g., Sell {atm + 50} CE, Sell {atm - 50} PE\n"
                    f"👉 Reason: Market indecisive + high volatility (VIX {vix_display})\n"
                    f"👉 Benefit: Profit from time decay + volatility crush\n"
                )
            else:
                suggestion = (
                    f"💡 **Wait / Small Range Bound Trade**\n"
                    f"👉 Reason: Neutral signal + low/moderate volatility (VIX {vix_display})\n"
                    f"👉 Action: Wait for clearer setup\n"
                )
        else:
            suggestion = (
                f"💡 **No clear edge**\n"
                f"👉 Reason: Signal unclear or conflicting signals\n"
                f"👉 Action: Observe or look for better setups\n"
            )
        
        return suggestion
    def detect_candlestick_patterns_with_volume(df):
        """
        Detect common candlestick patterns with volume confirmation.
        """
        patterns = {}
        avg_vol = df['Volume'].rolling(5).mean()
    
        # --- Bullish Engulfing ---
        cond = (
            (df['Close'] > df['Open']) &
            (df['Open'].shift(1) > df['Close'].shift(1)) &
            (df['Close'] > df['Open'].shift(1)) &
            (df['Open'] < df['Close'].shift(1)) &
            (df['Volume'] > avg_vol)
        )
        if cond.iloc[-1]:
            patterns['Bullish Engulfing'] = "✅ Strong bullish reversal with high volume."
    
        # --- Bearish Engulfing ---
        cond = (
            (df['Close'] < df['Open']) &
            (df['Open'].shift(1) < df['Close'].shift(1)) &
            (df['Close'] < df['Open'].shift(1)) &
            (df['Open'] > df['Close'].shift(1)) &
            (df['Volume'] > avg_vol)
        )
        if cond.iloc[-1]:
            patterns['Bearish Engulfing'] = "⚠️ Strong bearish reversal with high volume."
    
        # --- Hammer ---
        cond = (
            ((df['High'] - df['Low']) > 3 * (df['Open'] - df['Close']).abs()) &
            ((df['Close'] - df['Low']) / (0.001 + (df['High'] - df['Low'])) > 0.6) &
            (df['Volume'] > avg_vol)
        )
        if cond.iloc[-1]:
            patterns['Hammer'] = "✅ Bullish hammer — potential bottom reversal with volume support."
    
        # --- Inverted Hammer ---
        cond = (
            ((df['High'] - df['Low']) > 3 * (df['Open'] - df['Close']).abs()) &
            ((df['High'] - df['Close']) / (0.001 + (df['High'] - df['Low'])) > 0.6) &
            (df['Volume'] > avg_vol)
        )
        if cond.iloc[-1]:
            patterns['Inverted Hammer'] = "⚠️ Inverted hammer — potential reversal at low."
    
        # --- Shooting Star ---
        cond = (
            ((df['High'] - df['Low']) > 3 * (df['Open'] - df['Close']).abs()) &
            ((df['High'] - df['Close']) / (0.001 + (df['High'] - df['Low'])) > 0.6) &
            (df['Open'] > df['Close']) &
            (df['Volume'] > avg_vol)
        )
        if cond.iloc[-1]:
            patterns['Shooting Star'] = "⚠️ Shooting star — bearish potential at top."
    
        # --- Doji ---
        cond = (
            ((df['Close'] - df['Open']).abs() < 0.1 * (df['High'] - df['Low'])) &
            (df['Volume'] > avg_vol)
        )
        if cond.iloc[-1]:
            patterns['Doji'] = "⚖️ Doji with high volume — market indecision at key zone."
    
        # --- Morning Star ---
        cond = (
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Close'] > df['Open']) &
            (df['Close'] > (df['Open'].shift(2) + df['Close'].shift(2)) / 2) &
            (df['Volume'] > avg_vol)
        )
        if cond.iloc[-1]:
            patterns['Morning Star'] = "✅ Morning star — bullish reversal with volume confirmation."
    
        # --- Evening Star ---
        cond = (
            (df['Close'].shift(2) > df['Open'].shift(2)) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Close'] < df['Open']) &
            (df['Close'] < (df['Open'].shift(2) + df['Close'].shift(2)) / 2) &
            (df['Volume'] > avg_vol)
        )
        if cond.iloc[-1]:
            patterns['Evening Star'] = "⚠️ Evening star — bearish reversal with volume confirmation."
    
        # --- Piercing Line ---
        cond = (
            (df['Open'] < df['Close']) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Open'] < df['Close'].shift(1)) &
            (df['Close'] > (df['Open'].shift(1) + df['Close'].shift(1)) / 2) &
            (df['Volume'] > avg_vol)
        )
        if cond.iloc[-1]:
            patterns['Piercing Line'] = "✅ Piercing line — bullish reversal with volume support."
    
        # --- Dark Cloud Cover ---
        cond = (
            (df['Open'] > df['Close']) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Open'] > df['Close'].shift(1)) &
            (df['Close'] < (df['Open'].shift(1) + df['Close'].shift(1)) / 2) &
            (df['Volume'] > avg_vol)
        )
        if cond.iloc[-1]:
            patterns['Dark Cloud Cover'] = "⚠️ Dark cloud cover — bearish reversal with volume confirmation."
    
        return patterns

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
        df = detect_candlestick_patterns_with_volume(df)
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
            patterns = detect_candlestick_patterns_with_volume(df)
            if patterns:
                st.subheader("🕯️ Candlestick Patterns Detected")
                for pname, pdesc in patterns.items():
                    st.markdown(f"🔹 **{pname}:** {pdesc}")
            else:
                st.info("No major candlestick pattern detected on the latest candle.")

            clues.append(swing_msg)
            clues.append(positional_msg)

            return clues, signal, support, resistance
        clues_4h, signal_4h, support_4h, resistance_4h = analyze_df(df_4h, '4H')
        clues_1d, signal_1d, support_1d, resistance_1d = analyze_df(df_1d, '1D')
        clues_1h, signal_1h, support_1h, resistance_1h = analyze_df(df_1h, '1H')
        


        # === Compute weighted final signal ===
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
        
        if score >= 0.7:
            final = '💹 Ultra Strong Bullish (weighted)'
        elif score >= 0.4:
            final = '📈 Bullish bias (weighted)'
        elif score <= -0.7:
            final = '🔻 Ultra Strong Bearish (weighted)'
        elif score <= -0.4:
            final = '📉 Bearish bias (weighted)'
        else:
            final = '⚖️ Mixed / Neutral (weighted)'

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
        st.success(f"Final Combined Signal: {final}")
        if latest_vix and latest_vix > 20:
            st.warning(f"⚠️ VIX {latest_vix:.2f} is high — prefer non-directional strategies (Iron Condor etc).")
            
        if 'Bullish' in final and nifty_trend == 'down':
            st.warning(f"⚠️ {final} but Nifty down — caution advised!")
        elif 'Bearish' in final and nifty_trend == 'up':
            st.warning(f"⚠️ {final} but Nifty up — caution advised!")
        latest_price = df_1d['Close'].iloc[-1]
        vix_for_strategy = latest_vix if latest_vix is not None else 0
        nifty_change_pct = None
        if df_nifty is not None and not df_nifty.empty:
            nifty_change_pct = (df_nifty['Close'].iloc[-1] - df_nifty['Close'].iloc[0]) / df_nifty['Close'].iloc[0] * 100
        
        warnings_text = generate_market_warnings(latest_vix, nifty_change_pct)
        
        st.subheader("⚠️ Market Risk Warnings")
        st.markdown(warnings_text)
        strategy_suggestion = suggest_option_strategy(final, latest_price, vix_for_strategy)
        st.subheader("💡 Option Strategy Suggestion")
        st.markdown(strategy_suggestion)
        st.subheader("📏 Support/Resistance Alert")
        sr_alert = support_resistance_alert(latest_price, support_1d, resistance_1d)
        st.markdown(sr_alert)
        

def fetch_market_news_for_query(query, max_items=5):
    """
    Fetches Google News RSS headlines based on the search query.
    """
    rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '%20')}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    articles = []
    for entry in feed.entries[:max_items]:
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published if "published" in entry else "N/A"
        })
    return articles

def display_market_news(symbols):
    st.markdown("### 📰 Market News")

    # General market news
    st.markdown("#### 🏦 Nifty / Sensex / India Market")
    general_news = fetch_market_news_for_query("Nifty OR Sensex OR India stock market")
    if general_news:
        for art in general_news:
            st.markdown(f"- **[{art['title']}]({art['link']})**  \n_Published: {art['published']}_")
    else:
        st.info("No major headlines found for Nifty / Sensex.")

    # Per stock news
    for symbol in symbols:
        st.markdown(f"#### 📌 News for {symbol}")
        stock_news = fetch_market_news_for_query(symbol)
       
        if stock_news:
            for art in stock_news:
                st.markdown(f"- **[{art['title']}]({art['link']})**  \n_Published: {art['published']}_")
        else:
            st.info(f"No recent headlines found for {symbol}.")

# === Streamlit app code ===
st.title("📈 Stock Analyzer")

# User input for stock symbols
symbols = st.text_input("Enter stock symbols (comma-separated):", "INFY.NS").split(",")

# Run analysis
if st.button("Run Analysis"):
    clean_symbols = [s.strip() for s in symbols]
    stock_analyzer(clean_symbols)
    display_market_news(clean_symbols)
   
