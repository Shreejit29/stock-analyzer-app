import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands, AverageTrueRange

def get_manual_summary(symbol, clues_4h, signal_4h, clues_1d, signal_1d, clues_1w, signal_1w,
                       final, trade_description, latest_vix, nifty_trend,
                       resistance_gap_pct, support_gap_pct, bull_clues, bear_clues):
    
    lines = []
    lines.append(f"🔍 **{symbol.upper()} Summary**")

    # Core signal
    lines.append(f"📊 **Final Signal:** {final}")
    lines.append(f"🧭 **Suggested Trade:** {trade_description}")
    
    # Clue strength
    lines.append(f"📌 Bullish Clues: {bull_clues}, Bearish Clues: {bear_clues}")

    # Volume risk
    weak_volume_clue = any("Weak volume" in c for c in clues_4h + clues_1d + clues_1w)
    if weak_volume_clue:
        lines.append("⚠️ **Caution:** Weak volume detected — move may not sustain.")

    # Support/Resistance risk
    if 0 <= resistance_gap_pct <= 1.5:
        lines.append("⚠️ **Note:** Price is near resistance — potential rejection risk.")
    if 0 <= support_gap_pct <= 1.5:
        lines.append("⚠️ **Note:** Price is near support — may bounce or break.")

    # VIX and Nifty
    if float(latest_vix) < 12:
        lines.append("⚠️ **Market Risk:** Very low VIX — market complacency risk.")
    if "down" in nifty_trend.lower() and "Bullish" in final:
        lines.append("⚠️ **Caution:** Nifty trend is down — broad market may not support bullish setups.")

    # Summary logic
    if "Ultra Strong Bullish" in final:
        lines.append("✅ **Bias:** Strong upside potential across all timeframes.")
    elif "Moderate Bullish" in final:
        lines.append("🔼 **Bias:** Mild bullish edge, but watch for volume or resistance zones.")
    elif "Moderate Bearish" in final:
        lines.append("🔽 **Bias:** Weakness in price action — avoid long positions.")
    elif "Ultra Strong Bearish" in final:
        lines.append("⛔ **Bias:** Strong downside risk — consider short setups.")
    else:
        lines.append("⚖️ **Bias:** Mixed or unclear — better to wait for clarity.")

    return "\n\n".join(lines)

# Define Supertrend
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
# Compute VWAP
def compute_vwap(df):
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    return df
# Main Program
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
    # Support Resistance
    def calc_support_resistance(close_series, window=20):
        """
        Calculate support and resistance using rolling window min/max.
        """
        if len(close_series) < window:
            return close_series.min(), close_series.max()  # fallback
        support = close_series.rolling(window).min().iloc[-1]
        resistance = close_series.rolling(window).max().iloc[-1]
        return support, resistance
    # Alert of Support Resistance
    def support_resistance_alert(latest_price, support, resistance):
        support_gap_pct = (latest_price - support) / latest_price * 100
        resistance_gap_pct = (resistance - latest_price) / latest_price * 100
    
        alerts = []
    
        if 0 <= support_gap_pct < 2:
            alerts.append(f"⚠️ Price is within {support_gap_pct:.2f}% of support — risk of breakdown if breached.")
    
        if 0 <= resistance_gap_pct < 2:
            alerts.append(f"⚠️ Price is within {resistance_gap_pct:.2f}% of resistance — possible reversal zone.")
    
        if not alerts:
            return "✅ No immediate support/resistance barrier risk."
        else:
            return "\n".join(alerts)  # return just a string
    # Define Candlesticks
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
    # Compute Indicators
    def compute_indicators(df):
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Momentum
        df['RSI'] = RSIIndicator(close).rsi()
        macd_obj = MACD(close)
        df['MACD'] = macd_obj.macd()
        df['MACD_Signal'] = macd_obj.macd_signal()
    
        # Trend indicators
        df['EMA20'] = EMAIndicator(close, 20).ema_indicator()
        df['EMA50'] = EMAIndicator(close, 50).ema_indicator()
        df['EMA200'] = EMAIndicator(close, 200).ema_indicator()
        df['ADX'] = ADXIndicator(high, low, close).adx()
    
        # Volatility
        bb = BollingerBands(close)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['ATR'] = AverageTrueRange(high, low, close).average_true_range()
    
        # Volume
        df['OBV'] = OnBalanceVolumeIndicator(close, df['Volume']).on_balance_volume()
    
        # Custom indicators
        df = compute_supertrend(df)  # Your Supertrend function
        df = compute_vwap(df)        # Your VWAP function
        df = detect_candlestick_patterns(df)  # Already customized
    
        return df
    # Detect Trend Reversal
    def detect_trend_reversal(df):
        rsi = df['RSI']
        obv = df['OBV']
    
        if len(rsi) < 5 or len(obv) < 5:
            return "Not enough data"
    
        recent_rsi = rsi.tail(5)
        recent_obv = obv.tail(5)
    
        rsi_bull_cond = (recent_rsi.iloc[-1] > recent_rsi.iloc[-2]) and (recent_rsi.min() < 35)
        rsi_bear_cond = (recent_rsi.iloc[-1] < recent_rsi.iloc[-2]) and (recent_rsi.max() > 65)
    
        obv_bull_cond = (recent_obv.iloc[-1] > recent_obv.iloc[-2])
        obv_bear_cond = (recent_obv.iloc[-1] < recent_obv.iloc[-2])
    
        if rsi_bull_cond and obv_bull_cond:
            return "📈 Possible Bullish Reversal"
        elif rsi_bear_cond and obv_bear_cond:
            return "📉 Possible Bearish Reversal"
        else:
            return "⚖️ No clear reversal signal"
    # Gap at opning
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
    def detect_trap_signals(df, support, resistance):
        traps = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]
    
        # === Bull Trap: Price broke resistance but failed to hold and fell back
        if (
            prev['Close'] < resistance and
            latest['Close'] > resistance and
            latest['Close'] < resistance * 1.005 and
            latest['Volume'] < df['Volume'].tail(5).mean()
        ):
            traps.append("⚠️ Potential Bull Trap — Weak breakout above resistance with low volume")
    
        # === Bear Trap: Price broke support but reversed back up
        if (
            prev['Close'] > support and
            latest['Close'] < support and
            latest['Close'] > support * 0.995 and
            latest['Volume'] < df['Volume'].tail(5).mean()
        ):
            traps.append("⚠️ Potential Bear Trap — Breakdown below support failed to hold")
    
        # === Fakeout After Candlestick Reversal (e.g. Piercing Line but no follow-through)
        if latest['Piercing_Line'] or latest['Bullish_Engulfing']:
            if latest['Close'] < prev['Close']:
                traps.append("⚠️ Bullish candlestick but no follow-through — possible fakeout")
    
        if latest['Dark_Cloud_Cover'] or latest['Bearish_Engulfing']:
            if latest['Close'] > prev['Close']:
                traps.append("⚠️ Bearish candlestick but price moved up — fake bearish signal")
    
        return traps

    # Data Cleaning 
    def clean_yf_data(df):
        if df.empty:
            return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        if 'Close' not in df.columns:
            return None
        df.dropna(subset=['Close'], inplace=True)
        return df if not df.empty else None
    # === Download VIX and Nifty data ===
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
    
        # Download timeframes: 4H, 1D, and 1W (replacing 1H)
        df_4h = clean_yf_data(yf.download(symbol, period='6mo', interval='4h'))
        df_1d = clean_yf_data(yf.download(symbol, period='6mo', interval='1d'))
        df_1w = clean_yf_data(yf.download(symbol, period='2y', interval='1wk'))
    
        # Check for missing or invalid data
        if df_4h is None or df_1d is None or df_1w is None:
            st.warning(f"⚠️ Insufficient or invalid data for {symbol}. Skipping...")
            continue
    
        # Calculate support/resistance on appropriate timeframes
        support_4h, resistance_4h = calc_support_resistance(df_4h['Close'], window=60)
        support_1d, resistance_1d = calc_support_resistance(df_1d['Close'], window=120)
        support_1w, resistance_1w = calc_support_resistance(df_1w['Close'], window=40)
    
        # Compute indicators
        df_4h = compute_indicators(df_4h)
        df_1d = compute_indicators(df_1d)
        df_1w = compute_indicators(df_1w)

        latest_price = df_1d['Close'].iloc[-1]
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

            atr_mean = df['ATR'].tail(10).mean()
            if latest['ATR'] > 1.2 * atr_mean:
                clues.append('High Volatility')
            elif latest['ATR'] < 0.8 * atr_mean:
                clues.append('Low Volatility')
            else:
                clues.append('Normal Volatility')    
                      
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
            trap_clues = detect_trap_signals(df, support, resistance)
            clues.extend(trap_clues)

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
    
            # Remove StochRSI logic
            if latest['Supertrend_dir'] == 1:
                swing_msg = "🚀 Swing Bullish Signal (Supertrend)"
            elif latest['Supertrend_dir'] == -1:
                swing_msg = "⚠️ Swing Bearish Signal (Supertrend)"
            
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
        clues_1w, signal_1w, support_1w, resistance_1w = analyze_df(df_1w, '1W')
        latest_price = df_1d['Close'].iloc[-1]

        # Extract trap clues
        def extract_traps(clues):
            return [c for c in clues if 'Trap' in c or ('Breakout' in c and '⚠️' in c) or '🚨' in c]
        
        traps_4h = extract_traps(clues_4h)
        traps_1d = extract_traps(clues_1d)
        traps_1w = extract_traps(clues_1w)

        # === Decide trade type based on signal alignment ===
        def suggest_trade_timing(signal_1w, signal_1d, signal_4h):
            if 'Bullish' in signal_1w and 'Bullish' in signal_1d and 'Bullish' in signal_4h:
                return "🧭 Positional Buy Setup (>5 days)", "Positional"
            elif 'Bullish' in signal_1d and 'Bullish' in signal_4h:
                return "🔁 Swing Trade Opportunity (2–5 days)", "Swing"
            elif 'Bullish' in signal_4h:
                return "🕐 Short-Term Upside Bias", "Short-Term"
            elif 'Bearish' in signal_1w and 'Bearish' in signal_1d and 'Bearish' in signal_4h:
                return "🧭 Positional Short Setup (>5 days)", "Positional"
            elif 'Bearish' in signal_1d and 'Bearish' in signal_4h:
                return "🔁 Swing Short Opportunity (2–5 days)", "Swing"
            elif 'Bearish' in signal_4h:
                return "🕐 Short-Term Downside Bias", "Short-Term"
            else:
                return "⚠️ Unclear — Better to Wait", "Neutral"
        
        # Suggest trade based on signals
        trade_description, strategy_type = suggest_trade_timing(signal_1w, signal_1d, signal_4h)
        
        # Select support/resistance based on strategy type
        if strategy_type == "Short-Term":
            sr_support, sr_resistance = support_4h, resistance_4h
        elif strategy_type == "Swing":
            sr_support, sr_resistance = support_1d, resistance_1d
        else:  # Positional or fallback
            sr_support, sr_resistance = support_1w, resistance_1w
        # === Count clues ===
        bull_clues = sum('Bullish' in c or 'Up' in c for c in clues_4h + clues_1d + clues_1w)
        bear_clues = sum('Bearish' in c or 'Down' in c for c in clues_4h + clues_1d + clues_1w)
        total_clues = bull_clues + bear_clues
        
        confidence = (abs(bull_clues - bear_clues) / total_clues) if total_clues else 0
        confidence_percent = round(confidence * 100)
        
        # === Weighted Signal Score ===
        score = 0
        if 'Bullish' in signal_1w: score += 0.6
        if 'Bullish' in signal_1d: score += 0.3
        if 'Bullish' in signal_4h: score += 0.1
        if 'Bearish' in signal_1w: score -= 0.6
        if 'Bearish' in signal_1d: score -= 0.3
        if 'Bearish' in signal_4h: score -= 0.1
        
        bias = 'Bullish' if score > 0 else 'Bearish' if score < 0 else 'Neutral'
        confidence = round(abs(score) * 100)
        
        # === Adjust Confidence Based on Proximity to Support/Resistance ===
        resistance_gap_pct = (sr_resistance - latest_price) / latest_price * 100
        support_gap_pct = (latest_price - sr_support) / latest_price * 100
        
        if 0 <= resistance_gap_pct <= 1.5:
            confidence -= 10  # Price is near resistance — risky for entry
        if 0 <= support_gap_pct <= 1.5:
            confidence += 10  # Price is near support — possible bounce
        # === OBV Trend Confirmation (from 1D) ===
        obv_trend = df_1d['OBV'].iloc[-1] - df_1d['OBV'].iloc[-5]
        
        if obv_trend > 0 and latest_price > sr_resistance:
            confidence += 10  # ✅ Breakout with increasing volume (strong confirmation)
        elif obv_trend < 0 and latest_price >= sr_resistance:
            confidence -= 10  # ⚠️ Price at/above resistance but OBV dropping (divergence)
        
        # Clamp confidence within [0, 100]
        confidence = max(0, min(100, confidence))
        # === Final Signal
        if confidence >= 70:
            final = f"💹 Ultra Strong {bias} (Confidence: {confidence}%)"
        elif confidence >= 40:
            final = f"📈 Moderate {bias} Bias (Confidence: {confidence}%)"
        else:
            final = f"⚖️ Mixed/Neutral (Confidence: {confidence}%)"
        
        st.subheader(f"{symbol} 4H")
        for c in clues_4h:
            st.write(f"🔹 {c}")
        st.write(f"➡ 4H Signal: {signal_4h}")
        
        st.subheader(f"{symbol} 1D")
        for c in clues_1d:
            st.write(f"🔹 {c}")
        st.write(f"➡ 1D Signal: {signal_1d}")
        
        st.subheader(f"{symbol} 1W")
        for c in clues_1w:
            st.write(f"🔹 {c}")
        st.write(f"➡ 1W Signal: {signal_1w}")
        
        # Show final recommendation
        st.markdown(f"## 🧠 Final Analysis: {final}")
        st.markdown(f"### 🧭 Suggested Trade: {trade_description}")

        st.info(f"VIX: {latest_vix:.2f} ({vix_comment}), Nifty Trend: {nifty_trend}")
        st.markdown(f"**🧮 Clue Breakdown**: Bullish clues = {bull_clues}, Bearish clues = {bear_clues}")
        st.progress(confidence)  # Confidence as a visual progress bar
        st.subheader("📢 Final Signal")
        st.success(final)
     
        if latest_vix and latest_vix > 20:
            st.warning(f"⚠️ VIX {latest_vix:.2f} is high — prefer non-directional strategies (Iron Condor etc).")
        if 'Bullish' in final and nifty_trend == 'down':
            st.warning(f"⚠️ {final} but Nifty down — caution advised!")
        elif 'Bearish' in final and nifty_trend == 'up':
            st.warning(f"⚠️ {final} but Nifty up — caution advised!")
        st.subheader("📊 Candlestick Patterns (4H)")
        st.markdown(candlestick_summary(df_4h))
        st.subheader("📊 Candlestick Patterns (1D)")
        st.markdown(candlestick_summary(df_1d))
        st.subheader("📊 Candlestick Patterns (1W)")
        st.markdown(candlestick_summary(df_1w))
       

        vix_for_strategy = latest_vix if latest_vix is not None else 0
        nifty_change_pct = None
        if df_nifty is not None and not df_nifty.empty:
            nifty_change_pct = (df_nifty['Close'].iloc[-1] - df_nifty['Close'].iloc[0]) / df_nifty['Close'].iloc[0] * 100
        warnings_text = generate_market_warnings(latest_vix, nifty_change_pct)     
        st.subheader("⚠️ Market Risk Warnings")
        st.markdown(warnings_text)

        st.markdown("**🟢 For Swing Trade:**")
        st.markdown(support_resistance_alert(latest_price, support_4h, resistance_4h))

        st.markdown("**🔵 For Positional Trade:**")
        st.markdown(support_resistance_alert(latest_price, support_1d, resistance_1d))
                
        st.markdown("**🟠 For Long Trade:**")
        st.markdown(support_resistance_alert(latest_price, support_1w, resistance_1w))
        summary = get_manual_summary(
            symbol=symbol,
            clues_4h=clues_4h, signal_4h=signal_4h,
            clues_1d=clues_1d, signal_1d=signal_1d,
            clues_1w=clues_1w, signal_1w=signal_1w,
            final=final,
            trade_description=trade_description,
            latest_vix=latest_vix,
            nifty_trend=nifty_trend,
            resistance_gap_pct=resistance_gap_pct,
            support_gap_pct=support_gap_pct,
            bull_clues=bull_clues,
            bear_clues=bear_clues
        )

        st.markdown(summary)


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

