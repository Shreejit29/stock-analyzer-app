import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from io import BytesIO
import openpyxl  # For reading Excel files (automatically used by Pandas)
import xlsxwriter  # For writing Excel files with formatting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def detect_market_phase(df):
    """
    Detect market phase based on EMA alignment and volume.
    Returns: (phase_name, confidence_text)
    """
    if len(df) < 50:
        return "Unknown", "📉 Not enough data to determine phase."

    ema20 = df['EMA20'].iloc[-1]
    ema50 = df['EMA50'].iloc[-1]
    ema200 = df['EMA200'].iloc[-1]
    close = df['Close'].iloc[-1]

    avg_vol = df['Volume'].tail(20).mean()
    recent_vol = df['Volume'].iloc[-1]

    if close < ema20 < ema50 and recent_vol < avg_vol:
        return "Accumulation", "📦 Sideways buildup — breakout likely ahead."
    elif close > ema20 > ema50 and ema50 > ema200:
        return "Markup", "📈 Trending Up — ride the move."
    elif close > ema20 and ema20 < ema50 and recent_vol > avg_vol:
        return "Distribution", "🎭 Choppy zone — trend may be ending."
    elif close < ema20 and ema20 < ema50 and ema50 < ema200:
        return "Markdown", "📉 Downtrend — look for short setups."
    else:
        return "Transition", "🔁 Mixed signals — avoid aggressive entries."
def market_phase_message(trade_type, signal, phase):
    if trade_type == "Swing":
        if "Bullish" in signal and phase == "Accumulation":
            return "💡 Breakout brewing — watch for volume spike"
        elif "Bullish" in signal and phase == "Markup":
            return "✅ Swing entry valid — ride short trend leg"
        elif "Bearish" in signal and phase == "Distribution":
            return "⚠️ Avoid — traps likely in choppy zone"
        elif "Bearish" in signal and phase == "Markdown":
            return "🚨 Quick breakdown — aggressive swing short possible"

    elif trade_type == "Positional":
        if "Bullish" in signal and phase == "Accumulation":
            return "🟢 Early entry — need breakout confirmation"
        elif "Bullish" in signal and phase == "Markup":
            return "✅ Strong trend — positional hold justified"
        elif "Bearish" in signal and phase == "Distribution":
            return "⚠️ Exit/reduce — trend exhaustion likely"
        elif "Bearish" in signal and phase == "Markdown":
            return "🔻 Stay short — trend continuation expected"

    elif trade_type == "Short-Term":
        if "Neutral" in signal and phase == "Accumulation":
            return "⏸️ Wait — breakout not confirmed yet"
        elif "Bullish" in signal and phase == "Markup":
            return "⚡ Fast upside scalp possible"
        elif "Neutral" in signal and phase == "Distribution":
            return "🎭 Scalp carefully — fakeouts likely"
        elif "Bearish" in signal and phase == "Markdown":
            return "📉 Short scalp may work — protect gains fast"

    return "🔍 Mixed scenario — better to wait"


def generate_summary(symbol, latest_price, signal_4h, signal_1d, signal_1w,
                     clues_4h, clues_1d, clues_1w,
                     final, trade_description, latest_vix, nifty_trend,
                     sr_support, sr_resistance, traps_4h, traps_1d, traps_1w,
                     additional_signals=None, market_phase=None, trade_response=None):

    # Count clues
    bull_clues = sum('Bullish' in c or 'Up' in c for c in clues_4h + clues_1d + clues_1w)
    bear_clues = sum('Bearish' in c or 'Down' in c for c in clues_4h + clues_1d + clues_1w)

    # Trap detection
    all_traps = traps_4h + traps_1d + traps_1w
    trap_note = "🚫 No trap signals detected." if not all_traps else f"🚨 Trap Alert: {len(all_traps)} suspicious breakout signal(s)."

    # Support/resistance proximity
    support_gap = round((latest_price - sr_support) / latest_price * 100, 2)
    resistance_gap = round((sr_resistance - latest_price) / latest_price * 100, 2)

    support_note = f"⚠️ Price is near support ({support_gap}%) — watch for breakdown." if 0 <= support_gap <= 1.5 else ""
    resistance_note = f"⚠️ Price is near resistance ({resistance_gap}%) — possible reversal zone." if 0 <= resistance_gap <= 1.5 else ""

    # Risk notes
    vix_note = "⚠️ VIX <12 — market may be complacent." if latest_vix < 12 else ""
    nifty_note = "⚠️ Nifty is trending down — caution on longs." if "down" in nifty_trend.lower() else ""

    # Action suggestion
    if "Bullish" in final:
        action_note = f"✅ Look for breakout above {sr_resistance} with volume."
    elif "Bearish" in final:
        action_note = f"🔻 Watch for breakdown below {sr_support} with volume."
    else:
        action_note = "⏸️ Wait — no strong directional confirmation."
    
    # Format message
    summary = f"""
📊 *{symbol.upper()} - Trade Summary*
📉 *Latest Price*: ₹{latest_price:.2f}
📌 *Bias*: {final}
🎯 *Suggested Trade*: {trade_description}

🧭 *Signal Map*:
• 4H ➤ {signal_4h}
• 1D ➤ {signal_1d}
• 1W ➤ {signal_1w}

📚 *Clue Count*: 🟢 {bull_clues} Bullish | 🔴 {bear_clues} Bearish

{trap_note}
{support_note}
{resistance_note}
{vix_note}
{nifty_note}


📈 *Action Plan*:
{action_note}
""".strip()
    # 🧭 Market Phase
    if market_phase or trade_response:
        summary += "\n\n🧭 *Market Phase Analysis:*"
        if market_phase:
            summary += f"\n• Market Phase: {market_phase}"
        if trade_response:
            summary += f"\n• Phase Response: {trade_response}"                     


    if additional_signals:
      summary += "\n\n📌 *Additional Insights*:\n"
      for line in additional_signals:
        summary += f"{line}\n"
    return summary
def generate_additional_signals(clues_4h, clues_1d, clues_1w, latest_price, sr_support, sr_resistance, confidence_percent):
    lines = []

    # 1. Breakout/Breakdown Strength
    resistance_gap_pct = (sr_resistance - latest_price) / latest_price * 100
    support_gap_pct = (latest_price - sr_support) / latest_price * 100

    if latest_price > sr_resistance and resistance_gap_pct > 1.5 and 'OBV Up' in clues_4h + clues_1d:
        lines.append("💥 Strong breakout above resistance (volume confirmed).")
    elif latest_price < sr_support and support_gap_pct > 1.5 and 'OBV Down' in clues_4h + clues_1d:
        lines.append("💥 Strong breakdown below support (volume confirmed).")

    # 2. EMA Compression Zone
    if 'EMA Bullish alignment' in clues_1d and 'EMA Bearish alignment' in clues_4h:
        lines.append("📊 Compression zone — conflicting EMA signals across timeframes.")
    elif 'EMA Bearish alignment' in clues_1d and 'EMA Bullish alignment' in clues_4h:
        lines.append("📊 Possible squeeze — EMA conflict may cause breakout soon.")

    # 3. Trend Confidence Boost
    if 'Strong Trend' in clues_1d and 'OBV Up' in clues_1d and 'MACD Bullish' in clues_1d:
        lines.append("🔒 Strong trend backed by volume and momentum.")

    # 4. Trap Recovery
    has_trap = any('Trap' in clue for clue in clues_4h + clues_1d + clues_1w)
    if has_trap and 'MACD Bullish' in clues_1d and 'OBV Up' in clues_1d:
        lines.append("🧲 Trap likely absorbed — recovery in progress.")

    # 5. Trade Plan
    if confidence_percent >= 70 and resistance_gap_pct > 2:
        lines.append("🛒 Entry: Watch pullback above support.")
        lines.append("🎯 Target: Next resistance / swing high.")
        lines.append("🛡️ Risk Level: Moderate")
    elif confidence_percent >= 70 and support_gap_pct < 1:
        lines.append("🛑 Entry risky — too close to support. Avoid until breakout.")
        lines.append("🛡️ Risk Level: High")
    elif confidence_percent < 40:
        lines.append("⚠️ Signal weak — wait for confirmation before trading.")

    return lines

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
def stock_analyzer(symbols, summary_only=False):
    summary_table = []
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
    def detect_obv_divergence(df):
      price_change = df['Close'].iloc[-1] - df['Close'].iloc[-5]
      obv_change = df['OBV'].iloc[-1] - df['OBV'].iloc[-5]
  
      if price_change > 0 and obv_change < 0:
          return "⚠️ Bearish OBV Divergence — Price rising but volume not supporting"
      elif price_change < 0 and obv_change > 0:
          return "⚠️ Bullish OBV Divergence — Price falling but OBV rising"
      return None
    def detect_volume_cluster(df):
        recent_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].iloc[-10:-1].mean()
        price_range = df['High'].iloc[-1] - df['Low'].iloc[-1]
        if recent_vol > 1.5 * avg_vol and price_range < df['Close'].iloc[-1] * 0.01:
            return "🔍 Volume Cluster Detected — High activity in tight range (possible smart money)"
        return None

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
            obv_div = detect_obv_divergence(df)
            if obv_div:
                clues.append(obv_div)
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
            vol_cluster = detect_volume_cluster(df)
            if vol_cluster:
                clues.append(vol_cluster)

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
        # Detect market phase for all timeframes
        phase_4h, _ = detect_market_phase(df_4h)
        phase_1d, _ = detect_market_phase(df_1d)
        phase_1w, _ = detect_market_phase(df_1w)
        
        # Choose the best phase based on strategy
        if strategy_type == "Short-Term":
            selected_phase = phase_4h
        elif strategy_type == "Swing":
            selected_phase = phase_1d
        else:
            selected_phase = phase_1w
        
        # Get response to show in summary
        trade_response = market_phase_message(strategy_type, final, selected_phase)


        if summary_only:
          additional_signals = generate_additional_signals(clues_4h, clues_1d, clues_1w, latest_price, sr_support, sr_resistance, confidence_percent)
          summary = generate_summary(
              symbol, latest_price, signal_4h, signal_1d, signal_1w,
              clues_4h, clues_1d, clues_1w, final, trade_description,
              latest_vix, nifty_trend, sr_support, sr_resistance,
              traps_4h, traps_1d, traps_1w,
              additional_signals=additional_signals, market_phase=selected_phase, trade_response=trade_response
            )
          st.markdown(summary, unsafe_allow_html=False)
          
           
          # Count clues
          bull_clues = sum('Bullish' in c or 'Up' in c for c in clues_4h + clues_1d + clues_1w)
          bear_clues = sum('Bearish' in c or 'Down' in c for c in clues_4h + clues_1d + clues_1w)
          # ✅ Append only — no rendering here
          if "Bullish" in final:
              action_note = f"✅ Look for breakout above {sr_resistance} with volume."
          elif "Bearish" in final:
              action_note = f"🔻 Watch for breakdown below {sr_support} with volume."
          else:
              action_note = "⏸️ Wait — no strong directional confirmation."
          
          summary_table.append({
              "Symbol": symbol.upper(),
              "Price": latest_price,
              "Bull Clue" : bull_clues,
              "Bear Clue" : bear_clues,
              "Trade Type": trade_description,
              "Trap Signals": " | ".join(traps_4h + traps_1d + traps_1w) if (traps_4h or traps_1d or traps_1w) else "None",
              "Smart Signal": " | ".join(additional_signals) if additional_signals else "None",
              "Final Signal": final,
              "Action Plan": action_note,
              "Market Phase": selected_phase,                   
              "Phase Response": trade_response 
              })
          if summary_table:
            st.markdown("### 📋 Final Summary Table (All Stocks)")
            st.table(summary_table)  
          # 🔽 Save table to Excel (in memory)
          df_summary = pd.DataFrame(summary_table)
          excel_buffer = BytesIO()
          with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
              df_summary.to_excel(writer, index=False, sheet_name="Summary")
          excel_buffer.seek(0)

          # 🔽 Provide download button
          st.download_button(
              label="📥 Download Summary as Excel",
              data=excel_buffer,
              file_name="stock_summary.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
              key=f"summary_download_{len(summary_table)}"  # ✅ Unique key
          )

            
        else:
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
  
          # st.info(f"VIX: {latest_vix:.2f} ({vix_comment}), Nifty Trend: {nifty_trend}")
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
                  
          st.markdown("**🟠 For Short term Trade:**")
          st.markdown(support_resistance_alert(latest_price, support_1w, resistance_1w))
          additional_signals = generate_additional_signals(
              clues_4h, clues_1d, clues_1w,
              latest_price, sr_support, sr_resistance,
              confidence_percent
          )
          
          if additional_signals:
              st.subheader("🔍 Extra Smart Signal Clues")
              for line in additional_signals:
                  st.write(line)

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


# Choose mode
col1, col2 = st.columns(2)
run_full = col1.button("🔍 Full Analysis")
run_summary = col2.button("📤 WhatsApp Summary Only")

clean_symbols = [s.strip() for s in symbols]

if run_full:
    stock_analyzer(clean_symbols, summary_only=False)

if run_summary:
    stock_analyzer(clean_symbols, summary_only=True)


