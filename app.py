# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time # Per throttling richieste
import math

# --- Configurazione Globale ---
SYMBOLS = ["BTC", "ETH", "BNB", "SOL", "XRP", "RNDR", "FET", "RAY", "SUI", "ONDO", "ARB"]
NUM_COINS = len(SYMBOLS)
VS_CURRENCY = "usd"
CACHE_TTL = 300
DAYS_HISTORY_DAILY = 365
DAYS_HISTORY_HOURLY = 7

SYMBOL_TO_ID_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "RNDR": "render-token",
    "FET": "artificial-superintelligence-alliance",
    "RAY": "raydium", "SUI": "sui", "ONDO": "ondo", "ARB": "arbitrum"
}
ID_TO_SYMBOL_MAP = {v: k for k, v in SYMBOL_TO_ID_MAP.items()}
COINGECKO_IDS_LIST = [SYMBOL_TO_ID_MAP[s] for s in SYMBOLS if s in SYMBOL_TO_ID_MAP]

RSI_PERIOD = 14; MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9
MA_SHORT = 20; MA_LONG = 50; VWAP_PERIOD = 14

# --- Funzioni API CoinGecko ---

@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento dati di mercato (CoinGecko)...")
def get_coingecko_market_data(ids_list, currency):
    ids_string = ",".join(ids_list)
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': currency, 'ids': ids_string, 'order': 'market_cap_desc',
              'per_page': str(len(ids_list)), 'page': 1, 'sparkline': False,
              'price_change_percentage': '1h,24h,7d'}
    timestamp = datetime.now()
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status(); data = response.json()
        df = pd.DataFrame(data);
        if not df.empty: df.set_index('id', inplace=True)
        return df, timestamp
    except Exception as e: st.error(f"Errore API Mercato CoinGecko: {e}"); return pd.DataFrame(), timestamp

@st.cache_data(ttl=CACHE_TTL * 2, show_spinner=False)
def get_coingecko_historical_data(coin_id, currency, days, interval='daily'):
    time.sleep(0.5)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': currency, 'days': str(days),
              'interval': interval if interval == 'hourly' else 'daily', 'precision': 'full'}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status(); data = response.json()
        if not data or 'prices' not in data or not data['prices']: return pd.DataFrame()
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
        prices_df.set_index('timestamp', inplace=True)
        hist_df = prices_df
        if 'total_volumes' in data and data['total_volumes']:
             volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
             volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
             volumes_df.set_index('timestamp', inplace=True)
             hist_df = prices_df.join(volumes_df, how='outer').interpolate(method='time').ffill().bfill()
        else: hist_df['volume'] = 0
        hist_df['high'] = hist_df['close']; hist_df['low'] = hist_df['close']; hist_df['open'] = hist_df['close'].shift(1)
        hist_df = hist_df[~hist_df.index.duplicated(keep='last')]
        hist_df.dropna(subset=['close'], inplace=True)
        return hist_df
    except Exception as e: return pd.DataFrame()

# --- Funzioni Calcolo Indicatori (Manuali con Pandas) ---

def calculate_rsi_manual(series, period=RSI_PERIOD):
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan
    series = series.dropna(); len_series = len(series)
    if len_series < period + 1: return np.nan
    delta = series.diff(); gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    if len(avg_gain.dropna()) < 1 or len(avg_loss.dropna()) < 1 : return np.nan
    last_avg_gain = avg_gain.iloc[-1]; last_avg_loss = avg_loss.iloc[-1]
    if pd.isna(last_avg_gain) or pd.isna(last_avg_loss): return np.nan
    if last_avg_loss == 0: return 100.0 if last_avg_gain > 0 else 50.0
    rs = last_avg_gain / last_avg_loss; rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd_manual(series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan, np.nan, np.nan
    series = series.dropna()
    if len(series) < slow : return np.nan, np.nan, np.nan
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    if len(macd_line.dropna()) < signal: return macd_line.iloc[-1], np.nan, np.nan
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    last_macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else np.nan
    last_signal = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else np.nan
    last_hist = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else np.nan
    return last_macd, last_signal, last_hist

def calculate_sma_manual(series, period):
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan
    series = series.dropna()
    if len(series) < period: return np.nan
    return series.rolling(window=period).mean().iloc[-1]

def calculate_vwap_manual(df, period=VWAP_PERIOD):
    if not all(col in df.columns for col in ['close', 'volume']) or df['close'].isna().all() or df['volume'].isna().all(): return np.nan
    df_period = df.iloc[-period:].copy(); df_period.dropna(subset=['close', 'volume'], inplace=True)
    if df_period.empty or df_period['volume'].sum() == 0: return np.nan
    vwap = (df_period['close'] * df_period['volume']).sum() / df_period['volume'].sum()
    return vwap

# --- Funzione Segnale GPT Migliorata (SINTASSI CORRETTA) ---
def generate_gpt_signal(rsi_1d, rsi_1h, rsi_1w, macd_hist, ma_short, ma_long, current_price):
    """Genera un segnale più articolato (ma ancora basato su regole)."""
    required_inputs = [rsi_1d, macd_hist, ma_short, ma_long, current_price]
    if any(pd.isna(x) for x in required_inputs): return "⚪️ N/D"
    score = 0
    # Logica punteggio (invariata)
    if current_price > ma_long: score += 1
    else: score -= 1
    if ma_short > ma_long: score += 2
    else: score -= 2
    if macd_hist > 0: score += 2
    else: score -= 2
    if rsi_1d < 30: score += 2
    elif rsi_1d < 40: score += 1
    elif rsi_1d > 70: score -= 2
    elif rsi_1d > 60: score -= 1
    if not pd.isna(rsi_1w):
        if rsi_1w < 30: score += 1
        elif rsi_1w > 70: score -= 1
    if not pd.isna(rsi_1h):
        if rsi_1h > 60: score += 1
        elif rsi_1h < 40: score -= 1
    # Mappatura Score (CORRETTA SU PIÙ RIGHE)
    if score >= 5:
        return "⚡️ Strong Buy"
    elif score >= 2:
        return "🟢 Buy"
    elif score <= -5:
        return "🚨 Strong Sell"
    elif score <= -2:
        return "🔴 Sell"
    else:
        return "🟡 Hold"

# --- Funzione per "Gemini Alert" (Sintassi già corretta) ---
def generate_gemini_alert(ma_short, ma_long, macd_hist, rsi_1d):
    """Genera un alert specifico basato su forte confluenza di segnali DAILY."""
    if pd.isna(ma_short) or pd.isna(ma_long) or pd.isna(macd_hist) or pd.isna(rsi_1d): return "⏳"
    is_uptrend = ma_short > ma_long; is_momentum_positive = macd_hist > 0; is_not_overbought = rsi_1d < 70
    if is_uptrend and is_momentum_positive and is_not_overbought: return "✅ BUY Signal"
    is_downtrend = ma_short < ma_long; is_momentum_negative = macd_hist < 0; is_not_oversold = rsi_1d > 30
    if is_downtrend and is_momentum_negative and is_not_oversold: return "❌ SELL Signal"
    return "➖"

# --- Configurazione Pagina Streamlit ---
st.set_page_config(layout="wide", page_title="Crypto Tech Dashboard Pro", page_icon="📈")
st.title("📈 Crypto Technical Dashboard Pro (CoinGecko API)")
last_cg_update_placeholder = st.empty()
st.caption(f"Dati live/storici da CoinGecko. Cache live: {CACHE_TTL}s, Cache storico: {CACHE_TTL*2}s. Periodo Storico Usato: {DAYS_HISTORY_DAILY}d (daily), {DAYS_HISTORY_HOURLY}d (hourly).")

# --- Logica Principale ---
market_data_df, last_cg_update = get_coingecko_market_data(COINGECKO_IDS_LIST, VS_CURRENCY)
last_cg_update_placeholder.markdown(f"*Prezzi Live aggiornati alle: **{last_cg_update.strftime('%Y-%m-%d %H:%M:%S')}***")

if market_data_df.empty: st.error("Errore caricamento dati CoinGecko."); st.stop()

results = []; fetch_errors = []
progress_bar = st.progress(0, text="Analisi Criptovalute...")
coin_ids_ordered = market_data_df.index.tolist()

for i, coin_id in enumerate(coin_ids_ordered):
    live_data = market_data_df.loc[coin_id]
    symbol = live_data.get('symbol', coin_id).upper(); name = live_data.get('name', coin_id)
    rank = live_data.get('market_cap_rank', 'N/A'); current_price = live_data.get('current_price', np.nan)
    change_1h = live_data.get('price_change_percentage_1h_in_currency', np.nan)
    change_24h = live_data.get('price_change_percentage_24h_in_currency', np.nan)
    change_7d = live_data.get('price_change_percentage_7d_in_currency', np.nan)
    volume_24h = live_data.get('total_volume', np.nan); market_cap = live_data.get('market_cap', np.nan)

    hist_daily_df = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_DAILY, interval='daily')
    hist_hourly_df = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_HOURLY, interval='hourly')

    rsi1d, rsi1h, rsi1w = np.nan, np.nan, np.nan
    macd_line, macd_signal, macd_hist = np.nan, np.nan, np.nan
    ma_short, ma_long = np.nan, np.nan; vwap = np.nan

    if not hist_daily_df.empty:
        rsi1d = calculate_rsi_manual(hist_daily_df['close'])
        macd_line, macd_signal, macd_hist = calculate_macd_manual(hist_daily_df['close'])
        ma_short = calculate_sma_manual(hist_daily_df['close'], MA_SHORT)
        ma_long = calculate_sma_manual(hist_daily_df['close'], MA_LONG)
        vwap = calculate_vwap_manual(hist_daily_df)
        try:
            df_weekly = hist_daily_df['close'].resample('W-MON').last()
            if len(df_weekly) >= RSI_PERIOD + 1: rsi1w = calculate_rsi_manual(df_weekly)
            else: fetch_errors.append(f"Dati insuff. ({len(df_weekly)} sett.) per RSI 1w per {symbol}.")
        except Exception as e: fetch_errors.append(f"Errore resampling settimanale per {symbol}: {e}")
        min_len_needed = max(RSI_PERIOD + 1, MACD_SLOW, MA_LONG) # MACD slow è lungo
        if len(hist_daily_df) < min_len_needed: fetch_errors.append(f"Dati Daily insuff. ({len(hist_daily_df)}/{min_len_needed}) per indicatori per {symbol}.")
    else: fetch_errors.append(f"Nessun dato storico Daily da CoinGecko per {symbol}.")

    if not hist_hourly_df.empty:
        if len(hist_hourly_df) >= RSI_PERIOD + 1: rsi1h = calculate_rsi_manual(hist_hourly_df['close'])
        else: fetch_errors.append(f"Dati Hourly insuff. ({len(hist_hourly_df)} ore) per RSI 1h per {symbol}.")
    else: fetch_errors.append(f"Nessun dato storico Hourly da CoinGecko per {symbol}.")

    gpt_signal = generate_gpt_signal(rsi1d, rsi1h, rsi1w, macd_hist, ma_short, ma_long, current_price)
    gemini_alert = generate_gemini_alert(ma_short, ma_long, macd_hist, rsi1d)

    results.append({
        "Rank": rank, "Symbol": symbol, "Name": name, "Gemini Alert": gemini_alert,
        f"Prezzo ({VS_CURRENCY.upper()})": current_price, "% 1h": change_1h, "% 24h": change_24h, "% 7d": change_7d,
        "GPT Signal": gpt_signal, "RSI (1h)": rsi1h, "RSI (1d)": rsi1d, "RSI (1w)": rsi1w,
        "MACD Hist (1d)": macd_hist, f"MA({MA_SHORT}d)": ma_short, f"MA({MA_LONG}d)": ma_long,
        "VWAP (1d)": vwap, f"Volume 24h ({VS_CURRENCY.upper()})": volume_24h, f"Market Cap ({VS_CURRENCY.upper()})": market_cap,
    })
    progress_bar.progress((i + 1) / len(coin_ids_ordered), text=f"Analisi Criptovalute... ({symbol})")

progress_bar.empty()

if fetch_errors:
    with st.expander("ℹ️ Note sul Recupero Dati Storici/Calcolo Indicatori", expanded=False):
        unique_errors = sorted(list(set(fetch_errors)));
        for error_msg in unique_errors: st.info(error_msg)

if results:
    df = pd.DataFrame(results); df.set_index('Rank', inplace=True)
    # GPT Signal già aggiunto sopra

    cols_order = [
        "Symbol", "Name", "Gemini Alert", f"Prezzo ({VS_CURRENCY.upper()})",
        "% 1h", "% 24h", "% 7d", "GPT Signal", "RSI (1h)", "RSI (1d)", "RSI (1w)",
        "MACD Hist (1d)", f"MA({MA_SHORT}d)", f"MA({MA_LONG}d)", "VWAP (1d)",
        f"Volume 24h ({VS_CURRENCY.upper()})", f"Market Cap ({VS_CURRENCY.upper()})"
    ]
    cols_to_show = [col for col in cols_order if col in df.columns]
    df_display = df[cols_to_show].copy()

    # Formattazione manuale
    formatters = {}; currency_col = f"Prezzo ({VS_CURRENCY.upper()})"; market_cap_col = f"Market Cap ({VS_CURRENCY.upper()})"
    volume_col = f"Volume 24h ({VS_CURRENCY.upper()})"; pct_cols = ["% 1h", "% 24h", "% 7d"]
    for col in df_display.columns:
        if df_display[col].dtype == 'float64' or df_display[col].dtype == 'int64':
            if col == currency_col: formatters[col] = lambda x: f"${x:,.4f}" if pd.notna(x) else "N/A"
            elif col in pct_cols: formatters[col] = lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
            elif col == market_cap_col: formatters[col] = lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
            elif col == volume_col: formatters[col] = lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
            elif "RSI" in col: formatters[col] = lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
            elif "MACD" in col: formatters[col] = lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            elif "MA" in col or "VWAP" in col: formatters[col] = lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
            else: formatters[col] = lambda x: str(x) if pd.notna(x) else "N/A"
            df_display[col] = df_display[col].apply(formatters[col])
        elif df_display[col].dtype == 'object': df_display[col].fillna("N/A", inplace=True)
    df_display.fillna("N/A", inplace=True) # Catchall finale per sicurezza

    def highlight_pct_col(col):
        colors = [''] * len(col)
        for i, val in enumerate(col):
            if isinstance(val, str) and val.endswith('%'):
                try: num = float(val.replace('%','')); colors[i] = 'color: green' if num > 0 else 'color: red' if num < 0 else ''
                except ValueError: pass
        return colors
    def highlight_gemini_alert(col):
        colors = [''] * len(col)
        for i, val in enumerate(col):
            if val == "✅ BUY Signal": colors[i] = 'background-color: #28a745; color: white;'
            elif val == "❌ SELL Signal": colors[i] = 'background-color: #dc3545; color: white;'
        return colors

    styled_df = df_display.style
    for col_name in pct_cols:
        if col_name in df_display.columns: styled_df = styled_df.apply(highlight_pct_col, subset=[col_name], axis=0)
    if "Gemini Alert" in df_display.columns: styled_df = styled_df.apply(highlight_gemini_alert, subset=["Gemini Alert"], axis=0)

    st.dataframe(styled_df, use_container_width=True)
else: st.warning("Nessun risultato da visualizzare.")

# --- Legenda Aggiornata ---
st.divider()
with st.expander("📘 Legenda Indicatori Tecnici e Segnali", expanded=True):
    # (Contenuto legenda invariato rispetto alla versione precedente - già aggiornato)
    st.markdown("""
    Questa sezione spiega gli indicatori e i segnali visualizzati nella tabella. Ricorda che l'analisi tecnica è uno strumento e non una garanzia di risultati futuri. **Questa dashboard è solo a scopo informativo e non costituisce consulenza finanziaria.**

    **Indicatori di Momentum:**
    * **RSI (Relative Strength Index - 1h, 1d, 1w):**
        * *Cos'è:* Oscillatore (0-100) che misura la velocità e l'entità delle recenti variazioni di prezzo per valutare condizioni di ipercomprato o ipervenduto e la forza del momentum.
        * *Interpretazione:* `> 70`=Ipercomprato (possibile pullback), `< 30`=Ipervenduto (possibile rimbalzo). Le divergenze prezzo/RSI possono segnalare inversioni.
        * *Timeframe:* **1h** (brevissimo termine), **1d** (breve-medio), **1w** (medio-lungo). Confrontarli dà un quadro completo. (Nota: Dati 1h/1w dipendono dalla disponibilità API).
    * **MACD Hist (Moving Average Convergence Divergence Histogram - 1d):**
        * *Cos'è:* Differenza tra linea MACD (EMA 12 - EMA 26) e Signal Line (EMA 9 del MACD). Misura il momentum del trend.
        * *Interpretazione:* Istogramma `> 0` = Momentum rialzista; Istogramma `< 0` = Momentum ribassista. Crossover dello zero e divergenze sono segnali chiave.

    **Indicatori di Trend:**
    * **MA (Simple Moving Average - 20d, 50d):**
        * *Cos'è:* Media mobile semplice dei prezzi di chiusura. Identifica la direzione del trend e potenziali livelli di supporto/resistenza dinamici.
        * *Interpretazione:* Prezzo sopra MA = trend rialzista; Prezzo sotto = ribassista. Crossover MA20d/MA50d ("Golden/Death Cross") indicano possibili cambi di trend a medio termine.
    * **VWAP (Volume Weighted Average Price - 1d):**
        * *Cos'è:* Prezzo medio ponderato per il volume. Indica il prezzo "medio" scambiato rispetto ai volumi.
        * *Interpretazione:* Prezzo sopra VWAP = forza acquirente; Prezzo sotto = pressione venditrice. Usato come riferimento, specialmente intraday. *(Nota: Calcolo approssimato su dati giornalieri qui).*

    **Segnali Combinati (Esemplificativi - NON CONSULENZA FINANZIARIA):**
    * **GPT Signal:**
        * *Cos'è:* Sintesi direzionale generale basata su una **combinazione pesata** di più indicatori (MAs, MACD, RSI 1d/1w/1h).
        * *Scopo:* Offrire una visione d'insieme rapida del sentiment tecnico combinato. **Va interpretato con cautela.**
    * **Gemini Alert:**
        * *Cos'è:* Alert più **specifico** che si attiva solo con una **forte confluenza** di segnali DAILY.
        * *Logica:* `✅ BUY` se MA20>MA50 *E* MACD Hist>0 *E* RSI<70. `❌ SELL` se MA20<MA50 *E* MACD Hist<0 *E* RSI>30. `➖` Altrimenti. `⏳` Dati insuff.
        * *Scopo:* Evidenziare potenziali punti di ingresso/uscita basati su una specifica strategia di confluenza. **Massima cautela, non è una garanzia.**

    **Generale:**
    * **N/A:** Dato/Indicatore non disponibile o non calcolabile (es. dati storici insuff.).
    * **%, 1h, 24h, 7d:** Variazione percentuale del prezzo rispetto a 1 ora, 24 ore, 7 giorni fa.
    """)

# --- Footer con Bottone Aggiorna e Disclaimer ---
st.divider()
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔄 Aggiorna", help="Forza l'aggiornamento dei dati"):
        st.cache_data.clear(); st.rerun() # Messo su una riga per brevità qui
with col2:
    st.caption(f"Ultimo aggiornamento live: {last_cg_update.strftime('%H:%M:%S')} | Disclaimer: Non è consulenza finanziaria.")