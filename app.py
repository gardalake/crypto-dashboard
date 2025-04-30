# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta # Importa timedelta
import time
import math

# --- Password Protection ---
# NOTA: Questo è un metodo MOLTO basilare e NON sicuro per produzione.
# La password è visibile nel codice sorgente.
# Per sicurezza reale, usare metodi di autenticazione più robusti.
def check_password():
    """Restituisce True se la password è corretta, False altrimenti."""
    # Usa st.session_state per ricordare se la password è già stata inserita correttamente
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        password = st.text_input("🔑 Password", type="password")
        if password == "Leonardo": # La tua password
            st.session_state.password_correct = True
            st.rerun() # Ricarica la pagina per mostrare il contenuto
        elif password: # Se l'utente inserisce qualcosa ma è sbagliato
             st.warning("Password errata.")
             st.stop() # Ferma l'esecuzione
        else: # Se il campo è vuoto
             st.stop() # Ferma l'esecuzione finché non viene inserita la password
    return True

# --- Esegui controllo password all'inizio ---
if not check_password():
    st.stop() # Non dovrebbe mai arrivare qui se check_password usa st.stop(), ma per sicurezza

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

# --- Funzioni API CoinGecko (Invariate) ---
@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento dati di mercato (CoinGecko)...")
def get_coingecko_market_data(ids_list, currency):
    ids_string = ",".join(ids_list)
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': currency, 'ids': ids_string, 'order': 'market_cap_desc',
              'per_page': str(len(ids_list)), 'page': 1, 'sparkline': False,
              'price_change_percentage': '1h,24h,7d'} # CoinGecko NON fornisce 15m qui
    timestamp = datetime.now()
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status(); data = response.json()
        df = pd.DataFrame(data);
        if not df.empty: df.set_index('id', inplace=True)
        return df, timestamp
    except requests.exceptions.RequestException as req_ex:
        status_code = req_ex.response.status_code if req_ex.response is not None else "N/A"
        st.error(f"Errore API Mercato CoinGecko (Status: {status_code}): {req_ex}")
        return pd.DataFrame(), timestamp
    except Exception as e:
        st.error(f"Errore Processamento Dati Mercato CoinGecko: {e}")
        return pd.DataFrame(), timestamp

@st.cache_data(ttl=CACHE_TTL * 2, show_spinner=False)
def get_coingecko_historical_data(coin_id, currency, days, interval='daily'):
    time.sleep(1.5) # Manteniamo delay alto per rate limit
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': currency, 'days': str(days),
              'interval': interval if interval == 'hourly' else 'daily', 'precision': 'full'}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status(); data = response.json()
        if not data or 'prices' not in data or not data['prices']: return pd.DataFrame(), "No Prices Data"
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
        if hist_df.empty: return pd.DataFrame(), "Processed Empty"
        return hist_df, "Success"
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 429: return pd.DataFrame(), "Rate Limited (429)"
        else: return pd.DataFrame(), f"HTTP Error {http_err.response.status_code}"
    except Exception as e: return pd.DataFrame(), f"Generic Error: {type(e).__name__}"

# --- Funzioni Calcolo Indicatori (Manuali - Invariate) ---
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
    series = series.dropna(); len_series = len(series)
    if len_series < slow : return np.nan, np.nan, np.nan
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

# --- Funzioni Segnale (Invariate) ---
def generate_gpt_signal(rsi_1d, rsi_1h, rsi_1w, macd_hist, ma_short, ma_long, current_price):
    required_inputs = [rsi_1d, macd_hist, ma_short, ma_long, current_price]
    if any(pd.isna(x) for x in required_inputs): return "⚪️ N/D"
    score = 0
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
    if score >= 5: return "⚡️ Strong Buy"
    elif score >= 2: return "🟢 Buy"
    elif score <= -5: return "🚨 Strong Sell"
    elif score <= -2: return "🔴 Sell"
    else: return "🟡 Hold"

def generate_gemini_alert(ma_short, ma_long, macd_hist, rsi_1d):
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
# Spostata caption più specifica qui sotto
# st.caption(f"Dati live/storici da CoinGecko. Cache live: {CACHE_TTL}s, Cache storico: {CACHE_TTL*2}s...")

# --- Logica Principale ---
market_data_df, last_cg_update_utc = get_coingecko_market_data(COINGECKO_IDS_LIST, VS_CURRENCY)

# --- AGGIUSTAMENTO TIMESTAMP PER ROMA (UTC+2) ---
last_cg_update_rome = last_cg_update_utc + timedelta(hours=2)
last_cg_update_placeholder.markdown(f"*Prezzi Live aggiornati alle: **{last_cg_update_rome.strftime('%Y-%m-%d %H:%M:%S')} (Ora di Roma)***")

if market_data_df.empty: st.error("Errore caricamento dati CoinGecko."); st.stop()

results = []; fetch_errors = []
progress_bar = st.progress(0, text="Analisi Criptovalute...")
coin_ids_ordered = market_data_df.index.tolist()

for i, coin_id in enumerate(coin_ids_ordered):
    # ... (recupero dati live da market_data_df come prima) ...
    live_data = market_data_df.loc[coin_id]
    symbol = live_data.get('symbol', coin_id).upper(); name = live_data.get('name', coin_id)
    rank = live_data.get('market_cap_rank', 'N/A'); current_price = live_data.get('current_price', np.nan)
    change_1h = live_data.get('price_change_percentage_1h_in_currency', np.nan)
    change_24h = live_data.get('price_change_percentage_24h_in_currency', np.nan)
    change_7d = live_data.get('price_change_percentage_7d_in_currency', np.nan)
    volume_24h = live_data.get('total_volume', np.nan)
    # market_cap = live_data.get('market_cap', np.nan) # Market Cap non più necessario

    hist_daily_df, status_daily = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_DAILY, interval='daily')
    hist_hourly_df, status_hourly = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_HOURLY, interval='hourly')

    if status_daily != "Success": fetch_errors.append(f"{symbol}: Dati Daily - {status_daily}")
    if status_hourly != "Success": fetch_errors.append(f"{symbol}: Dati Hourly - {status_hourly}")

    rsi1d, rsi1h, rsi1w = np.nan, np.nan, np.nan
    macd_line, macd_signal, macd_hist = np.nan, np.nan, np.nan
    ma_short, ma_long = np.nan, np.nan; vwap = np.nan

    if status_daily == "Success" and not hist_daily_df.empty:
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
        min_len_needed = max(RSI_PERIOD + 1, MACD_SLOW, MA_LONG)
        if len(hist_daily_df) < min_len_needed: fetch_errors.append(f"Dati Daily insuff. ({len(hist_daily_df)}/{min_len_needed}) per indicatori per {symbol}.")

    if status_hourly == "Success" and not hist_hourly_df.empty:
        if len(hist_hourly_df) >= RSI_PERIOD + 1: rsi1h = calculate_rsi_manual(hist_hourly_df['close'])
        else: fetch_errors.append(f"Dati Hourly insuff. ({len(hist_hourly_df)} ore) per RSI 1h per {symbol}.")

    gpt_signal = generate_gpt_signal(rsi1d, rsi1h, rsi1w, macd_hist, ma_short, ma_long, current_price)
    gemini_alert = generate_gemini_alert(ma_short, ma_long, macd_hist, rsi1d)

    # --- Assembla Risultati (con % 15m come N/A) ---
    results.append({
        "Rank": rank, "Symbol": symbol, "Name": name, "Gemini Alert": gemini_alert,
        f"Prezzo ({VS_CURRENCY.upper()})": current_price,
        "% 15m": "N/A", # AGGIUNTA COLONNA % 15m (N/A)
        "% 1h": change_1h, "% 24h": change_24h, "% 7d": change_7d,
        "GPT Signal": gpt_signal, "RSI (1h)": rsi1h, "RSI (1d)": rsi1d, "RSI (1w)": rsi1w,
        "MACD Hist (1d)": macd_hist, f"MA({MA_SHORT}d)": ma_short, f"MA({MA_LONG}d)": ma_long,
        "VWAP (1d)": vwap, f"Volume 24h ({VS_CURRENCY.upper()})": volume_24h,
        # "Market Cap (...)" # RIMOSSA
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

    # NUOVO Ordine colonne con % 15m e GPT Signal spostato, Market Cap rimosso
    cols_order = [
        "Symbol", "Name", "Gemini Alert", "GPT Signal", # Segnali spostati
        f"Prezzo ({VS_CURRENCY.upper()})",
        "% 15m", "% 1h", "% 24h", "% 7d", # Percentuali raggruppate
        "RSI (1h)", "RSI (1d)", "RSI (1w)", "MACD Hist (1d)",
        f"MA({MA_SHORT}d)", f"MA({MA_LONG}d)", "VWAP (1d)",
        f"Volume 24h ({VS_CURRENCY.upper()})"
    ]
    cols_to_show = [col for col in cols_order if col in df.columns]
    df_display = df[cols_to_show].copy()

    # Formattazione manuale (invariata ma ora esclude Market Cap)
    formatters = {}; currency_col = f"Prezzo ({VS_CURRENCY.upper()})"; # market_cap_col rimosso
    volume_col = f"Volume 24h ({VS_CURRENCY.upper()})"; pct_cols = ["% 15m", "% 1h", "% 24h", "% 7d"] # Aggiunto % 15m
    for col in df_display.columns:
        # Applica formattazione solo se colonna è numerica
        if pd.api.types.is_numeric_dtype(df_display[col].infer_objects().dtype):
             if col == currency_col: formatters[col] = lambda x: f"${x:,.4f}" if pd.notna(x) else "N/A"
             elif col in pct_cols: formatters[col] = lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
             #elif col == market_cap_col: formatters[col] = lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A" # Rimosso
             elif col == volume_col: formatters[col] = lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
             elif "RSI" in col: formatters[col] = lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
             elif "MACD" in col: formatters[col] = lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
             elif "MA" in col or "VWAP" in col: formatters[col] = lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
             else: formatters[col] = lambda x: str(x) if pd.notna(x) else "N/A"
             # Applica la formattazione alla colonna
             df_display[col] = df_display[col].apply(formatters[col])
        elif df_display[col].dtype == 'object': # Gestisci colonne oggetto
             df_display[col].fillna("N/A", inplace=True)
    df_display.fillna("N/A", inplace=True) # Catchall finale

    def highlight_pct_col(col):
        colors = [''] * len(col)
        for i, val in enumerate(col):
            if isinstance(val, str) and val.endswith('%') and val != 'N/A': # Ignora N/A
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
    st.markdown("""
    Questa sezione spiega gli indicatori e i segnali visualizzati nella tabella. Ricorda che l'analisi tecnica è uno strumento e non una garanzia di risultati futuri. **Questa dashboard è solo a scopo informativo e non costituisce consulenza finanziaria.**

    **Variazioni Percentuali:**
    * **%, 15m, 1h, 24h, 7d:** Variazione percentuale del prezzo rispetto a 15 minuti (N/A*), 1 ora, 24 ore, 7 giorni fa. *(Nota: % 15m attualmente non disponibile tramite API gratuita CoinGecko utilizzata).*

    **Indicatori di Momentum:**
    * **RSI (Relative Strength Index - 1h, 1d, 1w):**
        * *Cos'è:* Oscillatore (0-100) che misura la velocità e l'entità delle recenti variazioni di prezzo. Identifica ipercomprato/ipervenduto e forza del momentum.
        * *Interpretazione:* `> 70`=Ipercomprato, `< 30`=Ipervenduto. Divergenze/Conferme con il prezzo sono importanti.
        * *Timeframe:* **1h** (brevissimo), **1d** (breve-medio), **1w** (medio-lungo). (Dati 1h/1w dipendono da API).
    * **MACD Hist (MACD Histogram - 1d):**
        * *Cos'è:* Differenza tra linea MACD (EMA 12 - EMA 26) e Signal Line (EMA 9 del MACD). Misura il momentum del trend giornaliero.
        * *Interpretazione:* Hist `> 0` = Momentum rialzista; Hist `< 0` = Momentum ribassista. Crossover dello zero e divergenze sono segnali chiave.

    **Indicatori di Trend:**
    * **MA (Simple Moving Average - 20d, 50d):**
        * *Cos'è:* Media mobile semplice prezzi chiusura. Identifica trend e livelli supporto/resistenza dinamici.
        * *Interpretazione:* Prezzo vs MA; Crossover MA20d/MA50d ("Golden/Death Cross").
    * **VWAP (Volume Weighted Average Price - 1d):**
        * *Cos'è:* Prezzo medio ponderato per volume giornaliero.
        * *Interpretazione:* Prezzo sopra/sotto VWAP indica forza/debolezza relativa ai volumi. *(Calcolo approssimato qui).*

    **Segnali Combinati (Esemplificativi - NON CONSULENZA FINANZIARIA):**
    * **Gemini Alert:**
        * *Cos'è:* Alert specifico basato su **forte confluenza DAILY**: `✅ BUY` se MA20>MA50 *E* MACD Hist>0 *E* RSI<70. `❌ SELL` se MA20<MA50 *E* MACD Hist<0 *E* RSI>30. `➖` Altrimenti. `⏳` Dati insuff.
    * **GPT Signal:**
        * *Cos'è:* Sintesi generale basata su combinazione pesata di più indicatori (MAs, MACD, RSI 1d/1w/1h). Offre visione d'insieme. **Interpretare con cautela.**

    **Generale:**
    * **N/A:** Dato/Indicatore non disponibile o non calcolabile (es. dati storici insuff., limitazioni API).
    """)

# --- Footer con Bottone Aggiorna e Disclaimer ---
st.divider()
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔄 Aggiorna", help="Forza l'aggiornamento dei dati"):
        st.cache_data.clear(); st.rerun()
with col2:
    st.caption(f"Ultimo aggiornamento live: {last_cg_update_rome.strftime('%H:%M:%S')} (Ora di Roma) | Disclaimer: Non è consulenza finanziaria.")