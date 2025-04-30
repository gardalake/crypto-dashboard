# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time # Per throttling richieste (anche se la cache aiuta)
import math # Per calcoli VWAP

# --- Configurazione Globale ---
SYMBOLS = ["BTC", "ETH", "BNB", "SOL", "XRP"]
VS_CURRENCY = "usd"
CACHE_TTL = 300
DAYS_HISTORY = 365 # Giorni di storico sufficienti per MA(50) e altri

SYMBOL_TO_ID_MAP = {"BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin", "SOL": "solana", "XRP": "ripple"}
ID_TO_SYMBOL_MAP = {v: k for k, v in SYMBOL_TO_ID_MAP.items()}
COINGECKO_IDS_LIST = list(SYMBOL_TO_ID_MAP.values())

# Periodi Indicatori
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MA_SHORT = 20
MA_LONG = 50
VWAP_PERIOD = 14

# --- Funzioni API CoinGecko (Invariate) ---

@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento prezzi live (CoinGecko)...")
def get_coingecko_current_prices(ids_list, currency):
    ids_string = ",".join(ids_list)
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": ids_string, "vs_currencies": currency, "include_24hr_change": "true"}
    prices = {}; changes = {}; timestamp = datetime.now()
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status(); data = response.json()
        for coin_id, price_data in data.items():
            prices[coin_id] = price_data.get(currency, np.nan)
            changes[coin_id] = price_data.get(f"{currency}_24h_change", np.nan)
        return prices, changes, timestamp
    except Exception as e: # Gestione generica per brevità
        st.error(f"Errore API Prezzi CoinGecko: {e}")
        return {}, {}, timestamp

@st.cache_data(ttl=CACHE_TTL * 2, show_spinner=False)
def get_coingecko_historical_data(coin_id, currency, days):
    time.sleep(0.3)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': currency, 'days': str(days), 'interval': 'daily', 'precision': 'full'}
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
             hist_df = prices_df.join(volumes_df, how='inner')
        else: hist_df['volume'] = 0

        hist_df['high'] = hist_df['close'] # Stima
        hist_df['low'] = hist_df['close']  # Stima
        hist_df['open'] = hist_df['close'].shift(1) # Stima

        hist_df = hist_df[~hist_df.index.duplicated(keep='last')]
        hist_df.dropna(subset=['close'], inplace=True) # Rimuovi giorni senza prezzo
        return hist_df

    except Exception as e: return pd.DataFrame()

# --- Funzioni Calcolo Indicatori (Manuali con Pandas - Invariate) ---

def calculate_rsi_manual(series, period=RSI_PERIOD):
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan
    series = series.dropna(); len_series = len(series)
    if len_series < period + 1: return np.nan

    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Verifica se le medie sono state calcolate (richiedono 'period' punti)
    if len(avg_gain.dropna()) < 1 or len(avg_loss.dropna()) < 1 : return np.nan

    last_avg_gain = avg_gain.iloc[-1]; last_avg_loss = avg_loss.iloc[-1]
    if pd.isna(last_avg_gain) or pd.isna(last_avg_loss): return np.nan
    if last_avg_loss == 0: return 100.0 if last_avg_gain > 0 else 50.0
    rs = last_avg_gain / last_avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd_manual(series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan, np.nan, np.nan
    series = series.dropna()
    if len(series) < slow : return np.nan, np.nan, np.nan # Richiede almeno 'slow' periodi per EMA

    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    # Controlla se la linea MACD ha abbastanza punti per calcolare il segnale
    if len(macd_line.dropna()) < signal: return macd_line.iloc[-1], np.nan, np.nan

    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

def calculate_sma_manual(series, period):
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan
    series = series.dropna()
    if len(series) < period: return np.nan
    return series.rolling(window=period).mean().iloc[-1]

def calculate_vwap_manual(df, period=VWAP_PERIOD):
    if not all(col in df.columns for col in ['close', 'volume']) or df['close'].isna().all() or df['volume'].isna().all(): return np.nan
    df_period = df.iloc[-period:].copy()
    df_period.dropna(subset=['close', 'volume'], inplace=True)
    if df_period.empty or df_period['volume'].sum() == 0: return np.nan
    vwap = (df_period['close'] * df_period['volume']).sum() / df_period['volume'].sum()
    return vwap

# --- Funzione Segnale GPT Semplificato (SINTASSI CORRETTA) ---
def generate_signal(rsi_val, macd_hist_val, ma_short_val, ma_long_val):
    """Genera un segnale esemplificativo basato su alcuni indicatori."""
    if pd.isna(rsi_val) or pd.isna(macd_hist_val) or pd.isna(ma_short_val) or pd.isna(ma_long_val):
        return "⚪️ N/D" # Non disponibile se mancano dati

    score = 0
    # RSI - Corretto su più righe
    if rsi_val < 30:
        score += 2
    elif rsi_val < 40:
        score += 1
    elif rsi_val > 70:
        score -= 2
    elif rsi_val > 60:
        score -= 1

    # MACD Histogram
    if macd_hist_val > 0:
        score += 1
    else:
        score -= 1

    # MA Crossover
    if ma_short_val > ma_long_val:
        score += 1
    else:
        score -= 1

    # Mappa score
    if score >= 3:
        return "🔶 Strong Buy"
    elif score >= 1:
        return "🟢 Buy"
    elif score <= -3:
        return "🔻 Strong Sell"
    elif score <= -1:
        return "🔴 Sell"
    else:
        return "🟡 Hold"

# --- Configurazione Pagina Streamlit ---
st.set_page_config(layout="wide", page_title="Crypto Tech Dashboard (CoinGecko)", page_icon="🦎")
st.title("🦎 Crypto Technical Dashboard (CoinGecko API - Manual Indicators)")
st.caption(f"Dati live e storici da CoinGecko API. Cache: {CACHE_TTL}s.")

# --- Logica Principale ---
cg_prices_dict, cg_changes_dict, last_cg_update = get_coingecko_current_prices(COINGECKO_IDS_LIST, VS_CURRENCY)

results = []; fetch_errors = []
progress_bar = st.progress(0, text="Analisi Criptovalute...")

for i, coin_id in enumerate(COINGECKO_IDS_LIST):
    symbol = ID_TO_SYMBOL_MAP.get(coin_id, coin_id.capitalize())
    current_price = cg_prices_dict.get(coin_id, np.nan)
    change_24h = cg_changes_dict.get(coin_id, np.nan)
    hist_df = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY)

    rsi1d, macd_line, macd_signal, macd_hist, ma_short, ma_long, vwap = [np.nan] * 7

    if not hist_df.empty:
        rsi1d = calculate_rsi_manual(hist_df['close'])
        macd_line, macd_signal, macd_hist = calculate_macd_manual(hist_df['close'])
        ma_short = calculate_sma_manual(hist_df['close'], MA_SHORT)
        ma_long = calculate_sma_manual(hist_df['close'], MA_LONG)
        vwap = calculate_vwap_manual(hist_df)

        if pd.isna(rsi1d) and len(hist_df) < RSI_PERIOD + 1 : fetch_errors.append(f"Dati insuff. ({len(hist_df)}) per RSI per {symbol}.")
        # Aggiungere altri controlli se necessario
    else:
        fetch_errors.append(f"Nessun dato storico valido restituito da CoinGecko per {symbol} ({coin_id}).")

    # --- Assembla Risultati ---
    results.append({
        "Symbol": symbol,
        f"Prezzo ({VS_CURRENCY.upper()})": current_price,
        "% 24h": change_24h,
        "RSI (1d)": rsi1d,
        "MACD Hist (1d)": macd_hist,
        f"MA({MA_SHORT}d)": ma_short,
        f"MA({MA_LONG}d)": ma_long,
        "VWAP (1d)": vwap,
        "SRSI k (1d)": "N/A", "RSI (1h)": "N/A", "RSI (1w)": "N/A", "RSI (1mo)": "N/A",
        "Doda Stoch": "N/A", "GChannel": "N/A", "Volume Flow": "N/A",
    })
    progress_bar.progress((i + 1) / len(COINGECKO_IDS_LIST), text=f"Analisi Criptovalute... ({symbol})")

progress_bar.empty()

# --- Mostra Errori/Info Raccolti ---
if fetch_errors:
    with st.expander("ℹ️ Note sul Recupero Dati Storici/Calcolo Indicatori", expanded=False):
        unique_errors = sorted(list(set(fetch_errors)))
        for error_msg in unique_errors: st.info(error_msg)

# --- Crea e Visualizza DataFrame ---
if results:
    df = pd.DataFrame(results)
    # Calcola segnale DOPO aver assemblato df e PRIMA di formattare/mostrare
    df['GPT Signal'] = df.apply(lambda row: generate_signal(
                                    row.get('RSI (1d)'), # Usa .get per sicurezza se colonna manca
                                    row.get('MACD Hist (1d)'),
                                    row.get(f'MA({MA_SHORT}d)'),
                                    row.get(f'MA({MA_LONG}d)')
                                ), axis=1)

    cols_order = ["Symbol", f"Prezzo ({VS_CURRENCY.upper()})", "% 24h", "GPT Signal", "RSI (1d)", "MACD Hist (1d)", f"MA({MA_SHORT}d)", f"MA({MA_LONG}d)", "VWAP (1d)"]
    cols_to_show = [col for col in cols_order if col in df.columns]
    df_display = df[cols_to_show].copy()

    # Formattazione manuale
    numeric_cols = df_display.select_dtypes(include=np.number).columns
    formatters = {}
    currency_col = f"Prezzo ({VS_CURRENCY.upper()})"
    pct_col = "% 24h"
    for col in numeric_cols:
        if col == currency_col: formatters[col] = lambda x: f"${x:,.4f}" if pd.notna(x) else "N/A"
        elif col == pct_col: formatters[col] = lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
        elif "RSI" in col: formatters[col] = lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        elif "MACD" in col: formatters[col] = lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        elif "MA" in col or "VWAP" in col: formatters[col] = lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
    for col, func in formatters.items():
        if col in df_display.columns: df_display[col] = df_display[col].apply(func)
    df_display.fillna("N/A", inplace=True)

    def highlight_pct_val(val):
        if isinstance(val, str) and val.endswith('%'):
            try: num = float(val.replace('%','')); return 'color: green' if num > 0 else 'color: red' if num < 0 else ''
            except ValueError: return ''
        return ''

    st.dataframe(df_display.style.map(highlight_pct_val, subset=[pct_col]), use_container_width=True, hide_index=True)
else:
    st.warning("Nessun risultato da visualizzare.")

# --- Legenda ---
st.markdown("### ℹ️ Indicatori Tecnici - Legenda")
st.markdown(f"""
* **RSI (Relative Strength Index):** Momentum (0-100). >70 Ipercomprato, <30 Ipervenduto. (Calcolato su dati giornalieri, periodo {RSI_PERIOD})
* **MACD Hist (MACD Histogram):** Momentum trend. (Calcolato su dati giornalieri)
* **MA (Simple Moving Average):** Media mobile semplice prezzo. (Calcolata su dati giornalieri, periodi {MA_SHORT} e {MA_LONG})
* **VWAP (Volume Weighted Average Price):** Prezzo medio ponderato volume. (Calcolato su dati giornalieri)
* **GPT Signal:** Segnale *esemplificativo* basato su RSI, MACD Hist, MA Crossover. **Non è consulenza finanziaria.**
* *Altri indicatori (SRSI, RSI 1h/1w/1mo, Doda Stoch, etc):* Non implementati (N/A).
""")
st.markdown(f"⏱️ Prezzi live aggiornati da CoinGecko circa alle: **{last_cg_update.strftime('%Y-%m-%d %H:%M:%S')}** (Cache: {CACHE_TTL}s)")