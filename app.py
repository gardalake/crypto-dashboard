# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time
import math
import yfinance as yf # Per dati mercato tradizionale
import feedparser # Per News RSS

# --- Configurazione Globale ---

# Mappa Simbolo -> ID CoinGecko per facile gestione
# AGGIUNGI/RIMUOVI/MODIFICA coppie qui per cambiare le coin monitorate
SYMBOL_TO_ID_MAP = {
    # Originali
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "RNDR": "render-token",
    "FET": "artificial-superintelligence-alliance", # ID post-merge ASI Alliance
    "RAY": "raydium", "SUI": "sui", "ONDO": "ondo", "ARB": "arbitrum",
    # Nuove Aggiunte
    "TAO": "bittensor", "LINK": "chainlink", "AVAX": "avalanche-2",
    "HBAR": "hedera-hashgraph", "PEPE": "pepe", "UNI": "uniswap",
    "TIA": "celestia", "JUP": "jupiter-aggregator", # Assunto Jupiter Aggregator
    "IMX": "immutable-x", "TRUMP": "maga", # Assunto MAGA (TRUMP)
    "NEAR": "near-protocol", "AERO": "aerodrome-finance", "TRON": "tron",
    "AERGO": "aergo", "ADA": "cardano", "MKR": "maker"
}

# Deriva dinamicamente la lista di simboli e ID dalla mappa
SYMBOLS = list(SYMBOL_TO_ID_MAP.keys())
COINGECKO_IDS_LIST = list(SYMBOL_TO_ID_MAP.values())
NUM_COINS = len(SYMBOLS)

# Ticker per mercati tradizionali (yfinance)
TRAD_TICKERS = ['^GSPC', '^IXIC', 'GC=F', 'UVXY', 'TQQQ', # Indici/VIX/Oro
                'NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR'] # Azioni

VS_CURRENCY = "usd" # Valuta di riferimento
CACHE_TTL = 1800 # NUOVO: Cache di 30 minuti (1800 sec)
DAYS_HISTORY_DAILY = 365; DAYS_HISTORY_HOURLY = 7
RSI_PERIOD = 14; SRSI_PERIOD = 14; SRSI_K = 3; SRSI_D = 3
MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9
MA_SHORT = 20; MA_LONG = 50; VWAP_PERIOD = 14

# URL Feed RSS per notizie (esempio: Cointelegraph)
NEWS_FEED_URL = "https://cointelegraph.com/rss"
NUM_NEWS_ITEMS = 5 # Numero di notizie da mostrare

# --- Password Protection ---
def check_password():
    if "password_correct" not in st.session_state: st.session_state.password_correct = False
    if not st.session_state.password_correct:
        password = st.text_input("🔑 Password", type="password", key="password_input")
        if password == "Leonardo": # Replace "YourSecurePassword" with the actual password
             st.session_state.password_correct = True
             # Avoid automatically rerunning if password was just entered correctly
             # Check query params to prevent loop if rerun=false is already set
             if st.query_params.get("rerun") != "false":
                 st.query_params["rerun"] = "false" # Set a flag to prevent immediate rerun after password entry
                 st.rerun() # Rerun to hide the password input form
        elif password: # Only show warning if a password was entered
            st.warning("Password errata.")
            st.stop() # Stop execution if password was incorrect
        else: # Stop execution if no password entered yet
            st.stop()
    return True # Return True if password check is passed or already correct

# Check password at the beginning
if not check_password():
    st.stop() # Ensure app stops if password check fails


# --- Funzione Helper Formattazione Numeri Grandi ---
def format_large_number(num):
    if pd.isna(num) or not isinstance(num, (int, float)): return "N/A"
    if abs(num) < 1_000_000: return f"{num:,.0f}"
    elif abs(num) < 1_000_000_000: return f"{num / 1_000_000:.1f}M"
    elif abs(num) < 1_000_000_000_000: return f"{num / 1_000_000_000:.1f}B"
    else: return f"{num / 1_000_000_000_000:.2f}T"

# --- Funzioni API CoinGecko ---
@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento dati di mercato (CoinGecko)...")
def get_coingecko_market_data(ids_list, currency):
    ids_string = ",".join(ids_list)
    url = "https://api.coingecko.com/api/v3/coins/markets"
    # Richiedi anche % 30d e 1y
    params = {'vs_currency': currency, 'ids': ids_string, 'order': 'market_cap_desc',
              'per_page': str(len(ids_list)), 'page': 1, 'sparkline': False,
              'price_change_percentage': '1h,24h,7d,30d,1y'}
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
    except Exception as e: st.error(f"Errore Processamento Dati Mercato CoinGecko: {e}"); return pd.DataFrame(), timestamp

@st.cache_data(ttl=CACHE_TTL * 2, show_spinner=False)
def get_coingecko_historical_data(coin_id, currency, days, interval='daily'):
    time.sleep(1.5) # Manteniamo delay alto per rate limiting API
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
             # Join preserving the index type and handling potential NaNs during join
             hist_df = prices_df.join(volumes_df, how='outer')
             # Interpolate after join, then fill remaining NaNs
             hist_df = hist_df.interpolate(method='time').ffill().bfill()
        else: hist_df['volume'] = 0 # Assign volume column if not present
        # Approximate OHLC data - Can be refined if API provides it directly
        hist_df['high'] = hist_df['close']; hist_df['low'] = hist_df['close']; hist_df['open'] = hist_df['close'].shift(1)
        # Ensure no duplicate indices remain after join/processing
        hist_df = hist_df[~hist_df.index.duplicated(keep='last')]
        hist_df.dropna(subset=['close'], inplace=True) # Ensure rows with NaN close prices are dropped
        if hist_df.empty: return pd.DataFrame(), "Processed Empty"
        return hist_df, "Success"
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 429: return pd.DataFrame(), "Rate Limited (429)"
        else: return pd.DataFrame(), f"HTTP Error {http_err.response.status_code}"
    except Exception as e: return pd.DataFrame(), f"Generic Error: {type(e).__name__}"

# --- Funzioni Dati Mercato Generale ---
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_fear_greed_index_cmc():
    # NOTE: This endpoint seems unofficial or changed. Using alternative.
    # url = "https://pro-api.coinmarketcap.com/v1/crypto/fear-and-greed" # Original, might need API key
    url_alt = "https://api.alternative.me/fng/?limit=1" # Alternative.me endpoint
    try:
        response = requests.get(url_alt, timeout=10)
        response.raise_for_status(); data = response.json()
        if data and data.get("data") and isinstance(data["data"], list) and len(data["data"]) > 0:
             latest_data = data["data"][0]; value = latest_data.get("value"); desc = latest_data.get("value_classification")
             if value is not None and desc is not None: return f"{int(value)} ({desc})"
        return "N/A (Alt Data)"
    except requests.exceptions.RequestException as req_ex:
        status_code = req_ex.response.status_code if req_ex.response is not None else "N/A"
        return f"N/A (Err {status_code})"
    except Exception: return "N/A (Errore)"


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_global_market_data_cg(currency):
    url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status(); data = response.json().get('data', {})
        total_mcap = data.get('total_market_cap', {}).get(currency.lower(), np.nan)
        return total_mcap
    except Exception: return np.nan

def get_etf_flow(): return "N/A" # Placeholder - Requires specific API/data source

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_traditional_market_data_yf(tickers):
    data = {}
    for ticker_sym in tickers:
        try:
            time.sleep(0.2); ticker_obj = yf.Ticker(ticker_sym);
            # Fetch more robustly: try today's data first, then history
            info = ticker_obj.info
            price = info.get('regularMarketPrice', info.get('currentPrice'))
            if price is None: # If live price isn't available, get last close
                hist = ticker_obj.history(period='5d', interval='1d')
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                else: # Fallback if history is also empty
                    price = info.get('previousClose', np.nan)
            data[ticker_sym] = price
        except Exception: data[ticker_sym] = np.nan # Handle potential errors for individual tickers
    return data

# --- NUOVA Funzione per News RSS ---
@st.cache_data(ttl=900, show_spinner="Caricamento notizie...") # Cache 15 min per news
def get_crypto_news(feed_url, num_items=NUM_NEWS_ITEMS):
    """Recupera e parsifica un feed RSS."""
    try:
        feed = feedparser.parse(feed_url)
        # Check for errors during parsing
        if feed.bozo:
            # Log warning but proceed if possible
            st.warning(f"Possibile errore nel parsing del feed RSS: {feed.bozo_exception}")
        if not feed.entries:
             return [] # Return empty list if no entries found
        return feed.entries[:num_items] # Ritorna le prime N notizie
    except Exception as e:
        st.error(f"Errore durante il recupero del feed RSS ({feed_url}): {e}")
        return [] # Ritorna lista vuota in caso di errore grave

# --- Funzioni Calcolo Indicatori (Manuali) ---
def calculate_rsi_manual(series, period=RSI_PERIOD):
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan
    series = series.dropna()
    if len(series) < period + 1: return np.nan # Need at least period+1 points for diff
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rename("gain")
    loss = -delta.where(delta < 0, 0.0).rename("loss")

    # Use Exponential Moving Average (EMA) for RSI calculation
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Check if calculation produced valid results
    if avg_loss.iloc[-1] == 0: # Avoid division by zero
        return 100.0 if avg_gain.iloc[-1] > 0 else 50.0 # RSI is 100 if only gains, 50 if no change

    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_stoch_rsi(series, rsi_period=RSI_PERIOD, stoch_period=SRSI_PERIOD, k_smooth=SRSI_K, d_smooth=SRSI_D):
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan, np.nan
    series = series.dropna()
    if len(series) < rsi_period + 1: return np.nan, np.nan

    # Calculate RSI Series first
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    # Avoid division by zero for RS calculation
    rs = avg_gain / avg_loss.replace(0, np.nan) # Replace 0 avg_loss with NaN to handle cases
    rsi_series = (100.0 - (100.0 / (1.0 + rs))).dropna()

    if len(rsi_series) < stoch_period: return np.nan, np.nan # Need enough RSI values for Stoch calc

    # Calculate Stochastic RSI
    min_rsi = rsi_series.rolling(window=stoch_period).min()
    max_rsi = rsi_series.rolling(window=stoch_period).max()
    range_rsi = max_rsi - min_rsi
    # Avoid division by zero if min == max
    stoch_rsi_k_raw = 100 * (rsi_series - min_rsi) / range_rsi.replace(0, np.nan)
    stoch_rsi_k_raw = stoch_rsi_k_raw.dropna()

    if len(stoch_rsi_k_raw) < k_smooth : return np.nan, np.nan # Need enough raw K values

    # Smooth %K and %D
    stoch_rsi_k = stoch_rsi_k_raw.rolling(window=k_smooth).mean()
    if len(stoch_rsi_k.dropna()) < d_smooth : return stoch_rsi_k.iloc[-1], np.nan # Return K if D cannot be calculated

    stoch_rsi_d = stoch_rsi_k.rolling(window=d_smooth).mean()

    # Get the last valid values
    last_k = stoch_rsi_k.iloc[-1] if not pd.isna(stoch_rsi_k.iloc[-1]) else np.nan
    last_d = stoch_rsi_d.iloc[-1] if not pd.isna(stoch_rsi_d.iloc[-1]) else np.nan
    return last_k, last_d

def calculate_macd_manual(series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan, np.nan, np.nan
    series = series.dropna()
    if len(series) < slow : return np.nan, np.nan, np.nan # Need enough data for slow EMA

    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow

    # Need enough MACD line values to calculate signal line
    if len(macd_line.dropna()) < signal: return macd_line.iloc[-1], np.nan, np.nan # Return MACD if signal cannot be calculated

    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    # Get the last valid values
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
    # Ensure required columns exist and are not all NaN
    required_cols = ['close', 'volume']
    if not all(col in df.columns for col in required_cols): return np.nan
    if df['close'].isna().all() or df['volume'].isna().all(): return np.nan

    # Use the last 'period' rows that have valid close and volume
    df_valid = df.dropna(subset=required_cols)
    if len(df_valid) < period: return np.nan # Not enough valid data points for the period

    df_period = df_valid.iloc[-period:]
    if df_period.empty or df_period['volume'].sum() == 0: return np.nan # Avoid division by zero

    # Calculate VWAP
    vwap = (df_period['close'] * df_period['volume']).sum() / df_period['volume'].sum()
    return vwap

# --- Funzione Raggruppata Indicatori ---
def compute_all_indicators(symbol, hist_daily_df, hist_hourly_df, fetch_errors_list):
    indicators = {
        "RSI (1h)": np.nan, "RSI (1d)": np.nan, "RSI (1w)": np.nan, "RSI (1mo)": np.nan,
        "SRSI %K (1d)": np.nan, "SRSI %D (1d)": np.nan,
        "MACD Line (1d)": np.nan, "MACD Signal (1d)": np.nan, "MACD Hist (1d)": np.nan,
        f"MA({MA_SHORT}d)": np.nan, f"MA({MA_LONG}d)": np.nan,
        "VWAP (1d)": np.nan,
        # Placeholders for future/other indicators
        "Doda Stoch": "N/A", "GChannel": "N/A", "Volume Flow": "N/A"
    }
    # Minimum lengths required for calculations
    min_len_daily_full = max(RSI_PERIOD + 1, SRSI_PERIOD + RSI_PERIOD + 1, MACD_SLOW + MACD_SIGNAL, MA_LONG)
    min_len_rsi_base = RSI_PERIOD + 1

    # --- Daily Indicators ---
    if not hist_daily_df.empty and 'close' in hist_daily_df.columns:
        close_daily = hist_daily_df['close'].dropna()
        len_daily = len(close_daily)

        if len_daily >= min_len_daily_full:
            indicators["RSI (1d)"] = calculate_rsi_manual(close_daily)
            indicators["SRSI %K (1d)"], indicators["SRSI %D (1d)"] = calculate_stoch_rsi(close_daily)
            macd_l, macd_s, macd_h = calculate_macd_manual(close_daily)
            indicators["MACD Line (1d)"] = macd_l
            indicators["MACD Signal (1d)"] = macd_s
            indicators["MACD Hist (1d)"] = macd_h
            indicators[f"MA({MA_SHORT}d)"] = calculate_sma_manual(close_daily, MA_SHORT)
            indicators[f"MA({MA_LONG}d)"] = calculate_sma_manual(close_daily, MA_LONG)
            indicators["VWAP (1d)"] = calculate_vwap_manual(hist_daily_df) # VWAP needs full df
        else:
            fetch_errors_list.append(f"{symbol}: Dati Daily insuff. ({len_daily}/{min_len_daily_full}) per ind. base.")
            # Attempt individual calculations if enough data exists for them
            if len_daily >= min_len_rsi_base: indicators["RSI (1d)"] = calculate_rsi_manual(close_daily)
            if len_daily >= MA_SHORT: indicators[f"MA({MA_SHORT}d)"] = calculate_sma_manual(close_daily, MA_SHORT)
            if len_daily >= MA_LONG: indicators[f"MA({MA_LONG}d)"] = calculate_sma_manual(close_daily, MA_LONG)
            # Add checks for SRSI and MACD minimum lengths as well if needed

        # --- Weekly RSI ---
        try:
            # Ensure index is datetime before resampling
            if pd.api.types.is_datetime64_any_dtype(close_daily.index):
                df_weekly = close_daily.resample('W-MON').last() # Resample to weekly, taking last price of week
                if len(df_weekly.dropna()) >= min_len_rsi_base:
                    indicators["RSI (1w)"] = calculate_rsi_manual(df_weekly)
                else:
                    fetch_errors_list.append(f"{symbol}: Dati insuff. ({len(df_weekly.dropna())}/{min_len_rsi_base} sett.) per RSI 1w.")
            else: fetch_errors_list.append(f"{symbol}: Indice Daily non è datetime per RSI 1w.")
        except Exception as e: fetch_errors_list.append(f"{symbol}: Errore resampling weekly: {e}")

        # --- Monthly RSI ---
        try:
            if pd.api.types.is_datetime64_any_dtype(close_daily.index):
                df_monthly = close_daily.resample('ME').last() # Resample to end of month
                if len(df_monthly.dropna()) >= min_len_rsi_base:
                    indicators["RSI (1mo)"] = calculate_rsi_manual(df_monthly)
                else:
                    fetch_errors_list.append(f"{symbol}: Dati insuff. ({len(df_monthly.dropna())}/{min_len_rsi_base} mesi) per RSI 1mo.")
            else: fetch_errors_list.append(f"{symbol}: Indice Daily non è datetime per RSI 1mo.")
        except Exception as e: fetch_errors_list.append(f"{symbol}: Errore resampling monthly: {e}")

    # --- Hourly RSI ---
    if not hist_hourly_df.empty and 'close' in hist_hourly_df.columns:
        close_hourly = hist_hourly_df['close'].dropna()
        len_hourly = len(close_hourly)
        if len_hourly >= min_len_rsi_base:
            indicators["RSI (1h)"] = calculate_rsi_manual(close_hourly)
        else:
            fetch_errors_list.append(f"{symbol}: Dati Hourly insuff. ({len_hourly}/{min_len_rsi_base} ore) per RSI 1h.")

    return indicators


# --- Funzioni Segnale (Sintassi Corretta) ---
def generate_gpt_signal(rsi_1d, rsi_1h, rsi_1w, macd_hist, ma_short, ma_long, srsi_k, srsi_d, current_price):
    required_inputs = [rsi_1d, macd_hist, ma_short, ma_long, current_price]
    # Check if any essential input is NaN
    if any(pd.isna(x) for x in required_inputs):
        return "⚪️ N/D" # Not Determinable if core data missing

    score = 0

    # Price vs Long MA
    if current_price > ma_long:
        score += 1
    else:
        score -= 1

    # MA Cross
    if ma_short > ma_long:
        score += 2 # Golden cross tendency
    else:
        score -= 2 # Death cross tendency

    # MACD Histogram
    if macd_hist > 0:
        score += 2 # Bullish momentum
    else:
        score -= 2 # Bearish momentum

    # Daily RSI Levels
    if rsi_1d < 30:
        score += 2 # Oversold
    elif rsi_1d < 40:
        score += 1 # Approaching oversold
    elif rsi_1d > 70:
        score -= 2 # Overbought
    elif rsi_1d > 60:
        score -= 1 # Approaching overbought

    # Weekly RSI (Bonus points)
    if not pd.isna(rsi_1w):
        if rsi_1w < 30:
            score += 1 # Weekly oversold adds confirmation
        elif rsi_1w > 70:
            score -= 1 # Weekly overbought adds confirmation

    # Hourly RSI (Fine-tuning)
    if not pd.isna(rsi_1h):
        if rsi_1h > 60: # Changed from >60 as positive signal seems counterintuitive here
            score -= 1 # Hourly overbought might indicate pullback soon
        elif rsi_1h < 40:
            score += 1 # Hourly oversold might indicate bounce soon

    # Stochastic RSI (Confirmation/Entry Signal)
    if not pd.isna(srsi_k) and not pd.isna(srsi_d):
        if srsi_k < 20 and srsi_d < 20:
            score += 1 # SRSI oversold
        elif srsi_k > 80 and srsi_d > 80:
            score -= 1 # SRSI overbought

    # Determine Signal based on Score
    if score >= 5:
        return "⚡️ Strong Buy"
    elif score >= 2:
        return "🟢 Buy"
    elif score <= -5:
        return "🚨 Strong Sell"
    elif score <= -2:
        return "🔴 Sell"
    # Add nuanced Hold conditions based on RSI
    elif score >= 0: # Neutral to slightly bullish score
        # Consider 'Close to Buy' if RSI is low but score isn't strongly bullish yet
        if not pd.isna(rsi_1d) and rsi_1d < 45 and rsi_1d > 30:
             return "⏳ CTB" # Close To Buy
        else:
             return "🟡 Hold"
    else: # Neutral to slightly bearish score
        # Consider 'Close to Sell' if RSI is high but score isn't strongly bearish yet
        if not pd.isna(rsi_1d) and rsi_1d > 55 and rsi_1d < 70:
             return "⚠️ CTS" # Close To Sell
        else:
             return "🟡 Hold"


def generate_gemini_alert(ma_short, ma_long, macd_hist, rsi_1d):
    # Check for missing essential data
    if pd.isna(ma_short) or pd.isna(ma_long) or pd.isna(macd_hist) or pd.isna(rsi_1d):
        return "⚪️ N/D"

    # Define conditions clearly
    is_uptrend = ma_short > ma_long
    is_momentum_positive = macd_hist > 0
    is_not_overbought = rsi_1d < 70 # Using standard 70 level

    is_downtrend = ma_short < ma_long
    is_momentum_negative = macd_hist < 0
    is_not_oversold = rsi_1d > 30 # Using standard 30 level

    # Combine conditions for alerts
    if is_uptrend and is_momentum_positive and is_not_overbought:
        return "⚡️ Strong Buy" # Confluence of positive signals
    elif is_downtrend and is_momentum_negative and is_not_oversold:
        return "🚨 Strong Sell" # Confluence of negative signals
    else:
        return "🟡 Hold" # Default to Hold if strong conditions not met


# --- Layout App Streamlit ---
st.set_page_config(layout="wide", page_title="Crypto Technical Dashboard Pro", page_icon="📈")

# --- Titolo e Bottone Aggiorna ---
col_title, col_button_placeholder, col_button = st.columns([4, 1, 1])
with col_title: st.title("📈 Crypto Technical Dashboard Pro") # TITOLO AGGIORNATO
with col_button:
    st.write("") # Spazio per allineare verticalmente
    if st.button("🔄 Aggiorna", help=f"Forza l'aggiornamento dei dati (cache: {CACHE_TTL/60:.0f} min)"):
        # Clear specific caches instead of all if needed
        st.cache_data.clear() # Clear all cached functions used by the app
        st.rerun() # Rerun the script

# Timestamp placeholder
last_cg_update_placeholder = st.empty()
# Display cache info more clearly
st.caption(f"Dati CoinGecko (cache {CACHE_TTL/60:.0f} min live / {CACHE_TTL*2/60:.0f} min storico). Mercato Trad. (cache {CACHE_TTL/60:.0f} min). Notizie (cache 15 min).")

# --- Sezione Market Overview (Layout Modificato) ---
st.markdown("---")
st.subheader("🌐 Market Overview")
# Fetch dati mercato generale in parallel? (Consider if performance is an issue)
fear_greed = get_fear_greed_index_cmc()
total_mcap = get_global_market_data_cg(VS_CURRENCY)
etf_flow = get_etf_flow() # Placeholder remains
trad_data = get_traditional_market_data_yf(TRAD_TICKERS) # Include ora le azioni

# --- Riga 1 Overview --- (5 Colonne)
mkt_col1, mkt_col2, mkt_col3, mkt_col4, mkt_col5 = st.columns(5)
with mkt_col1: st.metric(label="Fear & Greed (Alt.)", value=fear_greed, help="Indice da Alternative.me (0=Extreme Fear, 100=Extreme Greed)")
with mkt_col2: st.metric(label=f"Total Crypto M.Cap", value=f"${format_large_number(total_mcap)}", help=f"({VS_CURRENCY.upper()}) - CoinGecko Global Data")
with mkt_col3: st.metric(label="Crypto ETFs Flow (Daily)", value=etf_flow, help="Dato N/A - fonte API gratuita non trovata.")
with mkt_col4: st.metric(label="S&P 500 (^GSPC)", value=f"{trad_data.get('^GSPC', np.nan):,.2f}" if not pd.isna(trad_data.get('^GSPC')) else "N/A", delta=None) # Add delta later if needed
with mkt_col5: st.metric(label="Nasdaq (^IXIC)", value=f"{trad_data.get('^IXIC', np.nan):,.2f}" if not pd.isna(trad_data.get('^IXIC')) else "N/A", delta=None)

# --- Riga 2 Overview --- (3 Colonne)
mkt_col6, mkt_col7, mkt_col8 = st.columns(3)
with mkt_col6: st.metric(label="Gold (GC=F)", value=f"{trad_data.get('GC=F', np.nan):,.2f}" if not pd.isna(trad_data.get('GC=F')) else "N/A", delta=None)
with mkt_col7: st.metric(label="UVXY (Volatility)", value=f"{trad_data.get('UVXY', np.nan):,.2f}" if not pd.isna(trad_data.get('UVXY')) else "N/A", delta=None)
with mkt_col8: st.metric(label="TQQQ (Nasdaq 3x)", value=f"{trad_data.get('TQQQ', np.nan):,.2f}" if not pd.isna(trad_data.get('TQQQ')) else "N/A", delta=None)

# --- Riga 3 Overview: Azioni --- (4 Colonne)
st.markdown("<h6>Titoli Principali (Prezzi):</h6>", unsafe_allow_html=True)
stock_col1, stock_col2, stock_col3, stock_col4 = st.columns(4)
stock_tickers_row = ['NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR'] # Ensure these match TRAD_TICKERS
cols_stock = [stock_col1, stock_col2, stock_col3, stock_col4] * (len(stock_tickers_row) // 4 + (len(stock_tickers_row) % 4 > 0)) # Distribute tickers in columns

for idx, ticker in enumerate(stock_tickers_row):
    col_index = idx % 4
    with cols_stock[col_index]:
         price = trad_data.get(ticker, np.nan)
         st.metric(label=ticker, value=f"{price:,.2f}" if not pd.isna(price) else "N/A", delta=None) # Display price

st.markdown("---")

# --- Logica Principale Dashboard Crypto ---
market_data_df, last_cg_update_utc = get_coingecko_market_data(COINGECKO_IDS_LIST, VS_CURRENCY)

# Display timestamp robustly
if last_cg_update_utc:
    try:
        # Assuming your location (San Donato Milanese) uses Rome time (CEST/CET)
        # Find the correct timezone object (e.g., 'Europe/Rome')
        from zoneinfo import ZoneInfo # Use modern zoneinfo
        rome_tz = ZoneInfo("Europe/Rome")
        last_cg_update_local = last_cg_update_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(rome_tz)
        last_cg_update_placeholder.markdown(f"*Prezzi Live aggiornati alle: **{last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S %Z')}***")
    except ImportError:
        # Fallback if zoneinfo is not available (older Python?)
        last_cg_update_rome_approx = last_cg_update_utc + timedelta(hours=2) # Approximate CEST
        last_cg_update_placeholder.markdown(f"*Prezzi Live aggiornati alle: **{last_cg_update_rome_approx.strftime('%Y-%m-%d %H:%M:%S')} (Ora approx. Roma)***")
    except Exception as e:
        last_cg_update_placeholder.markdown(f"*Errore conversione timestamp: {e}*")
else:
    last_cg_update_placeholder.markdown("*Timestamp aggiornamento non disponibile.*")


if market_data_df.empty:
    st.error("Errore caricamento dati crypto da CoinGecko. Controllare API o connessione.")
    st.stop() # Stop execution if market data fails

results = []; fetch_errors = []
progress_bar = st.progress(0, text=f"Analisi 0/{NUM_COINS} Criptovalute...")

# Ensure processing happens based on the order received from API (usually market cap)
coin_ids_ordered = market_data_df.index.tolist()

for i, coin_id in enumerate(coin_ids_ordered):
    # Update progress text more informatively
    progress_bar.progress((i + 1) / len(coin_ids_ordered), text=f"Analisi {i+1}/{NUM_COINS} Criptovalute... ({coin_id})")

    # Check if data for this coin_id exists in the fetched market data
    if coin_id not in market_data_df.index:
        fetch_errors.append(f"{coin_id}: Dati live non trovati nel batch fetch.")
        continue # Skip to the next coin if live data is missing

    live_data = market_data_df.loc[coin_id]

    # Safely get data points, providing defaults or logging errors
    symbol = live_data.get('symbol', coin_id).upper()
    name = live_data.get('name', coin_id)
    rank = live_data.get('market_cap_rank', 'N/A')
    current_price = live_data.get('current_price', np.nan)
    volume_24h = live_data.get('total_volume', np.nan)
    # Safely get price change percentages
    change_1h = live_data.get('price_change_percentage_1h_in_currency', np.nan)
    change_24h = live_data.get('price_change_percentage_24h_in_currency', np.nan)
    change_7d = live_data.get('price_change_percentage_7d_in_currency', np.nan)
    change_30d = live_data.get('price_change_percentage_30d_in_currency', np.nan)
    change_1y = live_data.get('price_change_percentage_1y_in_currency', np.nan)

    # Fetch historical data
    hist_daily_df, status_daily = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_DAILY, interval='daily')
    hist_hourly_df, status_hourly = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_HOURLY, interval='hourly')

    # Log fetch errors clearly
    if status_daily != "Success": fetch_errors.append(f"{symbol}: Dati Daily - {status_daily}")
    if status_hourly != "Success": fetch_errors.append(f"{symbol}: Dati Hourly - {status_hourly}")

    # Compute indicators using fetched data
    indicators = compute_all_indicators(symbol, hist_daily_df, hist_hourly_df, fetch_errors) # Pass errors list

    # Generate signals based on indicators and live price
    gpt_signal = generate_gpt_signal(
        indicators.get("RSI (1d)"), indicators.get("RSI (1h)"), indicators.get("RSI (1w)"),
        indicators.get("MACD Hist (1d)"), indicators.get(f"MA({MA_SHORT}d)"), indicators.get(f"MA({MA_LONG}d)"),
        indicators.get("SRSI %K (1d)"), indicators.get("SRSI %D (1d)"),
        current_price
    )
    gemini_alert = generate_gemini_alert(
        indicators.get(f"MA({MA_SHORT}d)"), indicators.get(f"MA({MA_LONG}d)"),
        indicators.get("MACD Hist (1d)"), indicators.get("RSI (1d)")
    )

    # Assemble results for this coin into a dictionary
    results.append({
        "Rank": rank, "Symbol": symbol, "Name": name, "Gemini Alert": gemini_alert, "GPT Signal": gpt_signal,
        f"Prezzo ({VS_CURRENCY.upper()})": current_price,
        "% 1h": change_1h, "% 24h": change_24h, "% 7d": change_7d, "% 30d": change_30d, "% 1y": change_1y,
        "RSI (1h)": indicators.get("RSI (1h)"), "RSI (1d)": indicators.get("RSI (1d)"),
        "RSI (1w)": indicators.get("RSI (1w)"), "RSI (1mo)": indicators.get("RSI (1mo)"),
        "SRSI %K (1d)": indicators.get("SRSI %K (1d)"), "SRSI %D (1d)": indicators.get("SRSI %D (1d)"),
        "MACD Hist (1d)": indicators.get("MACD Hist (1d)"),
        f"MA({MA_SHORT}d)": indicators.get(f"MA({MA_SHORT}d)"), f"MA({MA_LONG}d)": indicators.get(f"MA({MA_LONG}d)"),
        "VWAP (1d)": indicators.get("VWAP (1d)"),
        f"Volume 24h ({VS_CURRENCY.upper()})": volume_24h,
        # Placeholders for future indicators
        "Doda Stoch": "N/A", "GChannel": "N/A", "Volume Flow": "N/A",
    })

# Clear the progress bar after the loop finishes
progress_bar.empty()

# --- Crea e Visualizza DataFrame ---
if results:
    df = pd.DataFrame(results)
    # Try setting Rank as index, handle potential errors (e.g., duplicate ranks)
    try:
        df.set_index('Rank', inplace=True, drop=True)
        # Sort by rank if it's numeric, otherwise keep API order
        if pd.api.types.is_numeric_dtype(df.index):
            df.sort_index(inplace=True)
    except KeyError:
        st.warning("Colonna 'Rank' non trovata per impostare l'indice.")
    except Exception as e:
        st.warning(f"Errore impostando/ordinando indice per Rank: {e}")


    # Define column order explicitly
    cols_order = [
        "Symbol", "Name", "Gemini Alert", "GPT Signal",
        f"Prezzo ({VS_CURRENCY.upper()})",
        "% 1h", "% 24h", "% 7d", "% 30d", "% 1y",
        "RSI (1h)", "RSI (1d)", "RSI (1w)", "RSI (1mo)",
        "SRSI %K (1d)", "SRSI %D (1d)",
        "MACD Hist (1d)", f"MA({MA_SHORT}d)", f"MA({MA_LONG}d)", "VWAP (1d)",
        f"Volume 24h ({VS_CURRENCY.upper()})"
        # Add other placeholder columns if they exist in `results` and should be shown
        # "Doda Stoch", "GChannel", "Volume Flow"
    ]
    # Filter df_display to only include columns that actually exist in df and are in cols_order
    cols_to_show = [col for col in cols_order if col in df.columns]
    df_display = df[cols_to_show].copy() # Create a copy for display formatting

    # --- Formatting Logic ---
    # Define column types for formatting
    currency_col = f"Prezzo ({VS_CURRENCY.upper()})"
    volume_col = f"Volume 24h ({VS_CURRENCY.upper()})"
    pct_cols = ["% 1h", "% 24h", "% 7d", "% 30d", "% 1y"]
    rsi_srsi_cols = [col for col in df_display.columns if "RSI" in col or "SRSI" in col]
    macd_cols = [col for col in df_display.columns if "MACD" in col]
    ma_vwap_cols = [col for col in df_display.columns if "MA" in col or "VWAP" in col]

    # Create a dictionary for streamlit's format argument
    formatters = {}
    for col in df_display.columns:
        # Use original df's dtypes for decisions, apply formatting to df_display
        original_dtype = df[col].dtype if col in df else None

        if col == currency_col:
             # Format as currency with 4 decimal places for precision
             formatters[col] = lambda x: f"${x:,.4f}" if pd.notna(x) else "N/A"
        elif col in pct_cols:
             # Format as percentage with sign and 2 decimal places
             formatters[col] = lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
        elif col == volume_col:
             # Format large numbers for volume
             formatters[col] = lambda x: f"${format_large_number(x)}" if pd.notna(x) else "N/A"
        elif col in rsi_srsi_cols:
             # Format RSI/SRSI with 1 decimal place
             formatters[col] = lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        elif col in macd_cols:
             # Format MACD values with 4 decimal places
             formatters[col] = lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        elif col in ma_vwap_cols:
             # Format MAs/VWAP like price, but maybe fewer decimals (e.g., 2)
             formatters[col] = lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
        # Add more conditions for other column types if necessary

        # Apply the formatter using .style.format - preferred method
        # This avoids converting numeric columns to strings prematurely

    # --- Styling Logic ---
    def highlight_pct_col_style(val):
        """Styles individual percentage values."""
        if isinstance(val, (int, float)) and pd.notna(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'grey'
            return f'color: {color}'
        return '' # Default style for N/A or non-numeric

    def highlight_signal_style(val):
        """Styles individual signal strings."""
        style = 'color: #6c757d;' # Default Hold color
        font_weight = 'normal'
        if isinstance(val, str):
            if "Strong Buy" in val: style = 'color: #198754;'; font_weight = 'bold'; # Dark Green
            elif "Buy" in val and "Strong" not in val: style = 'color: #28a745;' # Lighter Green
            elif "Strong Sell" in val: style = 'color: #dc3545;'; font_weight = 'bold'; # Red
            elif "Sell" in val and "Strong" not in val: style = 'color: #fd7e14;' # Orange-Red
            elif "CTB" in val: style = 'color: #20c997;' # Teal
            elif "CTS" in val: style = 'color: #ffc107;' # Amber/Yellow
            # elif "Hold" in val: style = 'color: #6c757d;' # Grey (default)
            elif "N/D" in val: style = 'color: #adb5bd;' # Lighter Grey
        return f'{style} font-weight: {font_weight};'

    # Apply styles using Styler.applymap
    styled_df = df_display.style.format(formatters, na_rep="N/A") # Apply formatters first

    # Apply conditional styling per column type
    for col in pct_cols:
        if col in df_display.columns:
            styled_df = styled_df.applymap(highlight_pct_col_style, subset=[col])

    if "Gemini Alert" in df_display.columns:
        styled_df = styled_df.applymap(highlight_signal_style, subset=["Gemini Alert"])
    if "GPT Signal" in df_display.columns:
        styled_df = styled_df.applymap(highlight_signal_style, subset=["GPT Signal"])

    # Display the styled DataFrame
    # use_container_width=True ensures it fits the page width
    st.dataframe(styled_df, use_container_width=True)

else:
    st.warning("Nessun risultato crypto da visualizzare. Controllare API o simboli.")


# --- NUOVA SEZIONE NEWS ---
st.markdown("---")
st.subheader("📰 Ultime Notizie Crypto (Cointelegraph Feed)")
news_items = get_crypto_news(NEWS_FEED_URL)

if news_items:
    for item in news_items:
        # Safely get title and link
        title = item.get('title', 'Titolo non disponibile')
        link = item.get('link', '#')

        # Try to parse and format publication date
        pub_date_str = ""
        if hasattr(item, 'published_parsed') and item.published_parsed:
            try:
                # Convert feedparser's time tuple to datetime object
                pub_dt_utc = datetime.fromtimestamp(time.mktime(item.published_parsed))
                # Convert to local time (e.g., Rome)
                try:
                    from zoneinfo import ZoneInfo
                    rome_tz = ZoneInfo("Europe/Rome")
                    pub_dt_local = pub_dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(rome_tz)
                    # Format date nicely (e.g., 01 May, 15:30 CEST)
                    pub_date_str = f" - *{pub_dt_local.strftime('%d %b, %H:%M %Z')}*"
                except ImportError:
                    # Fallback timezone conversion
                    pub_dt_local_approx = pub_dt_utc + timedelta(hours=2) # Approximate CEST
                    pub_date_str = f" - *{pub_dt_local_approx.strftime('%d %b, %H:%M')} (approx)*"
                except Exception: pass # Ignore timezone conversion errors silently
            except Exception: pass # Ignore date parsing errors silently

        # Display news item using Markdown
        st.markdown(f"- [{title}]({link}){pub_date_str}")
else:
    st.warning("Impossibile caricare le notizie dal feed RSS o nessuna notizia trovata.")


# --- Expander per errori/note di Fetch ---
if fetch_errors:
    with st.expander("ℹ️ Note Recupero Dati / Calcolo Indicatori", expanded=False):
        # Show unique errors only
        unique_errors = sorted(list(set(fetch_errors)))
        for error_msg in unique_errors:
            st.info(error_msg) # Use st.info or st.warning based on severity


# --- Legenda Aggiornata ---
st.divider()
with st.expander("📘 Legenda Indicatori Tecnici e Segnali", expanded=False):
    # Using Markdown for better formatting control
    st.markdown("""
    *Disclaimer: Questa dashboard è solo a scopo informativo e non costituisce consulenza finanziaria.*

    **Market Overview:**
    * **Fear & Greed (Alt.):** Indice sentiment da Alternative.me (0=Extreme Fear, 100=Extreme Greed).
    * **Total Crypto M.Cap:** Capitalizzazione totale mercato crypto (CoinGecko).
    * **Crypto ETFs Flow:** Flusso netto giornaliero ETF crypto spot (Dato **N/A**).
    * **S&P 500, Nasdaq, Gold, UVXY, TQQQ:** Prezzi indicativi mercato tradizionale (yfinance).
    * **Titoli Principali:** Prezzi indicativi azioni selezionate (yfinance).

    **Tabella Crypto:**
    * **Variazioni Percentuali (%):** Rispetto a 1h, 24h, 7d, 30d, 1y (dati CoinGecko).
    * **Indicatori Momentum:**
        * **RSI (1h, 1d, 1w, 1mo):** Relative Strength Index. Misura velocità e cambio movimenti prezzo (0-100). Convenzionalmente: `>70` Ipercomprato (Overbought), `<30` Ipervenduto (Oversold). Dati 1w/1mo calcolati da dati daily.
        * **SRSI %K / %D (1d):** Stochastic RSI. Applica Stocastico all'RSI per segnali più frequenti (0-100). `>80` Overbought, `<20` Oversold.
        * **MACD Hist (1d):** Moving Average Convergence Divergence Histogram. Differenza tra linea MACD e linea Segnale. `>0` (verde) Momentum rialzista, `<0` (rosso) Momentum ribassista.
    * **Indicatori Trend:**
        * **MA (SMA - 20d, 50d):** Simple Moving Average. Medie mobili semplici prezzo chiusura. Usate per identificare trend e livelli Supporto/Resistenza. Incrocio (MA20 vs MA50) può segnalare cambio trend (Golden/Death Cross).
        * **VWAP (1d):** Volume-Weighted Average Price. Prezzo medio ponderato per volumi (calcolato su ultimi 14 gg). Considerato da alcuni un indicatore di "fair value" intraday/breve periodo.
    * **Segnali Combinati (Esemplificativi - NON CONSULENZA FINANZIARIA):**
        * **Gemini Alert:** Alert basato su confluenza di segnali **Daily**:
            * `⚡️ Strong Buy`: MA20 > MA50 **e** MACD Hist > 0 **e** RSI < 70.
            * `🚨 Strong Sell`: MA20 < MA50 **e** MACD Hist < 0 **e** RSI > 30.
            * `🟡 Hold`: Condizioni Strong non soddisfatte.
            * `⚪️ N/D`: Dati insufficienti per il calcolo.
        * **GPT Signal:** Punteggio basato su una sintesi pesata di più indicatori (MAs, MACD, RSIs Daily/Weekly/Hourly, SRSI). Interpretazione del punteggio:
            * `⚡️ Strong Buy` (Score >= 5)
            * `🟢 Buy` (Score 2-4)
            * `⏳ CTB` (Score 0-1, RSI basso): Close To Buy - Potenziale inversione rialzista.
            * `🟡 Hold` (Score 0-1, RSI non basso / Score -1-0, RSI non alto)
            * `⚠️ CTS` (Score -1-0, RSI alto): Close To Sell - Potenziale inversione ribassista.
            * `🔴 Sell` (Score -4 to -2)
            * `🚨 Strong Sell` (Score <= -5)
            * `⚪️ N/D`: Dati insufficienti per il calcolo. **Usare con estrema cautela.**
    * **Generale:**
        * **N/A:** Dato non disponibile, non applicabile, o errore nel calcolo/recupero.
    """)

# --- Footer ---
st.divider()
st.caption("Disclaimer: Questa dashboard è solo a scopo informativo e di intrattenimento. Non costituisce consulenza finanziaria. Fare sempre le proprie ricerche (DYOR).")