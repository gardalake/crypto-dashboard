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
# Import zoneinfo for timezone handling if available (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Fallback for older Python versions (though Streamlit Cloud usually has newer versions)
    # Installation of 'tzdata' might be needed in requirements.txt
    st.warning("Modulo 'zoneinfo' non trovato. Usando fallback UTC+2 per Roma. Considera aggiornamento Python o aggiunta 'tzdata' a requirements.txt")
    ZoneInfo = None # Define as None to handle conditional logic later

# --- Configurazione Globale ---

# Mappa Simbolo -> ID CoinGecko per facile gestione
# AGGIUNGI/RIMUOVI/MODIFICA coppie qui per cambiare le coin monitorate
SYMBOL_TO_ID_MAP = {
    # Originali
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "RNDR": "render-token",
    "FET": "fetch-ai", # ID Changed per CoinGecko as of May 2024 (ASI merge not reflected yet typically) - Check this ID!
    "RAY": "raydium", "SUI": "sui", "ONDO": "ondo-finance", # Check 'ondo' vs 'ondo-finance' on CoinGecko
    "ARB": "arbitrum",
    # Nuove Aggiunte
    "TAO": "bittensor", "LINK": "chainlink", "AVAX": "avalanche-2",
    "HBAR": "hedera-hashgraph", "PEPE": "pepe", "UNI": "uniswap",
    "TIA": "celestia", "JUP": "jupiter-aggregator", # Assunto Jupiter Aggregator - Check this ID!
    "IMX": "immutable-x", "TRUMP": "maga", # Assunto MAGA (TRUMP) - Check this ID!
    "NEAR": "near", # Check 'near' vs 'near-protocol' on CoinGecko
    "AERO": "aerodrome-finance", "TRON": "tron",
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
CACHE_TTL = 1800 # Cache di 30 minuti (1800 sec)
DAYS_HISTORY_DAILY = 365; DAYS_HISTORY_HOURLY = 7
RSI_PERIOD = 14; SRSI_PERIOD = 14; SRSI_K = 3; SRSI_D = 3
MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9
MA_SHORT = 20; MA_LONG = 50; VWAP_PERIOD = 14

# URL Feed RSS per notizie (Coingelegraph)
NEWS_FEED_URL = "https://cointelegraph.com/rss"
NUM_NEWS_ITEMS = 5 # Numero di notizie da mostrare

# --- Password Protection ---
# Ensure password check function is robust
def check_password():
    # Initialize session state if not already done
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    # If password not yet correct, show input field
    if not st.session_state.password_correct:
        password = st.text_input("🔑 Password", type="password", key="password_input")

        # Check if the entered password is correct
        # Replace "Leonardo" with your actual, secure password
        # Consider using environment variables for secrets in deployed apps: st.secrets["app_password"]
        correct_password = "Leonardo" # Replace with your password or st.secrets["PASSWORD"]
        if password == correct_password:
            st.session_state.password_correct = True
            # Use query params to prevent immediate rerun loop after password entry
            if st.query_params.get("logged_in") != "true":
                st.query_params["logged_in"] = "true"
                st.rerun() # Rerun to clear password field and proceed
        # Only show warning if a password was entered and it was wrong
        elif password:
            st.warning("Password errata.")
            st.stop() # Stop execution if password incorrect
        # Stop execution if no password entered yet (initial load)
        else:
            st.stop()
    # Return True if password check passed or already correct
    return True

# Check password at the very start of the script execution
if not check_password():
    st.stop() # Stop if password check fails

# --- Funzione Helper Formattazione Numeri Grandi ---
def format_large_number(num):
    if pd.isna(num) or not isinstance(num, (int, float)): return "N/A"
    if abs(num) < 1_000_000: return f"{num:,.0f}" # Format thousands with comma
    elif abs(num) < 1_000_000_000: return f"{num / 1_000_000:.1f}M"
    elif abs(num) < 1_000_000_000_000: return f"{num / 1_000_000_000:.1f}B"
    else: return f"{num / 1_000_000_000_000:.2f}T"

# --- Funzioni API CoinGecko ---
@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento dati di mercato (CoinGecko)...")
def get_coingecko_market_data(ids_list, currency):
    ids_string = ",".join(ids_list)
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': currency,
        'ids': ids_string,
        'order': 'market_cap_desc',
        'per_page': str(len(ids_list)),
        'page': 1,
        'sparkline': False,
        'price_change_percentage': '1h,24h,7d,30d,1y', # Request needed percentages
        'precision': 'full' # Request full precision for prices
    }
    timestamp = datetime.now(ZoneInfo("UTC") if ZoneInfo else None) # Record timestamp in UTC
    try:
        response = requests.get(url, params=params, timeout=20) # Increased timeout slightly
        response.raise_for_status() # Raises HTTPError for bad responses (4XX, 5XX)
        data = response.json()
        if not data: # Handle empty response list
             st.warning("API CoinGecko ha restituito dati vuoti per il mercato live.")
             return pd.DataFrame(), timestamp
        df = pd.DataFrame(data)
        if not df.empty:
             # Set index AFTER ensuring df is not empty
             df.set_index('id', inplace=True)
        return df, timestamp
    except requests.exceptions.HTTPError as http_err:
        # Specific handling for 429 Rate Limit error
        if http_err.response.status_code == 429:
             st.error("Errore API CoinGecko: Limite richieste (429) raggiunto. Attendi qualche minuto prima di aggiornare.")
        else:
             st.error(f"Errore HTTP API Mercato CoinGecko (Status: {http_err.response.status_code}): {http_err}")
        return pd.DataFrame(), timestamp # Return empty DataFrame on error
    except requests.exceptions.RequestException as req_ex:
        # Handle other request errors (timeout, connection issues)
        st.error(f"Errore Richiesta API Mercato CoinGecko: {req_ex}")
        return pd.DataFrame(), timestamp
    except Exception as e:
        # Handle unexpected errors during processing
        st.error(f"Errore Processamento Dati Mercato CoinGecko: {e}")
        return pd.DataFrame(), timestamp

@st.cache_data(ttl=CACHE_TTL * 2, show_spinner=False) # Longer cache for historical
def get_coingecko_historical_data(coin_id, currency, days, interval='daily'):
    # Add a small delay before each historical request to avoid rapid firing
    time.sleep(1.2) # Delay reduced slightly, but still present
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': currency,
        'days': str(days),
        'interval': interval if interval == 'hourly' else 'daily', # Use 'daily' or 'hourly'
        'precision': 'full' # Request full precision
    }
    try:
        response = requests.get(url, params=params, timeout=25) # Slightly longer timeout for potentially larger data
        response.raise_for_status()
        data = response.json()
        # Check if expected keys exist and are not empty
        if not data or 'prices' not in data or not data['prices']:
             # Return specific status if no price data available
             return pd.DataFrame(), f"No Prices Data ({coin_id})"

        # Process Prices
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms', utc=True) # Ensure UTC
        prices_df.set_index('timestamp', inplace=True)

        # Process Volumes if available
        if 'total_volumes' in data and data['total_volumes']:
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms', utc=True) # Ensure UTC
            volumes_df.set_index('timestamp', inplace=True)
            # Join prices and volumes
            hist_df = prices_df.join(volumes_df, how='outer')
        else:
            # If no volume data, create volume column with zeros
            hist_df = prices_df.copy()
            hist_df['volume'] = 0.0

        # Interpolate missing values (e.g., volume for a timestamp with price)
        hist_df = hist_df.interpolate(method='time')
        # Forward fill then backward fill remaining NaNs
        hist_df = hist_df.ffill().bfill()

        # Approximate OHLC data (since API doesn't provide it directly in market_chart)
        # Note: This is a simplification. Real OHLC would require more granular data.
        hist_df['high'] = hist_df['close']
        hist_df['low'] = hist_df['close']
        hist_df['open'] = hist_df['close'].shift(1)
        # Fill the first NaN 'open' value
        hist_df['open'].fillna(hist_df['close'], inplace=True)


        # Ensure no duplicate indices remain and index is sorted
        hist_df = hist_df[~hist_df.index.duplicated(keep='last')].sort_index()

        # Drop rows where 'close' price is still NaN after processing
        hist_df.dropna(subset=['close'], inplace=True)

        if hist_df.empty:
            return pd.DataFrame(), f"Processed Empty ({coin_id})"

        return hist_df, "Success"

    except requests.exceptions.HTTPError as http_err:
        # Provide more context in error messages
        if http_err.response.status_code == 429:
            return pd.DataFrame(), f"Rate Limited (429) ({coin_id})"
        elif http_err.response.status_code == 404:
            return pd.DataFrame(), f"Not Found (404) ({coin_id})"
        else:
            return pd.DataFrame(), f"HTTP Error {http_err.response.status_code} ({coin_id})"
    except requests.exceptions.RequestException as req_ex:
        return pd.DataFrame(), f"Request Error ({req_ex}) ({coin_id})"
    except Exception as e:
        return pd.DataFrame(), f"Generic Error ({type(e).__name__}) ({coin_id})"


# --- Funzioni Dati Mercato Generale ---
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_fear_greed_index():
    # Using Alternative.me API as it's generally more reliable without keys
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and data.get("data") and isinstance(data["data"], list) and len(data["data"]) > 0:
             latest_data = data["data"][0]
             value = latest_data.get("value")
             desc = latest_data.get("value_classification")
             if value is not None and desc is not None:
                 # Ensure value is integer before formatting
                 return f"{int(value)} ({desc})"
        return "N/A" # Return N/A if data format is unexpected
    except requests.exceptions.RequestException as req_ex:
        status_code = req_ex.response.status_code if req_ex.response is not None else "N/A"
        st.warning(f"Errore F&G Index (Alt.me Status: {status_code}): {req_ex}")
        return "N/A"
    except Exception as e:
        st.warning(f"Errore Processamento F&G Index (Alt.me): {e}")
        return "N/A"


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_global_market_data_cg(currency):
    url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get('data', {})
        total_mcap = data.get('total_market_cap', {}).get(currency.lower(), np.nan)
        # Optionally get BTC dominance or other global metrics
        # btc_dominance = data.get('market_cap_percentage', {}).get('btc', np.nan)
        return total_mcap #, btc_dominance
    except requests.exceptions.RequestException as req_ex:
        st.warning(f"Errore API Global CoinGecko: {req_ex}")
        return np.nan #, np.nan
    except Exception as e:
        st.warning(f"Errore Processamento Global CoinGecko: {e}")
        return np.nan #, np.nan

# Placeholder for ETF flow - requires a reliable (often paid) data source
def get_etf_flow(): return "N/A"

@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento dati mercato tradizionale (Yahoo Finance)...")
def get_traditional_market_data_yf(tickers):
    data = {}
    # Download data more efficiently using yfinance's bulk download
    try:
        yf_data = yf.download(tickers, period='5d', interval='1d', progress=False)
        if yf_data.empty:
            st.warning("yfinance non ha restituito dati per i ticker tradizionali.")
            return {ticker: np.nan for ticker in tickers}

        # Get the last closing price for each ticker
        last_close = yf_data['Close'].iloc[-1]
        data = last_close.to_dict()

        # Fill any missing tickers with NaN (if download failed for some)
        for ticker in tickers:
            if ticker not in data:
                data[ticker] = np.nan
                st.warning(f"Dati non trovati per il ticker tradizionale: {ticker}")

    except Exception as e:
        st.error(f"Errore durante il recupero dati da yfinance: {e}")
        # Fallback: return NaN for all tickers on error
        data = {ticker: np.nan for ticker in tickers}

    return data


# --- NUOVA Funzione per News RSS ---
@st.cache_data(ttl=900, show_spinner="Caricamento notizie...") # Cache 15 min per news
def get_crypto_news(feed_url, num_items=NUM_NEWS_ITEMS):
    """Recupera e parsifica un feed RSS."""
    try:
        # Use a timeout for the feed parsing request
        feed = feedparser.parse(feed_url, request_headers={'User-Agent': 'Mozilla/5.0'}) # Add user agent
        # Check for errors during parsing more robustly
        if feed.bozo:
            # Log warning but proceed if possible, check exception type
            exc = feed.get('bozo_exception', Exception('Unknown feedparser error'))
            st.warning(f"Possibile errore/warning nel parsing del feed RSS ({feed_url}): {exc}")
        # Check if entries exist before slicing
        if not feed.entries:
             st.info(f"Nessuna notizia trovata nel feed RSS: {feed_url}")
             return []
        return feed.entries[:num_items] # Ritorna le prime N notizie
    except Exception as e:
        # Catch broader exceptions during the request/parsing
        st.error(f"Errore grave durante il recupero/parsing del feed RSS ({feed_url}): {e}")
        return [] # Ritorna lista vuota in caso di errore


# --- Funzioni Calcolo Indicatori (Manuali con validazione input) ---
def calculate_rsi_manual(series, period=RSI_PERIOD):
    # Validate input
    if not isinstance(series, pd.Series) or series.empty or series.isna().all():
        return np.nan
    series = series.dropna()
    if len(series) < period + 1:
        return np.nan # Need at least period+1 points for diff

    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rename("gain")
    loss = -delta.where(delta < 0, 0.0).rename("loss")

    # Use Exponential Moving Average (EMA) for RSI calculation
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Check if EMA calculation produced results before accessing iloc[-1]
    if avg_gain.isna().all() or avg_loss.isna().all():
        return np.nan

    last_avg_loss = avg_loss.iloc[-1]
    last_avg_gain = avg_gain.iloc[-1]

    # Avoid division by zero
    if last_avg_loss == 0:
        return 100.0 if last_avg_gain > 0 else 50.0 # RSI is 100 if only gains, 50 if no change

    rs = last_avg_gain / last_avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Clamp RSI between 0 and 100 as floating point math can sometimes exceed bounds
    return max(0.0, min(100.0, rsi))


def calculate_stoch_rsi(series, rsi_period=RSI_PERIOD, stoch_period=SRSI_PERIOD, k_smooth=SRSI_K, d_smooth=SRSI_D):
    # Validate input
    if not isinstance(series, pd.Series) or series.empty or series.isna().all():
        return np.nan, np.nan
    series = series.dropna()
    if len(series) < rsi_period + stoch_period: # Adjusted minimum length check
        return np.nan, np.nan

    # Calculate RSI Series first (reusing calculate_rsi_manual logic is complex, recalculate here)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    # Handle potential division by zero for RS calculation
    rs = avg_gain / avg_loss.replace(0, np.nan) # Replace 0 avg_loss with NaN
    rsi_series = (100.0 - (100.0 / (1.0 + rs))).dropna()

    if len(rsi_series) < stoch_period:
        return np.nan, np.nan # Need enough RSI values for Stoch calc

    # Calculate Stochastic RSI components
    min_rsi = rsi_series.rolling(window=stoch_period).min()
    max_rsi = rsi_series.rolling(window=stoch_period).max()
    range_rsi = max_rsi - min_rsi

    # Avoid division by zero if min == max over the period
    stoch_rsi_k_raw = 100 * (rsi_series - min_rsi) / range_rsi.replace(0, np.nan)
    stoch_rsi_k_raw = stoch_rsi_k_raw.dropna() # Drop NaNs resulting from division by zero or initial NaNs

    # Need enough data points for smoothing windows
    if len(stoch_rsi_k_raw) < k_smooth : return np.nan, np.nan

    # Smooth %K and %D using Simple Moving Average (SMA)
    stoch_rsi_k = stoch_rsi_k_raw.rolling(window=k_smooth).mean()
    if len(stoch_rsi_k.dropna()) < d_smooth :
         # Return last K if D cannot be calculated
         return stoch_rsi_k.iloc[-1] if not pd.isna(stoch_rsi_k.iloc[-1]) else np.nan, np.nan

    stoch_rsi_d = stoch_rsi_k.rolling(window=d_smooth).mean()

    # Get the last valid values, clamping between 0 and 100
    last_k = stoch_rsi_k.iloc[-1]
    last_d = stoch_rsi_d.iloc[-1]
    last_k = max(0.0, min(100.0, last_k)) if pd.notna(last_k) else np.nan
    last_d = max(0.0, min(100.0, last_d)) if pd.notna(last_d) else np.nan

    return last_k, last_d


def calculate_macd_manual(series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    # Validate input
    if not isinstance(series, pd.Series) or series.empty or series.isna().all():
        return np.nan, np.nan, np.nan
    series = series.dropna()
    if len(series) < slow + signal -1 : # Ensure enough data for all calculations
        return np.nan, np.nan, np.nan

    # Calculate EMAs
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    # Calculate MACD Line
    macd_line = ema_fast - ema_slow

    # Calculate Signal Line (EMA of MACD Line)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    # Calculate Histogram
    histogram = macd_line - signal_line

    # Get the last valid values
    last_macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else np.nan
    last_signal = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else np.nan
    last_hist = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else np.nan

    return last_macd, last_signal, last_hist


def calculate_sma_manual(series, period):
    # Validate input
    if not isinstance(series, pd.Series) or series.empty or series.isna().all():
        return np.nan
    series = series.dropna()
    if len(series) < period:
        return np.nan # Not enough data points for the period
    # Calculate SMA and return the last value
    return series.rolling(window=period).mean().iloc[-1]


def calculate_vwap_manual(df, period=VWAP_PERIOD):
    # Validate input DataFrame and columns
    required_cols = ['close', 'volume']
    if not isinstance(df, pd.DataFrame) or df.empty or not all(col in df.columns for col in required_cols):
        return np.nan
    # Drop rows where essential data is missing for the calculation period
    df_valid = df.dropna(subset=required_cols)
    if len(df_valid) < period:
        return np.nan # Not enough valid data points

    # Select the last 'period' rows with valid data
    df_period = df_valid.iloc[-period:]

    # Check for zero volume sum to avoid division by zero
    total_volume = df_period['volume'].sum()
    if total_volume == 0:
        # If volume is zero, VWAP is undefined; could return last close or NaN
        return df_period['close'].iloc[-1] if not df_period.empty else np.nan

    # Calculate VWAP: Sum(Price * Volume) / Sum(Volume)
    vwap = (df_period['close'] * df_period['volume']).sum() / total_volume
    return vwap


# --- Funzione Raggruppata Indicatori ---
def compute_all_indicators(symbol, hist_daily_df, hist_hourly_df, fetch_errors_list):
    """Computes all technical indicators for a given symbol."""
    indicators = {
        "RSI (1h)": np.nan, "RSI (1d)": np.nan, "RSI (1w)": np.nan, "RSI (1mo)": np.nan,
        "SRSI %K (1d)": np.nan, "SRSI %D (1d)": np.nan,
        "MACD Line (1d)": np.nan, "MACD Signal (1d)": np.nan, "MACD Hist (1d)": np.nan,
        f"MA({MA_SHORT}d)": np.nan, f"MA({MA_LONG}d)": np.nan,
        "VWAP (1d)": np.nan,
        # Placeholders for future/other indicators
        "Doda Stoch": "N/A", "GChannel": "N/A", "Volume Flow": "N/A"
    }
    # Define minimum data lengths required for reliable calculations
    # These can be fine-tuned based on indicator requirements
    min_len_daily_full = max(MACD_SLOW + MACD_SIGNAL, MA_LONG) + 5 # Add buffer
    min_len_rsi_base = RSI_PERIOD + 1
    min_len_srsi_base = RSI_PERIOD + SRSI_PERIOD + 5 # Add buffer
    min_len_macd_base = MACD_SLOW + MACD_SIGNAL + 5 # Add buffer

    # --- Daily Indicators ---
    if not hist_daily_df.empty and 'close' in hist_daily_df.columns:
        close_daily = hist_daily_df['close'].dropna()
        len_daily = len(close_daily)

        # Check overall length for comprehensive calculations
        if len_daily >= min_len_daily_full:
            indicators["RSI (1d)"] = calculate_rsi_manual(close_daily, RSI_PERIOD)
            indicators["SRSI %K (1d)"], indicators["SRSI %D (1d)"] = calculate_stoch_rsi(close_daily, RSI_PERIOD, SRSI_PERIOD, SRSI_K, SRSI_D)
            macd_l, macd_s, macd_h = calculate_macd_manual(close_daily, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            indicators["MACD Line (1d)"] = macd_l
            indicators["MACD Signal (1d)"] = macd_s
            indicators["MACD Hist (1d)"] = macd_h
            indicators[f"MA({MA_SHORT}d)"] = calculate_sma_manual(close_daily, MA_SHORT)
            indicators[f"MA({MA_LONG}d)"] = calculate_sma_manual(close_daily, MA_LONG)
            indicators["VWAP (1d)"] = calculate_vwap_manual(hist_daily_df, VWAP_PERIOD) # VWAP needs full df
        else:
            # If not enough data for all, try calculating individually if possible
            fetch_errors_list.append(f"{symbol}: Dati Daily insuff. ({len_daily}/{min_len_daily_full}) per tutti gli ind.")
            if len_daily >= min_len_rsi_base: indicators["RSI (1d)"] = calculate_rsi_manual(close_daily, RSI_PERIOD)
            if len_daily >= min_len_srsi_base: indicators["SRSI %K (1d)"], indicators["SRSI %D (1d)"] = calculate_stoch_rsi(close_daily, RSI_PERIOD, SRSI_PERIOD, SRSI_K, SRSI_D)
            if len_daily >= min_len_macd_base:
                 macd_l, macd_s, macd_h = calculate_macd_manual(close_daily, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
                 indicators["MACD Line (1d)"] = macd_l; indicators["MACD Signal (1d)"] = macd_s; indicators["MACD Hist (1d)"] = macd_h
            if len_daily >= MA_SHORT: indicators[f"MA({MA_SHORT}d)"] = calculate_sma_manual(close_daily, MA_SHORT)
            if len_daily >= MA_LONG: indicators[f"MA({MA_LONG}d)"] = calculate_sma_manual(close_daily, MA_LONG)
            if len_daily >= VWAP_PERIOD: indicators["VWAP (1d)"] = calculate_vwap_manual(hist_daily_df, VWAP_PERIOD)


        # --- Weekly & Monthly RSI from Daily Data ---
        if len_daily > min_len_rsi_base and pd.api.types.is_datetime64_any_dtype(close_daily.index):
            try: # Weekly RSI
                # Resample daily closes to weekly (Monday end), get last price of week
                df_weekly = close_daily.resample('W-MON').last()
                if len(df_weekly.dropna()) >= min_len_rsi_base:
                    indicators["RSI (1w)"] = calculate_rsi_manual(df_weekly, RSI_PERIOD)
                else: fetch_errors_list.append(f"{symbol}: Dati Weekly insuff. ({len(df_weekly.dropna())}/{min_len_rsi_base}) per RSI 1w.")
            except Exception as e: fetch_errors_list.append(f"{symbol}: Errore resampling/calcolo RSI weekly: {e}")

            try: # Monthly RSI
                # Resample daily closes to monthly end, get last price of month
                df_monthly = close_daily.resample('ME').last() # 'ME' for Month End
                if len(df_monthly.dropna()) >= min_len_rsi_base:
                    indicators["RSI (1mo)"] = calculate_rsi_manual(df_monthly, RSI_PERIOD)
                else: fetch_errors_list.append(f"{symbol}: Dati Monthly insuff. ({len(df_monthly.dropna())}/{min_len_rsi_base}) per RSI 1mo.")
            except Exception as e: fetch_errors_list.append(f"{symbol}: Errore resampling/calcolo RSI monthly: {e}")
        elif len_daily <= min_len_rsi_base:
             fetch_errors_list.append(f"{symbol}: Dati Daily insuff. per calcolare RSI Weekly/Monthly.")

    # --- Hourly RSI ---
    if not hist_hourly_df.empty and 'close' in hist_hourly_df.columns:
        close_hourly = hist_hourly_df['close'].dropna()
        len_hourly = len(close_hourly)
        if len_hourly >= min_len_rsi_base:
            indicators["RSI (1h)"] = calculate_rsi_manual(close_hourly, RSI_PERIOD)
        else:
            fetch_errors_list.append(f"{symbol}: Dati Hourly insuff. ({len_hourly}/{min_len_rsi_base}) per RSI 1h.")

    return indicators


# --- Funzioni Segnale (Sintassi Corretta e Logica Raffinata) ---
def generate_gpt_signal(rsi_1d, rsi_1h, rsi_1w, macd_hist, ma_short, ma_long, srsi_k, srsi_d, current_price):
    """Generates a composite signal based on multiple indicators."""
    # Define required inputs for the core signal logic
    required_inputs = [rsi_1d, macd_hist, ma_short, ma_long, current_price]
    # Check if any essential input is NaN (Not a Number)
    if any(pd.isna(x) for x in required_inputs):
        return "⚪️ N/D" # Not Determinable if core data missing

    # Initialize signal score
    score = 0

    # Factor 1: Price vs Long MA (Basic Trend Filter)
    if current_price > ma_long:
        score += 1
    else: # current_price <= ma_long
        score -= 1

    # Factor 2: MA Cross (Trend Confirmation)
    if ma_short > ma_long:
        score += 2 # Golden cross tendency
    else: # ma_short <= ma_long
        score -= 2 # Death cross tendency

    # Factor 3: MACD Histogram (Momentum)
    if macd_hist > 0:
        score += 2 # Bullish momentum
    else: # macd_hist <= 0
        score -= 2 # Bearish momentum

    # Factor 4: Daily RSI (Overbought/Oversold)
    if rsi_1d < 30:
        score += 2 # Clearly Oversold
    elif rsi_1d < 40:
        score += 1 # Approaching Oversold
    elif rsi_1d > 70:
        score -= 2 # Clearly Overbought
    elif rsi_1d > 60:
        score -= 1 # Approaching Overbought

    # Factor 5: Weekly RSI (Longer-term Confirmation - Bonus)
    if pd.notna(rsi_1w): # Only consider if weekly RSI is available
        if rsi_1w < 40: # Using 40 for weekly oversold influence
            score += 1
        elif rsi_1w > 60: # Using 60 for weekly overbought influence
            score -= 1

    # Factor 6: Hourly RSI (Shorter-term Timing - Fine-tuning)
    if pd.notna(rsi_1h): # Only consider if hourly RSI is available
        if rsi_1h < 30: # Hourly oversold might suggest bounce
            score += 1
        elif rsi_1h > 70: # Hourly overbought might suggest pullback
            score -= 1

    # Factor 7: Stochastic RSI (Overbought/Oversold Confirmation)
    if pd.notna(srsi_k) and pd.notna(srsi_d): # Ensure both K and D are available
        # SRSI Oversold condition
        if srsi_k < 20 and srsi_d < 20:
             # Optional: Check if K is crossing D upwards for stronger signal?
             # if srsi_k > srsi_d: score += 1 # Example: add points if K crosses D up
             score += 1
        # SRSI Overbought condition
        elif srsi_k > 80 and srsi_d > 80:
             # Optional: Check if K is crossing D downwards?
             # if srsi_k < srsi_d: score -= 1
             score -= 1

    # Determine Signal based on Final Score Thresholds
    if score >= 5: return "⚡️ Strong Buy"
    elif score >= 2: return "🟢 Buy"
    elif score <= -5: return "🚨 Strong Sell"
    elif score <= -2: return "🔴 Sell"
    # Nuanced Hold conditions based on score and RSI context
    elif score > 0: # Slightly Bullish Score (e.g., 0, 1)
        # If score is weakly positive AND daily RSI is NOT overbought, lean towards CTB/Hold
        return "⏳ CTB" if pd.notna(rsi_1d) and rsi_1d < 55 else "🟡 Hold"
    else: # Slightly Bearish Score (e.g., -1)
        # If score is weakly negative AND daily RSI is NOT oversold, lean towards CTS/Hold
        return "⚠️ CTS" if pd.notna(rsi_1d) and rsi_1d > 45 else "🟡 Hold"


def generate_gemini_alert(ma_short, ma_long, macd_hist, rsi_1d):
    """Generates a specific alert based on Daily MA cross, MACD, and RSI."""
    # Check for missing essential data
    if pd.isna(ma_short) or pd.isna(ma_long) or pd.isna(macd_hist) or pd.isna(rsi_1d):
        return "⚪️ N/D"

    # Define conditions clearly for readability
    is_uptrend_ma = ma_short > ma_long
    is_momentum_positive = macd_hist > 0
    # Relaxed RSI condition for Buy: not extremely overbought
    is_not_extremely_overbought = rsi_1d < 80

    is_downtrend_ma = ma_short < ma_long
    is_momentum_negative = macd_hist < 0
    # Relaxed RSI condition for Sell: not extremely oversold
    is_not_extremely_oversold = rsi_1d > 20

    # Combine conditions for alerts
    # Strong Buy: Uptrend MA, Positive Momentum, Not extremely Overbought RSI
    if is_uptrend_ma and is_momentum_positive and is_not_extremely_overbought:
        return "⚡️ Strong Buy"
    # Strong Sell: Downtrend MA, Negative Momentum, Not extremely Oversold RSI
    elif is_downtrend_ma and is_momentum_negative and is_not_extremely_oversold:
        return "🚨 Strong Sell"
    # Otherwise, default to Hold
    else:
        return "🟡 Hold"


# --- Layout App Streamlit ---
st.set_page_config(layout="wide", page_title="Crypto Technical Dashboard Pro", page_icon="📈")

# --- Titolo e Bottone Aggiorna ---
col_title, col_button_placeholder, col_button = st.columns([4, 1, 1])
with col_title: st.title("📈 Crypto Technical Dashboard Pro")
with col_button:
    st.write("") # Spacer for alignment
    if st.button("🔄 Aggiorna", help=f"Forza aggiornamento dati (cache max {CACHE_TTL/60:.0f} min)", key="refresh_button"):
        # Clear specific caches if needed, or all data caches
        st.cache_data.clear()
        # Clear query params to allow potential rerun after password re-entry if needed
        st.query_params.clear()
        st.rerun()

# Timestamp placeholder - updated after data fetch
last_update_placeholder = st.empty()
# Display cache info
st.caption(f"Cache: Dati Mercato Live ({CACHE_TTL/60:.0f} min), Storico ({CACHE_TTL*2/60:.0f} min), Tradizionale ({CACHE_TTL/60:.0f} min), Notizie (15 min).")

# --- Sezione Market Overview ---
st.markdown("---")
st.subheader("🌐 Market Overview")

# Fetch general market data
fear_greed_value = get_fear_greed_index()
total_market_cap = get_global_market_data_cg(VS_CURRENCY)
# btc_dominance_value = get_global_market_data_cg(VS_CURRENCY)[1] # If dominance is returned
etf_flow_value = get_etf_flow() # Placeholder
traditional_market_data = get_traditional_market_data_yf(TRAD_TICKERS)

# --- Riga 1 Overview --- (Adjust columns as needed)
mkt_col1, mkt_col2, mkt_col3, mkt_col4, mkt_col5 = st.columns(5)
with mkt_col1: st.metric(label="Fear & Greed Index", value=fear_greed_value, help="Fonte: Alternative.me (0=Paura Estrema, 100=Euforia Estrema)")
with mkt_col2: st.metric(label=f"Total Crypto M.Cap ({VS_CURRENCY.upper()})", value=f"${format_large_number(total_market_cap)}", help="Fonte: CoinGecko Global")
# with mkt_col_new: st.metric(label="BTC Dominance", value=f"{btc_dominance_value:.1f}%" if pd.notna(btc_dominance_value) else "N/A", help="Fonte: CoinGecko Global")
with mkt_col3: st.metric(label="Crypto ETFs Flow (Daily)", value=etf_flow_value, help="Dato N/A - Fonte non disponibile gratuitamente.")
with mkt_col4: st.metric(label="S&P 500 (^GSPC)", value=f"{traditional_market_data.get('^GSPC', np.nan):,.2f}" if pd.notna(traditional_market_data.get('^GSPC')) else "N/A")
with mkt_col5: st.metric(label="Nasdaq (^IXIC)", value=f"{traditional_market_data.get('^IXIC', np.nan):,.2f}" if pd.notna(traditional_market_data.get('^IXIC')) else "N/A")

# --- Riga 2 Overview ---
mkt_col6, mkt_col7, mkt_col8 = st.columns(3)
with mkt_col6: st.metric(label="Gold (GC=F)", value=f"{traditional_market_data.get('GC=F', np.nan):,.2f}" if pd.notna(traditional_market_data.get('GC=F')) else "N/A")
with mkt_col7: st.metric(label="UVXY (Volatility)", value=f"{traditional_market_data.get('UVXY', np.nan):,.2f}" if pd.notna(traditional_market_data.get('UVXY')) else "N/A")
with mkt_col8: st.metric(label="TQQQ (Nasdaq 3x)", value=f"{traditional_market_data.get('TQQQ', np.nan):,.2f}" if pd.notna(traditional_market_data.get('TQQQ')) else "N/A")

# --- Riga 3 Overview: Azioni Principali ---
st.markdown("<h6>Titoli Principali (Prezzi):</h6>", unsafe_allow_html=True)
stock_col1, stock_col2, stock_col3, stock_col4 = st.columns(4)
# Ensure these tickers are in TRAD_TICKERS list
stock_tickers_row = ['NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR']
cols_stock = [stock_col1, stock_col2, stock_col3, stock_col4] * ( (len(stock_tickers_row) + 3) // 4 ) # Distribute tickers

for idx, ticker in enumerate(stock_tickers_row):
    col_index = idx % 4
    # Use the correct column from the list
    current_col = cols_stock[col_index]
    with current_col:
         price = traditional_market_data.get(ticker, np.nan)
         st.metric(label=ticker, value=f"{price:,.2f}" if pd.notna(price) else "N/A") # Display price

st.markdown("---")

# --- Logica Principale Dashboard Crypto ---
st.subheader(f"📊 Analisi Tecnica Crypto ({NUM_COINS} Asset)")

# Fetch live market data for all selected coins
market_data_df, last_cg_update_utc = get_coingecko_market_data(COINGECKO_IDS_LIST, VS_CURRENCY)

# Display timestamp robustly using zoneinfo
if last_cg_update_utc and ZoneInfo:
    try:
        # Current location: San Donato Milanese -> Europe/Rome timezone
        local_tz = ZoneInfo("Europe/Rome")
        last_cg_update_local = last_cg_update_utc.astimezone(local_tz)
        # Format timestamp including timezone name (e.g., CEST)
        last_update_placeholder.markdown(f"*Dati live aggiornati alle: **{last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S %Z')}***")
    except Exception as e:
        last_update_placeholder.markdown(f"*Errore conversione timestamp: {e}*")
# Fallback if zoneinfo failed or timestamp missing
elif last_cg_update_utc:
     last_cg_update_rome_approx = last_cg_update_utc + timedelta(hours=2) # Approximate CEST
     last_update_placeholder.markdown(f"*Dati live aggiornati alle: **{last_cg_update_rome_approx.strftime('%Y-%m-%d %H:%M:%S')} (Ora approx. Roma)***")
else:
     last_update_placeholder.markdown("*Timestamp aggiornamento dati live non disponibile.*")


# Check if market data fetch failed (e.g., due to 429 error)
if market_data_df.empty:
    st.warning("Impossibile caricare i dati di mercato live. La tabella non può essere generata. Riprova più tardi.")
    # Optionally display fetch errors collected so far
    # if fetch_errors: ... (code below handles this)
    st.stop() # Stop execution if core data is missing

# --- Processing Loop for Each Coin ---
results = [] # List to store results for each coin
fetch_errors = [] # List to collect errors during historical fetch/indicator calc

# Use st.spinner for better progress indication during the loop
with st.spinner(f"Recupero dati storici e calcolo indicatori per {NUM_COINS} crypto..."):
    # Ensure processing happens based on the order received from API (usually market cap)
    coin_ids_ordered = market_data_df.index.tolist()

    for i, coin_id in enumerate(coin_ids_ordered):
        # Check if data for this coin_id exists in the fetched market data
        if coin_id not in market_data_df.index:
            fetch_errors.append(f"{coin_id}: Dati live non trovati nel batch fetch.")
            continue # Skip to the next coin

        live_data = market_data_df.loc[coin_id]

        # Safely get live data points
        symbol = live_data.get('symbol', coin_id).upper()
        name = live_data.get('name', coin_id)
        rank = live_data.get('market_cap_rank', 'N/A')
        current_price = live_data.get('current_price', np.nan)
        volume_24h = live_data.get('total_volume', np.nan)
        change_1h = live_data.get('price_change_percentage_1h_in_currency', np.nan)
        change_24h = live_data.get('price_change_percentage_24h_in_currency', np.nan)
        change_7d = live_data.get('price_change_percentage_7d_in_currency', np.nan)
        change_30d = live_data.get('price_change_percentage_30d_in_currency', np.nan)
        change_1y = live_data.get('price_change_percentage_1y_in_currency', np.nan)

        # Fetch historical data (errors handled within the function)
        hist_daily_df, status_daily = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_DAILY, interval='daily')
        hist_hourly_df, status_hourly = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_HOURLY, interval='hourly')

        # Append fetch status messages to errors list *only if not success*
        if status_daily != "Success": fetch_errors.append(f"{symbol}: Daily - {status_daily}")
        if status_hourly != "Success": fetch_errors.append(f"{symbol}: Hourly - {status_hourly}")

        # Compute indicators (pass the errors list to potentially add more)
        indicators = compute_all_indicators(symbol, hist_daily_df, hist_hourly_df, fetch_errors)

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

        # Assemble results row, ensuring all keys exist
        results.append({
            "Rank": rank, "Symbol": symbol, "Name": name,
            "Gemini Alert": gemini_alert, "GPT Signal": gpt_signal,
            f"Prezzo ({VS_CURRENCY.upper()})": current_price,
            "% 1h": change_1h, "% 24h": change_24h, "% 7d": change_7d, "% 30d": change_30d, "% 1y": change_1y,
            "RSI (1h)": indicators.get("RSI (1h)"), "RSI (1d)": indicators.get("RSI (1d)"),
            "RSI (1w)": indicators.get("RSI (1w)"), "RSI (1mo)": indicators.get("RSI (1mo)"),
            "SRSI %K (1d)": indicators.get("SRSI %K (1d)"), "SRSI %D (1d)": indicators.get("SRSI %D (1d)"),
            "MACD Hist (1d)": indicators.get("MACD Hist (1d)"),
            f"MA({MA_SHORT}d)": indicators.get(f"MA({MA_SHORT}d)"), f"MA({MA_LONG}d)": indicators.get(f"MA({MA_LONG}d)"),
            "VWAP (1d)": indicators.get("VWAP (1d)"),
            f"Volume 24h ({VS_CURRENCY.upper()})": volume_24h,
            # Ensure placeholder keys are present if needed for column consistency
            # "Doda Stoch": indicators.get("Doda Stoch", "N/A"),
            # "GChannel": indicators.get("GChannel", "N/A"),
            # "Volume Flow": indicators.get("Volume Flow", "N/A"),
        })
        # Update spinner text (optional, might slow down if too frequent)
        # st.spinner(f"Processing {symbol} ({i+1}/{NUM_COINS})...")


# --- Crea e Visualizza DataFrame ---
if results:
    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(results)

    # Set Rank as index (handle potential errors like non-unique ranks)
    try:
        # Convert Rank to numeric if possible, coercing errors to NaN
        df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
        # Drop rows where Rank became NaN if necessary, or handle them
        # df.dropna(subset=['Rank'], inplace=True)
        df.set_index('Rank', inplace=True, drop=True)
        # Sort by Rank (numeric index)
        df.sort_index(inplace=True)
    except KeyError:
        st.warning("Colonna 'Rank' non trovata nei risultati.")
    except Exception as e:
        st.warning(f"Errore impostando/ordinando per Rank: {e}. Mostrando in ordine API.")

    # Define column order for display explicitly
    cols_order = [
        "Symbol", "Name", "Gemini Alert", "GPT Signal",
        f"Prezzo ({VS_CURRENCY.upper()})",
        "% 1h", "% 24h", "% 7d", "% 30d", "% 1y",
        "RSI (1h)", "RSI (1d)", "RSI (1w)", "RSI (1mo)",
        "SRSI %K (1d)", "SRSI %D (1d)",
        "MACD Hist (1d)", f"MA({MA_SHORT}d)", f"MA({MA_LONG}d)", "VWAP (1d)",
        f"Volume 24h ({VS_CURRENCY.upper()})"
        # Add other placeholder columns if they exist and should be shown
    ]
    # Filter df to include only desired columns that actually exist
    cols_to_show = [col for col in cols_order if col in df.columns]
    df_display = df[cols_to_show].copy() # Create copy for display formatting

    # --- Formatting Logic (using style.format for better type handling) ---
    formatters = {}
    currency_col = f"Prezzo ({VS_CURRENCY.upper()})"
    volume_col = f"Volume 24h ({VS_CURRENCY.upper()})"
    pct_cols = ["% 1h", "% 24h", "% 7d", "% 30d", "% 1y"]
    rsi_srsi_cols = [col for col in df_display.columns if "RSI" in col or "SRSI" in col]
    macd_cols = [col for col in df_display.columns if "MACD" in col]
    ma_vwap_cols = [col for col in df_display.columns if "MA" in col or "VWAP" in col]

    # Define format strings or functions for each column type
    formatters[currency_col] = "${:,.4f}" # Price format
    for col in pct_cols: formatters[col] = "{:+.2f}%" # Percentage format
    formatters[volume_col] = lambda x: f"${format_large_number(x)}" # Volume format using helper
    for col in rsi_srsi_cols: formatters[col] = "{:.1f}" # RSI/SRSI format
    for col in macd_cols: formatters[col] = "{:.4f}" # MACD format
    for col in ma_vwap_cols: formatters[col] = "{:,.2f}" # MA/VWAP format

    # Apply formatting and handle NaNs
    styled_df = df_display.style.format(formatters, na_rep="N/A", precision=4)

    # --- Styling Logic ---
    def highlight_pct_col_style(val):
        """Styles individual percentage values (numeric)."""
        if pd.isna(val): return ''
        color = 'green' if val > 0 else 'red' if val < 0 else '#6c757d' # Grey for 0
        return f'color: {color};'

    def highlight_signal_style(val):
        """Styles individual signal strings."""
        style = 'color: #6c757d;' # Default Hold color (grey)
        font_weight = 'normal'
        if isinstance(val, str):
            if "Strong Buy" in val: style = 'color: #198754;'; font_weight = 'bold';
            elif "Buy" in val and "Strong" not in val: style = 'color: #28a745;'; font_weight = 'normal';
            elif "Strong Sell" in val: style = 'color: #dc3545;'; font_weight = 'bold';
            elif "Sell" in val and "Strong" not in val: style = 'color: #fd7e14;'; font_weight = 'normal'; # Orange-Red
            elif "CTB" in val: style = 'color: #20c997;'; font_weight = 'normal'; # Teal
            elif "CTS" in val: style = 'color: #ffc107; color: #000;'; font_weight = 'normal'; # Amber/Yellow (dark text)
            # Hold is default grey
            elif "N/D" in val: style = 'color: #adb5bd;'; font_weight = 'normal'; # Lighter Grey
        return f'{style} font-weight: {font_weight};'

    # Apply conditional styling using applymap
    for col in pct_cols:
        if col in df_display.columns:
            # Apply to the original numeric data before formatting turns it to string
            styled_df = styled_df.applymap(highlight_pct_col_style, subset=[col])

    if "Gemini Alert" in df_display.columns:
        styled_df = styled_df.applymap(highlight_signal_style, subset=["Gemini Alert"])
    if "GPT Signal" in df_display.columns:
        styled_df = styled_df.applymap(highlight_signal_style, subset=["GPT Signal"])

    # Display the styled DataFrame
    st.dataframe(styled_df, use_container_width=True) # Fit container width

else:
    st.warning("Nessun risultato crypto valido da visualizzare dopo l'elaborazione.")

# --- Expander per errori/note di Fetch/Calcolo ---
# Moved below main table display, shown only if errors occurred
if fetch_errors:
    with st.expander("ℹ️ Note Recupero Dati / Calcolo Indicatori", expanded=False):
        # Show unique errors only for clarity
        unique_errors = sorted(list(set(fetch_errors)))
        # Limit number of errors displayed to avoid clutter
        max_errors_to_show = 15
        for i, error_msg in enumerate(unique_errors):
            if i < max_errors_to_show:
                 st.info(error_msg)
            elif i == max_errors_to_show:
                 st.info(f"... e altri {len(unique_errors) - max_errors_to_show} errori.")
                 break


# --- SEZIONE NEWS ---
st.markdown("---")
st.subheader("📰 Ultime Notizie Crypto (Cointelegraph Feed)")
news_items = get_crypto_news(NEWS_FEED_URL)

if news_items:
    # Display news items using Markdown for links and basic formatting
    for item in news_items:
        title = item.get('title', 'Titolo non disponibile')
        link = item.get('link', '#')
        # Try to parse and format publication date
        pub_date_str = ""
        if hasattr(item, 'published_parsed') and item.published_parsed:
            try:
                pub_dt_utc = datetime.fromtimestamp(time.mktime(item.published_parsed))
                if ZoneInfo: # Use zoneinfo if available
                     local_tz = ZoneInfo("Europe/Rome")
                     pub_dt_local = pub_dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(local_tz)
                     pub_date_str = f" - *{pub_dt_local.strftime('%d %b, %H:%M %Z')}*"
                else: # Fallback
                    pub_dt_local_approx = pub_dt_utc + timedelta(hours=2)
                    pub_date_str = f" - *{pub_dt_local_approx.strftime('%d %b, %H:%M')} (approx)*"
            except Exception: pass # Ignore date parsing/conversion errors

        st.markdown(f"- [{title}]({link}){pub_date_str}")
else:
    st.warning("Impossibile caricare le notizie dal feed RSS o nessuna notizia trovata.")


# --- Legenda Aggiornata ---
st.divider()
with st.expander("📘 Legenda Indicatori Tecnici e Segnali", expanded=False):
    st.markdown("""
    *Disclaimer: Questa dashboard è solo a scopo informativo e non costituisce consulenza finanziaria.*

    **Market Overview:**
    * **Fear & Greed Index:** Indice sentiment da Alternative.me (0=Paura Estrema, 100=Euforia Estrema).
    * **Total Crypto M.Cap:** Capitalizzazione totale mercato crypto (Fonte: CoinGecko).
    * **Crypto ETFs Flow:** Flusso netto giornaliero ETF crypto spot (Dato **N/A** - fonte non disp.).
    * **S&P 500, Nasdaq, Gold, UVXY, TQQQ:** Prezzi indicativi mercato tradizionale (Fonte: Yahoo Finance).
    * **Titoli Principali:** Prezzi indicativi azioni selezionate (Fonte: Yahoo Finance).

    **Tabella Analisi Tecnica:**
    * **Variazioni Percentuali (%):** Rispetto a 1h, 24h, 7d, 30d, 1y (Fonte: CoinGecko).
    * **Indicatori Momentum:**
        * **RSI (1h/1d/1w/1mo):** Relative Strength Index (0-100). `>70` Ipercomprato, `<30` Ipervenduto. 1w/1mo calcolati da dati daily.
        * **SRSI %K / %D (1d):** Stochastic RSI (0-100). `>80` Ipercomprato, `<20` Ipervenduto. Segnali più frequenti dell'RSI.
        * **MACD Hist (1d):** Moving Average Convergence Divergence Histogram. `>0` Momentum rialzista, `<0` Momentum ribassista.
    * **Indicatori Trend:**
        * **MA (20d, 50d):** Simple Moving Average (Media Mobile Semplice). Incrocio MA20/MA50 può indicare cambio trend.
        * **VWAP (1d):** Volume-Weighted Average Price (Prezzo Medio Ponderato per Volumi, ultimi 14gg).
    * **Segnali Combinati (Esemplificativi - NON CONSULENZA):**
        * **Gemini Alert:** Alert specifico DAILY: `⚡️ Strong Buy` (MA20>MA50 & MACD>0 & RSI<80). `🚨 Strong Sell` (MA20<MA50 & MACD<0 & RSI>20). `🟡 Hold` Altrimenti. `⚪️ N/D` Dati insuff.
        * **GPT Signal:** Punteggio combinato (MAs, MACD, RSIs, SRSI). Interpretazione: `⚡️ Strong Buy` (Score >= 5), `🟢 Buy` (2-4), `⏳ CTB` (Close To Buy - Score > 0 & RSI<55), `🟡 Hold` (Neutro), `⚠️ CTS` (Close To Sell - Score < 0 & RSI>45), `🔴 Sell` (-4 to -2), `🚨 Strong Sell` (Score <= -5). `⚪️ N/D` Dati insuff. **Usare con cautela.**
    * **Generale:** **N/A:** Dato non disponibile o errore.
    """)

# --- Footer ---
st.divider()
st.caption("Disclaimer: Strumento a scopo informativo/didattico. Non costituisce consulenza finanziaria. DYOR (Do Your Own Research).")