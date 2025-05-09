# Version: v1.2 - Fix Styler Apply/Map Usage, SRSI subset fix, Ensure full code
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Other existing imports
from alpha_vantage.timeseries import TimeSeries
import logging
import io

# --- START: Logging Configuration ---
log_stream = io.StringIO()
logging.basicConfig(
    stream=log_stream,
    level=logging.INFO, # Set to INFO for production, DEBUG for development
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)
logger.info("Logging configured for UI.")
# --- END: Logging Configuration ---

# Import zoneinfo
try:
    from zoneinfo import ZoneInfo
    logger.info("Module 'zoneinfo' imported.")
except ImportError:
    logger.warning("Module 'zoneinfo' not found. Using offset approximation for Rome timezone.")
    st.warning("Module 'zoneinfo' not found. Using offset approximation for Rome timezone.")
    ZoneInfo = None

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Crypto Technical Dashboard Pro", page_icon="📈")

# --- CSS ---
st.markdown("""<style>div[data-testid="stMetricValue"] { font-size: 14px !important; }</style>""", unsafe_allow_html=True)
logger.info("CSS applied.")

# --- Global Configuration ---
logger.info("Starting global configuration.")
SYMBOL_TO_ID_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "RNDR": "render-token",
    "RAY": "raydium", "SUI": "sui", "ONDO": "ondo-finance", "ARB": "arbitrum",
    "TAO": "bittensor", "LINK": "chainlink", "HBAR": "hedera-hashgraph",
    "IMX": "immutable-x", "TRUMP": "official-trump", "AERO": "aerodrome-finance", "MKR": "maker",
}
SYMBOLS = list(SYMBOL_TO_ID_MAP.keys())
COINGECKO_IDS_LIST = list(SYMBOL_TO_ID_MAP.values())
NUM_COINS = len(SYMBOLS)
FIRE_ICON_THRESHOLD = 8
logger.info(f"Number of coins configured: {NUM_COINS}")
TRAD_TICKERS_AV = ['SPY', 'QQQ', 'GLD', 'SLV', 'UNG', 'UVXY', 'TQQQ', 'NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR', 'COIN', 'MSTR']
logger.info(f"Traditional tickers configured (Alpha Vantage): {TRAD_TICKERS_AV}")
VS_CURRENCY = "usd"
CACHE_TTL, CACHE_HIST_TTL, CACHE_CHART_TTL, CACHE_TRAD_TTL = 1800, 3600, 1800, 14400
DAYS_HISTORY_DAILY, DAYS_HISTORY_HOURLY = 365, 7
RSI_PERIOD, RSI_OB, RSI_OS = 14, 70.0, 30.0
SRSI_PERIOD, SRSI_K, SRSI_D, SRSI_OB, SRSI_OS = 14, 3, 3, 80.0, 20.0
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MA_SHORT, MA_MEDIUM, MA_LONG, MA_XLONG = 7, 20, 50, 30
BB_PERIOD, BB_STD_DEV = 20, 2.0
VWAP_PERIOD = 14
logger.info("Finished global configuration.")

# --- FUNCTION DEFINITIONS (General) ---

def check_password():
    logger.debug("Executing check_password.")
    if "password_correct" not in st.session_state: st.session_state.password_correct = False
    if not st.session_state.password_correct:
        pwd_col, btn_col = st.columns([3, 1])
        with pwd_col: password = st.text_input("🔑 Password", type="password", key="password_input_field")
        with btn_col: st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True); login_button_pressed = st.button("Login", key="login_button")
        should_check = login_button_pressed or (password and password != "")
        if not should_check: logger.debug("Waiting for password input or button click."); st.stop()
        else:
            correct_password = "Leonardo"
            if password == correct_password:
                logger.info("Password correct."); st.session_state.password_correct = True
                if st.query_params.get("logged_in") != "true": st.query_params["logged_in"] = "true"; st.rerun()
            else: logger.warning("Incorrect password entered."); st.warning("Incorrect password."); st.stop()
    logger.debug("Password check passed."); return True

def format_large_number(num):
    if pd.isna(num) or not isinstance(num, (int, float)): return "N/A"
    num_abs = abs(num); sign = "-" if num < 0 else ""
    if num_abs < 1_000_000: return f"{sign}{num_abs:,.0f}"
    elif num_abs < 1_000_000_000: return f"{sign}{num_abs / 1_000_000:.1f}M"
    elif num_abs < 1_000_000_000_000: return f"{sign}{num_abs / 1_000_000_000:.1f}B"
    else: return f"{sign}{num_abs / 1_000_000_000_000:.2f}T"

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading market data (CoinGecko)...")
def get_coingecko_market_data(ids_list, currency):
    logger.info(f"Attempting CoinGecko live data fetch for {len(ids_list)} IDs.")
    ids_string = ",".join(ids_list); url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': currency, 'ids': ids_string, 'order': 'market_cap_desc', 'per_page': str(len(ids_list)), 'page': 1, 'sparkline': False, 'price_change_percentage': '1h,24h,7d,30d,1y', 'precision': 'full'}
    timestamp_utc = datetime.now(ZoneInfo("UTC") if ZoneInfo else None)
    if 'api_warning_shown' not in st.session_state: st.session_state.api_warning_shown = False
    try:
        logger.debug(f"Requesting URL: {url} with params: {params}"); response = requests.get(url, params=params, timeout=20); response.raise_for_status()
        data = response.json();
        if not data: logger.warning("CoinGecko Live API: Empty data received."); st.warning("CoinGecko Live API: Empty data received."); return pd.DataFrame(), timestamp_utc
        df = pd.DataFrame(data);
        if not df.empty: df.set_index('id', inplace=True)
        st.session_state["api_warning_shown"] = False; logger.info(f"CoinGecko live data fetched for {len(df)} coins."); return df, timestamp_utc
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code; logger.warning(f"HTTP Error CoinGecko Market API (Status: {status_code}): {http_err}")
        if status_code == 429 and not st.session_state.get("api_warning_shown", False): st.warning("Warning CoinGecko API (Live): Rate limit (429) reached. Data might be outdated."); st.session_state["api_warning_shown"] = True
        elif status_code != 429 or st.session_state.get("api_warning_shown", False): st.error(f"HTTP Error CoinGecko Market API (Status: {status_code}): {http_err}")
        return pd.DataFrame(), timestamp_utc
    except requests.exceptions.RequestException as req_ex: logger.error(f"Request Error CoinGecko Market API: {req_ex}"); st.error(f"Request Error CoinGecko Market API: {req_ex}"); return pd.DataFrame(), timestamp_utc
    except Exception as e: logger.exception("Error Processing CoinGecko Market Data:"); st.error(f"Error Processing CoinGecko Market Data: {e}"); return pd.DataFrame(), timestamp_utc

@st.cache_data(ttl=CACHE_CHART_TTL, show_spinner=False)
def get_coingecko_historical_data_for_chart(coin_id, currency, days):
    logger.debug(f"CHART: Starting historical fetch for {coin_id} (daily, {days}d)."); url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': currency, 'days': str(days), 'interval': 'daily', 'precision': 'full'}; status_msg = f"Unknown Error ({coin_id}, daily chart)"
    try:
        logger.debug(f"CHART: Requesting URL: {url} with params: {params}"); response = requests.get(url, params=params, timeout=25); response.raise_for_status(); data = response.json()
        if not data or 'prices' not in data or not data['prices']: status_msg = f"No Prices Data ({coin_id}, daily chart)"; logger.warning(status_msg); return pd.DataFrame(), status_msg
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'close']); prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms', utc=True); prices_df.set_index('timestamp', inplace=True); hist_df = prices_df
        if 'total_volumes' in data and data['total_volumes']:
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume']); volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms', utc=True); volumes_df.set_index('timestamp', inplace=True); hist_df = prices_df.join(volumes_df, how='outer')
        else: hist_df['volume'] = 0.0
        hist_df = hist_df.interpolate(method='time').ffill().bfill(); hist_df['high'] = hist_df['close']; hist_df['low'] = hist_df['close']; hist_df['open'] = hist_df['close'].shift(1)
        if not hist_df.empty: hist_df.loc[hist_df.index[0], 'open'] = hist_df['close'].iloc[0]
        hist_df = hist_df[~hist_df.index.duplicated(keep='last')].sort_index(); hist_df.dropna(subset=['close'], inplace=True)
        if hist_df.empty: status_msg = f"Processed Empty ({coin_id}, daily chart)"; logger.warning(status_msg); return pd.DataFrame(), status_msg
        status_msg = "Success"; logger.info(f"CHART: Historical data fetched for {coin_id} (daily), {len(hist_df)} rows.")
        return_cols = ['open', 'high', 'low', 'close', 'volume']; hist_df_final = hist_df[[col for col in return_cols if col in hist_df.columns]]; return hist_df_final, status_msg
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code;
        if status_code == 429: status_msg = f"Rate Limited (429) ({coin_id}, daily chart)"
        elif status_code == 404: status_msg = f"Not Found (404) ({coin_id}, daily chart)"
        else: status_msg = f"HTTP Error {status_code} ({coin_id}, daily chart)"
        logger.warning(f"CHART: HTTP Error CoinGecko History API: {status_msg}"); return pd.DataFrame(), status_msg
    except requests.exceptions.RequestException as req_ex: status_msg = f"Request Error ({req_ex}) ({coin_id}, daily chart)"; logger.error(f"CHART: Request Error CoinGecko History API: {status_msg}"); return pd.DataFrame(), status_msg
    except Exception as e: status_msg = f"Generic Error ({type(e).__name__}) ({coin_id}, daily chart)"; logger.exception(f"CHART: Error processing CoinGecko History API for {coin_id}:"); return pd.DataFrame(), status_msg

@st.cache_data(ttl=CACHE_HIST_TTL, show_spinner=False)
def get_coingecko_historical_data(coin_id, currency, days, interval='daily'):
    logger.debug(f"TABLE: Starting historical fetch for {coin_id} ({interval}), 6s delay..."); time.sleep(6.0); logger.debug(f"TABLE: Delay ended for {coin_id} ({interval}), starting API call.")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"; params = {'vs_currency': currency, 'days': str(days), 'interval': interval if interval == 'hourly' else 'daily', 'precision': 'full'}
    status_msg = f"Unknown Error ({coin_id}, {interval})"
    try:
        logger.debug(f"TABLE: Requesting URL: {url} with params: {params}"); response = requests.get(url, params=params, timeout=25); response.raise_for_status(); data = response.json()
        if not data or 'prices' not in data or not data['prices']: status_msg = f"No Prices Data ({coin_id}, {interval})"; logger.warning(status_msg); return pd.DataFrame(), status_msg
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'close']); prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms', utc=True); prices_df.set_index('timestamp', inplace=True); hist_df = prices_df
        if 'total_volumes' in data and data['total_volumes']:
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume']); volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms', utc=True); volumes_df.set_index('timestamp', inplace=True); hist_df = prices_df.join(volumes_df, how='outer')
        else: hist_df['volume'] = 0.0
        hist_df = hist_df.interpolate(method='time').ffill().bfill(); hist_df['high'] = hist_df['close']; hist_df['low'] = hist_df['close']; hist_df['open'] = hist_df['close'].shift(1)
        if not hist_df.empty: hist_df.loc[hist_df.index[0], 'open'] = hist_df['close'].iloc[0]
        hist_df = hist_df[~hist_df.index.duplicated(keep='last')].sort_index(); hist_df.dropna(subset=['close'], inplace=True)
        if hist_df.empty: status_msg = f"Processed Empty ({coin_id}, {interval})"; logger.warning(status_msg); return pd.DataFrame(), status_msg
        status_msg = "Success"; logger.info(f"TABLE: Historical data fetched for {coin_id} ({interval}), {len(hist_df)} rows."); return hist_df, status_msg
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code;
        if status_code == 429: status_msg = f"Rate Limited (429) ({coin_id}, {interval})"
        elif status_code == 404: status_msg = f"Not Found (404) ({coin_id}, {interval})"
        else: status_msg = f"HTTP Error {status_code} ({coin_id}, {interval})"
        logger.warning(f"TABLE: HTTP Error CoinGecko History API: {status_msg}"); return pd.DataFrame(), status_msg
    except requests.exceptions.RequestException as req_ex: status_msg = f"Request Error ({req_ex}) ({coin_id}, {interval})"; logger.error(f"TABLE: Request Error CoinGecko History API: {status_msg}"); return pd.DataFrame(), status_msg
    except Exception as e: status_msg = f"Generic Error ({type(e).__name__}) ({coin_id}, {interval})"; logger.exception(f"TABLE: Error processing CoinGecko History API for {coin_id} ({interval}):"); return pd.DataFrame(), status_msg

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_fear_greed_index():
    logger.info("Attempting Fear & Greed Index fetch."); url = "https://api.alternative.me/fng/?limit=1"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data and isinstance(data.get("data"), list) and len(data["data"]) > 0:
             latest_data = data["data"][0]; value = latest_data.get("value"); desc = latest_data.get("value_classification")
             if value is not None and desc is not None: logger.info(f"F&G Index: {value} ({desc})."); return f"{int(value)} ({desc})"
        logger.warning("Unexpected F&G Index data format received from API."); return "N/A"
    except requests.exceptions.RequestException as req_ex: status_code = req_ex.response.status_code if req_ex.response is not None else "N/A"; msg = f"API Error F&G Index (Alternative.me Status: {status_code}): {req_ex}"; logger.warning(msg); st.warning(msg); return "N/A"
    except Exception as e: msg = f"Error Processing F&G Index (Alternative.me): {e}"; logger.exception(msg); st.warning(msg); return "N/A"

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_global_market_data_cg(currency):
    logger.info("Attempting Global CoinGecko data fetch."); url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json().get('data', {})
        total_mcap = data.get('total_market_cap', {}).get(currency.lower(), np.nan)
        if pd.notna(total_mcap): logger.info(f"Global CoinGecko M.Cap ({currency.upper()}): {total_mcap}.")
        else: logger.warning(f"Global CoinGecko M.Cap ({currency.upper()}) not found in response.")
        return total_mcap
    except requests.exceptions.RequestException as req_ex: msg = f"Global CoinGecko API Error: {req_ex}"; logger.warning(msg); st.warning(msg); return np.nan
    except Exception as e: msg = f"Error Processing Global CoinGecko Data: {e}"; logger.exception(msg); st.warning(msg); return np.nan

def get_etf_flow(): logger.debug("get_etf_flow called (placeholder)."); return "N/A"

@st.cache_data(ttl=CACHE_TRAD_TTL, show_spinner="Loading traditional market data (Alpha Vantage)...")
def get_traditional_market_data_av(tickers):
    """Fetches quote data from Alpha Vantage for traditional tickers."""
    logger.info(f"Attempting Alpha Vantage fetch for {len(tickers)} tickers."); data = {ticker: {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'} for ticker in tickers}; api_key = None
    try: api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]; logger.info("Alpha Vantage API key read from secrets.")
    except KeyError: logger.error("Secret 'ALPHA_VANTAGE_API_KEY' not defined."); st.error("Configuration Error: Alpha Vantage API key not found in secrets. Traditional market data unavailable."); return data
    except Exception as e: logger.exception("Unexpected error reading Alpha Vantage secrets:"); st.error(f"Error reading Alpha Vantage secrets: {e}"); return data
    if not api_key: logger.error("Alpha Vantage API key found but is empty."); st.error("Configuration Error: Alpha Vantage API key is empty in secrets. Traditional market data unavailable."); return data
    ts = TimeSeries(key=api_key, output_format='pandas'); calls_made = 0; max_calls_per_minute = 5; max_calls_this_run = 25; delay_between_calls = (60.0 / max_calls_per_minute) + 1.0
    for ticker_sym in tickers:
        if calls_made >= max_calls_this_run: msg = f"AV call limit for this run ({max_calls_this_run}) reached. Stopping fetch for {ticker_sym}."; logger.warning(msg); st.warning(msg); break
        try:
            logger.info(f"AV Fetch for {ticker_sym} (Call #{calls_made+1}/{max_calls_this_run}, Pause {delay_between_calls:.1f}s)..."); time.sleep(delay_between_calls); quote_data, meta_data = ts.get_quote_endpoint(symbol=ticker_sym); calls_made += 1; logger.debug(f"AV response for {ticker_sym}: Head:\n{quote_data.head()}")
            if not quote_data.empty:
                try: data[ticker_sym]['price'] = float(quote_data['05. price'].iloc[0])
                except (KeyError, ValueError, IndexError, TypeError): pass
                try: data[ticker_sym]['change'] = float(quote_data['09. change'].iloc[0])
                except (KeyError, ValueError, IndexError, TypeError): pass
                try: data[ticker_sym]['change_percent'] = quote_data['10. change percent'].iloc[0]
                except (KeyError, IndexError, TypeError): pass
                logger.info(f"AV data for {ticker_sym} fetched OK.")
            else: logger.warning(f"Empty response from AV for {ticker_sym}."); st.warning(f"Empty response from AV for {ticker_sym}."); data[ticker_sym] = {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'}
        except ValueError as ve:
             msg = f"Alpha Vantage Error (ValueError) for {ticker_sym}: {ve}"; logger.warning(msg); st.warning(msg); ve_str = str(ve).lower()
             if "call frequency" in ve_str or "api key" in ve_str or "limit" in ve_str or "premium" in ve_str: logger.error("Critical Alpha Vantage API key/limit error detected. Stopping fetch for traditional markets."); st.error("Alpha Vantage API key/limit error detected. Stopping fetch."); break
             data[ticker_sym] = {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'}
        except Exception as e: msg = f"Generic error fetching Alpha Vantage for {ticker_sym}: {e}"; logger.exception(msg); st.warning(msg); data[ticker_sym] = {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'}
    logger.info(f"Finished Alpha Vantage fetch. Made {calls_made} calls."); return data

# --- Indicator Calculation Functions (Manual for Table) ---
def calculate_rsi_manual(series: pd.Series, period: int = RSI_PERIOD) -> float:
    if not isinstance(series, pd.Series) or series.empty or series.isna().all(): return np.nan
    series = series.dropna();
    if len(series) < period + 1: return np.nan
    delta = series.diff(); gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean(); avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    if avg_gain.isna().all() or avg_loss.isna().all(): return np.nan
    last_avg_loss = avg_loss.iloc[-1]; last_avg_gain = avg_gain.iloc[-1]
    if pd.isna(last_avg_loss) or pd.isna(last_avg_gain): return np.nan
    if last_avg_loss == 0: return 100.0 if last_avg_gain > 0 else 50.0
    rs = last_avg_gain / last_avg_loss; rsi = 100.0 - (100.0 / (1.0 + rs))
    return max(0.0, min(100.0, rsi))

def calculate_stoch_rsi(series: pd.Series, rsi_period: int = RSI_PERIOD, stoch_period: int = SRSI_PERIOD, k_smooth: int = SRSI_K, d_smooth: int = SRSI_D) -> tuple[float, float]:
    if not isinstance(series, pd.Series) or series.empty or series.isna().all(): return np.nan, np.nan
    series = series.dropna();
    if len(series) < rsi_period + stoch_period + max(k_smooth, d_smooth) -1 : return np.nan, np.nan
    delta = series.diff(); gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean(); avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan); rsi_series = (100.0 - (100.0 / (1.0 + rs))).dropna()
    if len(rsi_series) < stoch_period: return np.nan, np.nan
    min_rsi = rsi_series.rolling(window=stoch_period, min_periods=stoch_period).min(); max_rsi = rsi_series.rolling(window=stoch_period, min_periods=stoch_period).max()
    range_rsi = max_rsi - min_rsi; stoch_rsi_k_raw = 100 * (rsi_series - min_rsi) / range_rsi.replace(0, np.nan); stoch_rsi_k_raw = stoch_rsi_k_raw.dropna()
    if len(stoch_rsi_k_raw) < k_smooth: return np.nan, np.nan
    stoch_rsi_k = stoch_rsi_k_raw.rolling(window=k_smooth, min_periods=k_smooth).mean()
    if len(stoch_rsi_k.dropna()) < d_smooth: last_k_val = stoch_rsi_k.iloc[-1]; k_final = max(0.0, min(100.0, last_k_val)) if pd.notna(last_k_val) else np.nan; return k_final, np.nan
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_smooth, min_periods=d_smooth).mean()
    last_k = stoch_rsi_k.iloc[-1]; last_d = stoch_rsi_d.iloc[-1]
    k_final = max(0.0, min(100.0, last_k)) if pd.notna(last_k) else np.nan; d_final = max(0.0, min(100.0, last_d)) if pd.notna(last_d) else np.nan
    return k_final, d_final

def calculate_macd_manual(series: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> tuple[float, float, float]:
    if not isinstance(series, pd.Series) or series.empty or series.isna().all(): return np.nan, np.nan, np.nan
    series = series.dropna();
    if len(series) < slow + signal - 1: return np.nan, np.nan, np.nan
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean(); ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow; signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    last_macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else np.nan; last_signal = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else np.nan
    last_hist = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else np.nan
    return last_macd, last_signal, last_hist

def calculate_sma_manual(series: pd.Series, period: int) -> float:
    if not isinstance(series, pd.Series) or series.empty or series.isna().all(): return np.nan
    series = series.dropna();
    if len(series) < period: return np.nan
    return series.rolling(window=period, min_periods=period).mean().iloc[-1]

def calculate_vwap_manual(df_slice: pd.DataFrame, period: int = VWAP_PERIOD) -> float:
    required_cols = ['close', 'volume'];
    if not isinstance(df_slice, pd.DataFrame) or df_slice.empty or not all(col in df_slice.columns for col in required_cols): return np.nan
    df_valid_slice = df_slice[required_cols].dropna();
    if len(df_valid_slice) < period: return np.nan
    df_period = df_valid_slice.iloc[-period:]
    pv = df_period['close'] * df_period['volume']; total_volume = df_period['volume'].sum()
    if total_volume == 0 or pd.isna(total_volume): return df_period['close'].iloc[-1] if not df_period.empty else np.nan
    vwap = pv.sum() / total_volume
    return vwap

def calculate_bbands_manual(series: pd.Series, period: int = BB_PERIOD, std_dev: float = BB_STD_DEV) -> tuple[float, float, float, float, float, float]:
    """Calculates the last values for BBands: Mid, Upper, Lower, %B, Width, Width Change."""
    if not isinstance(series, pd.Series) or series.empty or series.isna().all(): return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    series = series.dropna(); min_len_bb_change = period + 1
    if len(series) >= period:
        middle_band_series = series.rolling(window=period, min_periods=period).mean(); std_series = series.rolling(window=period, min_periods=period).std()
        middle_band_now = middle_band_series.iloc[-1]; std_now = std_series.iloc[-1]
        if pd.notna(middle_band_now) and pd.notna(std_now):
            upper_band_now = middle_band_now + (std_dev * std_now); lower_band_now = middle_band_now - (std_dev * std_now)
            last_price = series.iloc[-1]; band_range_now = upper_band_now - lower_band_now
            percent_b_now = ((last_price - lower_band_now) / band_range_now) * 100 if band_range_now > 0 else np.nan
            bandwidth_now = (band_range_now / middle_band_now) * 100 if middle_band_now > 0 else np.nan
        else: return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else: return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    bandwidth_change = np.nan
    if len(series) >= min_len_bb_change:
        middle_band_prev = middle_band_series.iloc[-2]; std_prev = std_series.iloc[-2]
        if pd.notna(middle_band_prev) and pd.notna(std_prev):
            upper_band_prev = middle_band_prev + (std_dev * std_prev); lower_band_prev = middle_band_prev - (std_dev * std_prev)
            band_range_prev = upper_band_prev - lower_band_prev
            bandwidth_prev = (band_range_prev / middle_band_prev) * 100 if middle_band_prev > 0 else np.nan
            if pd.notna(bandwidth_now) and pd.notna(bandwidth_prev) and bandwidth_prev != 0: bandwidth_change = ((bandwidth_now - bandwidth_prev) / bandwidth_prev) * 100
    return middle_band_now, upper_band_now, lower_band_now, percent_b_now, bandwidth_now, bandwidth_change

def compute_all_indicators(symbol: str, hist_daily_df: pd.DataFrame) -> dict:
    """Calculates all technical indicators (last value) for the table."""
    indicators = { "RSI (1d)": np.nan, "RSI (1w)": np.nan, "RSI (1mo)": np.nan, "SRSI %K (1d)": np.nan, "SRSI %D (1d)": np.nan, "MACD Line (1d)": np.nan, "MACD Signal (1d)": np.nan, "MACD Hist (1d)": np.nan, f"MA({MA_SHORT}d)": np.nan, f"MA({MA_MEDIUM}d)": np.nan, f"MA({MA_LONG}d)": np.nan, f"MA({MA_XLONG}d)": np.nan, "BB %B": np.nan, "BB Width": np.nan, "BB Width %Chg": np.nan, "VWAP (1d)": np.nan, "VWAP %": np.nan }
    min_len_rsi_base = RSI_PERIOD + 1; min_len_srsi_base = RSI_PERIOD + SRSI_PERIOD + max(SRSI_K, SRSI_D) + 5; min_len_macd_base = MACD_SLOW + MACD_SIGNAL + 5;
    min_len_sma_short = MA_SHORT; min_len_sma_medium = MA_MEDIUM; min_len_sma_xlong = MA_XLONG; min_len_sma_long = MA_LONG; min_len_bb = BB_PERIOD + 1;
    min_len_vwap_base = VWAP_PERIOD; min_len_vwap_change = VWAP_PERIOD + 1

    if not hist_daily_df.empty and 'close' in hist_daily_df.columns:
        if 'volume' not in hist_daily_df.columns: logger.warning(f"{symbol}: TABLE: 'volume' column missing. VWAP N/A."); hist_daily_df['volume'] = np.nan
        close_daily = hist_daily_df['close'].dropna(); len_daily = len(close_daily); df_for_vwap = hist_daily_df[['close', 'volume']]

        if len_daily >= min_len_rsi_base: indicators["RSI (1d)"] = calculate_rsi_manual(close_daily, RSI_PERIOD)
        else: logger.warning(f"{symbol}: TABLE: Insuff data ({len_daily}/{min_len_rsi_base}) for RSI(1d)")
        if len_daily >= min_len_srsi_base: k, d = calculate_stoch_rsi(close_daily, RSI_PERIOD, SRSI_PERIOD, SRSI_K, SRSI_D); indicators["SRSI %K (1d)"] = k; indicators["SRSI %D (1d)"] = d
        else: logger.warning(f"{symbol}: TABLE: Insuff data ({len_daily}/{min_len_srsi_base}) for SRSI(1d)")
        if len_daily >= min_len_macd_base: macd_l, macd_s, macd_h = calculate_macd_manual(close_daily, MACD_FAST, MACD_SLOW, MACD_SIGNAL); indicators["MACD Line (1d)"] = macd_l; indicators["MACD Signal (1d)"] = macd_s; indicators["MACD Hist (1d)"] = macd_h
        else: logger.warning(f"{symbol}: TABLE: Insuff data ({len_daily}/{min_len_macd_base}) for MACD(1d)")
        if len_daily >= min_len_sma_short: indicators[f"MA({MA_SHORT}d)"] = calculate_sma_manual(close_daily, MA_SHORT)
        else: logger.warning(f"{symbol}: TABLE: Insuff data ({len_daily}/{min_len_sma_short}) for MA({MA_SHORT}d)")
        if len_daily >= min_len_sma_medium: indicators[f"MA({MA_MEDIUM}d)"] = calculate_sma_manual(close_daily, MA_MEDIUM)
        else: logger.warning(f"{symbol}: TABLE: Insuff data ({len_daily}/{min_len_sma_medium}) for MA({MA_MEDIUM}d)")
        if len_daily >= min_len_sma_xlong: indicators[f"MA({MA_XLONG}d)"] = calculate_sma_manual(close_daily, MA_XLONG)
        else: logger.warning(f"{symbol}: TABLE: Insuff data ({len_daily}/{min_len_sma_xlong}) for MA({MA_XLONG}d)")
        if len_daily >= min_len_sma_long: indicators[f"MA({MA_LONG}d)"] = calculate_sma_manual(close_daily, MA_LONG)
        else: logger.warning(f"{symbol}: TABLE: Insuff data ({len_daily}/{min_len_sma_long}) for MA({MA_LONG}d)")
        if len_daily >= min_len_bb:
             bb_mid, bb_up, bb_low, bb_pct_b, bb_width, bb_width_chg = calculate_bbands_manual(close_daily, BB_PERIOD, BB_STD_DEV)
             indicators["BB %B"] = bb_pct_b; indicators["BB Width"] = bb_width; indicators["BB Width %Chg"] = bb_width_chg
        else: logger.warning(f"{symbol}: TABLE: Insuff data ({len_daily}/{min_len_bb}) for Bollinger Bands")
        if len(df_for_vwap) >= min_len_vwap_base:
            indicators["VWAP (1d)"] = calculate_vwap_manual(df_for_vwap.iloc[-VWAP_PERIOD:], VWAP_PERIOD)
            if len(df_for_vwap) >= min_len_vwap_change:
                vwap_today = indicators["VWAP (1d)"]; vwap_yesterday = calculate_vwap_manual(df_for_vwap.iloc[-(VWAP_PERIOD + 1):-1], VWAP_PERIOD)
                if pd.notna(vwap_today) and pd.notna(vwap_yesterday) and vwap_yesterday != 0: indicators["VWAP %"] = ((vwap_today - vwap_yesterday) / vwap_yesterday) * 100
                else: logger.warning(f"{symbol}: TABLE: Cannot calculate VWAP % Change")
            else: logger.warning(f"{symbol}: TABLE: Insuff data ({len(df_for_vwap)}/{min_len_vwap_change}) for VWAP % Change(1d)")
        else: logger.warning(f"{symbol}: TABLE: Insuff data ({len(df_for_vwap)}/{min_len_vwap_base}) for VWAP(1d)")

        if len_daily > min_len_rsi_base and pd.api.types.is_datetime64_any_dtype(close_daily.index):
            try:
                df_weekly = close_daily.resample('W-MON').last()
                if len(df_weekly.dropna()) >= min_len_rsi_base: indicators["RSI (1w)"] = calculate_rsi_manual(df_weekly, RSI_PERIOD)
                else: logger.warning(f"{symbol}: TABLE: Insuff Weekly data ({len(df_weekly.dropna())}/{min_len_rsi_base}) for RSI(1w)")
            except Exception as e: logger.exception(f"{symbol}: TABLE: Error calculating weekly RSI:")
            try:
                df_monthly = close_daily.resample('ME').last()
                if len(df_monthly.dropna()) >= min_len_rsi_base: indicators["RSI (1mo)"] = calculate_rsi_manual(df_monthly, RSI_PERIOD)
                else: logger.warning(f"{symbol}: TABLE: Insuff Monthly data ({len(df_monthly.dropna())}/{min_len_rsi_base}) for RSI(1mo)")
            except Exception as e: logger.exception(f"{symbol}: TABLE: Error calculating monthly RSI:")
    else: logger.warning(f"{symbol}: TABLE: Empty daily historical data for indicator calculation.")
    return indicators

# --- Signal Functions (Refined v1.1) ---
def generate_gpt_signal(rsi_1d, rsi_1w, macd_hist, ma_short, ma_medium, ma_long, srsi_k, srsi_d, bb_pct_b, bb_width_chg, vwap_1d, current_price):
    """Generates a signal using more indicators ('GPT' style v1.1)."""
    required_inputs = [ rsi_1d, macd_hist, ma_short, ma_medium, ma_long, srsi_k, srsi_d, bb_pct_b, bb_width_chg, vwap_1d, current_price ]
    if any(pd.isna(x) for x in required_inputs): return "⚪️ N/A"
    score = 0
    price_vs_ma7 = current_price > ma_short; price_vs_ma20 = current_price > ma_medium; price_vs_ma50 = current_price > ma_long
    ma7_vs_ma20 = ma_short > ma_medium; ma20_vs_ma50 = ma_medium > ma_long
    if price_vs_ma7 and price_vs_ma20 and price_vs_ma50 and ma7_vs_ma20 and ma20_vs_ma50: score += 3
    elif price_vs_ma7 and ma7_vs_ma20: score += 1.5
    elif price_vs_ma50: score += 0.5
    if not price_vs_ma7 and not price_vs_ma20 and not price_vs_ma50 and not ma7_vs_ma20 and not ma20_vs_ma50: score -= 3
    elif not price_vs_ma7 and not ma7_vs_ma20: score -= 1.5
    elif not price_vs_ma50: score -= 0.5
    if current_price > vwap_1d: score += 1
    else: score -= 1
    if macd_hist > 0: score += 1.5
    else: score -= 1.5
    if rsi_1d < RSI_OS: score += 1.5
    elif rsi_1d < 40: score += 0.5
    elif rsi_1d > RSI_OB: score -= 1.5
    elif rsi_1d > 60: score -= 0.5
    if pd.notna(rsi_1w):
        if rsi_1w < 40: score += 0.5
        elif rsi_1w > 60: score -= 0.5
    if pd.notna(srsi_k) and pd.notna(srsi_d):
        if srsi_k < SRSI_OS and srsi_d < SRSI_OS: score += 1
        elif srsi_k > SRSI_OB and srsi_d > SRSI_OB: score -= 1
        elif srsi_k > srsi_d: score += 0.5
        elif srsi_k < srsi_d: score -= 0.5
    if bb_pct_b > 100: score -= 0.5
    elif bb_pct_b < 0: score += 0.5
    if bb_width_chg > 5: score += 0.5
    elif bb_width_chg < -5: score -= 0.25
    if score >= 6.0: return "⚡️ Strong Buy"
    elif score >= 3.0: return "🟢 Buy"
    elif score <= -6.0: return "🚨 Strong Sell"
    elif score <= -3.0: return "🔴 Sell"
    elif score > 0: return "⏳ CTB"
    elif score < 0: return "⚠️ CTS"
    else: return "🟡 Hold"

def generate_gemini_alert(ma_medium, ma_long, macd_hist, rsi_1d, vwap_1d, current_price):
    """Generates alert based on MA20/50 Cross, MACD, RSI, VWAP ('Gemini' style v1.1)."""
    required_inputs = [ma_medium, ma_long, macd_hist, rsi_1d, vwap_1d, current_price];
    if any(pd.isna(x) for x in required_inputs): return "⚪️ N/A"
    is_ma_cross_bullish = ma_medium > ma_long; is_ma_cross_bearish = ma_medium < ma_long
    is_momentum_positive = macd_hist > 0; is_momentum_negative = macd_hist < 0
    is_price_confirm_bullish = current_price > ma_medium and current_price > vwap_1d
    is_price_confirm_bearish = current_price < ma_medium and current_price < vwap_1d
    is_rsi_ok_bullish = rsi_1d < RSI_OB + 5
    is_rsi_ok_bearish = rsi_1d > RSI_OS - 5
    if is_ma_cross_bullish and is_momentum_positive and is_price_confirm_bullish and is_rsi_ok_bullish: return "⚡️ Strong Buy"
    elif is_ma_cross_bearish and is_momentum_negative and is_price_confirm_bearish and is_rsi_ok_bearish: return "🚨 Strong Sell"
    else: return "🟡 Hold"


# --- Chart Indicator Calculation Functions (Manual) ---
def calculate_sma_series(series: pd.Series, period: int) -> pd.Series:
    """Calculates SMA for the entire series."""
    if not isinstance(series, pd.Series) or series.empty: return pd.Series(index=series.index, dtype=float)
    return series.rolling(window=period, min_periods=period).mean()

def calculate_rsi_series(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculates RSI for the entire series."""
    if not isinstance(series, pd.Series) or series.empty: return pd.Series(index=series.index, dtype=float)
    series_valid = series.dropna()
    if len(series_valid) < period + 1: return pd.Series(index=series.index, dtype=float)
    delta = series_valid.diff(); gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean(); avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan); rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.loc[avg_loss == 0] = 100.0; rsi = rsi.clip(0, 100)
    return rsi.reindex(series.index)


# --- Chart Creation Function ---
def create_coin_chart(df, symbol):
    """Creates Plotly chart with Candlestick, MA, and RSI."""
    logger.info(f"CHART: Creating chart for {symbol} with {len(df)} rows.")
    required_ohlc = ['open', 'high', 'low', 'close'];
    if df.empty or not all(col in df.columns for col in required_ohlc): logger.warning(f"CHART: Empty DataFrame or missing OHLC for {symbol}. Columns: {df.columns.tolist()}"); return None
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        logger.warning(f"CHART: Index is not Datetime for {symbol}. Attempting conversion.");
        try: df.index = pd.to_datetime(df.index)
        except Exception as e: logger.error(f"CHART: Failed to convert index to Datetime for {symbol}: {e}"); return None

    # Calculate indicators using manual functions
    try:
        logger.debug(f"CHART: Calculating manual indicators (SMA, RSI) for {symbol}.")
        close_series = df['close'].dropna()
        if close_series.empty: raise ValueError("'close' series is empty after dropna()")
        df['MA_Medium'] = calculate_sma_series(close_series, MA_MEDIUM).reindex(df.index)
        df['MA_Long'] = calculate_sma_series(close_series, MA_LONG).reindex(df.index)
        df['RSI'] = calculate_rsi_series(close_series, RSI_PERIOD).reindex(df.index)
        logger.debug(f"CHART: Columns after manual calculations: {df.columns.tolist()}")
    except Exception as calc_err:
        logger.exception(f"CHART: Error during manual indicator calculation for {symbol}:")
        st.warning(f"Could not calculate indicators for {symbol} chart: {calc_err}")
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name=f'{symbol} Price (Daily)', increasing_line_color= 'green', decreasing_line_color= 'red'), row=1, col=1)
    if 'MA_Medium' in df.columns and df['MA_Medium'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['MA_Medium'], mode='lines', line=dict(color='blue', width=1), name=f'MA({MA_MEDIUM}d)'), row=1, col=1)
    else: logger.warning(f"CHART: MA_Medium column not found or empty for {symbol}")
    if 'MA_Long' in df.columns and df['MA_Long'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['MA_Long'], mode='lines', line=dict(color='orange', width=1), name=f'MA({MA_LONG}d)'), row=1, col=1)
    else: logger.warning(f"CHART: MA_Long column not found or empty for {symbol}")
    if 'RSI' in df.columns and df['RSI'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', line=dict(color='purple', width=1), name='RSI (14d)'), row=2, col=1)
        fig.add_hline(y=RSI_OB, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
        fig.add_hline(y=RSI_OS, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
    else: logger.warning(f"CHART: RSI column not found or empty for {symbol}"); fig.update_yaxes(title_text='RSI N/A', row=2, col=1)
    fig.update_layout(title=f'{symbol}/{VS_CURRENCY.upper()} Daily Technical Analysis', xaxis_title=None, yaxis_title=f'Price ({VS_CURRENCY.upper()})', yaxis2_title='RSI', xaxis_rangeslider_visible=False, legend_title_text='Indicators', height=600, margin=dict(l=50, r=50, t=50, b=50), hovermode="x unified" )
    fig.update_yaxes(autorange=True, row=1, col=1); logger.info(f"CHART: Plotly chart created for {symbol}."); return fig

# --- START OF MAIN APP EXECUTION ---
logger.info("Starting main UI execution.")
try:
    if not check_password(): st.stop()
    logger.info("Password check passed.")

    # --- Title, Refresh Button, Timestamp ---
    col_title, col_button_placeholder, col_button = st.columns([4, 1, 1])
    with col_title: st.title("📈 Crypto Technical Dashboard Pro")
    with col_button:
        st.write("")
        if st.button("🔄 Refresh", help="Force data refresh (clears cache)", key="refresh_button"):
            logger.info("Refresh button clicked.");
            if 'api_warning_shown' in st.session_state: del st.session_state['api_warning_shown']
            st.cache_data.clear(); st.query_params.clear(); st.rerun()

    last_update_placeholder = st.empty()
    st.caption(f"Cache TTL: Live ({CACHE_TTL/60:.0f}m), Table History ({CACHE_HIST_TTL/60:.0f}m), Chart History ({CACHE_CHART_TTL/60:.0f}m), Traditional ({CACHE_TRAD_TTL/3600:.0f}h).")

    # --- Market Overview Section ---
    st.markdown("---"); st.subheader("🌐 Market Overview")
    fear_greed_value = get_fear_greed_index(); total_market_cap = get_global_market_data_cg(VS_CURRENCY); etf_flow_value = get_etf_flow(); traditional_market_data = get_traditional_market_data_av(TRAD_TICKERS_AV)
    def format_delta(change_val, change_pct_str):
        delta_string = None;
        if pd.notna(change_val) and isinstance(change_pct_str, str) and change_pct_str not in ['N/A', '', None]:
            try: change_pct_val = float(change_pct_str.replace('%','').strip()); delta_string = f"{change_val:+.2f} ({change_pct_val:+.2f}%)"
            except (ValueError, AttributeError): delta_string = f"{change_val:+.2f} (?%)"
        elif pd.notna(change_val): delta_string = f"{change_val:+.2f}";
        return delta_string
    def render_metric(column, label, value_func=None, ticker=None, data_dict=None, help_text=None):
        value_str = "N/A"; delta_txt = None; d_color = "off"
        if ticker and data_dict:
            trad_info = data_dict.get(ticker, {}); price = trad_info.get('price', np.nan); change = trad_info.get('change', np.nan); change_pct = trad_info.get('change_percent', 'N/A')
            value_str = f"${price:,.2f}" if pd.notna(price) else "N/A"; delta_txt = format_delta(change, change_pct);
            if pd.notna(change): d_color = "normal"
        elif value_func:
            try: value_str = value_func(); value_str = str(value_str) if value_str is not None else "N/A"
            except Exception as e: logger.error(f"Error in value_func '{label}': {e}"); value_str = "Error"
            delta_txt = None; d_color = "off"
        else: value_str = "N/A";
        column.metric(label=label, value=value_str, delta=delta_txt, delta_color=d_color, help=help_text)
    overview_items_row1 = [ ("Fear & Greed Index", None, get_fear_greed_index, "Source: Alternative.me"), (f"Total Crypto M.Cap ({VS_CURRENCY.upper()})", None, lambda: f"${format_large_number(total_market_cap)}", "Source: CoinGecko"), ("Crypto ETFs Flow (Daily)", None, get_etf_flow, "Data N/A"), ("S&P 500 (SPY)", "SPY", None, "Source: AV (ETF)"), ("Nasdaq (QQQ)", "QQQ", None, "Source: AV (ETF)") ]
    overview_cols_1 = st.columns(len(overview_items_row1));
    for i, (label, ticker, func, help_text) in enumerate(overview_items_row1): render_metric(overview_cols_1[i], label, value_func=func, ticker=ticker, data_dict=traditional_market_data, help_text=help_text)
    overview_items_row2 = [ ("Gold (GLD)", "GLD", None, "Source: AV (ETF)"), ("Silver (SLV)", "SLV", None, "Source: AV (ETF)"), ("Natural Gas (UNG)", "UNG", None, "Source: AV (ETF)"), ("UVXY (Volatility)", "UVXY", None, "Source: AV"), ("TQQQ (Nasdaq 3x)", "TQQQ", None, "Source: AV") ]
    overview_cols_2 = st.columns(len(overview_items_row2));
    for i, (label, ticker, func, help_text) in enumerate(overview_items_row2): render_metric(overview_cols_2[i], label, value_func=func, ticker=ticker, data_dict=traditional_market_data, help_text=help_text)
    st.markdown("<h6>Major Stocks (Source: Alpha Vantage, 4h Cache):</h6>", unsafe_allow_html=True); stock_tickers_row_av = ['NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR', 'COIN', 'MSTR']
    num_stock_cols = 5; stock_cols = st.columns(num_stock_cols);
    for idx, ticker in enumerate(stock_tickers_row_av): render_metric(stock_cols[idx % num_stock_cols], label=ticker, ticker=ticker, data_dict=traditional_market_data, help_text=f"Ticker: {ticker}")
    st.markdown("---")

    # --- Main Crypto Technical Analysis Table ---
    st.subheader(f"📊 Crypto Technical Analysis ({NUM_COINS} Assets)"); logger.info("Starting live crypto data fetch for table."); market_data_df, last_cg_update_utc = get_coingecko_market_data(COINGECKO_IDS_LIST, VS_CURRENCY)

    # --- Timestamp Display ---
    if last_cg_update_utc:
        timestamp_display_str = "*Live CoinGecko data timestamp unavailable.*"
        try:
            if ZoneInfo:
                local_tz = ZoneInfo("Europe/Rome")
                if last_cg_update_utc.tzinfo is None: logger.debug("UTC timestamp is naive, adding UTC timezone."); last_cg_update_utc = last_cg_update_utc.replace(tzinfo=ZoneInfo("UTC"))
                last_cg_update_local = last_cg_update_utc.astimezone(local_tz)
                timestamp_display_str = f"*Live CoinGecko data updated at: **{last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S %Z')}***"
                logger.info(f"Displaying timestamp: {last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            else:
                logger.debug("ZoneInfo unavailable, using UTC+2 offset for Rome.")
                offset_hours = 2; last_cg_update_rome_approx = last_cg_update_utc + timedelta(hours=offset_hours)
                timestamp_display_str = f"*Live CoinGecko data updated at: **{last_cg_update_rome_approx.strftime('%Y-%m-%d %H:%M:%S')} (Approx. Rome Time)***"
                logger.info(f"Displaying timestamp (approx): {last_cg_update_rome_approx.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e: logger.exception("Error formatting/converting timestamp:"); timestamp_display_str = f"*Timestamp conversion error ({e}). UTC: {last_cg_update_utc.strftime('%Y-%m-%d %H:%M:%S')}*"
        last_update_placeholder.markdown(timestamp_display_str)
    else: logger.warning("Live CoinGecko data timestamp unavailable (last_cg_update_utc is None)."); last_update_placeholder.markdown("*Live CoinGecko data timestamp unavailable.*")

    table_results_df = pd.DataFrame();
    # --- Check Live Data and Process Table ---
    if market_data_df.empty:
        msg = "Critical Error: Could not load live CoinGecko data. Analysis table cannot be generated."
        if st.session_state.get("api_warning_shown", False): msg = "Technical Analysis table not generated: error loading live data (possible CoinGecko API limit reached)."
        logger.error(msg); st.error(msg)
    else:
        # --- Process Table ---
        logger.info(f"Live CoinGecko data OK ({len(market_data_df)} rows), starting table processing loop."); results = []; fetch_errors_for_display = []; process_start_time = time.time(); effective_num_coins = len(market_data_df.index)
        if effective_num_coins != NUM_COINS: logger.warning(f"Coin count from API ({effective_num_coins}) != configured ({NUM_COINS}). Processing {effective_num_coins}.")
        show_fire_icon = (market_data_df['price_change_percentage_1h_in_currency'].dropna() > 0).sum() >= FIRE_ICON_THRESHOLD
        logger.info(f"Show fire icon condition met: {show_fire_icon}")

        spinner_msg = f"Fetching history and calculating table indicators for {effective_num_coins} crypto... (~{(effective_num_coins * 6.0 / 60):.1f} min)"
        with st.spinner(spinner_msg):
            coin_ids_ordered = market_data_df.index.tolist(); logger.info(f"CoinGecko ID list for table: {coin_ids_ordered}"); actual_processed_count = 0
            for i, coin_id in enumerate(coin_ids_ordered):
                symbol = next((sym for sym, c_id in SYMBOL_TO_ID_MAP.items() if c_id == coin_id), "N/A"); logger.info(f"--- Processing Table {i+1}/{effective_num_coins}: {symbol} ({coin_id}) ---")
                try:
                    if symbol == "N/A": msg = f"{coin_id}: ID not in local map. Skipped."; logger.warning(msg); fetch_errors_for_display.append(msg); continue
                    live_data = market_data_df.loc[coin_id]; name = live_data.get('name', coin_id); rank = live_data.get('market_cap_rank', 'N/A'); current_price = live_data.get('current_price', np.nan); volume_24h = live_data.get('total_volume', np.nan)
                    change_1h=live_data.get('price_change_percentage_1h_in_currency',np.nan); change_24h=live_data.get('price_change_percentage_24h_in_currency',np.nan); change_7d=live_data.get('price_change_percentage_7d_in_currency',np.nan); change_30d=live_data.get('price_change_percentage_30d_in_currency',np.nan); change_1y=live_data.get('price_change_percentage_1y_in_currency',np.nan)
                    hist_daily_df_table, status_daily = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_DAILY, interval='daily')

                    indicators = {}; gpt_signal = "⚪️ N/A"; gemini_alert = "⚪️ N/A";
                    if status_daily != "Success":
                        fetch_errors_for_display.append(f"{symbol}: Daily History (Table) - {status_daily}");
                        logger.warning(f"{symbol}: Could not calculate table indicators/signals.")
                    else:
                        indicators = compute_all_indicators(symbol, hist_daily_df_table)
                        gpt_signal = generate_gpt_signal( indicators.get("RSI (1d)"), indicators.get("RSI (1w)"), indicators.get("MACD Hist (1d)"), indicators.get(f"MA({MA_SHORT}d)"), indicators.get(f"MA({MA_MEDIUM}d)"), indicators.get(f"MA({MA_LONG}d)"), indicators.get("SRSI %K (1d)"), indicators.get("SRSI %D (1d)"), indicators.get("BB %B"), indicators.get("BB Width %Chg"), indicators.get("VWAP (1d)"), current_price)
                        gemini_alert = generate_gemini_alert( indicators.get(f"MA({MA_MEDIUM}d)"), indicators.get(f"MA({MA_LONG}d)"), indicators.get("MACD Hist (1d)"), indicators.get("RSI (1d)"), indicators.get("VWAP (1d)"), current_price)

                    coingecko_link = f"https://www.coingecko.com/en/coins/{coin_id}";
                    results.append({ "Rank": rank, "Symbol": symbol, "Name": name, "MA/MACD Cross Alert": gemini_alert, "Composite Score": gpt_signal, f"Price ({VS_CURRENCY.upper()})": current_price, "% 1h": change_1h, "% 24h": change_24h, "% 7d": change_7d, "% 30d": change_30d, "% 1y": change_1y, "RSI (1d)": indicators.get("RSI (1d)"), "RSI (1w)": indicators.get("RSI (1w)"), "RSI (1mo)": indicators.get("RSI (1mo)"), "SRSI %K (1d)": indicators.get("SRSI %K (1d)"), "SRSI %D (1d)": indicators.get("SRSI %D (1d)"), "MACD Hist (1d)": indicators.get("MACD Hist (1d)"), f"MA({MA_SHORT}d)": indicators.get(f"MA({MA_SHORT}d)"), f"MA({MA_MEDIUM}d)": indicators.get(f"MA({MA_MEDIUM}d)"), f"MA({MA_XLONG}d)": indicators.get(f"MA({MA_XLONG}d)"), f"MA({MA_LONG}d)": indicators.get(f"MA({MA_LONG}d)"), "BB %B": indicators.get("BB %B"), "BB Width": indicators.get("BB Width"), "BB Width %Chg": indicators.get("BB Width %Chg"), "VWAP (1d)": indicators.get("VWAP (1d)"), "VWAP %": indicators.get("VWAP %"), f"Volume 24h ({VS_CURRENCY.upper()})": volume_24h, "Link": coingecko_link })
                    logger.info(f"--- Table processing for {symbol} completed. ---"); actual_processed_count += 1
                except Exception as coin_err: err_msg = f"Critical error processing table for {symbol} ({coin_id}): {coin_err}"; logger.exception(err_msg); fetch_errors_for_display.append(f"{symbol}: Critical Table Error - See Log")
        process_end_time = time.time(); total_time = process_end_time - process_start_time; logger.info(f"Finished crypto table loop. Processed {actual_processed_count}/{effective_num_coins} coins. Time: {total_time:.1f} sec"); st.sidebar.info(f"Table Processing Time: {total_time:.1f} sec")

        # --- Display Table ---
        if results:
            logger.info(f"Creating final table DataFrame with {len(results)} results.");
            try:
                table_results_df = pd.DataFrame(results); table_results_df['Rank'] = pd.to_numeric(table_results_df['Rank'], errors='coerce'); table_results_df.set_index('Rank', inplace=True, drop=True); table_results_df.sort_index(inplace=True)
                cols_order = [
                    "Symbol", "Name", "MA/MACD Cross Alert", "Composite Score",
                    f"Price ({VS_CURRENCY.upper()})", "% 1h", "% 24h", "% 7d", "% 30d", "% 1y",
                    "RSI (1d)", "RSI (1w)", "RSI (1mo)",
                    "SRSI %K (1d)", "SRSI %D (1d)",
                    "MACD Hist (1d)",
                    f"MA({MA_SHORT}d)", f"MA({MA_MEDIUM}d)", f"MA({MA_XLONG}d)", f"MA({MA_LONG}d)",
                    "BB %B", "BB Width", "BB Width %Chg",
                    "VWAP (1d)", "VWAP %",
                    f"Volume 24h ({VS_CURRENCY.upper()})", "Link"
                ]
                cols_to_show = [col for col in cols_order if col in table_results_df.columns];
                df_display = table_results_df[cols_to_show].copy()

                # --- Styling Functions (Return CSS Strings Only) ---
                def highlight_signal_style(val):
                    style = 'color: #6c757d; font-weight: normal;';
                    if isinstance(val, str):
                        if "Strong Buy" in val: style = 'color: #198754; font-weight: bold;'
                        elif "Buy" in val and "Strong" not in val: style = 'color: #28a745;'
                        elif "Strong Sell" in val: style = 'color: #dc3545; font-weight: bold;'
                        elif "Sell" in val and "Strong" not in val: style = 'color: #fd7e14;'
                        elif "CTB" in val: style = 'color: #20c997;'
                        elif "CTS" in val: style = 'color: #ffc107; color: #000;'
                        elif "Hold" in val: style = 'color: #6c757d;'
                        elif "N/A" in val or "N/D" in val : style = 'color: #adb5bd;'
                    elif pd.isna(val): style = 'color: #adb5bd;'
                    return style

                def highlight_pct_col_style(val):
                    if pd.isna(val) or not isinstance(val, (int, float)): return ''
                    color = 'green' if val > 0 else 'red' if val < 0 else '#6c757d'; return f'color: {color};'

                def style_rsi(val):
                    if pd.isna(val) or not isinstance(val, (int, float)): return ''
                    if val > RSI_OB: return 'color: #dc3545; font-weight: bold;'
                    elif val < RSI_OS: return 'color: #198754; font-weight: bold;'
                    else: return ''

                def style_macd_hist(val):
                    if pd.isna(val) or not isinstance(val, (int, float)): return ''
                    if val > 0: return 'color: green;'
                    elif val < 0: return 'color: red;'
                    else: return ''

                def style_stoch_rsi(row_subset):
                    """Applies row-wise style CSS string to SRSI cols - returns Series of CSS strings FOR THE SUBSET."""
                    k_col = "SRSI %K (1d)"; d_col = "SRSI %D (1d)"
                    default_style = ''; style_k = default_style; style_d = default_style
                    original_row_index = row_subset.name
                    if original_row_index not in table_results_df.index:
                         return pd.Series([default_style] * len(row_subset.index), index=row_subset.index)

                    k_val_num = table_results_df.loc[original_row_index, k_col] if k_col in table_results_df.columns else np.nan
                    d_val_num = table_results_df.loc[original_row_index, d_col] if d_col in table_results_df.columns else np.nan

                    if pd.notna(k_val_num) and pd.notna(d_val_num):
                        if k_val_num > SRSI_OB and d_val_num > SRSI_OB: style_str = 'color: #dc3545; font-weight: bold;'
                        elif k_val_num < SRSI_OS and d_val_num < SRSI_OS: style_str = 'color: #198754; font-weight: bold;'
                        elif k_val_num > d_val_num: style_str = 'color: #28a745;'
                        elif k_val_num < d_val_num: style_str = 'color: #fd7e14;'
                        else: style_str = default_style
                        style_k = style_str; style_d = style_str
                    # Returns a Series of styles that matches the index of the input `row_subset`
                    return pd.Series([style_k, style_d], index=row_subset.index)

                # --- Define Formatters ---
                formatters = {}
                currency_col = f"Price ({VS_CURRENCY.upper()})"; volume_col = f"Volume 24h ({VS_CURRENCY.upper()})"
                pct_cols_all = ["% 1h", "% 24h", "% 7d", "% 30d", "% 1y", "VWAP %", "BB Width", "BB Width %Chg", "BB %B"]
                rsi_cols_list = [c for c in df_display.columns if "RSI" in c and "%" not in c and "SRSI" not in c]
                srsi_value_cols = ["SRSI %K (1d)", "SRSI %D (1d)"]
                macd_hist_col = ["MACD Hist (1d)"]
                ma_vwap_cols = [c for c in df_display.columns if ("MA" in c or "VWAP" in c) and "%" not in c]

                if currency_col in df_display.columns: formatters[currency_col] = "${:,.4f}"
                if volume_col in df_display.columns: formatters[volume_col] = lambda x: format_large_number(x)
                for col in pct_cols_all:
                    if col in df_display.columns and col != '% 1h': formatters[col] = "{:+.2f}%"
                def format_1h_with_icon(val):
                    if pd.isna(val): return "N/A"
                    icon = "🔥 " if show_fire_icon and val > 0 else ""
                    return f"{icon}{val:+.2f}%"
                if '% 1h' in df_display.columns: formatters['% 1h'] = format_1h_with_icon
                for col in rsi_cols_list + srsi_value_cols:
                     if col in df_display.columns: formatters[col] = "{:.1f}"
                if macd_hist_col[0] in df_display.columns: formatters[macd_hist_col[0]] = "{:+.4f}"
                for col in ma_vwap_cols:
                     if col in df_display.columns: formatters[col] = "{:,.2f}"

                # --- Apply Styles THEN Formatting ---
                styled_table = df_display.style

                cols_for_pct_style = [col for col in pct_cols_all if col in df_display.columns];
                if cols_for_pct_style: styled_table = styled_table.map(highlight_pct_col_style, subset=cols_for_pct_style)

                signal_cols_to_style = ["MA/MACD Cross Alert", "Composite Score"]
                for col in signal_cols_to_style:
                     if col in df_display.columns: styled_table = styled_table.map(highlight_signal_style, subset=[col])

                rsi_cols_to_style = [col for col in rsi_cols_list if col in df_display.columns]
                if rsi_cols_to_style: styled_table = styled_table.map(style_rsi, subset=rsi_cols_to_style)

                if macd_hist_col[0] in df_display.columns: styled_table = styled_table.map(style_macd_hist, subset=macd_hist_col)

                srsi_cols_exist = all(col in df_display.columns for col in srsi_value_cols)
                if srsi_cols_exist:
                     logger.debug("Applying SRSI row-wise styling.")
                     styled_table = styled_table.apply(style_stoch_rsi, axis=1, subset=srsi_value_cols)

                # Apply formatting LAST
                styled_table = styled_table.format(formatters, na_rep="N/A") # Removed precision=4, handled by individual formatters

                # --- Display Table ---
                logger.info("Displaying styled table DataFrame.");
                st.dataframe(styled_table, use_container_width=True,
                             column_config={
                                 "MA/MACD Cross Alert": st.column_config.TextColumn(label="MA/MACD\nCross Alert", width="small"),
                                 "Composite Score": st.column_config.TextColumn(label="Composite\nScore", width="small"),
                                 "Link": st.column_config.LinkColumn("CoinGecko", help="CoinGecko Link", display_text="🔗 Link", width="small")
                             })

            except Exception as df_err: logger.exception("Error creating/styling table DataFrame:"); st.error(f"Error displaying table: {df_err}")
        else: logger.warning("No valid table results to display."); st.warning("No valid crypto results to display in the table.")

    # --- Chart Section ---
    st.divider(); st.subheader("💹 Detailed Coin Chart")
    sel_col, price_col = st.columns([3, 1])
    with sel_col: chart_symbol = st.selectbox("Select coin for chart:", options=SYMBOLS, index=0, key="chart_coin_selector")
    with price_col:
        st.write(""); st.write("")
        if chart_symbol:
            chart_coin_id = SYMBOL_TO_ID_MAP.get(chart_symbol)
            if chart_coin_id and not market_data_df.empty and chart_coin_id in market_data_df.index:
                 latest_price = market_data_df.loc[chart_coin_id].get('current_price', np.nan)
                 if pd.notna(latest_price): st.metric(label=f"Current Price {chart_symbol}", value=f"${latest_price:,.4f}")
                 else: st.caption(f"Current price N/A")
            else: st.caption(f"Current price N/A")

    chart_placeholder = st.empty()
    if chart_symbol:
        chart_coin_id = SYMBOL_TO_ID_MAP.get(chart_symbol)
        if chart_coin_id:
            logger.info(f"CHART: Attempting to load data for {chart_symbol} ({chart_coin_id}) chart.")
            with chart_placeholder:
                 with st.spinner(f"Loading data and chart for {chart_symbol}..."):
                    chart_hist_df, chart_status = get_coingecko_historical_data_for_chart(chart_coin_id, VS_CURRENCY, DAYS_HISTORY_DAILY)
                    if chart_status == "Success" and not chart_hist_df.empty:
                        fig = create_coin_chart(chart_hist_df.copy(), chart_symbol)
                        if fig: st.plotly_chart(fig, use_container_width=True); logger.info(f"CHART: Chart for {chart_symbol} displayed.")
                        else: st.error(f"Could not generate chart for {chart_symbol} (indicator calculation or internal error, see log).")
                    else: logger.error(f"CHART: Failed to load historical data for {chart_symbol}. Status: {chart_status}"); st.error(f"Could not load historical data for {chart_symbol} chart. ({chart_status})")
        else: st.error(f"CoinGecko ID not found for symbol {chart_symbol}."); logger.error(f"CHART: CoinGecko ID not found for {chart_symbol} in map.")

    # --- Legend (Improved v1.1) ---
    st.divider();
    with st.expander("📘 Indicator, Signal & Legend Guide", expanded=False):
        st.markdown("""
        *Disclaimer: This dashboard is provided for informational and educational purposes only and does not constitute financial advice.*

        **Market Overview:** (See above section for details)

        **Crypto Technical Analysis Table:**
        *   **Rank:** Market cap rank (CoinGecko).
        *   **MA/MACD Cross Alert / Composite Score:** **Experimental** signals. **NOT trading advice.** Signal logic refined in v1.1. (See colors below).
        *   **Price:** Current price ($) (CoinGecko).
        *   **% 1h...1y:** Price changes. <span style="color:red;">Red</span>=Negative, <span style="color:green;">Green</span>=Positive. 🔥 Icon on % 1h indicates market breadth (>=8 coins positive).
        *   **RSI (1d, 1w, 1mo):** Relative Strength Index (0-100).
            *   <span style="color:#dc3545; font-weight:bold;">Value > 70</span>: Overbought.
            *   <span style="color:#198754; font-weight:bold;">Value < 30</span>: Oversold.
        *   **SRSI %K / %D (1d):** Stochastic RSI (0-100).
            *   <span style="color:#198754; font-weight:bold;">Values (Bold Green)</span>: Oversold (K&D < 20).
            *   <span style="color:#dc3545; font-weight:bold;">Values (Bold Red)</span>: Overbought (K&D > 80).
            *   <span style="color:#28a745;">Values (Green)</span>: Bullish Crossover (K > D).
            *   <span style="color:#fd7e14;">Values (Orange)</span>: Bearish Crossover (K < D).
        *   **MACD Hist (1d):** MACD Histogram.
            *   <span style="color:green;">Value > 0 (Green)</span>: Bullish momentum.
            *   <span style="color:red;">Value < 0 (Red)</span>: Bearish momentum.
        *   **MA(7d/20d/30d/50d):** Simple Moving Averages. Trend lines. *Values not colored.*
        *   **BB %B / Width / Width %Chg:** Bollinger Bands (20d, 2 std dev). Measure volatility and price relative to range.
            *   **%B (%):** Price position relative to bands (%). >100 = Above Upper; <0 = Below Lower. <span style="color:red;">Red</span>/% <span style="color:green;">Green</span> color indicates value.
            *   **Width (%):** Tightness of bands (%). <span style="color:red;">Red</span>/% <span style="color:green;">Green</span> color indicates value.
            *   **Width %Chg (%):** Daily % change in Band Width. <span style="color:red;">Red</span>=Narrowing, <span style="color:green;">Green</span>=Widening.
        *   **VWAP (1d):** Volume Weighted Average Price. *Value not colored.*
        *   **VWAP %:** Daily % change of VWAP. <span style="color:red;">Red</span>=Decreasing, <span style="color:green;">Green</span>=Increasing.
        *   **Volume 24h:** Trading volume ($) (CoinGecko).
        *   **Link:** CoinGecko page link.
        *   **N/A:** Data Not Available.

        **Signal Column Colors & Meanings:**
        *   <span style="color:#198754; font-weight:bold;">⚡️ Strong Buy</span> / <span style="color:#28a745;">🟢 Buy</span>: Bullish technical conditions.
        *   <span style="color:#dc3545; font-weight:bold;">🚨 Strong Sell</span> / <span style="color:#fd7e14;">🔴 Sell</span>: Bearish technical conditions.
        *   <span style="color:#20c997;">⏳ CTB</span>: Consider To Buy (Monitor).
        *   <span style="color:#ffc107; color:#000;">⚠️ CTS</span>: Consider To Sell (Monitor).
        *   <span style="color:#6c757d;">🟡 Hold</span>: Neutral / Mixed conditions.
        *   <span style="color:#adb5bd;">⚪️ N/A</span>: Calculation failed.

        **Detailed Coin Chart:**
        *   Displays daily Candlesticks, Moving Averages (20d, 50d), and RSI (14d).

        **Important Notes:**
        *   Table data fetch is slowed (6s/coin). Initial load takes time.
        *   Chart data fetch is faster.
        *   Traditional Market data has **4h cache**.
        *   **DYOR (Do Your Own Research).**
        """, unsafe_allow_html=True)
    st.divider(); st.caption("Disclaimer: Informational/educational tool only. Not financial advice. DYOR.")
except Exception as main_exception: logger.exception("!!! UNHANDLED ERROR IN MAIN APP EXECUTION !!!"); st.error(f"Unexpected error: {main_exception}. Check the log.")

# --- Application Log Display ---
st.divider(); st.subheader("📄 Application Log"); st.caption("Logs generated during the last app run (INFO Level). Useful for monitoring.")
log_content = log_stream.getvalue(); st.text_area("Log:", value=log_content, height=300, key="log_display_area", help="Select All (Ctrl+A or Cmd+A) and Copy (Ctrl+C or Cmd+C) to analyze or share.")
logger.info("--- End of Streamlit script execution: app.py ---"); log_stream.close()
