# Version: v1.4.4 - Implement safe_formatter, add detailed pre-styling logs
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time
from alpha_vantage.timeseries import TimeSeries
import logging
import io

# --- START: Logging Configuration ---
log_stream = io.StringIO()
logging.basicConfig(
    stream=log_stream,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)
logger.info("Logging configured for UI (v1.4.4).")
# --- END: Logging Configuration ---

try:
    from zoneinfo import ZoneInfo
    logger.info("Module 'zoneinfo' imported.")
except ImportError:
    logger.warning("Module 'zoneinfo' not found. Using offset approximation for Rome timezone.")
    st.warning("Module 'zoneinfo' not found. Using offset approximation for Rome timezone.")
    ZoneInfo = None

st.set_page_config(layout="wide", page_title="Crypto Technical Dashboard Pro", page_icon="📈")
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
TRAD_TICKERS_AV = ['SPY', 'QQQ', 'GLD', 'SLV', 'UNG', 'UVXY', 'TQQQ', 'NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR', 'COIN', 'MSTR']
VS_CURRENCY = "usd"
CACHE_TTL, CACHE_HIST_TTL, CACHE_TRAD_TTL = 1800, 3600, 14400
DAYS_HISTORY_DAILY, DAYS_HISTORY_HOURLY = 365, 7
RSI_PERIOD, RSI_OB, RSI_OS = 14, 70.0, 30.0
SRSI_PERIOD, SRSI_K, SRSI_D, SRSI_OB, SRSI_OS = 14, 3, 3, 80.0, 20.0
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MA_SHORT, MA_MEDIUM, MA_LONG, MA_XLONG = 7, 20, 50, 30
BB_PERIOD, BB_STD_DEV = 20, 2.0
VWAP_PERIOD = 14
logger.info(f"Global config: {NUM_COINS} coins, Trad Tickers: {len(TRAD_TICKERS_AV)}")

# --- FUNCTION DEFINITIONS (General) ---
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
        response = requests.get(url, params=params, timeout=20); response.raise_for_status()
        data = response.json();
        if not data: logger.warning("CG Live API: Empty data."); return pd.DataFrame(), timestamp_utc
        df = pd.DataFrame(data);
        if not df.empty: df.set_index('id', inplace=True)
        numeric_cols = ['current_price', 'market_cap_rank', 'total_volume',
                        'price_change_percentage_1h_in_currency', 'price_change_percentage_24h_in_currency',
                        'price_change_percentage_7d_in_currency', 'price_change_percentage_30d_in_currency',
                        'price_change_percentage_1y_in_currency']
        for col in numeric_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        st.session_state["api_warning_shown"] = False; logger.info(f"CG live data fetched for {len(df)} coins."); return df, timestamp_utc
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code; logger.warning(f"HTTP Err CG Market API (Status: {status_code}): {http_err}")
        if status_code == 429 and not st.session_state.get("api_warning_shown", False): st.warning("Warn CG API (Live): Rate limit (429)."); st.session_state["api_warning_shown"] = True
        elif status_code != 429 or st.session_state.get("api_warning_shown", False): st.error(f"HTTP Err CG Market API (Status: {status_code}): {http_err}")
        return pd.DataFrame(), timestamp_utc
    except Exception as e: logger.exception("Err Processing CG Market Data:"); st.error(f"Err Proc CG Market Data: {e}"); return pd.DataFrame(), timestamp_utc

@st.cache_data(ttl=CACHE_HIST_TTL, show_spinner=False)
def get_coingecko_historical_data(coin_id, currency, days, interval='daily'):
    logger.debug(f"TABLE Hist: {coin_id} ({interval}), 6s delay..."); time.sleep(6.0)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"; params = {'vs_currency': currency, 'days': str(days), 'interval': interval, 'precision': 'full'}
    status_msg = f"Unknown Error ({coin_id}, {interval})"
    try:
        response = requests.get(url, params=params, timeout=25); response.raise_for_status(); data = response.json()
        if not data or 'prices' not in data or not data['prices']: status_msg = f"No Prices ({coin_id}, {interval})"; logger.warning(status_msg); return pd.DataFrame(), status_msg
        
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms', utc=True)
        prices_df.set_index('timestamp', inplace=True)
        hist_df = prices_df
        hist_df['close'] = pd.to_numeric(hist_df['close'], errors='coerce')

        if 'total_volumes' in data and data['total_volumes']:
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms', utc=True)
            volumes_df.set_index('timestamp', inplace=True)
            volumes_df['volume'] = pd.to_numeric(volumes_df['volume'], errors='coerce')
            hist_df = prices_df.join(volumes_df, how='outer')
        else:
            hist_df['volume'] = 0.0
        
        hist_df = hist_df.interpolate(method='time').ffill().bfill()
        hist_df['high'] = hist_df['close'] # Approximate if not available
        hist_df['low'] = hist_df['close']  # Approximate if not available
        hist_df['open'] = hist_df['close'].shift(1)
        if not hist_df.empty and pd.isna(hist_df.iloc[0]['open']):
             hist_df.loc[hist_df.index[0], 'open'] = hist_df.iloc[0]['close']
        
        for col in ['open', 'high', 'low', 'close', 'volume']: # Final numeric check
            if col in hist_df.columns: hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')

        hist_df = hist_df[~hist_df.index.duplicated(keep='last')].sort_index().dropna(subset=['close'])
        if hist_df.empty: status_msg = f"Processed Empty ({coin_id}, {interval})"; logger.warning(status_msg); return pd.DataFrame(), status_msg
        status_msg = "Success"; logger.info(f"TABLE Hist: {coin_id} ({interval}), {len(hist_df)} rows."); return hist_df, status_msg
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        status_msg = f"HTTP Err {status_code} ({coin_id}, {interval})"
        if status_code == 429: status_msg = f"Rate Limit (429) ({coin_id}, {interval})"
        logger.warning(f"TABLE Hist: CG API Err: {status_msg}"); return pd.DataFrame(), status_msg
    except Exception as e: status_msg = f"Generic Err ({type(e).__name__}) ({coin_id}, {interval})"; logger.exception(f"TABLE Hist: Err proc CG Hist for {coin_id}:"); return pd.DataFrame(), status_msg

# ... (get_fear_greed_index, get_global_market_data_cg, get_etf_flow, get_traditional_market_data_av - unchanged from v1.4.3, assuming they are not the source of *this specific table error*)
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
    logger.info(f"Attempting Alpha Vantage fetch for {len(tickers)} tickers."); data = {ticker: {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'} for ticker in tickers}; api_key = None
    try: api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]; logger.info("Alpha Vantage API key read from secrets.")
    except KeyError: logger.error("Secret 'ALPHA_VANTAGE_API_KEY' not defined."); st.error("Configuration Error: Alpha Vantage API key not found. Trad data unavailable."); return data
    except Exception as e: logger.exception("Unexpected error reading Alpha Vantage secrets:"); st.error(f"Error reading AV secrets: {e}"); return data
    if not api_key: logger.error("Alpha Vantage API key found but is empty."); st.error("Configuration Error: Alpha Vantage API key is empty. Trad data unavailable."); return data
    ts = TimeSeries(key=api_key, output_format='pandas'); calls_made = 0; max_calls_per_minute = 5; max_calls_this_run = 25; delay_between_calls = (60.0 / max_calls_per_minute) + 1.0
    for ticker_sym in tickers:
        if calls_made >= max_calls_this_run: msg = f"AV call limit for this run ({max_calls_this_run}) reached. Stop fetch: {ticker_sym}."; logger.warning(msg); st.warning(msg); break
        try:
            logger.info(f"AV Fetch for {ticker_sym} (Call #{calls_made+1}/{max_calls_this_run}, Pause {delay_between_calls:.1f}s)..."); time.sleep(delay_between_calls); quote_data, meta_data = ts.get_quote_endpoint(symbol=ticker_sym); calls_made += 1
            if not quote_data.empty:
                data[ticker_sym]['price'] = pd.to_numeric(quote_data['05. price'].iloc[0], errors='coerce')
                data[ticker_sym]['change'] = pd.to_numeric(quote_data['09. change'].iloc[0], errors='coerce')
                data[ticker_sym]['change_percent'] = str(quote_data['10. change percent'].iloc[0]) # Keep as string, handle in format_delta
                logger.info(f"AV data for {ticker_sym} fetched OK.")
            else: logger.warning(f"Empty response from AV for {ticker_sym}.");
        except ValueError as ve:
             msg = f"Alpha Vantage Error (ValueError) for {ticker_sym}: {ve}"; logger.warning(msg); st.warning(msg); ve_str = str(ve).lower()
             if "call frequency" in ve_str or "api key" in ve_str or "limit" in ve_str or "premium" in ve_str: logger.error("Critical AV API key/limit error. Stopping fetch."); st.error("AV API key/limit error. Stopping fetch."); break
        except Exception as e: msg = f"Generic error AV for {ticker_sym}: {e}"; logger.exception(msg); st.warning(msg);
    logger.info(f"Finished Alpha Vantage fetch. Made {calls_made} calls."); return data

# --- Indicator Calculation Functions (Ensuring numeric inputs/outputs) ---
def _ensure_numeric_series(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series): return pd.Series(dtype=float)
    return pd.to_numeric(series, errors='coerce').dropna()

def calculate_rsi_manual(series: pd.Series, period: int = RSI_PERIOD) -> float:
    series = _ensure_numeric_series(series)
    if len(series) < period + 1: return np.nan
    delta = series.diff(); gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean(); avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    if avg_gain.empty or avg_loss.empty or avg_gain.isna().all() or avg_loss.isna().all(): return np.nan # Added empty check
    last_avg_loss = avg_loss.iloc[-1]; last_avg_gain = avg_gain.iloc[-1]
    if pd.isna(last_avg_loss) or pd.isna(last_avg_gain): return np.nan
    if last_avg_loss == 0: return 100.0 if last_avg_gain > 0 else 50.0
    rs = last_avg_gain / last_avg_loss; rsi = 100.0 - (100.0 / (1.0 + rs))
    return max(0.0, min(100.0, rsi)) if pd.notna(rsi) else np.nan

def calculate_stoch_rsi(series: pd.Series, rsi_period: int = RSI_PERIOD, stoch_period: int = SRSI_PERIOD, k_smooth: int = SRSI_K, d_smooth: int = SRSI_D) -> tuple[float, float]:
    series = _ensure_numeric_series(series)
    if len(series) < rsi_period + stoch_period + max(k_smooth, d_smooth) -1 : return np.nan, np.nan
    delta = series.diff(); gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean(); avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    if avg_loss.replace(0, np.nan).isna().all(): return np.nan, np.nan # Prevent division by zero if all avg_loss is 0
    rs = avg_gain / avg_loss.replace(0, np.nan); rsi_series = (100.0 - (100.0 / (1.0 + rs))).dropna()
    if len(rsi_series) < stoch_period: return np.nan, np.nan
    min_rsi = rsi_series.rolling(window=stoch_period, min_periods=1).min(); max_rsi = rsi_series.rolling(window=stoch_period, min_periods=1).max() # min_periods=1 for robustness
    range_rsi = (max_rsi - min_rsi).replace(0, np.nan) # Avoid division by zero
    stoch_rsi_k_raw = (100 * (rsi_series - min_rsi) / range_rsi).dropna()
    if len(stoch_rsi_k_raw) < k_smooth: return np.nan, np.nan
    stoch_rsi_k = stoch_rsi_k_raw.rolling(window=k_smooth, min_periods=1).mean() # min_periods=1
    if stoch_rsi_k.empty: return np.nan, np.nan
    if len(stoch_rsi_k.dropna()) < d_smooth :
        last_k_val = stoch_rsi_k.iloc[-1] if not stoch_rsi_k.empty else np.nan
        k_final = max(0.0, min(100.0, last_k_val)) if pd.notna(last_k_val) else np.nan; return k_final, np.nan
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_smooth, min_periods=1).mean() # min_periods=1
    if stoch_rsi_d.empty: return (max(0.0, min(100.0, stoch_rsi_k.iloc[-1])) if pd.notna(stoch_rsi_k.iloc[-1]) else np.nan) if not stoch_rsi_k.empty else np.nan, np.nan

    last_k = stoch_rsi_k.iloc[-1] if not stoch_rsi_k.empty else np.nan
    last_d = stoch_rsi_d.iloc[-1] if not stoch_rsi_d.empty else np.nan
    k_final = max(0.0, min(100.0, last_k)) if pd.notna(last_k) else np.nan
    d_final = max(0.0, min(100.0, last_d)) if pd.notna(last_d) else np.nan
    return k_final, d_final


def calculate_macd_manual(series: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> tuple[float, float, float]:
    series = _ensure_numeric_series(series)
    if len(series) < slow + signal - 1: return np.nan, np.nan, np.nan
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    if macd_line.empty or signal_line.empty or histogram.empty: return np.nan, np.nan, np.nan
    last_macd = macd_line.iloc[-1] if pd.notna(macd_line.iloc[-1]) else np.nan
    last_signal = signal_line.iloc[-1] if pd.notna(signal_line.iloc[-1]) else np.nan
    last_hist = histogram.iloc[-1] if pd.notna(histogram.iloc[-1]) else np.nan
    return last_macd, last_signal, last_hist

def calculate_sma_manual(series: pd.Series, period: int) -> float:
    series = _ensure_numeric_series(series)
    if len(series) < period: return np.nan
    sma = series.rolling(window=period, min_periods=period).mean().iloc[-1]
    return sma if pd.notna(sma) else np.nan


def calculate_vwap_manual(df_slice: pd.DataFrame, period: int = VWAP_PERIOD) -> float:
    if not isinstance(df_slice, pd.DataFrame) or df_slice.empty or not all(col in df_slice.columns for col in ['close', 'volume']): return np.nan
    df_valid_slice = df_slice[['close', 'volume']].copy()
    df_valid_slice['close'] = pd.to_numeric(df_valid_slice['close'], errors='coerce')
    df_valid_slice['volume'] = pd.to_numeric(df_valid_slice['volume'], errors='coerce')
    df_valid_slice = df_valid_slice.dropna()
    if len(df_valid_slice) < period: return np.nan
    df_period = df_valid_slice.iloc[-period:]
    if df_period.empty: return np.nan
    pv = df_period['close'] * df_period['volume']; total_volume = df_period['volume'].sum()
    if total_volume == 0 or pd.isna(total_volume): return df_period['close'].iloc[-1] if not df_period.empty else np.nan
    vwap = pv.sum() / total_volume
    return vwap if pd.notna(vwap) else np.nan

def calculate_bbands_manual(series: pd.Series, period: int = BB_PERIOD, std_dev: float = BB_STD_DEV) -> tuple[float, float, float, float, float, float]:
    series = _ensure_numeric_series(series)
    min_len_bb_change = period + 1 # Need at least period+1 for change calculation
    if len(series) < period: return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    middle_band_series = series.rolling(window=period, min_periods=period).mean()
    std_series = series.rolling(window=period, min_periods=period).std(ddof=0) # ddof=0 for population std often used in BB

    if middle_band_series.empty or std_series.empty or middle_band_series.iloc[-1] is np.nan or std_series.iloc[-1] is np.nan:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    middle_band_now = middle_band_series.iloc[-1]
    std_now = std_series.iloc[-1]
    
    upper_band_now = middle_band_now + (std_dev * std_now)
    lower_band_now = middle_band_now - (std_dev * std_now)
    last_price = series.iloc[-1]
    
    band_range_now = upper_band_now - lower_band_now
    percent_b_now = ((last_price - lower_band_now) / band_range_now) * 100 if band_range_now > 0 else np.nan
    bandwidth_now = (band_range_now / middle_band_now) * 100 if middle_band_now > 0 else np.nan

    bandwidth_change = np.nan
    if len(series) >= min_len_bb_change and len(middle_band_series) >= 2 and len(std_series) >= 2: # Ensure there's a previous value
        middle_band_prev = middle_band_series.iloc[-2]
        std_prev = std_series.iloc[-2]
        if pd.notna(middle_band_prev) and pd.notna(std_prev):
            upper_band_prev = middle_band_prev + (std_dev * std_prev)
            lower_band_prev = middle_band_prev - (std_dev * std_prev)
            band_range_prev = upper_band_prev - lower_band_prev
            bandwidth_prev = (band_range_prev / middle_band_prev) * 100 if middle_band_prev > 0 else np.nan
            if pd.notna(bandwidth_now) and pd.notna(bandwidth_prev) and bandwidth_prev != 0:
                bandwidth_change = ((bandwidth_now - bandwidth_prev) / bandwidth_prev) * 100
    
    return middle_band_now, upper_band_now, lower_band_now, percent_b_now, bandwidth_now, bandwidth_change

def compute_all_indicators(symbol: str, hist_daily_df: pd.DataFrame) -> dict:
    indicators = { "RSI (1d)": np.nan, "RSI (1w)": np.nan, "RSI (1mo)": np.nan, "SRSI %K (1d)": np.nan, "SRSI %D (1d)": np.nan, "MACD Line (1d)": np.nan, "MACD Signal (1d)": np.nan, "MACD Hist (1d)": np.nan, f"MA({MA_SHORT}d)": np.nan, f"MA({MA_MEDIUM}d)": np.nan, f"MA({MA_LONG}d)": np.nan, f"MA({MA_XLONG}d)": np.nan, "BB %B": np.nan, "BB Width": np.nan, "BB Width %Chg": np.nan, "VWAP (1d)": np.nan, "VWAP %": np.nan }
    if hist_daily_df.empty or 'close' not in hist_daily_df.columns:
        logger.warning(f"{symbol}: TABLE: Empty/invalid daily historical data for indicator calculation.")
        return indicators

    close_series_numeric = pd.to_numeric(hist_daily_df['close'], errors='coerce')
    volume_series_numeric = pd.to_numeric(hist_daily_df.get('volume'), errors='coerce') if 'volume' in hist_daily_df else pd.Series([np.nan]*len(hist_daily_df), index=hist_daily_df.index)
    
    df_for_vwap_calc = pd.DataFrame({'close': close_series_numeric, 'volume': volume_series_numeric})
    close_daily = close_series_numeric.dropna(); len_daily = len(close_daily)
    
    min_len_rsi_base = RSI_PERIOD + 1; min_len_srsi_base = RSI_PERIOD + SRSI_PERIOD + max(SRSI_K, SRSI_D) + 5; min_len_macd_base = MACD_SLOW + MACD_SIGNAL + 5;
    min_len_sma_short = MA_SHORT; min_len_sma_medium = MA_MEDIUM; min_len_sma_xlong = MA_XLONG; min_len_sma_long = MA_LONG; min_len_bb = BB_PERIOD + 1;
    min_len_vwap_base = VWAP_PERIOD; min_len_vwap_change = VWAP_PERIOD + 1


    if len_daily >= min_len_rsi_base: indicators["RSI (1d)"] = calculate_rsi_manual(close_daily, RSI_PERIOD)
    if len_daily >= min_len_srsi_base: indicators["SRSI %K (1d)"], indicators["SRSI %D (1d)"] = calculate_stoch_rsi(close_daily, RSI_PERIOD, SRSI_PERIOD, SRSI_K, SRSI_D)
    if len_daily >= min_len_macd_base: indicators["MACD Line (1d)"], indicators["MACD Signal (1d)"], indicators["MACD Hist (1d)"] = calculate_macd_manual(close_daily, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    if len_daily >= min_len_sma_short: indicators[f"MA({MA_SHORT}d)"] = calculate_sma_manual(close_daily, MA_SHORT)
    if len_daily >= min_len_sma_medium: indicators[f"MA({MA_MEDIUM}d)"] = calculate_sma_manual(close_daily, MA_MEDIUM)
    if len_daily >= min_len_sma_xlong: indicators[f"MA({MA_XLONG}d)"] = calculate_sma_manual(close_daily, MA_XLONG)
    if len_daily >= min_len_sma_long: indicators[f"MA({MA_LONG}d)"] = calculate_sma_manual(close_daily, MA_LONG)
    if len_daily >= min_len_bb:  indicators["BB %B"], indicators["BB Width"], indicators["BB Width %Chg"] = calculate_bbands_manual(close_daily, BB_PERIOD, BB_STD_DEV)[3:6] # Get relevant parts

    vwap_df_ready = df_for_vwap_calc.dropna(subset=['close', 'volume'])
    if len(vwap_df_ready) >= min_len_vwap_base:
        indicators["VWAP (1d)"] = calculate_vwap_manual(vwap_df_ready.iloc[-VWAP_PERIOD:], VWAP_PERIOD) # Pass already sliced part
        if len(vwap_df_ready) >= min_len_vwap_change:
            vwap_today = indicators["VWAP (1d)"]
            vwap_yesterday = calculate_vwap_manual(vwap_df_ready.iloc[-(VWAP_PERIOD + 1):-1], VWAP_PERIOD)
            if pd.notna(vwap_today) and pd.notna(vwap_yesterday) and vwap_yesterday != 0: indicators["VWAP %"] = ((vwap_today - vwap_yesterday) / vwap_yesterday) * 100
    
    if len_daily > min_len_rsi_base and pd.api.types.is_datetime64_any_dtype(close_daily.index):
        try:
            df_weekly = close_daily.resample('W-MON').last().dropna()
            if len(df_weekly) >= min_len_rsi_base: indicators["RSI (1w)"] = calculate_rsi_manual(df_weekly, RSI_PERIOD)
        except Exception as e: logger.exception(f"{symbol}: TABLE: Err calc weekly RSI:")
        try:
            df_monthly = close_daily.resample('ME').last().dropna()
            if len(df_monthly) >= min_len_rsi_base: indicators["RSI (1mo)"] = calculate_rsi_manual(df_monthly, RSI_PERIOD)
        except Exception as e: logger.exception(f"{symbol}: TABLE: Err calc monthly RSI:")
    return indicators

def generate_gpt_signal(rsi_1d, rsi_1w, macd_hist, ma_short, ma_medium, ma_long, srsi_k, srsi_d, bb_pct_b, bb_width_chg, vwap_1d, current_price):
    try: current_price_num = float(current_price) if pd.notna(current_price) else np.nan
    except: current_price_num = np.nan
    required_inputs = [ rsi_1d, macd_hist, ma_short, ma_medium, ma_long, srsi_k, srsi_d, bb_pct_b, bb_width_chg, vwap_1d, current_price_num ]
    if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in required_inputs): return "⚪️ N/A" # More robust NaN check
    score = 0 # ... (rest of GPT signal logic - assumed correct if inputs are numeric)
    price_vs_ma7 = current_price_num > ma_short; price_vs_ma20 = current_price_num > ma_medium; price_vs_ma50 = current_price_num > ma_long
    ma7_vs_ma20 = ma_short > ma_medium; ma20_vs_ma50 = ma_medium > ma_long
    if price_vs_ma7 and price_vs_ma20 and price_vs_ma50 and ma7_vs_ma20 and ma20_vs_ma50: score += 3
    elif price_vs_ma7 and ma7_vs_ma20: score += 1.5
    elif price_vs_ma50: score += 0.5
    if not price_vs_ma7 and not price_vs_ma20 and not price_vs_ma50 and not ma7_vs_ma20 and not ma20_vs_ma50: score -= 3
    elif not price_vs_ma7 and not ma7_vs_ma20: score -= 1.5
    elif not price_vs_ma50: score -= 0.5
    if current_price_num > vwap_1d: score += 1
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
    if pd.notna(bb_pct_b): # Check if bb_pct_b is not NaN
        if bb_pct_b > 100: score -= 0.5
        elif bb_pct_b < 0: score += 0.5
    if pd.notna(bb_width_chg): # Check if bb_width_chg is not NaN
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
    try: current_price_num = float(current_price) if pd.notna(current_price) else np.nan
    except: current_price_num = np.nan
    required_inputs = [ma_medium, ma_long, macd_hist, rsi_1d, vwap_1d, current_price_num]
    if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in required_inputs): return "⚪️ N/A"
    # ... (rest of Gemini alert logic)
    is_ma_cross_bullish = ma_medium > ma_long; is_ma_cross_bearish = ma_medium < ma_long
    is_momentum_positive = macd_hist > 0; is_momentum_negative = macd_hist < 0
    is_price_confirm_bullish = current_price_num > ma_medium and current_price_num > vwap_1d
    is_price_confirm_bearish = current_price_num < ma_medium and current_price_num < vwap_1d
    is_rsi_ok_bullish = rsi_1d < RSI_OB + 5
    is_rsi_ok_bearish = rsi_1d > RSI_OS - 5
    if is_ma_cross_bullish and is_momentum_positive and is_price_confirm_bullish and is_rsi_ok_bullish: return "⚡️ Strong Buy"
    elif is_ma_cross_bearish and is_momentum_negative and is_price_confirm_bearish and is_rsi_ok_bearish: return "🚨 Strong Sell"
    else: return "🟡 Hold"

# --- START OF MAIN APP EXECUTION ---
logger.info("Starting main UI execution.")
try:
    # ... (Title, Refresh, Timestamp, Market Overview - unchanged from v1.4.3)
    col_title, col_button_placeholder, col_button = st.columns([4, 1, 1])
    with col_title: st.title("📈 Crypto Technical Dashboard Pro")
    with col_button:
        st.write("")
        if st.button("🔄 Refresh", help="Force data refresh (clears cache)", key="refresh_button"):
            logger.info("Refresh button clicked.");
            if 'api_warning_shown' in st.session_state: del st.session_state['api_warning_shown']
            st.cache_data.clear(); st.query_params.clear(); st.rerun()

    last_update_placeholder = st.empty()
    st.caption(f"Cache TTL: Live ({CACHE_TTL/60:.0f}m), Table History ({CACHE_HIST_TTL/60:.0f}m), Traditional ({CACHE_TRAD_TTL/3600:.0f}h).")

    st.markdown("---"); st.subheader("🌐 Market Overview")
    fear_greed_value = get_fear_greed_index(); total_market_cap = get_global_market_data_cg(VS_CURRENCY); etf_flow_value = get_etf_flow(); traditional_market_data = get_traditional_market_data_av(TRAD_TICKERS_AV)
    def format_delta(change_val, change_pct_str): # Used for traditional_market_data
        delta_string = None
        # Ensure change_val is numeric for formatting
        change_val_num = pd.to_numeric(change_val, errors='coerce')

        if pd.notna(change_val_num) and isinstance(change_pct_str, str) and change_pct_str not in ['N/A', '', None]:
            try:
                change_pct_val = float(change_pct_str.replace('%','').strip())
                delta_string = f"{change_val_num:+.2f} ({change_pct_val:+.2f}%)"
            except (ValueError, AttributeError):
                delta_string = f"{change_val_num:+.2f} (?%)"
        elif pd.notna(change_val_num):
            delta_string = f"{change_val_num:+.2f}"
        return delta_string

    def render_metric(column, label, value_func=None, ticker=None, data_dict=None, help_text=None):
        value_str = "N/A"; delta_txt = None; d_color = "off"
        if ticker and data_dict:
            trad_info = data_dict.get(ticker, {});
            price = pd.to_numeric(trad_info.get('price'), errors='coerce') # Ensure numeric
            change = pd.to_numeric(trad_info.get('change'), errors='coerce') # Ensure numeric
            change_pct = trad_info.get('change_percent', 'N/A')

            value_str = f"${price:,.2f}" if pd.notna(price) else "N/A"
            delta_txt = format_delta(change, change_pct)
            if pd.notna(change): d_color = "normal"
        elif value_func: # For F&G, MCap, ETF Flow
            try: value_str = value_func(); value_str = str(value_str) if value_str is not None else "N/A"
            except Exception as e: logger.error(f"Error in value_func '{label}': {e}"); value_str = "Error"
        column.metric(label=label, value=value_str, delta=delta_txt, delta_color=d_color, help=help_text)

    overview_items_row1 = [ ("Fear & Greed Index", None, get_fear_greed_index, "Source: Alternative.me"), (f"Total Crypto M.Cap ({VS_CURRENCY.upper()})", None, lambda: f"${format_large_number(total_market_cap)}", "Source: CoinGecko"), ("Crypto ETFs Flow (Daily)", None, get_etf_flow, "Data N/A"), ("S&P 500 (SPY)", "SPY", None, "Source: AV (ETF)"), ("Nasdaq (QQQ)", "QQQ", None, "Source: AV (ETF)") ]
    overview_cols_1 = st.columns(len(overview_items_row1));
    for i, (label, ticker, func, help_text) in enumerate(overview_items_row1): render_metric(overview_cols_1[i], label, value_func=func, ticker=ticker, data_dict=traditional_market_data, help_text=help_text)
    overview_items_row2 = [ ("Gold (GLD)", "GLD", None, "Source: AV (ETF)"), ("Silver (SLV)", "SLV", None, "Source: AV (ETF)"), ("Natural Gas (UNG)", "UNG", None, "Source: AV (ETF)"), ("UVXY (Volatility)", "UVXY", None, "Source: AV"), ("TQQQ (Nasdaq 3x)", "TQQQ", None, "Source: AV") ]
    overview_cols_2 = st.columns(len(overview_items_row2));
    for i, (label, ticker, func, help_text) in enumerate(overview_items_row2): render_metric(overview_cols_2[i], label, value_func=func, ticker=ticker, data_dict=traditional_market_data, help_text=help_text)
    st.markdown("<h6>Major Stocks (Source: Alpha Vantage):</h6>", unsafe_allow_html=True); stock_tickers_row_av = ['NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR', 'COIN', 'MSTR']
    num_stock_cols = 5; stock_cols = st.columns(num_stock_cols);
    for idx, ticker in enumerate(stock_tickers_row_av): render_metric(stock_cols[idx % num_stock_cols], label=ticker, ticker=ticker, data_dict=traditional_market_data, help_text=f"Ticker: {ticker}")
    st.markdown("---")


    st.subheader(f"📊 Crypto Technical Analysis ({NUM_COINS} Assets)")
    market_data_df, last_cg_update_utc = get_coingecko_market_data(COINGECKO_IDS_LIST, VS_CURRENCY)

    if last_cg_update_utc: # Timestamp display
        try:
            local_tz_name = "Europe/Rome"
            if ZoneInfo: local_tz = ZoneInfo(local_tz_name)
            else: local_tz = None # Fallback handled below

            if last_cg_update_utc.tzinfo is None: last_cg_update_utc = last_cg_update_utc.replace(tzinfo=ZoneInfo("UTC") if ZoneInfo else None)

            if local_tz: last_cg_update_local = last_cg_update_utc.astimezone(local_tz)
            else: offset_hours = 2; last_cg_update_local = last_cg_update_utc + timedelta(hours=offset_hours) # Approx Rome
            
            timestamp_display_str = f"*Live CoinGecko data updated at: **{last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S %Z') if local_tz else last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S')}***"
            if not local_tz: timestamp_display_str += " (Approx. Rome Time)"
            logger.info(f"Displaying timestamp: {last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S %Z') if local_tz else last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S')} ")
        except Exception as e: logger.exception("Err formatting timestamp:"); timestamp_display_str = f"*Timestamp conv err. UTC: {last_cg_update_utc}*"
        last_update_placeholder.markdown(timestamp_display_str)
    else: last_update_placeholder.markdown("*Live CoinGecko data timestamp unavailable.*")


    if market_data_df.empty:
        msg = "CRITICAL: No live CoinGecko data. Table cannot be generated."
        if st.session_state.get("api_warning_shown", False): msg = "Table not generated: Err loading live data (CoinGecko API limit?)."
        logger.error(msg); st.error(msg)
    else:
        results = []; fetch_errors = []; process_start_time = time.time()
        effective_num_coins = len(market_data_df.index)
        show_fire_icon = (market_data_df['price_change_percentage_1h_in_currency'].dropna() > 0).sum() >= FIRE_ICON_THRESHOLD if 'price_change_percentage_1h_in_currency' in market_data_df else False

        spinner_msg = f"Processing table for {effective_num_coins} crypto assets... (~{(effective_num_coins * 6.1 / 60):.1f} min)"
        with st.spinner(spinner_msg):
            for i, coin_id in enumerate(market_data_df.index):
                symbol = next((s for s, cid in SYMBOL_TO_ID_MAP.items() if cid == coin_id), "N/A")
                logger.info(f"--- Proc Table {i+1}/{effective_num_coins}: {symbol} ({coin_id}) ---")
                if symbol == "N/A": fetch_errors.append(f"{coin_id}: ID not in map."); continue
                
                live = market_data_df.loc[coin_id]
                hist_df, status = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_DAILY)
                
                current_price_val = pd.to_numeric(live.get('current_price'), errors='coerce') # Ensure numeric

                if status != "Success":
                    fetch_errors.append(f"{symbol}: Hist.Data - {status}")
                    indicators = compute_all_indicators(symbol, pd.DataFrame()) # Ensures all NaNs
                else:
                    indicators = compute_all_indicators(symbol, hist_df)

                gpt_s = generate_gpt_signal(indicators.get("RSI (1d)"), indicators.get("RSI (1w)"), indicators.get("MACD Hist (1d)"), indicators.get(f"MA({MA_SHORT}d)"), indicators.get(f"MA({MA_MEDIUM}d)"), indicators.get(f"MA({MA_LONG}d)"), indicators.get("SRSI %K (1d)"), indicators.get("SRSI %D (1d)"), indicators.get("BB %B"), indicators.get("BB Width %Chg"), indicators.get("VWAP (1d)"), current_price_val)
                gem_a = generate_gemini_alert(indicators.get(f"MA({MA_MEDIUM}d)"), indicators.get(f"MA({MA_LONG}d)"), indicators.get("MACD Hist (1d)"), indicators.get("RSI (1d)"), indicators.get("VWAP (1d)"), current_price_val)
                
                results.append({
                    "Rank": live.get('market_cap_rank', np.nan), "Symbol": symbol, "Name": live.get('name', coin_id),
                    "MA/MACD Cross Alert": gem_a, "Composite Score": gpt_s,
                    f"Price ({VS_CURRENCY.upper()})": current_price_val, # Already numeric or NaN
                    "% 1h": live.get('price_change_percentage_1h_in_currency', np.nan),
                    "% 24h": live.get('price_change_percentage_24h_in_currency', np.nan),
                    "% 7d": live.get('price_change_percentage_7d_in_currency', np.nan),
                    "% 30d": live.get('price_change_percentage_30d_in_currency', np.nan),
                    "% 1y": live.get('price_change_percentage_1y_in_currency', np.nan),
                    **indicators, # Unpack all calculated indicators (should be numeric/NaN)
                    f"Volume 24h ({VS_CURRENCY.upper()})": live.get('total_volume', np.nan),
                    "Link": f"https://www.coingecko.com/en/coins/{coin_id}"
                })
            logger.info(f"Table processing loop done. Time: {time.time() - process_start_time:.1f}s")

        if fetch_errors:
            st.sidebar.error("Data Fetch/Processing Issues:")
            for err_item in fetch_errors: st.sidebar.caption(f"- {err_item}")

        if results:
            try:
                table_df = pd.DataFrame(results)
                table_df['Rank'] = pd.to_numeric(table_df['Rank'], errors='coerce')
                table_df.set_index('Rank', inplace=True, drop=False) # Keep Rank column for display
                table_df.sort_index(inplace=True)

                cols_order = [
                    "Rank", "Symbol", "Name", "MA/MACD Cross Alert", "Composite Score",
                    f"Price ({VS_CURRENCY.upper()})", "% 1h", "% 24h", "% 7d", "% 30d", "% 1y",
                    "RSI (1d)", "RSI (1w)", "RSI (1mo)", "SRSI %K (1d)", "SRSI %D (1d)",
                    "MACD Hist (1d)", f"MA({MA_SHORT}d)", f"MA({MA_MEDIUM}d)",
                    f"MA({MA_XLONG}d)", f"MA({MA_LONG}d)", "BB %B", "BB Width", "BB Width %Chg",
                    "VWAP (1d)", "VWAP %", f"Volume 24h ({VS_CURRENCY.upper()})", "Link"
                ]
                df_display = table_df[[col for col in cols_order if col in table_df.columns]].copy()
                
                # Ensure all columns intended for numeric formatting are numeric
                numeric_format_cols = [
                    f"Price ({VS_CURRENCY.upper()})", "% 1h", "% 24h", "% 7d", "% 30d", "% 1y",
                    "RSI (1d)", "RSI (1w)", "RSI (1mo)", "SRSI %K (1d)", "SRSI %D (1d)",
                    "MACD Hist (1d)", f"MA({MA_SHORT}d)", f"MA({MA_MEDIUM}d)",
                    f"MA({MA_XLONG}d)", f"MA({MA_LONG}d)", "BB %B", "BB Width", "BB Width %Chg",
                    "VWAP (1d)", "VWAP %", f"Volume 24h ({VS_CURRENCY.upper()})"
                ]
                logger.info("--- Pre-coercion dtypes for df_display ---")
                logger.info(df_display.dtypes)

                for col in numeric_format_cols:
                    if col in df_display.columns:
                        # logger.info(f"Coercing column: {col}, original type: {df_display[col].dtype}")
                        # original_non_numeric = df_display[col][~pd.to_numeric(df_display[col], errors='coerce').notna() & df_display[col].notna()]
                        # if not original_non_numeric.empty:
                        #     logger.warning(f"Non-numeric values found in {col} before coercion: {original_non_numeric.unique()}")
                        df_display[col] = pd.to_numeric(df_display[col], errors='coerce')
                        # if df_display[col].dtype == 'object':
                        #    logger.error(f"COLUMN {col} IS STILL OBJECT AFTER COERCION!")

                logger.info("--- Post-coercion dtypes for df_display ---")
                logger.info(df_display.dtypes)
                # Example: Log a column that often has NaNs if a coin fails
                # if "RSI (1d)" in df_display.columns:
                #     logger.info(f"Sample 'RSI (1d)' after coercion:\n{df_display[['Symbol', 'RSI (1d)']].head(15).to_string()}")


                # --- Safe Formatter Definition ---
                def safe_formatter(fmt_str, col_name="<unknown>"): # Add col_name for logging
                    def _format(x):
                        if pd.isna(x): return "N/A"
                        if not isinstance(x, (int, float, np.number)): # If it's not numeric after all coercions
                            logger.warning(f"SafeFormatter: Non-numeric value '{x}' (type: {type(x)}) in column '{col_name}' encountered by formatter '{fmt_str}'. Coercing to N/A.")
                            return "N/A" # Force non-numeric to N/A here
                        try:
                            return fmt_str.format(x)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"SafeFormatter: Could not format value '{x}' (type: {type(x)}) in column '{col_name}' with '{fmt_str}'. Error: {e}. Returning as string or N/A.")
                            return str(x) # Fallback to string representation
                    return _format

                formatters = {}
                currency_col = f"Price ({VS_CURRENCY.upper()})"
                volume_col = f"Volume 24h ({VS_CURRENCY.upper()})"
                
                if currency_col in df_display.columns: formatters[currency_col] = safe_formatter("${:,.4f}", currency_col)
                if volume_col in df_display.columns: formatters[volume_col] = lambda x: format_large_number(x) # This one is already safe

                pct_cols_all = ["% 1h", "% 24h", "% 7d", "% 30d", "% 1y", "VWAP %", "BB Width", "BB Width %Chg", "BB %B"]
                for col in pct_cols_all:
                    if col in df_display.columns and col != '% 1h': formatters[col] = safe_formatter("{:+.2f}%", col)
                
                def format_1h_with_icon(val): # val should be numeric or NaN due to pre-coercion
                    if pd.isna(val): return "N/A"
                    if not isinstance(val, (int, float, np.number)): # Additional safety
                        logger.warning(f"format_1h_with_icon received non-numeric: {val} (type {type(val)})")
                        return "N/A"
                    icon = "🔥 " if show_fire_icon and val > 0 else ""
                    try:
                        return f"{icon}{val:+.2f}%"
                    except: # Should not happen if val is numeric
                        logger.error(f"format_1h_with_icon failed to format numeric val: {val}")
                        return "N/A"
                if '% 1h' in df_display.columns: formatters['% 1h'] = format_1h_with_icon

                rsi_cols_list = [c for c in df_display.columns if "RSI" in c and "%" not in c and "SRSI" not in c]
                srsi_value_cols = ["SRSI %K (1d)", "SRSI %D (1d)"]
                for col in rsi_cols_list + srsi_value_cols:
                     if col in df_display.columns: formatters[col] = safe_formatter("{:.1f}", col)
                
                macd_hist_col_name = "MACD Hist (1d)"
                if macd_hist_col_name in df_display.columns: formatters[macd_hist_col_name] = safe_formatter("{:+.4f}", macd_hist_col_name)
                
                ma_vwap_cols = [c for c in df_display.columns if ("MA" in c or "VWAP" in c) and "%" not in c]
                for col in ma_vwap_cols:
                     if col in df_display.columns: formatters[col] = safe_formatter("{:,.2f}", col)
                
                # --- Styling Functions (unchanged for now, should receive numeric/NaN) ---
                # ... (highlight_signal_style, highlight_pct_col_style, style_rsi, style_macd_hist, style_stoch_rsi) ...
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
                    if pd.isna(val): return 'color: #adb5bd;' 
                    if not isinstance(val, (int, float, np.number)): return '' 
                    color = 'green' if val > 0 else 'red' if val < 0 else '#6c757d'; return f'color: {color};'

                def style_rsi(val): 
                    if pd.isna(val): return 'color: #adb5bd;'
                    if not isinstance(val, (int, float, np.number)): return ''
                    if val > RSI_OB: return 'color: #dc3545; font-weight: bold;'
                    elif val < RSI_OS: return 'color: #198754; font-weight: bold;'
                    else: return ''

                def style_macd_hist(val): 
                    if pd.isna(val): return 'color: #adb5bd;'
                    if not isinstance(val, (int, float, np.number)): return ''
                    if val > 0: return 'color: green;'
                    elif val < 0: return 'color: red;'
                    else: return ''

                def style_stoch_rsi(row_subset): 
                    k_col = "SRSI %K (1d)"; d_col = "SRSI %D (1d)"
                    default_style = ''; style_k_css = default_style; style_d_css = default_style
                    
                    k_val_num = row_subset[k_col] if k_col in row_subset else np.nan
                    d_val_num = row_subset[d_col] if d_col in row_subset else np.nan

                    if pd.notna(k_val_num) and pd.notna(d_val_num):
                        if k_val_num > SRSI_OB and d_val_num > SRSI_OB: style_str = 'color: #dc3545; font-weight: bold;'
                        elif k_val_num < SRSI_OS and d_val_num < SRSI_OS: style_str = 'color: #198754; font-weight: bold;'
                        elif k_val_num > d_val_num: style_str = 'color: #28a745;'
                        elif k_val_num < d_val_num: style_str = 'color: #fd7e14;'
                        else: style_str = default_style 
                        style_k_css = style_str; style_d_css = style_str
                    elif pd.isna(k_val_num) or pd.isna(d_val_num): 
                        style_k_css = 'color: #adb5bd;' if pd.isna(k_val_num) else ''
                        style_d_css = 'color: #adb5bd;' if pd.isna(d_val_num) else ''
                    
                    output_styles = pd.Series('', index=row_subset.index, dtype=str)
                    if k_col in row_subset.index: output_styles[k_col] = style_k_css
                    if d_col in row_subset.index: output_styles[d_col] = style_d_css
                    return output_styles

                styled_table = df_display.style
                
                # Apply styling functions
                for col_name in ["MA/MACD Cross Alert", "Composite Score"]:
                    if col_name in df_display.columns: styled_table = styled_table.map(highlight_signal_style, subset=[col_name])
                
                cols_for_pct_style_apply = [col for col in pct_cols_all if col in df_display.columns]
                if cols_for_pct_style_apply: styled_table = styled_table.map(highlight_pct_col_style, subset=cols_for_pct_style_apply)

                rsi_cols_to_style_apply = [col for col in rsi_cols_list if col in df_display.columns]
                if rsi_cols_to_style_apply: styled_table = styled_table.map(style_rsi, subset=rsi_cols_to_style_apply)

                if macd_hist_col_name in df_display.columns:
                     styled_table = styled_table.map(style_macd_hist, subset=[macd_hist_col_name])

                srsi_cols_exist_apply = all(col in df_display.columns for col in srsi_value_cols)
                if srsi_cols_exist_apply:
                     styled_table = styled_table.apply(style_stoch_rsi, axis=1, subset=srsi_value_cols)
                
                # Apply formatting LAST
                styled_table = styled_table.format(formatters, na_rep="N/A")
                
                st.dataframe(styled_table, use_container_width=True, column_config={
                    "Rank": st.column_config.NumberColumn(format="%d"), # Format Rank as integer
                    "MA/MACD Cross Alert": st.column_config.TextColumn(label="MA/MACD\nCross Alert", width="small"),
                    "Composite Score": st.column_config.TextColumn(label="Composite\nScore", width="small"),
                    "Link": st.column_config.LinkColumn("CoinGecko", display_text="🔗 Link", width="small")
                })

            except Exception as df_err: logger.exception("Error creating/styling table DataFrame:"); st.error(f"Error displaying table: {df_err}")
        else: st.warning("No valid crypto results to display in the table.")

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
            *   <span style="color:#adb5bd;">Value (Light Grey)</span>: N/A.
        *   **SRSI %K / %D (1d):** Stochastic RSI (0-100).
            *   <span style="color:#198754; font-weight:bold;">Values (Bold Green)</span>: Oversold (K&D < 20).
            *   <span style="color:#dc3545; font-weight:bold;">Values (Bold Red)</span>: Overbought (K&D > 80).
            *   <span style="color:#28a745;">Values (Green)</span>: Bullish Crossover (K > D).
            *   <span style="color:#fd7e14;">Values (Orange)</span>: Bearish Crossover (K < D).
            *   <span style="color:#adb5bd;">Values (Light Grey)</span>: N/A.
        *   **MACD Hist (1d):** MACD Histogram.
            *   <span style="color:green;">Value > 0 (Green)</span>: Bullish momentum.
            *   <span style="color:red;">Value < 0 (Red)</span>: Bearish momentum.
            *   <span style="color:#adb5bd;">Value (Light Grey)</span>: N/A.
        *   **MA(7d/20d/30d/50d):** Simple Moving Averages. Trend lines. *Values not colored, N/A if insufficient data.*
        *   **BB %B / Width / Width %Chg:** Bollinger Bands (20d, 2 std dev). Measure volatility and price relative to range.
            *   **%B (%):** Price position relative to bands (%). >100 = Above Upper; <0 = Below Lower. <span style="color:red;">Red</span>/% <span style="color:green;">Green</span> color indicates value. <span style="color:#adb5bd;">N/A</span> if calc fails.
            *   **Width (%):** Tightness of bands (%). <span style="color:red;">Red</span>/% <span style="color:green;">Green</span> color indicates value. <span style="color:#adb5bd;">N/A</span> if calc fails.
            *   **Width %Chg (%):** Daily % change in Band Width. <span style="color:red;">Red</span>=Narrowing, <span style="color:green;">Green</span>=Widening. <span style="color:#adb5bd;">N/A</span> if calc fails.
        *   **VWAP (1d):** Volume Weighted Average Price. *Value not colored, N/A if insufficient data.*
        *   **VWAP %:** Daily % change of VWAP. <span style="color:red;">Red</span>=Decreasing, <span style="color:green;">Green</span>=Increasing. <span style="color:#adb5bd;">N/A</span> if calc fails.
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

        **Important Notes:**
        *   Table data fetch is slowed (6s/coin). Initial load takes time. **CoinGecko rate limits may still occur.**
        *   Traditional Market data has **4h cache**. Alpha Vantage free tier has daily limits (e.g., 25 calls/day).
        *   **DYOR (Do Your Own Research).**
        """, unsafe_allow_html=True)
    st.divider(); st.caption("Disclaimer: Informational/educational tool only. Not financial advice. DYOR.")

except Exception as main_exception: logger.exception("!!! UNHANDLED ERROR IN MAIN APP EXECUTION !!!"); st.error(f"Unexpected error: {main_exception}. Check the log.")

# --- Application Log Display ---
st.divider(); st.subheader("📄 Application Log"); st.caption("Logs generated during the last app run (INFO Level). Useful for monitoring.")
log_content = log_stream.getvalue(); st.text_area("Log:", value=log_content, height=300, key="log_display_area", help="Select All (Ctrl+A or Cmd+A) and Copy (Ctrl+C or Cmd+C) to analyze or share.")
logger.info("--- End of Streamlit script execution: app.py ---"); log_stream.close()
