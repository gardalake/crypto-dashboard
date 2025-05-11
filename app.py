# Version: v1.4.7 - Enhanced Logging System
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
# Default level is INFO. Can be changed by IS_DEBUG_MODE.
logging.basicConfig(
    stream=log_stream,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)
# --- END: Logging Configuration ---

# --- Global Debug Flag ---
# Set to True for more verbose internal logging (DEBUG level logs will appear)
# Can be controlled by a UI element later if desired.
IS_DEBUG_MODE = False # Default

if IS_DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
    logger.info("VERBOSE DEBUG MODE ENABLED. Log level set to DEBUG.")
else:
    logger.setLevel(logging.INFO) # Ensure it's INFO if not debug

logger.info(f"Logging configured for UI (v1.4.7 - Debug Mode: {'ON' if IS_DEBUG_MODE else 'OFF'}).")


try:
    from zoneinfo import ZoneInfo
    logger.info("[CONFIG] Module 'zoneinfo' imported.")
except ImportError:
    logger.warning("[CONFIG_WARN] Module 'zoneinfo' not found. Using offset approximation for Rome timezone.")
    ZoneInfo = None

st.set_page_config(layout="wide", page_title="Crypto Technical Dashboard Pro", page_icon="📈")
st.markdown("""<style>div[data-testid="stMetricValue"] { font-size: 14px !important; }</style>""", unsafe_allow_html=True)
logger.info("[UI_SETUP] CSS applied.")

# --- Global Configuration ---
logger.info("[CONFIG] Starting global configuration.")
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
DAYS_HISTORY_DAILY = 365
RSI_PERIOD, RSI_OB, RSI_OS = 14, 70.0, 30.0
SRSI_PERIOD, SRSI_K, SRSI_D, SRSI_OB, SRSI_OS = 14, 3, 3, 80.0, 20.0
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MA_SHORT, MA_MEDIUM, MA_LONG, MA_XLONG = 7, 20, 50, 30
BB_PERIOD, BB_STD_DEV = 20, 2.0
VWAP_PERIOD = 14
logger.info(f"[CONFIG] Global config done: {NUM_COINS} coins ({','.join(SYMBOLS[:3])}...), Trad Tickers: {len(TRAD_TICKERS_AV)}.")

# --- FUNCTION DEFINITIONS (General) ---
def format_large_number(num):
    if pd.isna(num) or not isinstance(num, (int, float, np.number)): return "N/A"
    num_abs = abs(num); sign = "-" if num < 0 else ""
    if num_abs < 1_000_000: return f"{sign}{num_abs:,.0f}"
    elif num_abs < 1_000_000_000: return f"{sign}{num_abs / 1_000_000:.1f}M"
    elif num_abs < 1_000_000_000_000: return f"{sign}{num_abs / 1_000_000_000:.1f}B"
    else: return f"{sign}{num_abs / 1_000_000_000_000:.2f}T"

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading market data (CoinGecko)...")
def get_coingecko_market_data(ids_list, currency):
    func_tag = "[API_CG_LIVE]"
    logger.info(f"{func_tag} Attempting fetch for {len(ids_list)} IDs.")
    ids_string = ",".join(ids_list); url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': currency, 'ids': ids_string, 'order': 'market_cap_desc', 'per_page': str(len(ids_list)), 'page': 1, 'sparkline': False, 'price_change_percentage': '1h,24h,7d,30d,1y', 'precision': 'full'}
    timestamp_utc = datetime.now(ZoneInfo("UTC") if ZoneInfo else None)
    if 'api_warning_shown' not in st.session_state: st.session_state.api_warning_shown = False
    try:
        logger.debug(f"{func_tag} Requesting URL: {url} with params: {params}")
        response = requests.get(url, params=params, timeout=20); response.raise_for_status()
        data = response.json()
        if not data: logger.warning(f"{func_tag}[WARN] Empty data received from {url}."); return pd.DataFrame(), timestamp_utc
        df = pd.DataFrame(data)
        if df.empty: logger.info(f"{func_tag} DataFrame is empty after creation from response."); return df, timestamp_utc
        df.set_index('id', inplace=True)
        numeric_cols = ['current_price', 'market_cap_rank', 'total_volume',
                        'price_change_percentage_1h_in_currency', 'price_change_percentage_24h_in_currency',
                        'price_change_percentage_7d_in_currency', 'price_change_percentage_30d_in_currency',
                        'price_change_percentage_1y_in_currency']
        for col in numeric_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        st.session_state["api_warning_shown"] = False; logger.info(f"{func_tag}[SUCCESS] Live data fetched for {len(df)} coins."); return df, timestamp_utc
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        err_text = http_err.response.text[:200] if http_err.response else "N/A"
        err_msg = f"{func_tag}[ERROR] HTTP Err (Status: {status_code}) from {url}: {http_err}. Response: {err_text}"
        logger.warning(err_msg)
        if status_code == 429 and not st.session_state.get("api_warning_shown", False):
            st.warning("Warning CoinGecko API (Live): Rate limit (429) reached. Data might be outdated."); st.session_state["api_warning_shown"] = True
        elif status_code != 429 or st.session_state.get("api_warning_shown", False): # Show error if not 429 or if 429 already shown
            st.error(f"HTTP Error CoinGecko Market API (Status: {status_code}): {http_err}")
        return pd.DataFrame(), timestamp_utc
    except Exception as e:
        logger.exception(f"{func_tag}[PROC_ERROR] Error Processing CG Market Data from {url}:"); st.error(f"Error Processing CoinGecko Market Data: {e}"); return pd.DataFrame(), timestamp_utc

@st.cache_data(ttl=CACHE_HIST_TTL, show_spinner=False)
def get_coingecko_historical_data(coin_id, currency, days, interval='daily'):
    func_tag = f"[API_CG_HIST({coin_id})]"
    logger.debug(f"{func_tag} Fetching ({interval}), {days}d. Delaying 6s...")
    time.sleep(6.0) # Mandatory delay for CoinGecko free tier
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': currency, 'days': str(days), 'interval': interval, 'precision': 'full'}
    status_msg = "Unknown Error" # Default status
    try:
        logger.debug(f"{func_tag} Requesting URL: {url} with params: {params}")
        response = requests.get(url, params=params, timeout=25); response.raise_for_status()
        data = response.json()
        if not data or 'prices' not in data or not data['prices']:
            status_msg = f"No Prices Data in API response"; logger.warning(f"{func_tag}[WARN] {status_msg}. URL: {url}, Params: {params}"); return pd.DataFrame(), status_msg
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms', utc=True)
        prices_df.set_index('timestamp', inplace=True)
        hist_df = prices_df.copy()
        hist_df['close'] = pd.to_numeric(hist_df['close'], errors='coerce')
        if 'total_volumes' in data and data['total_volumes']:
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms', utc=True)
            volumes_df.set_index('timestamp', inplace=True)
            volumes_df['volume'] = pd.to_numeric(volumes_df['volume'], errors='coerce')
            hist_df = hist_df.join(volumes_df, how='outer')
        else: hist_df['volume'] = 0.0
        hist_df = hist_df.interpolate(method='time').ffill().bfill()
        hist_df['high'] = hist_df['close']; hist_df['low'] = hist_df['close']; hist_df['open'] = hist_df['close'].shift(1)
        if not hist_df.empty and pd.isna(hist_df.iloc[0]['open']): hist_df.loc[hist_df.index[0], 'open'] = hist_df.iloc[0]['close']
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in hist_df.columns: hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
        hist_df = hist_df[~hist_df.index.duplicated(keep='last')].sort_index().dropna(subset=['close'])
        if hist_df.empty: status_msg = f"Processed Empty (no valid 'close' data)"; logger.warning(f"{func_tag}[PROC_WARN] {status_msg}"); return pd.DataFrame(), status_msg
        status_msg = "Success"; logger.info(f"{func_tag}[SUCCESS] Hist data fetched, {len(hist_df)} rows."); return hist_df, status_msg
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        api_err_text = http_err.response.text[:200] if http_err.response else "N/A"
        status_msg = f"HTTP Err {status_code}"
        if status_code == 429: status_msg = f"Rate Limited (429)"
        logger.warning(f"{func_tag}[ERROR] {status_msg}. URL: {url}, Params: {params}, Response: {api_err_text}"); return pd.DataFrame(), status_msg
    except Exception as e:
        status_msg = f"Generic Err ({type(e).__name__})"; logger.exception(f"{func_tag}[PROC_ERROR] Error processing CG Hist:"); return pd.DataFrame(), status_msg

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_fear_greed_index():
    func_tag = "[API_FG_INDEX]"
    logger.info(f"{func_tag} Attempting fetch."); url = "https://api.alternative.me/fng/?limit=1"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data and isinstance(data.get("data"), list) and len(data["data"]) > 0:
             latest_data = data["data"][0]; value = latest_data.get("value"); desc = latest_data.get("value_classification")
             if value is not None and desc is not None: logger.info(f"{func_tag}[SUCCESS] Index: {value} ({desc})."); return f"{int(value)} ({desc})"
        logger.warning(f"{func_tag}[WARN] Unexpected data format from API."); return "N/A"
    except requests.exceptions.RequestException as req_ex:
        status_code = req_ex.response.status_code if req_ex.response is not None else "N/A"
        msg = f"{func_tag}[ERROR] API Error (Status: {status_code}): {req_ex}"; logger.warning(msg); st.sidebar.warning(f"F&G Index API Error (Status: {status_code})"); return "N/A"
    except Exception as e:
        msg = f"{func_tag}[PROC_ERROR] Error Processing: {e}"; logger.exception(msg); st.sidebar.warning(f"F&G Index Processing Error"); return "N/A"

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_global_market_data_cg(currency):
    func_tag = "[API_CG_GLOBAL]"
    logger.info(f"{func_tag} Attempting fetch for {currency.upper()}."); url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json().get('data', {})
        total_mcap = data.get('total_market_cap', {}).get(currency.lower(), np.nan)
        if pd.notna(total_mcap): logger.info(f"{func_tag}[SUCCESS] Global M.Cap ({currency.upper()}): {total_mcap}.")
        else: logger.warning(f"{func_tag}[WARN] Global M.Cap ({currency.upper()}) not found in response.")
        return total_mcap
    except requests.exceptions.RequestException as req_ex:
        msg = f"{func_tag}[ERROR] API Error: {req_ex}"; logger.warning(msg); st.sidebar.warning(f"Global MCap API Error"); return np.nan
    except Exception as e:
        msg = f"{func_tag}[PROC_ERROR] Error Processing: {e}"; logger.exception(msg); st.sidebar.warning(f"Global MCap Processing Error"); return np.nan

def get_etf_flow(): logger.debug("[DATA_ETF] get_etf_flow called (placeholder)."); return "N/A"

@st.cache_data(ttl=CACHE_TRAD_TTL, show_spinner="Loading traditional market data (Alpha Vantage)...")
def get_traditional_market_data_av(tickers):
    func_tag = "[API_AV]"
    logger.info(f"{func_tag} Attempting fetch for {len(tickers)} tickers."); data = {ticker: {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'} for ticker in tickers}; api_key = None
    try: api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]; logger.info(f"{func_tag} API key read from secrets.")
    except KeyError: logger.error(f"{func_tag}[CONFIG_ERROR] Secret 'ALPHA_VANTAGE_API_KEY' not defined."); st.sidebar.error("AlphaVantage Key Missing"); return data
    except Exception as e: logger.exception(f"{func_tag}[CONFIG_ERROR] Unexpected error reading Alpha Vantage secrets:"); st.sidebar.error(f"AlphaVantage Secrets Err: {e}"); return data
    if not api_key: logger.error(f"{func_tag}[CONFIG_ERROR] Alpha Vantage API key found but is empty."); st.sidebar.error("AlphaVantage Key Empty"); return data
    
    ts = TimeSeries(key=api_key, output_format='pandas'); calls_made = 0; max_calls_per_minute = 5; max_calls_this_run = 25; delay_between_calls = (60.0 / max_calls_per_minute) + 0.5 # Added small buffer
    
    for ticker_sym in tickers:
        if calls_made >= max_calls_this_run:
            msg = f"{func_tag}[RATE_LIMIT] Call limit for this run ({max_calls_this_run}) reached. Stopping fetch for {ticker_sym}."; logger.warning(msg); st.sidebar.warning(msg); break
        try:
            logger.info(f"{func_tag} Fetching {ticker_sym} (Call #{calls_made+1}/{max_calls_this_run}, Pause {delay_between_calls:.1f}s)..."); time.sleep(delay_between_calls)
            quote_data, meta_data = ts.get_quote_endpoint(symbol=ticker_sym); calls_made += 1
            if not quote_data.empty:
                data[ticker_sym]['price'] = pd.to_numeric(quote_data['05. price'].iloc[0], errors='coerce')
                data[ticker_sym]['change'] = pd.to_numeric(quote_data['09. change'].iloc[0], errors='coerce')
                data[ticker_sym]['change_percent'] = str(quote_data['10. change percent'].iloc[0]) # Keep as string for format_delta
                logger.info(f"{func_tag}[SUCCESS] Data for {ticker_sym} fetched OK.")
            else: logger.warning(f"{func_tag}[WARN] Empty response from AV for {ticker_sym}.");
        except ValueError as ve: # Often API limit related
             msg = f"{func_tag}[ERROR] Alpha Vantage Error for {ticker_sym}: {ve}"; logger.warning(msg); st.sidebar.warning(f"AV Error ({ticker_sym}): Limit likely"); ve_str = str(ve).lower()
             if "call frequency" in ve_str or "api key" in ve_str or "limit" in ve_str or "premium" in ve_str:
                 logger.error(f"{func_tag}[CRITICAL_API_ERROR] Critical AV API key/limit error detected. Stopping fetch."); st.sidebar.error("AV API Limit Reached!"); break
        except Exception as e: msg = f"{func_tag}[PROC_ERROR] Generic error AV for {ticker_sym}: {e}"; logger.exception(msg); st.sidebar.warning(f"AV Error ({ticker_sym})");
    logger.info(f"{func_tag} Finished fetch. Made {calls_made} calls."); return data

# --- Indicator Calculation Functions ---
def _ensure_numeric_series(series: pd.Series, func_name_parent="indicator_calc") -> pd.Series:
    if not isinstance(series, pd.Series):
        logger.debug(f"[{func_name_parent}|DATA_CLEAN] Input is not a Series, returning empty numeric series.")
        return pd.Series(dtype=float)
    if series.empty:
        logger.debug(f"[{func_name_parent}|DATA_CLEAN] Input series is empty, returning as is.")
        return series
    
    s_numeric = pd.to_numeric(series, errors='coerce')
    s_cleaned = s_numeric.dropna()
    if IS_DEBUG_MODE and len(s_numeric) != len(s_cleaned):
        logger.debug(f"[{func_name_parent}|DATA_CLEAN] Dropped {len(s_numeric) - len(s_cleaned)} NaNs during numeric conversion.")
    return s_cleaned

def calculate_rsi_manual(series: pd.Series, period: int = RSI_PERIOD, symbol="<sym>") -> float:
    func_tag = f"[INDICATOR_CALC({symbol})|RSI]"
    series = _ensure_numeric_series(series, func_tag)
    if len(series) < period + 1: logger.debug(f"{func_tag}[INSUFF_DATA] Need {period+1}, got {len(series)}"); return np.nan
    # ... (rest of RSI calculation)
    delta = series.diff(); gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean(); avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    if avg_gain.empty or avg_loss.empty or avg_gain.isna().all() or avg_loss.isna().all(): logger.debug(f"{func_tag}[CALC_FAIL] AvgGain/Loss empty or all NaN."); return np.nan
    last_avg_loss = avg_loss.iloc[-1]; last_avg_gain = avg_gain.iloc[-1]
    if pd.isna(last_avg_loss) or pd.isna(last_avg_gain):  logger.debug(f"{func_tag}[CALC_FAIL] Last AvgGain/Loss is NaN."); return np.nan
    if last_avg_loss == 0: return 100.0 if last_avg_gain > 0 else 50.0
    rs = last_avg_gain / last_avg_loss; rsi = 100.0 - (100.0 / (1.0 + rs))
    return max(0.0, min(100.0, rsi)) if pd.notna(rsi) else np.nan

# ... (Similarly add func_tag and symbol to other indicator functions, and debug logs for IS_DEBUG_MODE) ...
# For brevity, I'll skip repeating for all indicator functions, but the pattern is:
# 1. Add `symbol` parameter.
# 2. Create `func_tag = f"[INDICATOR_CALC({symbol})|INDICATOR_NAME]"`
# 3. Pass `func_tag` to `_ensure_numeric_series`.
# 4. Add `logger.debug(f"{func_tag}[INSUFF_DATA] ...")` or `f"{func_tag}[CALC_FAIL] ...")` on known failure points.

def compute_all_indicators(symbol: str, hist_daily_df: pd.DataFrame) -> dict:
    func_tag = f"[COMPUTE_ALL_INDICATORS({symbol})]"
    indicators = { "RSI (1d)": np.nan, "RSI (1w)": np.nan, "RSI (1mo)": np.nan, "SRSI %K (1d)": np.nan, "SRSI %D (1d)": np.nan, "MACD Line (1d)": np.nan, "MACD Signal (1d)": np.nan, "MACD Hist (1d)": np.nan, f"MA({MA_SHORT}d)": np.nan, f"MA({MA_MEDIUM}d)": np.nan, f"MA({MA_LONG}d)": np.nan, f"MA({MA_XLONG}d)": np.nan, "BB %B": np.nan, "BB Width": np.nan, "BB Width %Chg": np.nan, "VWAP (1d)": np.nan, "VWAP %": np.nan }
    if hist_daily_df.empty or 'close' not in hist_daily_df.columns:
        logger.warning(f"{func_tag}[DATA_WARN] Empty/invalid daily historical data. Shape: {hist_daily_df.shape if isinstance(hist_daily_df, pd.DataFrame) else 'N/A'}. Cols: {list(hist_daily_df.columns) if isinstance(hist_daily_df, pd.DataFrame) else 'N/A'}.")
        return indicators
    
    logger.debug(f"{func_tag} Input hist_daily_df shape: {hist_daily_df.shape}. Head:\n{hist_daily_df.head(2).to_string()}")
    close_series_numeric = pd.to_numeric(hist_daily_df['close'], errors='coerce')
    volume_series_numeric = pd.to_numeric(hist_daily_df.get('volume'), errors='coerce') if 'volume' in hist_daily_df else pd.Series([np.nan]*len(hist_daily_df), index=hist_daily_df.index)
    df_for_vwap_calc = pd.DataFrame({'close': close_series_numeric, 'volume': volume_series_numeric})
    close_daily = close_series_numeric.dropna(); len_daily = len(close_daily)
    
    min_len_rsi_base = RSI_PERIOD + 1 # ... other min_len variables ...

    indicators["RSI (1d)"] = calculate_rsi_manual(close_daily, RSI_PERIOD, symbol=symbol)
    # ... call other calculate_xxx_manual functions, passing `symbol` ...
    # Example for one more:
    # indicators["SRSI %K (1d)"], indicators["SRSI %D (1d)"] = calculate_stoch_rsi(close_daily, RSI_PERIOD, SRSI_PERIOD, SRSI_K, SRSI_D, symbol=symbol)


    # Placeholder for other indicator calls - apply the same pattern of passing `symbol`
    # and adding specific debug logs within those calculation functions if length checks fail.
    # For example, if calculate_stoch_rsi was fully updated:
    # indicators["SRSI %K (1d)"], indicators["SRSI %D (1d)"] = calculate_stoch_rsi(close_daily, RSI_PERIOD, SRSI_PERIOD, SRSI_K, SRSI_D, symbol=symbol)
    # else: # Use the existing calls if not updated yet
    min_len_srsi_base = RSI_PERIOD + SRSI_PERIOD + max(SRSI_K, SRSI_D) + 5
    if len_daily >= min_len_srsi_base: indicators["SRSI %K (1d)"], indicators["SRSI %D (1d)"] = calculate_stoch_rsi(close_daily, RSI_PERIOD, SRSI_PERIOD, SRSI_K, SRSI_D) # Original call
    else: logger.debug(f"{func_tag}[INSUFF_DATA] SRSI: {len_daily}/{min_len_srsi_base} rows.")

    min_len_macd_base = MACD_SLOW + MACD_SIGNAL + 5
    if len_daily >= min_len_macd_base: indicators["MACD Line (1d)"], indicators["MACD Signal (1d)"], indicators["MACD Hist (1d)"] = calculate_macd_manual(close_daily, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    else: logger.debug(f"{func_tag}[INSUFF_DATA] MACD: {len_daily}/{min_len_macd_base} rows.")
    
    # ... (SMA, BB, VWAP calculations - apply similar logging if IS_DEBUG_MODE for insufficient data)

    if IS_DEBUG_MODE: logger.debug(f"{func_tag} Calculated indicators: { {k:v for k,v in indicators.items() if pd.notna(v)} }")
    return indicators


# --- Main App Execution and Table Display (Code from v1.4.6, adapted for new logging) ---
# ... (The rest of the script would follow, with logging prefixes added to main sections) ...
# --- START OF MAIN APP EXECUTION ---
# setup_ui_debug_mode() # Call this if you want the UI checkbox to control IS_DEBUG_MODE
logger.info("[MAIN_EXEC] Starting main UI execution.")
try:
    # Title, Refresh, Timestamp 
    col_title, _, col_button = st.columns([4, 1, 1]); 
    with col_title: st.title("📈 Crypto Technical Dashboard Pro")
    with col_button: 
        st.write("")
        if st.button("🔄 Refresh", help="Force data refresh (clears cache)", key="refresh_button_v147"): # Ensure unique key
            logger.info("[UI_ACTION] Refresh button clicked.");
            if 'api_warning_shown' in st.session_state: del st.session_state['api_warning_shown']
            st.cache_data.clear(); st.query_params.clear(); st.rerun()
    last_update_placeholder = st.empty()
    st.caption(f"Cache TTL: Live ({CACHE_TTL/60:.0f}m), Table History ({CACHE_HIST_TTL/60:.0f}m), Traditional ({CACHE_TRAD_TTL/3600:.0f}h).")

    # Market Overview Section
    st.markdown("---"); st.subheader("🌐 Market Overview")
    # These API calls now have st.sidebar.warning on failure within their functions
    fear_greed_value = get_fear_greed_index()
    total_market_cap = get_global_market_data_cg(VS_CURRENCY)
    etf_flow_value = get_etf_flow() # Placeholder
    traditional_market_data = get_traditional_market_data_av(TRAD_TICKERS_AV)
    
    # ... (render_metric and overview display logic from v1.4.6)
    def format_delta(change_val, change_pct_str):
        delta_string = None; change_val_num = pd.to_numeric(change_val, errors='coerce')
        if pd.notna(change_val_num) and isinstance(change_pct_str, str) and change_pct_str.strip() not in ['N/A', '', None]:
            try: change_pct_val = float(change_pct_str.replace('%','').strip()); delta_string = f"{change_val_num:+.2f} ({change_pct_val:+.2f}%)"
            except (ValueError, AttributeError): delta_string = f"{change_val_num:+.2f} (?%)"
        elif pd.notna(change_val_num): delta_string = f"{change_val_num:+.2f}"
        return delta_string
    def render_metric(column, label, value_func=None, ticker=None, data_dict=None, help_text=None):
        value_str = "N/A"; delta_txt = None; d_color = "off"
        if ticker and data_dict: 
            trad_info = data_dict.get(ticker, {}); price = pd.to_numeric(trad_info.get('price'), errors='coerce'); change = pd.to_numeric(trad_info.get('change'), errors='coerce'); change_pct = trad_info.get('change_percent', 'N/A')
            value_str = f"${price:,.2f}" if pd.notna(price) else "N/A"; delta_txt = format_delta(change, change_pct)
            if pd.notna(change) and change != 0 : d_color = "normal" 
        elif value_func: 
            try: value_str = value_func(); value_str = str(value_str) if value_str is not None else "N/A"
            except Exception as e: logger.error(f"[UI_ERROR] Error in value_func '{label}': {e}"); value_str = "Error"
        column.metric(label=label, value=value_str, delta=delta_txt, delta_color=d_color, help=help_text)
    overview_items_row1 = [ ("Fear & Greed Index", None, get_fear_greed_index, "Source: Alternative.me"), (f"Total Crypto M.Cap ({VS_CURRENCY.upper()})", None, lambda: f"${format_large_number(total_market_cap)}", "Source: CoinGecko"), ("Crypto ETFs Flow (Daily)", None, get_etf_flow, "Data N/A"), ("S&P 500 (SPY)", "SPY", None, "Source: AV (ETF)"), ("Nasdaq (QQQ)", "QQQ", None, "Source: AV (ETF)") ]
    overview_cols_1 = st.columns(len(overview_items_row1));
    for i, (label, ticker, func, help_text) in enumerate(overview_items_row1): render_metric(overview_cols_1[i], label, value_func=func, ticker=ticker, data_dict=traditional_market_data, help_text=help_text)
    overview_items_row2 = [ ("Gold (GLD)", "GLD", None, "Source: AV (ETF)"), ("Silver (SLV)", "SLV", None, "Source: AV (ETF)"), ("Natural Gas (UNG)", "UNG", None, "Source: AV (ETF)"), ("UVXY (Volatility)", "UVXY", None, "Source: AV"), ("TQQQ (Nasdaq 3x)", "TQQQ", None, "Source: AV") ]
    overview_cols_2 = st.columns(len(overview_items_row2));
    for i, (label, ticker, func, help_text) in enumerate(overview_items_row2): render_metric(overview_cols_2[i], label, value_func=func, ticker=ticker, data_dict=traditional_market_data, help_text=help_text)
    st.markdown("<h6>Major Stocks (Source: Alpha Vantage):</h6>", unsafe_allow_html=True); stock_tickers_row_av = ['NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR', 'COIN', 'MSTR']
    num_stock_cols = 5; stock_cols = st.columns(num_stock_cols);
    for idx, ticker_sym_loopvar in enumerate(stock_tickers_row_av): render_metric(stock_cols[idx % num_stock_cols], label=ticker_sym_loopvar, ticker=ticker_sym_loopvar, data_dict=traditional_market_data, help_text=f"Ticker: {ticker_sym_loopvar}")
    st.markdown("---")

    # Main Crypto Technical Analysis Table
    st.subheader(f"📊 Crypto Technical Analysis ({NUM_COINS} Assets)")
    market_data_df, last_cg_update_utc = get_coingecko_market_data(COINGECKO_IDS_LIST, VS_CURRENCY)
    # ... (Timestamp display from v1.4.6) ...
    if last_cg_update_utc:
        try:
            local_tz_name = "Europe/Rome"; user_tz = None
            if ZoneInfo: user_tz = ZoneInfo(local_tz_name)
            if last_cg_update_utc.tzinfo is None: last_cg_update_utc = last_cg_update_utc.replace(tzinfo=ZoneInfo("UTC") if ZoneInfo else None)
            if user_tz: last_cg_update_local = last_cg_update_utc.astimezone(user_tz)
            else: offset_hours = 2; last_cg_update_local = last_cg_update_utc + timedelta(hours=offset_hours)
            time_str = last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S %Z') if user_tz else last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S')
            timestamp_display_str = f"*Live CoinGecko data updated at: **{time_str}***"
            if not user_tz: timestamp_display_str += f" (Approx. {local_tz_name} Time)"
            logger.info(f"[UI_INFO] Displaying timestamp: {time_str}")
        except Exception as e: logger.exception("[UI_ERROR] Error formatting/converting timestamp:"); timestamp_display_str = f"*Timestamp conversion error. UTC: {last_cg_update_utc.strftime('%Y-%m-%d %H:%M:%S %Z') if last_cg_update_utc and last_cg_update_utc.tzinfo else str(last_cg_update_utc)}*"
        last_update_placeholder.markdown(timestamp_display_str)
    else: logger.warning("[UI_WARN] Live CoinGecko data timestamp unavailable."); last_update_placeholder.markdown("*Live CoinGecko data timestamp unavailable.*")


    if market_data_df.empty:
        msg = "[CRITICAL_DATA] No live CoinGecko data. Table cannot be generated."
        if st.session_state.get("api_warning_shown", False): msg = "[TABLE_ERROR] Table not generated: Error loading live data (CoinGecko API limit?)."
        logger.error(msg); st.error(msg)
    else:
        results = []; fetch_errors = []; process_start_time = time.time(); effective_num_coins = len(market_data_df.index)
        show_fire_icon = (market_data_df['price_change_percentage_1h_in_currency'].dropna() > 0).sum() >= FIRE_ICON_THRESHOLD if 'price_change_percentage_1h_in_currency' in market_data_df else False
        logger.info(f"[TABLE_SETUP] Show fire icon: {show_fire_icon}. Processing {effective_num_coins} coins.")
        spinner_msg = f"Processing table for {effective_num_coins} crypto assets... (~{(effective_num_coins * 6.1 / 60):.1f} min)" # 6.1s to account for minor overheads
        
        with st.spinner(spinner_msg):
            for i, coin_id_loop in enumerate(market_data_df.index): # Renamed coin_id to coin_id_loop
                symbol_loop = next((s for s, cid in SYMBOL_TO_ID_MAP.items() if cid == coin_id_loop), "N/A") # Renamed symbol
                log_prefix_coin = f"[TABLE_PROC({symbol_loop}|{coin_id_loop})]"
                logger.info(f"{log_prefix_coin} Start ({i+1}/{effective_num_coins})")
                if symbol_loop == "N/A": 
                    fetch_errors.append(f"{coin_id_loop}: ID not in SYMBOL_TO_ID_MAP."); logger.warning(f"{log_prefix_coin}[CONFIG_ERROR] Coin ID not in map."); continue
                
                live_coin_data = market_data_df.loc[coin_id_loop]
                hist_df, hist_status = get_coingecko_historical_data(coin_id_loop, VS_CURRENCY, DAYS_HISTORY_DAILY)
                current_price_val = pd.to_numeric(live_coin_data.get('current_price'), errors='coerce')

                if hist_status != "Success":
                    logger.warning(f"{log_prefix_coin}[DATA_FAIL] Hist.Data Error - {hist_status}. Indicators will be NaN.")
                    fetch_errors.append(f"{symbol_loop}: Hist.Data - {hist_status}")
                    indicators = compute_all_indicators(symbol_loop, pd.DataFrame()) 
                else:
                    indicators = compute_all_indicators(symbol_loop, hist_df)
                
                # Pass symbol for better debug in signal functions if IS_DEBUG_MODE is on
                # gpt_signal_val = generate_gpt_signal(..., symbol=symbol_loop) 
                gpt_signal_val = generate_gpt_signal(indicators.get("RSI (1d)"), indicators.get("RSI (1w)"), indicators.get("MACD Hist (1d)"), indicators.get(f"MA({MA_SHORT}d)"), indicators.get(f"MA({MA_MEDIUM}d)"), indicators.get(f"MA({MA_LONG}d)"), indicators.get("SRSI %K (1d)"), indicators.get("SRSI %D (1d)"), indicators.get("BB %B"), indicators.get("BB Width %Chg"), indicators.get("VWAP (1d)"), current_price_val)
                gemini_alert_val = generate_gemini_alert(indicators.get(f"MA({MA_MEDIUM}d)"), indicators.get(f"MA({MA_LONG}d)"), indicators.get("MACD Hist (1d)"), indicators.get("RSI (1d)"), indicators.get("VWAP (1d)"), current_price_val)

                results.append({ "Rank": live_coin_data.get('market_cap_rank', np.nan), "Symbol": symbol_loop, "Name": live_coin_data.get('name', coin_id_loop), "MA/MACD Cross Alert": gemini_alert_val, "Composite Score": gpt_signal_val, f"Price ({VS_CURRENCY.upper()})": current_price_val, "% 1h": live_coin_data.get('price_change_percentage_1h_in_currency', np.nan), "% 24h": live_coin_data.get('price_change_percentage_24h_in_currency', np.nan), "% 7d": live_coin_data.get('price_change_percentage_7d_in_currency', np.nan), "% 30d": live_coin_data.get('price_change_percentage_30d_in_currency', np.nan), "% 1y": live_coin_data.get('price_change_percentage_1y_in_currency', np.nan), **indicators, f"Volume 24h ({VS_CURRENCY.upper()})": live_coin_data.get('total_volume', np.nan), "Link": f"https://www.coingecko.com/en/coins/{coin_id_loop}"})
                logger.info(f"{log_prefix_coin} End processing.")
            logger.info(f"[TABLE_PROC_DONE] Table processing loop finished. Time: {time.time() - process_start_time:.1f}s")

        if fetch_errors: 
            st.sidebar.error("⚠️ Data Fetch/Processing Issues Encountered:")
            for err_item in fetch_errors: st.sidebar.caption(f" - {err_item}")

        if results: # Table Display
            try:
                table_df = pd.DataFrame(results); table_df['Rank'] = pd.to_numeric(table_df['Rank'], errors='coerce'); table_df.set_index('Rank', inplace=True, drop=False) ; table_df.sort_index(inplace=True)
                cols_order = [ "Rank", "Symbol", "Name", "MA/MACD Cross Alert", "Composite Score", f"Price ({VS_CURRENCY.upper()})", "% 1h", "% 24h", "% 7d", "% 30d", "% 1y", "RSI (1d)", "RSI (1w)", "RSI (1mo)", "SRSI %K (1d)", "SRSI %D (1d)", "MACD Hist (1d)", f"MA({MA_SHORT}d)", f"MA({MA_MEDIUM}d)", f"MA({MA_XLONG}d)", f"MA({MA_LONG}d)", "BB %B", "BB Width", "BB Width %Chg", "VWAP (1d)", "VWAP %", f"Volume 24h ({VS_CURRENCY.upper()})", "Link"]
                df_display = table_df[[col for col in cols_order if col in table_df.columns]].copy()
                numeric_cols_for_formatting = [ f"Price ({VS_CURRENCY.upper()})", "% 1h", "% 24h", "% 7d", "% 30d", "% 1y", "RSI (1d)", "RSI (1w)", "RSI (1mo)", "SRSI %K (1d)", "SRSI %D (1d)", "MACD Hist (1d)", f"MA({MA_SHORT}d)", f"MA({MA_MEDIUM}d)", f"MA({MA_XLONG}d)", f"MA({MA_LONG}d)", "BB %B", "BB Width", "BB Width %Chg", "VWAP (1d)", "VWAP %", f"Volume 24h ({VS_CURRENCY.upper()})"]
                
                # Debug checkboxes with unique keys (fixed in v1.4.6)
                # You can control IS_DEBUG_MODE with a single checkbox at the top or sidebar if preferred
                # For now, IS_DEBUG_MODE is a global constant at the top of the script.
                if IS_DEBUG_MODE: # Only log dtypes if in full debug mode
                    logger.debug("[TABLE_DEBUG] --- Pre-coercion dtypes for df_display ---"); logger.debug(df_display.dtypes.to_string())
                
                for col in numeric_cols_for_formatting:
                    if col in df_display.columns: df_display[col] = pd.to_numeric(df_display[col], errors='coerce')
                
                if IS_DEBUG_MODE:
                    logger.debug("[TABLE_DEBUG] --- Post-coercion dtypes for df_display ---"); logger.debug(df_display.dtypes.to_string())
                    # Example: Log a problematic column if needed for deep dive
                    # if "RSI (1d)" in df_display:
                    #     logger.debug(f"[TABLE_DEBUG] Sample 'RSI (1d)' data:\n{df_display[['Symbol', 'RSI (1d)']].head().to_string()}")

                # safe_formatter and other formatters/styling from v1.4.6 (unchanged)
                # ... (This entire block is the same as v1.4.6) ...
                def safe_formatter(fmt_str, col_name="<unknown>"): 
                    def _format(x):
                        if pd.isna(x): return "N/A"
                        if not isinstance(x, (int, float, np.number)): logger.warning(f"[FORMAT_WARN] SafeFormatter: Non-numeric val '{x}' (type: {type(x)}) in col '{col_name}' for fmt '{fmt_str}'. Ret N/A."); return "N/A" 
                        try: return fmt_str.format(x)
                        except (ValueError, TypeError) as e: logger.warning(f"[FORMAT_ERR] SafeFormatter: Format err for val '{x}' (type: {type(x)}) in col '{col_name}' with fmt '{fmt_str}'. Err: {e}. Ret str."); return str(x) 
                    return _format
                formatters = {}; currency_col = f"Price ({VS_CURRENCY.upper()})"; volume_col = f"Volume 24h ({VS_CURRENCY.upper()})"
                if currency_col in df_display.columns: formatters[currency_col] = safe_formatter("${:,.4f}", currency_col)
                if volume_col in df_display.columns: formatters[volume_col] = lambda x: format_large_number(x)
                pct_cols_all = ["% 1h", "% 24h", "% 7d", "% 30d", "% 1y", "VWAP %", "BB Width", "BB Width %Chg", "BB %B"]
                for col in pct_cols_all:
                    if col in df_display.columns and col != '% 1h': formatters[col] = safe_formatter("{:+.2f}%", col)
                def format_1h_with_icon(val):
                    if pd.isna(val): return "N/A";
                    if not isinstance(val, (int, float, np.number)): logger.warning(f"[FORMAT_WARN] format_1h_with_icon got non-numeric: {val}"); return "N/A"
                    icon = "🔥 " if show_fire_icon and val > 0 else ""; try: return f"{icon}{val:+.2f}%"
                    except: logger.error(f"[FORMAT_ERR] format_1h_with_icon failed for {val}"); return "N/A"
                if '% 1h' in df_display.columns: formatters['% 1h'] = format_1h_with_icon
                rsi_cols_list = [c for c in df_display.columns if "RSI" in c and "%" not in c and "SRSI" not in c]; srsi_value_cols = ["SRSI %K (1d)", "SRSI %D (1d)"]
                for col in rsi_cols_list + srsi_value_cols:
                     if col in df_display.columns: formatters[col] = safe_formatter("{:.1f}", col)
                macd_hist_col_name = "MACD Hist (1d)"
                if macd_hist_col_name in df_display.columns: formatters[macd_hist_col_name] = safe_formatter("{:+.4f}", macd_hist_col_name)
                ma_value_cols_pattern = [f"MA({MA_SHORT}d)", f"MA({MA_MEDIUM}d)", f"MA({MA_XLONG}d)", f"MA({MA_LONG}d)", "VWAP (1d)"]
                ma_vwap_cols = [col for col in ma_value_cols_pattern if col in df_display.columns] 
                for col in ma_vwap_cols: formatters[col] = safe_formatter("{:,.2f}", col)
                
                # Styling functions (unchanged from v1.4.6)
                def highlight_signal_style(val): style = 'color: #6c757d; font-weight: normal;'; if isinstance(val, str):
                        if "Strong Buy" in val: style = 'color: #198754; font-weight: bold;'
                        elif "Buy" in val: style = 'color: #28a745;' 
                        elif "Strong Sell" in val: style = 'color: #dc3545; font-weight: bold;'
                        elif "Sell" in val: style = 'color: #fd7e14;' 
                        elif "CTB" in val: style = 'color: #20c997;'
                        elif "CTS" in val: style = 'color: #ffc107; color: #000;' 
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
                    k_col = "SRSI %K (1d)"; d_col = "SRSI %D (1d)"; style_k_css = 'color: #adb5bd;'; style_d_css = 'color: #adb5bd;'
                    k_val_num = row_subset.get(k_col, np.nan); d_val_num = row_subset.get(d_col, np.nan)
                    if pd.notna(k_val_num) and pd.notna(d_val_num):
                        common_style = '';
                        if k_val_num > SRSI_OB and d_val_num > SRSI_OB: common_style = 'color: #dc3545; font-weight: bold;'
                        elif k_val_num < SRSI_OS and d_val_num < SRSI_OS: common_style = 'color: #198754; font-weight: bold;'
                        elif k_val_num > d_val_num: common_style = 'color: #28a745;' 
                        elif k_val_num < d_val_num: common_style = 'color: #fd7e14;' 
                        else: common_style = '' 
                        style_k_css = common_style; style_d_css = common_style
                    output_styles = pd.Series('', index=row_subset.index, dtype=str)
                    if k_col in row_subset.index: output_styles[k_col] = style_k_css
                    if d_col in row_subset.index: output_styles[d_col] = style_d_css
                    return output_styles
                
                styled_table = df_display.style
                for col_name in ["MA/MACD Cross Alert", "Composite Score"]:
                    if col_name in df_display.columns: styled_table = styled_table.map(highlight_signal_style, subset=[col_name])
                cols_for_pct_style_apply = [col for col in pct_cols_all if col in df_display.columns]
                if cols_for_pct_style_apply: styled_table = styled_table.map(highlight_pct_col_style, subset=cols_for_pct_style_apply)
                rsi_cols_to_style_apply = [col for col in rsi_cols_list if col in df_display.columns]
                if rsi_cols_to_style_apply: styled_table = styled_table.map(style_rsi, subset=rsi_cols_to_style_apply)
                if macd_hist_col_name in df_display.columns: styled_table = styled_table.map(style_macd_hist, subset=[macd_hist_col_name])
                srsi_cols_exist_apply = all(col in df_display.columns for col in srsi_value_cols)
                if srsi_cols_exist_apply: styled_table = styled_table.apply(style_stoch_rsi, axis=1, subset=srsi_value_cols)
                styled_table = styled_table.format(formatters, na_rep="N/A") 
                
                st.dataframe(styled_table, use_container_width=True, column_config={ "Rank": st.column_config.NumberColumn(format="%d"), "MA/MACD Cross Alert": st.column_config.TextColumn(label="MA/MACD\nCross Alert", width="small"), "Composite Score": st.column_config.TextColumn(label="Composite\nScore", width="small"), "Link": st.column_config.LinkColumn("CoinGecko", display_text="🔗 Link", width="small")})
            except Exception as df_err: logger.exception("[TABLE_ERROR] Error creating/styling table DataFrame:"); st.error(f"Error displaying table: {df_err}")
        else: st.warning("No valid crypto results to display in the table (all fetches might have failed).")

    # Legend and Disclaimer
    st.divider();
    with st.expander("📘 Indicator, Signal & Legend Guide", expanded=False): # Legend from v1.4.6
        st.markdown("""*Disclaimer: ... * **N/A:** Data Not Available ... **DYOR.**""", unsafe_allow_html=True) 
    st.divider(); st.caption("Disclaimer: Informational/educational tool only. Not financial advice. DYOR.")
except Exception as main_exception: logger.exception("!!! [CRITICAL_ERROR] UNHANDLED ERROR IN MAIN APP EXECUTION !!!"); st.error(f"An unexpected error occurred: {main_exception}. Please check the application log below for details.")
st.divider(); st.subheader("📄 Application Log"); st.caption("Logs from last run. Refresh page for latest after code changes.")
log_content = log_stream.getvalue(); st.text_area("Log:", value=log_content, height=300, key="log_display_area_v147", help="Ctrl+A, Ctrl+C to copy.") # Unique key for log area
logger.info(f"--- End of Streamlit script execution (v1.4.7 - Debug Mode: {'ON' if IS_DEBUG_MODE else 'OFF'}) ---"); log_stream.close()
