# Versione: v0.6 - Add Heuristic Pred, Latest Price Display, Remove Error Expander, Improve Legend
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
# import pandas_ta as ta # Removed
import requests
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import math
# import yfinance as yf
# import feedparser
from alpha_vantage.timeseries import TimeSeries
import logging
import io

# --- INIZIO: Configurazione Logging in UI ---
log_stream = io.StringIO()
logging.basicConfig(
    stream=log_stream,
    level=logging.INFO, # Set to INFO for production, DEBUG for development
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)
logger.info("Logging configurato per UI.")
# --- FINE: Configurazione Logging in UI ---

# Import zoneinfo
try:
    from zoneinfo import ZoneInfo
    logger.info("Modulo 'zoneinfo' importato.")
except ImportError:
    logger.warning("Modulo 'zoneinfo' non trovato. Verrà usata approssimazione per fuso orario Roma.")
    st.warning("Modulo 'zoneinfo' non trovato. Verrà usata approssimazione per fuso orario Roma.")
    ZoneInfo = None

# --- Layout App Streamlit ---
st.set_page_config(layout="wide", page_title="Crypto Technical Dashboard Pro", page_icon="📈")

# --- CSS ---
st.markdown("""<style>div[data-testid="stMetricValue"] { font-size: 14px !important; }</style>""", unsafe_allow_html=True)
logger.info("CSS applicato.")

# --- Configurazione Globale ---
logger.info("Inizio configurazione globale.")
SYMBOL_TO_ID_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "RNDR": "render-token",
    "RAY": "raydium", "SUI": "sui", "ONDO": "ondo-finance", "ARB": "arbitrum",
    "TAO": "bittensor", "LINK": "chainlink", "HBAR": "hedera-hashgraph",
    "IMX": "immutable-x", "TRUMP": "official-trump", "AERO": "aerodrome-finance",
    "MKR": "maker",
}
SYMBOLS = list(SYMBOL_TO_ID_MAP.keys())
COINGECKO_IDS_LIST = list(SYMBOL_TO_ID_MAP.values())
NUM_COINS = len(SYMBOLS)
logger.info(f"Numero coins configurate: {NUM_COINS}")
TRAD_TICKERS_AV = [
    'SPY', 'QQQ', 'GLD', 'SLV', 'UNG', 'UVXY', 'TQQQ', 'NVDA', 'GOOGL', 'AAPL',
    'META', 'TSLA', 'MSFT', 'TSM', 'PLTR', 'COIN', 'MSTR'
]
logger.info(f"Tickers tradizionali configurati (Alpha Vantage): {TRAD_TICKERS_AV}")
VS_CURRENCY = "usd"
CACHE_TTL = 1800  # 30 min
CACHE_HIST_TTL = CACHE_TTL * 2 # 60 min (Used for table indicators)
CACHE_CHART_TTL = CACHE_TTL # 30 min (Separate shorter cache for chart data)
CACHE_TRAD_TTL = 14400 # 4h (Alpha Vantage)
# CACHE_FORECAST_TTL Removed
DAYS_HISTORY_DAILY = 365
DAYS_HISTORY_HOURLY = 7
# DAYS_HISTORY_FORECAST Removed
# FORECAST_DAYS Removed
RSI_PERIOD = 14
SRSI_PERIOD = 14
SRSI_K = 3
SRSI_D = 3
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MA_SHORT = 20
MA_LONG = 50
VWAP_PERIOD = 14
HEURISTIC_PRED_PERIOD = 3 # Days for simple price change prediction
logger.info("Fine configurazione globale.")

# --- DEFINIZIONI FUNZIONI (Generali) ---

def check_password():
    logger.debug("Esecuzione check_password.")
    if "password_correct" not in st.session_state: st.session_state.password_correct = False
    if not st.session_state.password_correct:
        pwd_col, btn_col = st.columns([3, 1])
        with pwd_col: password = st.text_input("🔑 Password", type="password", key="password_input_field")
        with btn_col: st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True); login_button_pressed = st.button("Accedi", key="login_button")
        should_check = login_button_pressed or (password and password != "")
        if not should_check: logger.debug("In attesa di input password o click bottone."); st.stop()
        else:
            correct_password = "Leonardo"
            if password == correct_password:
                logger.info("Password corretta."); st.session_state.password_correct = True
                if st.query_params.get("logged_in") != "true": st.query_params["logged_in"] = "true"; st.rerun()
            else: logger.warning("Password errata inserita."); st.warning("Password errata."); st.stop()
    logger.debug("Check password superato."); return True

def format_large_number(num):
    if pd.isna(num) or not isinstance(num, (int, float)): return "N/A"
    num_abs = abs(num); sign = "-" if num < 0 else ""
    if num_abs < 1_000_000: return f"{sign}{num_abs:,.0f}"
    elif num_abs < 1_000_000_000: return f"{sign}{num_abs / 1_000_000:.1f}M"
    elif num_abs < 1_000_000_000_000: return f"{sign}{num_abs / 1_000_000_000:.1f}B"
    else: return f"{sign}{num_abs / 1_000_000_000_000:.2f}T"

@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento dati di mercato (CoinGecko)...")
def get_coingecko_market_data(ids_list, currency):
    logger.info(f"Tentativo fetch dati live CoinGecko per {len(ids_list)} IDs.")
    ids_string = ",".join(ids_list); url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': currency, 'ids': ids_string, 'order': 'market_cap_desc', 'per_page': str(len(ids_list)), 'page': 1, 'sparkline': False, 'price_change_percentage': '1h,24h,7d,30d,1y', 'precision': 'full'}
    timestamp_utc = datetime.now(ZoneInfo("UTC") if ZoneInfo else None)
    if 'api_warning_shown' not in st.session_state: st.session_state.api_warning_shown = False
    try:
        logger.debug(f"Requesting URL: {url} with params: {params}"); response = requests.get(url, params=params, timeout=20); response.raise_for_status()
        data = response.json();
        if not data: logger.warning("API CoinGecko live: Dati vuoti ricevuti."); st.warning("API CoinGecko live: Dati vuoti ricevuti."); return pd.DataFrame(), timestamp_utc
        df = pd.DataFrame(data);
        if not df.empty: df.set_index('id', inplace=True)
        st.session_state["api_warning_shown"] = False; logger.info(f"Dati live CoinGecko recuperati per {len(df)} coins."); return df, timestamp_utc
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code; logger.warning(f"Errore HTTP API Mercato CoinGecko (Status: {status_code}): {http_err}")
        if status_code == 429 and not st.session_state.get("api_warning_shown", False): st.warning("Attenzione API CoinGecko (Live): Limite richieste (429) raggiunto. I dati potrebbero non essere aggiornati."); st.session_state["api_warning_shown"] = True
        elif status_code != 429 or st.session_state.get("api_warning_shown", False): st.error(f"Errore HTTP API Mercato CoinGecko (Status: {status_code}): {http_err}")
        return pd.DataFrame(), timestamp_utc
    except requests.exceptions.RequestException as req_ex: logger.error(f"Errore Richiesta API Mercato CoinGecko: {req_ex}"); st.error(f"Errore Richiesta API Mercato CoinGecko: {req_ex}"); return pd.DataFrame(), timestamp_utc
    except Exception as e: logger.exception("Errore Processamento Dati Mercato CoinGecko:"); st.error(f"Errore Processamento Dati Mercato CoinGecko: {e}"); return pd.DataFrame(), timestamp_utc

@st.cache_data(ttl=CACHE_CHART_TTL, show_spinner=False)
def get_coingecko_historical_data_for_chart(coin_id, currency, days):
    logger.debug(f"CHART: Inizio fetch storico per {coin_id} (daily, {days}d)."); url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': currency, 'days': str(days), 'interval': 'daily', 'precision': 'full'}; status_msg = f"Errore Sconosciuto ({coin_id}, daily chart)"
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
        status_msg = "Success"; logger.info(f"CHART: Dati storici CoinGecko per {coin_id} (daily) recuperati, {len(hist_df)} righe.")
        return_cols = ['open', 'high', 'low', 'close', 'volume']; hist_df_final = hist_df[[col for col in return_cols if col in hist_df.columns]]; return hist_df_final, status_msg
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code;
        if status_code == 429: status_msg = f"Rate Limited (429) ({coin_id}, daily chart)"
        elif status_code == 404: status_msg = f"Not Found (404) ({coin_id}, daily chart)"
        else: status_msg = f"HTTP Error {status_code} ({coin_id}, daily chart)"
        logger.warning(f"CHART: Errore HTTP API Storico CoinGecko: {status_msg}"); return pd.DataFrame(), status_msg
    except requests.exceptions.RequestException as req_ex: status_msg = f"Request Error ({req_ex}) ({coin_id}, daily chart)"; logger.error(f"CHART: Errore Richiesta API Storico CoinGecko: {status_msg}"); return pd.DataFrame(), status_msg
    except Exception as e: status_msg = f"Generic Error ({type(e).__name__}) ({coin_id}, daily chart)"; logger.exception(f"CHART: Errore Processamento API Storico CoinGecko per {coin_id}:"); return pd.DataFrame(), status_msg

@st.cache_data(ttl=CACHE_HIST_TTL, show_spinner=False)
def get_coingecko_historical_data(coin_id, currency, days, interval='daily'):
    logger.debug(f"TABLE: Inizio fetch storico per {coin_id} ({interval}), pausa 6s..."); time.sleep(6.0); logger.debug(f"TABLE: Fine pausa per {coin_id} ({interval}), inizio chiamata API.")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"; params = {'vs_currency': currency, 'days': str(days), 'interval': interval if interval == 'hourly' else 'daily', 'precision': 'full'}
    status_msg = f"Errore Sconosciuto ({coin_id}, {interval})"
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
        status_msg = "Success"; logger.info(f"TABLE: Dati storici CoinGecko per {coin_id} ({interval}) recuperati, {len(hist_df)} righe."); return hist_df, status_msg
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code;
        if status_code == 429: status_msg = f"Rate Limited (429) ({coin_id}, {interval})"
        elif status_code == 404: status_msg = f"Not Found (404) ({coin_id}, {interval})"
        else: status_msg = f"HTTP Error {status_code} ({coin_id}, {interval})"
        logger.warning(f"TABLE: Errore HTTP API Storico CoinGecko: {status_msg}"); return pd.DataFrame(), status_msg
    except requests.exceptions.RequestException as req_ex: status_msg = f"Request Error ({req_ex}) ({coin_id}, {interval})"; logger.error(f"TABLE: Errore Richiesta API Storico CoinGecko: {status_msg}"); return pd.DataFrame(), status_msg
    except Exception as e: status_msg = f"Generic Error ({type(e).__name__}) ({coin_id}, {interval})"; logger.exception(f"TABLE: Errore Processamento API Storico CoinGecko per {coin_id} ({interval}):"); return pd.DataFrame(), status_msg

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_fear_greed_index():
    logger.info("Tentativo fetch Fear & Greed Index."); url = "https://api.alternative.me/fng/?limit=1"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data and isinstance(data.get("data"), list) and len(data["data"]) > 0:
             latest_data = data["data"][0]; value = latest_data.get("value"); desc = latest_data.get("value_classification")
             if value is not None and desc is not None: logger.info(f"F&G Index: {value} ({desc})."); return f"{int(value)} ({desc})"
        logger.warning("Formato dati F&G Index inatteso ricevuto da API."); return "N/A"
    except requests.exceptions.RequestException as req_ex: status_code = req_ex.response.status_code if req_ex.response is not None else "N/A"; msg = f"Errore API F&G Index (Alternative.me Status: {status_code}): {req_ex}"; logger.warning(msg); st.warning(msg); return "N/A"
    except Exception as e: msg = f"Errore Processamento F&G Index (Alternative.me): {e}"; logger.exception(msg); st.warning(msg); return "N/A"

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_global_market_data_cg(currency):
    logger.info("Tentativo fetch dati Global CoinGecko."); url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json().get('data', {})
        total_mcap = data.get('total_market_cap', {}).get(currency.lower(), np.nan)
        if pd.notna(total_mcap): logger.info(f"Global CoinGecko M.Cap ({currency.upper()}): {total_mcap}.")
        else: logger.warning(f"Global CoinGecko M.Cap ({currency.upper()}) non trovato nella risposta.")
        return total_mcap
    except requests.exceptions.RequestException as req_ex: msg = f"Errore API Global CoinGecko: {req_ex}"; logger.warning(msg); st.warning(msg); return np.nan
    except Exception as e: msg = f"Errore Processamento Global CoinGecko: {e}"; logger.exception(msg); st.warning(msg); return np.nan

def get_etf_flow(): logger.debug("get_etf_flow chiamato (placeholder)."); return "N/A"

@st.cache_data(ttl=CACHE_TRAD_TTL, show_spinner="Caricamento dati mercato tradizionale (Alpha Vantage)...")
def get_traditional_market_data_av(tickers):
    logger.info(f"Tentativo fetch dati Alpha Vantage per {len(tickers)} tickers."); data = {ticker: {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'} for ticker in tickers}; api_key = None
    try: api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]; logger.info("Chiave API Alpha Vantage letta dai secrets.")
    except KeyError: logger.error("Secret 'ALPHA_VANTAGE_API_KEY' non definito."); st.error("Errore Configurazione: Chiave API Alpha Vantage non trovata."); return data
    except Exception as e: logger.exception("Errore lettura secrets Alpha Vantage:"); st.error(f"Errore lettura secrets Alpha Vantage: {e}"); return data
    if not api_key: logger.error("Chiave API Alpha Vantage vuota."); st.error("Errore Configurazione: Chiave API Alpha Vantage vuota."); return data
    ts = TimeSeries(key=api_key, output_format='pandas'); calls_made = 0; max_calls_per_minute = 5; max_calls_this_run = 25; delay_between_calls = (60.0 / max_calls_per_minute) + 1.0
    for ticker_sym in tickers:
        if calls_made >= max_calls_this_run: msg = f"Limite chiamate AV ({max_calls_this_run}) raggiunto. Stop fetch per {ticker_sym}."; logger.warning(msg); st.warning(msg); break
        try:
            logger.info(f"Fetch AV per {ticker_sym} (Call #{calls_made+1}/{max_calls_this_run}, Pausa {delay_between_calls:.1f}s)..."); time.sleep(delay_between_calls); quote_data, meta_data = ts.get_quote_endpoint(symbol=ticker_sym); calls_made += 1; logger.debug(f"Risposta AV per {ticker_sym}: Head:\n{quote_data.head()}")
            if not quote_data.empty:
                try: data[ticker_sym]['price'] = float(quote_data['05. price'].iloc[0])
                except (KeyError, ValueError, IndexError, TypeError): pass
                try: data[ticker_sym]['change'] = float(quote_data['09. change'].iloc[0])
                except (KeyError, ValueError, IndexError, TypeError): pass
                try: data[ticker_sym]['change_percent'] = quote_data['10. change percent'].iloc[0]
                except (KeyError, IndexError, TypeError): pass
                logger.info(f"Dati AV per {ticker_sym} recuperati OK.")
            else: logger.warning(f"Risposta vuota da AV per {ticker_sym}."); st.warning(f"Risposta vuota da AV per {ticker_sym}."); data[ticker_sym] = {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'}
        except ValueError as ve:
             msg = f"Errore Alpha Vantage (ValueError) per {ticker_sym}: {ve}"; logger.warning(msg); st.warning(msg); ve_str = str(ve).lower()
             if "call frequency" in ve_str or "api key" in ve_str or "limit" in ve_str or "premium" in ve_str: logger.error("Errore chiave/limite API AV. Interruzione fetch."); st.error("Errore chiave/limite API AV. Interruzione fetch."); break
             data[ticker_sym] = {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'}
        except Exception as e: msg = f"Errore generico fetch AV per {ticker_sym}: {e}"; logger.exception(msg); st.warning(msg); data[ticker_sym] = {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'}
    logger.info(f"Fine fetch dati Alpha Vantage. Effettuate {calls_made} chiamate."); return data

# --- Funzioni Calcolo Indicatori (Manuali per la tabella) ---
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

def compute_all_indicators(symbol: str, hist_daily_df: pd.DataFrame) -> dict:
    """Calcola tutti gli indicatori tecnici (ultimo valore) per la tabella."""
    indicators = {"RSI (1d)": np.nan, "RSI (1w)": np.nan, "RSI (1mo)": np.nan, "SRSI %K (1d)": np.nan, "SRSI %D (1d)": np.nan,"MACD Line (1d)": np.nan, "MACD Signal (1d)": np.nan, "MACD Hist (1d)": np.nan, f"MA({MA_SHORT}d)": np.nan, f"MA({MA_LONG}d)": np.nan, "VWAP (1d)": np.nan, "VWAP % Change (1d)": np.nan }
    min_len_rsi_base = RSI_PERIOD + 1; min_len_srsi_base = RSI_PERIOD + SRSI_PERIOD + max(SRSI_K, SRSI_D) + 5; min_len_macd_base = MACD_SLOW + MACD_SIGNAL + 5; min_len_sma_short = MA_SHORT; min_len_sma_long = MA_LONG; min_len_vwap_base = VWAP_PERIOD; min_len_vwap_change = VWAP_PERIOD + 1

    if not hist_daily_df.empty and 'close' in hist_daily_df.columns:
        if 'volume' not in hist_daily_df.columns: logger.warning(f"{symbol}: TAB: Colonna 'volume' mancante. VWAP N/A."); hist_daily_df['volume'] = np.nan
        close_daily = hist_daily_df['close'].dropna(); len_daily = len(close_daily); df_for_vwap = hist_daily_df[['close', 'volume']]

        # --- Calcoli indicatori giornalieri ---
        if len_daily >= min_len_rsi_base: indicators["RSI (1d)"] = calculate_rsi_manual(close_daily, RSI_PERIOD)
        else: logger.warning(f"{symbol}: TAB: Dati insuff. ({len_daily}/{min_len_rsi_base}) per RSI(1d)")
        if len_daily >= min_len_srsi_base: k, d = calculate_stoch_rsi(close_daily, RSI_PERIOD, SRSI_PERIOD, SRSI_K, SRSI_D); indicators["SRSI %K (1d)"] = k; indicators["SRSI %D (1d)"] = d
        else: logger.warning(f"{symbol}: TAB: Dati insuff. ({len_daily}/{min_len_srsi_base}) per SRSI(1d)")
        if len_daily >= min_len_macd_base: macd_l, macd_s, macd_h = calculate_macd_manual(close_daily, MACD_FAST, MACD_SLOW, MACD_SIGNAL); indicators["MACD Line (1d)"] = macd_l; indicators["MACD Signal (1d)"] = macd_s; indicators["MACD Hist (1d)"] = macd_h
        else: logger.warning(f"{symbol}: TAB: Dati insuff. ({len_daily}/{min_len_macd_base}) per MACD(1d)")
        if len_daily >= min_len_sma_short: indicators[f"MA({MA_SHORT}d)"] = calculate_sma_manual(close_daily, MA_SHORT)
        else: logger.warning(f"{symbol}: TAB: Dati insuff. ({len_daily}/{min_len_sma_short}) per MA({MA_SHORT}d)")
        if len_daily >= min_len_sma_long: indicators[f"MA({MA_LONG}d)"] = calculate_sma_manual(close_daily, MA_LONG)
        else: logger.warning(f"{symbol}: TAB: Dati insuff. ({len_daily}/{min_len_sma_long}) per MA({MA_LONG}d)")
        if len(df_for_vwap) >= min_len_vwap_base:
            indicators["VWAP (1d)"] = calculate_vwap_manual(df_for_vwap.iloc[-VWAP_PERIOD:], VWAP_PERIOD)
            if len(df_for_vwap) >= min_len_vwap_change:
                vwap_today = indicators["VWAP (1d)"]; vwap_yesterday = calculate_vwap_manual(df_for_vwap.iloc[-(VWAP_PERIOD + 1):-1], VWAP_PERIOD)
                if pd.notna(vwap_today) and pd.notna(vwap_yesterday) and vwap_yesterday != 0: indicators["VWAP % Change (1d)"] = ((vwap_today - vwap_yesterday) / vwap_yesterday) * 100
                else: logger.warning(f"{symbol}: TAB: Impossibile calcolare VWAP % Change")
            else: logger.warning(f"{symbol}: TAB: Dati insuff. ({len(df_for_vwap)}/{min_len_vwap_change}) per VWAP % Change(1d)")
        else: logger.warning(f"{symbol}: TAB: Dati insuff. ({len(df_for_vwap)}/{min_len_vwap_base}) per VWAP(1d)")

        # --- Calcoli indicatori Weekly/Monthly ---
        if len_daily > min_len_rsi_base and pd.api.types.is_datetime64_any_dtype(close_daily.index):
            try: # Weekly RSI
                df_weekly = close_daily.resample('W-MON').last()
                if len(df_weekly.dropna()) >= min_len_rsi_base: indicators["RSI (1w)"] = calculate_rsi_manual(df_weekly, RSI_PERIOD)
                else: logger.warning(f"{symbol}: TAB: Dati Weekly insuff. ({len(df_weekly.dropna())}/{min_len_rsi_base}) per RSI(1w)")
            except Exception as e: logger.exception(f"{symbol}: TAB: Errore calcolo RSI weekly:")
            try: # Monthly RSI
                df_monthly = close_daily.resample('ME').last()
                if len(df_monthly.dropna()) >= min_len_rsi_base: indicators["RSI (1mo)"] = calculate_rsi_manual(df_monthly, RSI_PERIOD)
                else: logger.warning(f"{symbol}: TAB: Dati Monthly insuff. ({len(df_monthly.dropna())}/{min_len_rsi_base}) per RSI(1mo)")
            except Exception as e: logger.exception(f"{symbol}: TAB: Errore calcolo RSI monthly:")
    else: logger.warning(f"{symbol}: TAB: Dati giornalieri vuoti per calcolo indicatori.")
    return indicators

# --- Funzioni Segnale (v0.3 Logic - Correct Syntax) ---
def generate_gpt_signal(rsi_1d, rsi_1w, macd_hist, ma_short, ma_long, srsi_k, srsi_d, vwap_1d, current_price):
    """Genera un segnale basato su una combinazione di indicatori (stile 'GPT' - v0.3 Logic - Correct Syntax)."""
    required_inputs = [rsi_1d, macd_hist, ma_short, ma_long, vwap_1d, current_price]
    if any(pd.isna(x) for x in required_inputs): return "⚪️ N/A"
    score = 0
    if current_price > ma_long: score += 1
    else: score -= 1
    if ma_short > ma_long: score += 2
    else: score -= 2
    if current_price > vwap_1d: score += 1
    else: score -= 1
    if macd_hist > 0: score += 2
    else: score -= 2
    if rsi_1d < 30: score += 2
    elif rsi_1d < 40: score += 1
    elif rsi_1d > 70: score -= 2
    elif rsi_1d > 60: score -= 1
    if pd.notna(rsi_1w):
        if rsi_1w < 40: score += 1
        elif rsi_1w > 60: score -= 1
    if pd.notna(srsi_k) and pd.notna(srsi_d):
        if srsi_k < 20 and srsi_d < 20: score += 1
        elif srsi_k > 80 and srsi_d > 80: score -= 1
        elif srsi_k > srsi_d: score += 0.5
        elif srsi_k < srsi_d: score -= 0.5
    if score >= 5.5: return "⚡️ Strong Buy"
    elif score >= 2.5: return "🟢 Buy"
    elif score <= -5.5: return "🚨 Strong Sell"
    elif score <= -2.5: return "🔴 Sell"
    elif score > 0: return "⏳ CTB" if rsi_1d < 60 and current_price > vwap_1d else "🟡 Hold"
    else: return "⚠️ CTS" if rsi_1d > 40 and current_price < vwap_1d else "🟡 Hold"

def generate_gemini_alert(ma_short, ma_long, macd_hist, rsi_1d, vwap_1d, current_price):
    """Genera un alert basato su MA Crossover, MACD, RSI e VWAP (stile 'Gemini' - v0.3 Logic)."""
    required_inputs = [ma_short, ma_long, macd_hist, rsi_1d, vwap_1d, current_price];
    if any(pd.isna(x) for x in required_inputs): return "⚪️ N/A"
    is_uptrend_ma = ma_short > ma_long; is_downtrend_ma = ma_short < ma_long; is_momentum_positive = macd_hist > 0; is_momentum_negative = macd_hist < 0
    is_not_extremely_overbought = rsi_1d < 80; is_not_extremely_oversold = rsi_1d > 20; is_price_above_vwap = current_price > vwap_1d; is_price_below_vwap = current_price < vwap_1d
    if is_uptrend_ma and is_momentum_positive and is_not_extremely_overbought and is_price_above_vwap: return "⚡️ Strong Buy"
    elif is_downtrend_ma and is_momentum_negative and is_not_extremely_oversold and is_price_below_vwap: return "🚨 Strong Sell"
    else: return "🟡 Hold"

# --- NUOVA Funzione Predizione Euristica (v0.6) ---
def generate_heuristic_prediction(hist_daily_df: pd.DataFrame, periods: int = HEURISTIC_PRED_PERIOD) -> str:
    """Genera una previsione euristica basata sul cambiamento di prezzo negli ultimi 'periods' giorni."""
    if hist_daily_df is None or not isinstance(hist_daily_df, pd.DataFrame) or hist_daily_df.empty or 'close' not in hist_daily_df.columns:
        logger.warning("Pred Heur: Dati input non validi.")
        return "⚪️ N/A"
    if len(hist_daily_df) < periods + 1:
        logger.warning(f"Pred Heur: Dati insuff. ({len(hist_daily_df)}/{periods+1})")
        return "⚪️ N/A"
    try:
        change = hist_daily_df['close'].pct_change(periods=periods).iloc[-1]
        if pd.isna(change):
            logger.warning(f"Pred Heur: Risultato pct_change è NaN.")
            return "⚪️ N/A"
        threshold = 0.01 # Soglia +/- 1%
        if change > threshold: return "⬆️ Up"
        elif change < -threshold: return "⬇️ Down"
        else: return "➡️ Flat"
    except IndexError:
         logger.warning(f"Pred Heur: Errore indice durante pct_change.")
         return "⚪️ N/A"
    except Exception as e:
        logger.exception(f"Pred Heur: Errore calcolo: {e}")
        return "⚪️ N/A"


# --- Funzioni Calcolo Indicatori per Grafico (Manuali v0.5) ---
def calculate_sma_series(series: pd.Series, period: int) -> pd.Series:
    """Calcola la SMA per l'intera serie."""
    if not isinstance(series, pd.Series) or series.empty: return pd.Series(index=series.index, dtype=float)
    return series.rolling(window=period, min_periods=period).mean()

def calculate_rsi_series(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calcola l'RSI per l'intera serie."""
    if not isinstance(series, pd.Series) or series.empty: return pd.Series(index=series.index, dtype=float)
    series_valid = series.dropna()
    if len(series_valid) < period + 1: return pd.Series(index=series.index, dtype=float)
    delta = series_valid.diff()
    gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.loc[avg_loss == 0] = 100.0
    rsi = rsi.clip(0, 100)
    return rsi.reindex(series.index)


# --- Funzione Creazione Grafico (Manuali v0.5) ---
def create_coin_chart(df, symbol):
    """Crea un grafico Plotly con Candlestick, MA e RSI (calcoli manuali)."""
    logger.info(f"CHART: Creazione grafico per {symbol} con {len(df)} righe.")
    required_ohlc = ['open', 'high', 'low', 'close'];
    if df.empty or not all(col in df.columns for col in required_ohlc): logger.warning(f"CHART: DataFrame vuoto o OHLC mancanti per {symbol}. Colonne: {df.columns.tolist()}"); return None
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        logger.warning(f"CHART: Index non Datetime per {symbol}. Tentativo conversione.");
        try: df.index = pd.to_datetime(df.index)
        except Exception as e: logger.error(f"CHART: Fallita conversione index a Datetime per {symbol}: {e}"); return None

    # Calcola indicatori usando le funzioni manuali
    try:
        logger.debug(f"CHART: Calcolo indicatori manuali (SMA, RSI) per {symbol}.")
        close_series = df['close'].dropna()
        if close_series.empty: raise ValueError("Serie 'close' è vuota dopo dropna()")
        df['MA_Short'] = calculate_sma_series(close_series, MA_SHORT).reindex(df.index)
        df['MA_Long'] = calculate_sma_series(close_series, MA_LONG).reindex(df.index)
        df['RSI'] = calculate_rsi_series(close_series, RSI_PERIOD).reindex(df.index)
        logger.debug(f"CHART: Colonne dopo calcoli manuali: {df.columns.tolist()}")
    except Exception as calc_err:
        logger.exception(f"CHART: Errore durante calcolo indicatori manuali per {symbol}:")
        st.warning(f"Impossibile calcolare indicatori per il grafico di {symbol}: {calc_err}")
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name=f'{symbol} Prezzo (Daily)', increasing_line_color= 'green', decreasing_line_color= 'red'), row=1, col=1)
    if 'MA_Short' in df.columns and df['MA_Short'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['MA_Short'], mode='lines', line=dict(color='blue', width=1), name=f'MA({MA_SHORT}d)'), row=1, col=1)
    else: logger.warning(f"CHART: Colonna MA_Short non trovata o vuota per {symbol}")
    if 'MA_Long' in df.columns and df['MA_Long'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['MA_Long'], mode='lines', line=dict(color='orange', width=1), name=f'MA({MA_LONG}d)'), row=1, col=1)
    else: logger.warning(f"CHART: Colonna MA_Long non trovata o vuota per {symbol}")
    if 'RSI' in df.columns and df['RSI'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', line=dict(color='purple', width=1), name='RSI (14d)'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1); fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
    else: logger.warning(f"CHART: Colonna RSI non trovata o vuota per {symbol}"); fig.update_yaxes(title_text='RSI N/A', row=2, col=1)
    fig.update_layout(title=f'{symbol}/{VS_CURRENCY.upper()} Analisi Tecnica (Daily)', xaxis_title=None, yaxis_title='Prezzo (USD)', yaxis2_title='RSI', xaxis_rangeslider_visible=False, legend_title_text='Indicatori', height=600, margin=dict(l=50, r=50, t=50, b=50), hovermode="x unified" )
    fig.update_yaxes(autorange=True, row=1, col=1); logger.info(f"CHART: Grafico Plotly per {symbol} creato."); return fig

# --- INIZIO ESECUZIONE PRINCIPALE APP ---
logger.info("Inizio esecuzione UI principale.")
try:
    if not check_password(): st.stop()
    logger.info("Password check superato.")

    # --- TITOLO, BOTTONE REFRESH, TIMESTAMP ---
    col_title, col_button_placeholder, col_button = st.columns([4, 1, 1])
    with col_title:
        st.title("📈 Crypto Technical Dashboard Pro")
    with col_button:
        st.write("") # Spacer
        if st.button("🔄 Aggiorna", help="Forza aggiornamento dati (cancella cache)", key="refresh_button"):
            logger.info("Bottone Aggiorna cliccato.")
            if 'api_warning_shown' in st.session_state:
                del st.session_state['api_warning_shown']
            st.cache_data.clear()
            st.query_params.clear()
            st.rerun()

    last_update_placeholder = st.empty()
    st.caption(f"Cache: Live ({CACHE_TTL/60:.0f}m), Storico Tabella ({CACHE_HIST_TTL/60:.0f}m), Grafico ({CACHE_CHART_TTL/60:.0f}m), Tradizionale ({CACHE_TRAD_TTL/3600:.0f}h).")


    # --- SEZIONE MARKET OVERVIEW ---
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
            except Exception as e: logger.error(f"Errore value_func '{label}': {e}"); value_str = "Errore"
            delta_txt = None; d_color = "off"
        else: value_str = "N/A";
        column.metric(label=label, value=value_str, delta=delta_txt, delta_color=d_color, help=help_text)
    overview_items_row1 = [ ("Fear & Greed Index", None, get_fear_greed_index, "Fonte: Alternative.me"), (f"Total Crypto M.Cap ({VS_CURRENCY.upper()})", None, lambda: f"${format_large_number(total_market_cap)}", "Fonte: CoinGecko"), ("Crypto ETFs Flow (Daily)", None, get_etf_flow, "Dato N/A"), ("S&P 500 (SPY)", "SPY", None, "Fonte: AV (ETF)"), ("Nasdaq (QQQ)", "QQQ", None, "Fonte: AV (ETF)") ]
    overview_cols_1 = st.columns(len(overview_items_row1));
    for i, (label, ticker, func, help_text) in enumerate(overview_items_row1): render_metric(overview_cols_1[i], label, value_func=func, ticker=ticker, data_dict=traditional_market_data, help_text=help_text)
    overview_items_row2 = [ ("Gold (GLD)", "GLD", None, "Fonte: AV (ETF)"), ("Silver (SLV)", "SLV", None, "Fonte: AV (ETF)"), ("Natural Gas (UNG)", "UNG", None, "Fonte: AV (ETF)"), ("UVXY (Volatility)", "UVXY", None, "Fonte: AV"), ("TQQQ (Nasdaq 3x)", "TQQQ", None, "Fonte: AV") ]
    overview_cols_2 = st.columns(len(overview_items_row2));
    for i, (label, ticker, func, help_text) in enumerate(overview_items_row2): render_metric(overview_cols_2[i], label, value_func=func, ticker=ticker, data_dict=traditional_market_data, help_text=help_text)
    st.markdown("<h6>Titoli Principali (Fonte: Alpha Vantage, Cache 4h):</h6>", unsafe_allow_html=True); stock_tickers_row_av = ['NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR', 'COIN', 'MSTR']
    num_stock_cols = 5; stock_cols = st.columns(num_stock_cols);
    for idx, ticker in enumerate(stock_tickers_row_av): render_metric(stock_cols[idx % num_stock_cols], label=ticker, ticker=ticker, data_dict=traditional_market_data, help_text=f"Ticker: {ticker}")
    st.markdown("---")

    # --- LOGICA PRINCIPALE DASHBOARD CRYPTO (Tabella) ---
    st.subheader(f"📊 Analisi Tecnica Crypto ({NUM_COINS} Asset)"); logger.info("Inizio recupero dati crypto live per tabella."); market_data_df, last_cg_update_utc = get_coingecko_market_data(COINGECKO_IDS_LIST, VS_CURRENCY)

    # --- Gestione Timestamp (Corretto) ---
    if last_cg_update_utc:
        timestamp_display_str = "*Timestamp dati live CoinGecko non disponibile.*"
        try:
            if ZoneInfo:
                local_tz = ZoneInfo("Europe/Rome")
                if last_cg_update_utc.tzinfo is None:
                    logger.debug("Timestamp UTC non timezone-aware, aggiungo TZ UTC.")
                    last_cg_update_utc = last_cg_update_utc.replace(tzinfo=ZoneInfo("UTC"))
                last_cg_update_local = last_cg_update_utc.astimezone(local_tz)
                timestamp_display_str = f"*Dati live CoinGecko aggiornati alle: **{last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S %Z')}***"
                logger.info(f"Timestamp visualizzato: {last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            else:
                logger.debug("ZoneInfo non disponibile, uso offset UTC+2 per Roma.")
                offset_hours = 2
                last_cg_update_rome_approx = last_cg_update_utc + timedelta(hours=offset_hours)
                timestamp_display_str = f"*Dati live CoinGecko aggiornati alle: **{last_cg_update_rome_approx.strftime('%Y-%m-%d %H:%M:%S')} (Ora approx. Roma)***"
                logger.info(f"Timestamp visualizzato (approx): {last_cg_update_rome_approx.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            logger.exception("Errore durante formattazione/conversione timestamp:")
            timestamp_display_str = f"*Errore conversione timestamp ({e}). Ora UTC: {last_cg_update_utc.strftime('%Y-%m-%d %H:%M:%S')}*"
        last_update_placeholder.markdown(timestamp_display_str)
    else:
        logger.warning("Timestamp dati live CoinGecko non disponibile (last_cg_update_utc è None).")
        last_update_placeholder.markdown("*Timestamp dati live CoinGecko non disponibile.*")
    # --- Fine Blocco Timestamp ---

    table_results_df = pd.DataFrame(); # Initialize results df
    # --- Verifica Dati Live e Elaborazione Tabella (Corretto) ---
    if market_data_df.empty:
        msg = "Errore critico: Impossibile caricare dati live CoinGecko. Tabella analisi non generata."
        if st.session_state.get("api_warning_shown", False):
             msg = "Tabella Analisi Tecnica non generata: errore caricamento dati live (possibile limite API CoinGecko)."
        logger.error(msg)
        st.error(msg)
    else:
        # --- Elaborazione Tabella ---
        logger.info(f"Dati live CoinGecko OK ({len(market_data_df)} righe), inizio ciclo elaborazione tabella."); results = []; fetch_errors_for_display = []; process_start_time = time.time(); effective_num_coins = len(market_data_df.index)
        if effective_num_coins != NUM_COINS: logger.warning(f"Numero coin API ({effective_num_coins}) != configurate ({NUM_COINS}). Processando {effective_num_coins}.")
        estimated_wait_secs = effective_num_coins * 1 * 6.0; estimated_wait_mins = estimated_wait_secs / 60; spinner_msg = f"Recupero dati storici e calcolo indicatori tabella per {effective_num_coins} crypto... (~{estimated_wait_mins:.1f} min)"
        with st.spinner(spinner_msg):
            coin_ids_ordered = market_data_df.index.tolist(); logger.info(f"Lista ID CoinGecko per tabella: {coin_ids_ordered}"); actual_processed_count = 0
            for i, coin_id in enumerate(coin_ids_ordered):
                symbol = next((sym for sym, c_id in SYMBOL_TO_ID_MAP.items() if c_id == coin_id), "N/A"); logger.info(f"--- Elaborazione Tabella {i+1}/{effective_num_coins}: {symbol} ({coin_id}) ---")
                try:
                    if symbol == "N/A": msg = f"{coin_id}: ID non in mappa locale. Saltato."; logger.warning(msg); fetch_errors_for_display.append(msg); continue
                    live_data = market_data_df.loc[coin_id]; name = live_data.get('name', coin_id); rank = live_data.get('market_cap_rank', 'N/A'); current_price = live_data.get('current_price', np.nan); volume_24h = live_data.get('total_volume', np.nan)
                    change_1h=live_data.get('price_change_percentage_1h_in_currency',np.nan); change_24h=live_data.get('price_change_percentage_24h_in_currency',np.nan); change_7d=live_data.get('price_change_percentage_7d_in_currency',np.nan); change_30d=live_data.get('price_change_percentage_30d_in_currency',np.nan); change_1y=live_data.get('price_change_percentage_1y_in_currency',np.nan)
                    hist_daily_df_table, status_daily = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_DAILY, interval='daily')

                    indicators = {}; gpt_signal = "⚪️ N/A"; gemini_alert = "⚪️ N/A"; heuristic_pred = "⚪️ N/A" # Defaults
                    if status_daily != "Success":
                        fetch_errors_for_display.append(f"{symbol}: Storico Daily (Tabella) - {status_daily}");
                        logger.warning(f"{symbol}: Impossibile calcolare indicatori/segnali tabella.")
                    else:
                        indicators = compute_all_indicators(symbol, hist_daily_df_table)
                        gpt_signal = generate_gpt_signal( indicators.get("RSI (1d)"), indicators.get("RSI (1w)"), indicators.get("MACD Hist (1d)"), indicators.get(f"MA({MA_SHORT}d)"), indicators.get(f"MA({MA_LONG}d)"), indicators.get("SRSI %K (1d)"), indicators.get("SRSI %D (1d)"), indicators.get("VWAP (1d)"), current_price)
                        gemini_alert = generate_gemini_alert( indicators.get(f"MA({MA_SHORT}d)"), indicators.get(f"MA({MA_LONG}d)"), indicators.get("MACD Hist (1d)"), indicators.get("RSI (1d)"), indicators.get("VWAP (1d)"), current_price)
                        heuristic_pred = generate_heuristic_prediction(hist_daily_df_table)

                    coingecko_link = f"https://www.coingecko.com/en/coins/{coin_id}";
                    results.append({ "Rank": rank, "Symbol": symbol, "Name": name, "Gemini Alert": gemini_alert, "GPT Signal": gpt_signal, "Heuristic Pred.": heuristic_pred, f"Prezzo ({VS_CURRENCY.upper()})": current_price, "% 1h": change_1h, "% 24h": change_24h, "% 7d": change_7d, "% 30d": change_30d, "% 1y": change_1y, "RSI (1d)": indicators.get("RSI (1d)"), "RSI (1w)": indicators.get("RSI (1w)"), "RSI (1mo)": indicators.get("RSI (1mo)"), "SRSI %K (1d)": indicators.get("SRSI %K (1d)"), "SRSI %D (1d)": indicators.get("SRSI %D (1d)"), "MACD Hist (1d)": indicators.get("MACD Hist (1d)"), f"MA({MA_SHORT}d)": indicators.get(f"MA({MA_SHORT}d)"), f"MA({MA_LONG}d)": indicators.get(f"MA({MA_LONG}d)"), "VWAP (1d)": indicators.get("VWAP (1d)"), "VWAP % Change (1d)": indicators.get("VWAP % Change (1d)"), f"Volume 24h ({VS_CURRENCY.upper()})": volume_24h, "Link": coingecko_link })
                    logger.info(f"--- Elaborazione Tabella {symbol} completata. ---"); actual_processed_count += 1
                except Exception as coin_err: err_msg = f"Errore grave elaborazione tabella {symbol} ({coin_id}): {coin_err}"; logger.exception(err_msg); fetch_errors_for_display.append(f"{symbol}: Errore Grave Tabella - Vedi Log")
        process_end_time = time.time(); total_time = process_end_time - process_start_time; logger.info(f"Fine ciclo tabella crypto. Processate {actual_processed_count}/{effective_num_coins}. Tempo: {total_time:.1f} sec"); st.sidebar.info(f"Tempo elab. Tabella: {total_time:.1f} sec")

        # --- Visualizzazione Tabella ---
        if results:
            logger.info(f"Creazione DataFrame tabella con {len(results)} risultati.");
            try:
                table_results_df = pd.DataFrame(results); table_results_df['Rank'] = pd.to_numeric(table_results_df['Rank'], errors='coerce'); table_results_df.set_index('Rank', inplace=True, drop=True); table_results_df.sort_index(inplace=True)
                cols_order = [ "Symbol", "Name", "Gemini Alert", "GPT Signal", "Heuristic Pred.", f"Prezzo ({VS_CURRENCY.upper()})", "% 1h", "% 24h", "% 7d", "% 30d", "% 1y", "RSI (1d)", "RSI (1w)", "RSI (1mo)", "SRSI %K (1d)", "SRSI %D (1d)", "MACD Hist (1d)", f"MA({MA_SHORT}d)", f"MA({MA_LONG}d)", "VWAP (1d)", "VWAP % Change (1d)", f"Volume 24h ({VS_CURRENCY.upper()})", "Link" ]
                cols_to_show = [col for col in cols_order if col in table_results_df.columns]; df_display_table = table_results_df[cols_to_show].copy()
                formatters = {}; currency_col = f"Prezzo ({VS_CURRENCY.upper()})"; volume_col = f"Volume 24h ({VS_CURRENCY.upper()})"; pct_cols = ["% 1h", "% 24h", "% 7d", "% 30d", "% 1y", "VWAP % Change (1d)"]; rsi_srsi_cols = [c for c in df_display_table.columns if ("RSI" in c or "SRSI" in c)]; macd_cols = [c for c in df_display_table.columns if "MACD" in c]; ma_vwap_cols = [c for c in df_display_table.columns if ("MA" in c or "VWAP" in c) and "%" not in c]
                if currency_col in df_display_table.columns: formatters[currency_col] = "${:,.4f}";
                if volume_col in df_display_table.columns: formatters[volume_col] = lambda x: f"${format_large_number(x)}";
                for col in pct_cols:
                    if col in df_display_table.columns: formatters[col] = "{:+.2f}%"
                for col in rsi_srsi_cols:
                    if col in df_display_table.columns: formatters[col] = "{:.1f}"
                for col in macd_cols:
                    if col in df_display_table.columns: formatters[col] = "{:.4f}"
                for col in ma_vwap_cols:
                    if col in df_display_table.columns: formatters[col] = "{:,.2f}"
                styled_table = df_display_table.style.format(formatters, na_rep="N/A", precision=4, subset=list(formatters.keys()))

                # --- Funzione Stile Segnale CORRETTA ---
                def highlight_signal_style(val):
                    style = 'color: #6c757d;'
                    font_weight = 'normal'
                    if isinstance(val, str):
                        if "Strong Buy" in val:
                            style = 'color: #198754;'
                            font_weight = 'bold'
                        elif "Buy" in val and "Strong" not in val:
                            style = 'color: #28a745;'
                        elif "Strong Sell" in val:
                            style = 'color: #dc3545;'
                            font_weight = 'bold'
                        elif "Sell" in val and "Strong" not in val:
                            style = 'color: #fd7e14;'
                        elif "CTB" in val:
                            style = 'color: #20c997;'
                        elif "CTS" in val:
                            style = 'color: #ffc107; color: #000;'
                        elif "Hold" in val:
                            style = 'color: #6c757d;'
                        elif "N/A" in val:
                            style = 'color: #adb5bd;'
                    return f'{style} font-weight: {font_weight};'
                def highlight_pct_col_style(val):
                    if pd.isna(val) or not isinstance(val, (int, float)): return ''; color = 'green' if val > 0 else 'red' if val < 0 else '#6c757d'; return f'color: {color};'

                cols_for_pct_style = [col for col in pct_cols if col in df_display_table.columns];
                if cols_for_pct_style: styled_table = styled_table.applymap(highlight_pct_col_style, subset=cols_for_pct_style)
                signal_cols_to_style = ["Gemini Alert", "GPT Signal"] # Exclude Heuristic Pred from specific styling
                for col in signal_cols_to_style:
                     if col in df_display_table.columns:
                         styled_table = styled_table.applymap(highlight_signal_style, subset=[col])

                logger.info("Visualizzazione DataFrame tabella."); st.dataframe(styled_table, use_container_width=True, column_config={"Link": st.column_config.LinkColumn("CoinGecko", help="Link CoinGecko", display_text="🔗 Link", width="small")})
            except Exception as df_err: logger.exception("Errore creazione/styling DataFrame tabella:"); st.error(f"Errore visualizzazione tabella: {df_err}")
        else: logger.warning("Nessun risultato tabella valido da visualizzare."); st.warning("Nessun risultato crypto valido da visualizzare nella tabella.")

        # --- EXPANDER ERRORI RIMOSSO (v0.6) ---
        # fetch_errors_unique_display = sorted(list(set(fetch_errors_for_display)));
        # if fetch_errors_unique_display:
        #     with st.expander("❗️ Errori Elaborazione Dati Tabella", expanded=False):
        #         # ... (codice rimosso) ...

    # --- SEZIONE GRAFICO ---
    st.divider(); st.subheader("💹 Grafico Dettaglio Coin")
    chart_symbol = st.selectbox("Seleziona una coin:", options=SYMBOLS, index=0, key="chart_coin_selector")

    # --- Visualizza Prezzo Corrente per Coin Selezionata (Nuovo v0.6) ---
    if chart_symbol:
        chart_coin_id = SYMBOL_TO_ID_MAP.get(chart_symbol)
        if chart_coin_id and not market_data_df.empty and chart_coin_id in market_data_df.index:
             latest_price = market_data_df.loc[chart_coin_id].get('current_price', np.nan)
             if pd.notna(latest_price):
                 # Usare colonne per mettere il prezzo accanto al selettore
                 col1, col2 = st.columns([3, 1])
                 with col1:
                     pass # Il selectbox è già qui
                 with col2:
                     st.metric(label=f"Prezzo Corrente {chart_symbol}", value=f"${latest_price:,.4f}")
             else:
                 st.caption(f"Prezzo corrente per {chart_symbol} non disponibile.")
        elif chart_coin_id: # Handle case where market data is empty but ID exists
             st.caption(f"Prezzo corrente per {chart_symbol} non disponibile (dati live mancanti).")


    chart_placeholder = st.empty() # Placeholder per grafico/messaggi
    if chart_symbol:
        chart_coin_id = SYMBOL_TO_ID_MAP.get(chart_symbol) # Defined above, re-check for safety
        if chart_coin_id:
            logger.info(f"CHART: Tentativo caricamento dati per grafico {chart_symbol} ({chart_coin_id}).")
            with chart_placeholder:
                 with st.spinner(f"Caricamento dati e grafico per {chart_symbol}..."):
                    chart_hist_df, chart_status = get_coingecko_historical_data_for_chart(chart_coin_id, VS_CURRENCY, DAYS_HISTORY_DAILY)
                    if chart_status == "Success" and not chart_hist_df.empty:
                        fig = create_coin_chart(chart_hist_df.copy(), chart_symbol) # Pass a copy
                        if fig: st.plotly_chart(fig, use_container_width=True); logger.info(f"CHART: Grafico per {chart_symbol} visualizzato.")
                        else: st.error(f"Impossibile generare grafico per {chart_symbol} (errore calcolo indicatori o interno, vedi log).")
                    else: logger.error(f"CHART: Fallito caricamento dati storici per {chart_symbol}. Status: {chart_status}"); st.error(f"Impossibile caricare dati storici per grafico di {chart_symbol}. ({chart_status})")
        else: st.error(f"ID CoinGecko non trovato per {chart_symbol}."); logger.error(f"CHART: ID CoinGecko non trovato per {chart_symbol} in mappa.")

    # --- SEZIONE FORECAST XGBOOST RIMOSSA (v0.6) ---

    # --- LEGENDA (Migliorata v0.6) ---
    st.divider();
    with st.expander("📘 Legenda Completa Indicatori e Segnali", expanded=False):
        st.markdown("""
        *Disclaimer: Questa dashboard è fornita solo a scopo informativo e didattico e non costituisce in alcun modo consulenza finanziaria.*

        **Market Overview:**
        *   **Fear & Greed Index:** Indice del sentiment di mercato crypto (0=Paura Estrema, 100=Avidità Estrema). Valori bassi indicano paura (potenziale opportunità), alti indicano avidità (potenziale rischio). Fonte: Alternative.me.
        *   **Total Crypto M.Cap:** Capitalizzazione di mercato totale ($) di tutte le criptovalute (Fonte: CoinGecko). Misura la dimensione generale del mercato crypto.
        *   **Crypto ETFs Flow:** Flusso netto giornaliero ($) negli ETF crypto spot. Positivo=Afflussi, Negativo=Deflussi. **Dato N/A in questa versione.**
        *   **S&P 500 (SPY), etc.:** Prezzi/Variazioni giornaliere mercati tradizionali. Fonte: Alpha Vantage (**cache 4h**, aggiornamento ritardato).

        **Tabella Analisi Tecnica Crypto:**
        *   **Rank:** Posizione per market cap (CoinGecko).
        *   **Gemini Alert / GPT Signal:** Segnali **esemplificativi/sperimentali**. **NON usare per trading.** Combinano MA, MACD, RSI, VWAP.
            *   ⚡️ Strong Buy / 🟢 Buy: Condizioni tecniche aggregate potenzialmente rialziste.
            *   🚨 Strong Sell / 🔴 Sell: Condizioni tecniche aggregate potenzialmente ribassiste.
            *   🟡 Hold: Condizioni neutre/miste.
            *   ⏳ CTB: Consider To Buy (potenziale rialzista, monitorare).
            *   ⚠️ CTS: Consider To Sell (potenziale ribassista, monitorare).
            *   ⚪️ N/A: Non disponibile (dati insuff.).
        *   **Heuristic Pred.:** Previsione **semplicistica/sperimentale** basata solo su variazione prezzo ultimi 3gg (default). **NON è AI, NON affidabile.**
            *   ⬆️ Up (>+1%), ⬇️ Down (<-1%), ➡️ Flat (-1% a +1%).
            *   ⚪️ N/A: Non calcolabile (dati insuff.).
        *   **Prezzo:** Prezzo corrente ($) (CoinGecko).
        *   **% 1h...1y:** Variazioni percentuali prezzo (CoinGecko).
        *   **RSI (1d, 1w, 1mo):** Relative Strength Index (Daily, Weekly, Monthly). Misura velocità e forza dei movimenti di prezzo. <30 Ipervenduto, >70 Ipercomprato (livelli indicativi). Confrontare timeframe dà idea della forza del trend.
        *   **SRSI %K / %D (1d):** Stochastic RSI (Daily). Oscillatore su RSI, più sensibile. <20 Ipervenduto, >80 Ipercomprato. Crossover K/D può dare segnali.
        *   **MACD Hist (1d):** MACD Histogram (Daily). Momentum. Positivo = Bullish, Negativo = Bearish. Altezza/Profondità = Forza.
        *   **MA(20d) / MA(50d):** Medie Mobili Semplici (Daily). Trend. Prezzo > MA = Bullish. Incrocio MA20>MA50 (Golden Cross) = Bullish; MA20<MA50 (Death Cross) = Bearish.
        *   **VWAP (1d):** Volume Weighted Average Price (Daily, 14 periodi). Prezzo medio ponderato per volume. Livello di riferimento; Prezzo > VWAP = Forza; Prezzo < VWAP = Debolezza.
        *   **VWAP % Change (1d):** Variazione % VWAP giornaliero rispetto al giorno precedente.
        *   **Volume 24h:** Volume scambiato ($) (CoinGecko).
        *   **Link:** Link a CoinGecko.
        *   **N/A:** Dato non disponibile.

        **Grafico Dettaglio Coin:**
        *   Visualizza Candele giornaliere, MA(20d, 50d), RSI(14d) per analisi visiva.

        **Note Generali:**
        *   Recupero dati per **tabella** rallentato (6s/coin).
        *   Recupero dati per **grafico** più veloce (nessuna pausa).
        *   Dati Mercati Tradizionali **cache 4h**.
        *   **DYOR (Do Your Own Research).**
        """)
    st.divider(); st.caption("Disclaimer: Strumento informativo/didattico. Non consulenza finanziaria. DYOR.")
except Exception as main_exception: logger.exception("!!! ERRORE NON GESTITO NELL'ESECUZIONE PRINCIPALE !!!"); st.error(f"Errore imprevisto: {main_exception}. Controlla il log.")

# --- VISUALIZZAZIONE LOG APPLICAZIONE ---
st.divider(); st.subheader("📄 Log Applicazione"); st.caption("Log generati durante l'ultima esecuzione (INFO Level). Utile per monitoraggio.")
log_content = log_stream.getvalue(); st.text_area("Log:", value=log_content, height=300, key="log_display_area", help="Seleziona (Ctrl+A) e copia (Ctrl+C) per analisi/condivisione.")
logger.info("--- Fine esecuzione script Streamlit app.py ---"); log_stream.close()
