# Versione: v14 - FIX DEFINITIVO SyntaxError try/except per Weekly RSI. Include log UI, AV attivo(opz.'d'), layout/colori, WLD/HYPE/FART, TRUMP ID fix, no news.
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time
import math
# import yfinance as yf
# import feedparser
from alpha_vantage.timeseries import TimeSeries # Per Alpha Vantage
import logging
import io

# --- INIZIO: Configurazione Logging in UI ---
log_stream = io.StringIO()
logging.basicConfig(
    stream=log_stream, level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', force=True
)
logger = logging.getLogger(__name__)
logger.info("Logging configurato per UI.")
# --- FINE: Configurazione Logging in UI ---

# Import zoneinfo
try: from zoneinfo import ZoneInfo; logger.info("Modulo 'zoneinfo' importato.")
except ImportError: logger.warning("Modulo 'zoneinfo' non trovato."); st.warning("Modulo 'zoneinfo' non trovato."); ZoneInfo = None

# --- Layout App Streamlit ---
st.set_page_config(layout="wide", page_title="Crypto Technical Dashboard Pro", page_icon="📈")

# --- CSS ---
st.markdown("""<style>div[data-testid="stMetricValue"] { font-size: 14px !important; }</style>""", unsafe_allow_html=True)
logger.info("CSS applicato.")

# --- Configurazione Globale ---
logger.info("Inizio configurazione globale.")
SYMBOL_TO_ID_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin", "SOL": "solana", "XRP": "ripple",
    "RNDR": "render-token", "FET": "fetch-ai", "RAY": "raydium", "SUI": "sui", "ONDO": "ondo-finance",
    "ARB": "arbitrum", "TAO": "bittensor", "LINK": "chainlink", "AVAX": "avalanche-2", "HBAR": "hedera-hashgraph",
    "PEPE": "pepe", "UNI": "uniswap", "TIA": "celestia", "JUP": "jupiter-aggregator", "IMX": "immutable-x",
    "TRUMP": "official-trump", # CORRETTO ID
    "NEAR": "near", "AERO": "aerodrome-finance", "TRON": "tron", "AERGO": "aergo", # Riaggiunto AERGO
    "ADA": "cardano", "MKR": "maker", "WLD": "worldcoin-org", "HYPE": "hyperliquid", "FART": "fartcoin" # Aggiunti WLD, HYPE, FART
}
SYMBOLS = list(SYMBOL_TO_ID_MAP.keys())
COINGECKO_IDS_LIST = list(SYMBOL_TO_ID_MAP.values())
NUM_COINS = len(SYMBOLS) # Ora 31
logger.info(f"Numero coins configurate: {NUM_COINS}")
TRAD_TICKERS_AV = ['SPY', 'QQQ', 'GLD', 'SLV', 'UNG', 'UVXY', 'TQQQ', 'NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR', 'COIN', 'MSTR']
logger.info(f"Tickers tradizionali configurati (Alpha Vantage): {TRAD_TICKERS_AV}")
VS_CURRENCY = "usd"; CACHE_TTL = 1800; CACHE_HIST_TTL = CACHE_TTL * 2; CACHE_TRAD_TTL = 14400 # AV Cache 4h
DAYS_HISTORY_DAILY = 365; DAYS_HISTORY_HOURLY = 7; RSI_PERIOD = 14; SRSI_PERIOD = 14; SRSI_K = 3; SRSI_D = 3
MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9; MA_SHORT = 20; MA_LONG = 50; VWAP_PERIOD = 14
logger.info("Fine configurazione globale.")

# --- DEFINIZIONI FUNZIONI ---

def check_password():
    logger.debug("Esecuzione check_password.")
    if "password_correct" not in st.session_state: st.session_state.password_correct = False
    if not st.session_state.password_correct:
        password = st.text_input("🔑 Password", type="password", key="password_input")
        correct_password = st.secrets.get("APP_PASSWORD", "Leonardo")
        if not correct_password: logger.error("Password APP_PASSWORD non configurata!"); st.error("Password APP_PASSWORD non configurata nei secrets!"); st.stop()
        if password == correct_password:
            logger.info("Password corretta."); st.session_state.password_correct = True
            if st.query_params.get("logged_in") != "true": st.query_params["logged_in"] = "true"; st.rerun()
        elif password: logger.warning("Password errata inserita."); st.warning("Password errata."); st.stop()
        else: logger.debug("Nessuna password inserita."); st.stop()
    logger.debug("Check password superato."); return True

def format_large_number(num):
    if pd.isna(num) or not isinstance(num, (int, float)): return "N/A"
    if abs(num) < 1_000_000: return f"{num:,.0f}"
    elif abs(num) < 1_000_000_000: return f"{num / 1_000_000:.1f}M"
    elif abs(num) < 1_000_000_000_000: return f"{num / 1_000_000_000:.1f}B"
    else: return f"{num / 1_000_000_000_000:.2f}T"

@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento dati di mercato (CoinGecko)...")
def get_coingecko_market_data(ids_list, currency):
    logger.info(f"Tentativo fetch dati live CoinGecko per {len(ids_list)} IDs.")
    ids_string = ",".join(ids_list); url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': currency, 'ids': ids_string, 'order': 'market_cap_desc','per_page': str(len(ids_list)), 'page': 1, 'sparkline': False,'price_change_percentage': '1h,24h,7d,30d,1y', 'precision': 'full'}
    timestamp_utc = datetime.now(ZoneInfo("UTC") if ZoneInfo else None)
    if 'api_warning_shown' not in st.session_state: st.session_state.api_warning_shown = False
    try:
        response = requests.get(url, params=params, timeout=20); response.raise_for_status(); data = response.json()
        if not data: logger.warning("API CoinGecko live: Dati vuoti ricevuti."); st.warning("API CoinGecko live: Dati vuoti."); return pd.DataFrame(), timestamp_utc
        df = pd.DataFrame(data);
        if not df.empty: df.set_index('id', inplace=True)
        st.session_state["api_warning_shown"] = False; logger.info(f"Dati live CoinGecko recuperati per {len(df)} coins."); return df, timestamp_utc
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 429: logger.warning("API CoinGecko (Live): Limite richieste (429)."); st.warning("Attenzione API CoinGecko (Live): Limite richieste (429) raggiunto."); st.session_state["api_warning_shown"] = True
        else: logger.error(f"Errore HTTP API Mercato CoinGecko (Status: {http_err.response.status_code}): {http_err}"); st.error(f"Errore HTTP API Mercato CoinGecko (Status: {http_err.response.status_code}): {http_err}")
        return pd.DataFrame(), timestamp_utc
    except requests.exceptions.RequestException as req_ex: logger.error(f"Errore Richiesta API Mercato CoinGecko: {req_ex}"); st.error(f"Errore Richiesta API Mercato CoinGecko: {req_ex}"); return pd.DataFrame(), timestamp_utc
    except Exception as e: logger.exception("Errore Processamento Dati Mercato CoinGecko:"); st.error(f"Errore Processamento Dati Mercato CoinGecko: {e}"); return pd.DataFrame(), timestamp_utc

@st.cache_data(ttl=CACHE_HIST_TTL, show_spinner=False)
def get_coingecko_historical_data(coin_id, currency, days, interval='daily'):
    logger.info(f"Tentativo fetch dati storici CoinGecko per {coin_id} ({days}d, {interval}).")
    time.sleep(4.0); url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': currency, 'days': str(days), 'interval': interval if interval == 'hourly' else 'daily', 'precision': 'full'}
    status_msg = f"Errore Sconosciuto ({coin_id})";
    try:
        response = requests.get(url, params=params, timeout=25); response.raise_for_status(); data = response.json()
        if not data or 'prices' not in data or not data['prices']: status_msg = f"No Prices Data ({coin_id})"; logger.warning(status_msg); return pd.DataFrame(), status_msg
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'close']); prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms', utc=True); prices_df.set_index('timestamp', inplace=True); hist_df = prices_df
        if 'total_volumes' in data and data['total_volumes']: volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume']); volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms', utc=True); volumes_df.set_index('timestamp', inplace=True); hist_df = prices_df.join(volumes_df, how='outer')
        else: hist_df['volume'] = 0.0
        hist_df = hist_df.interpolate(method='time').ffill().bfill(); hist_df['high'] = hist_df['close']; hist_df['low'] = hist_df['close']; hist_df['open'] = hist_df['close'].shift(1); hist_df['open'].fillna(hist_df['close'], inplace=True)
        hist_df = hist_df[~hist_df.index.duplicated(keep='last')].sort_index(); hist_df.dropna(subset=['close'], inplace=True)
        if hist_df.empty: status_msg = f"Processed Empty ({coin_id})"; logger.warning(status_msg); return pd.DataFrame(), status_msg
        status_msg = "Success"; logger.info(f"Dati storici CoinGecko per {coin_id} recuperati."); return hist_df, status_msg
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        if status_code == 429: status_msg = f"Rate Limited (429) ({coin_id})"
        elif status_code == 404: status_msg = f"Not Found (404) ({coin_id})"
        else: status_msg = f"HTTP Error {status_code} ({coin_id})"
        logger.warning(f"Errore HTTP API Storico CoinGecko: {status_msg}"); return pd.DataFrame(), status_msg
    except requests.exceptions.RequestException as req_ex: status_msg = f"Request Error ({req_ex}) ({coin_id})"; logger.error(f"Errore Richiesta API Storico CoinGecko: {status_msg}"); return pd.DataFrame(), status_msg
    except Exception as e: status_msg = f"Generic Error ({type(e).__name__}) ({coin_id})"; logger.exception(f"Errore Processamento API Storico CoinGecko per {coin_id}:"); return pd.DataFrame(), status_msg

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_fear_greed_index():
    logger.info("Tentativo fetch Fear & Greed Index.")
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data and data.get("data") and isinstance(data["data"], list) and len(data["data"]) > 0:
             latest_data = data["data"][0]; value = latest_data.get("value"); desc = latest_data.get("value_classification")
             if value is not None and desc is not None: logger.info(f"F&G Index: {value} ({desc})."); return f"{int(value)} ({desc})"
        logger.warning("Formato dati F&G Index inatteso."); return "N/A"
    except requests.exceptions.RequestException as req_ex:
        status_code = req_ex.response.status_code if req_ex.response is not None else "N/A"; msg = f"Errore F&G Index (Alt.me Status: {status_code}): {req_ex}"; logger.warning(msg); st.warning(msg); return "N/A"
    except Exception as e: msg = f"Errore Processamento F&G Index (Alt.me): {e}"; logger.exception(msg); st.warning(msg); return "N/A"

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_global_market_data_cg(currency):
    logger.info("Tentativo fetch dati Global CoinGecko.")
    url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json().get('data', {})
        total_mcap = data.get('total_market_cap', {}).get(currency.lower(), np.nan); logger.info(f"Global CoinGecko M.Cap: {total_mcap}."); return total_mcap
    except requests.exceptions.RequestException as req_ex: msg = f"Errore API Global CoinGecko: {req_ex}"; logger.warning(msg); st.warning(msg); return np.nan
    except Exception as e: msg = f"Errore Processamento Global CoinGecko: {e}"; logger.exception(msg); st.warning(msg); return np.nan

def get_etf_flow(): return "N/A"

@st.cache_data(ttl=CACHE_TRAD_TTL, show_spinner="Caricamento dati mercato tradizionale (Alpha Vantage)...")
def get_traditional_market_data_av(tickers):
    # [RIATTIVATA - con logging e cache 4h]
    logger.info(f"Tentativo fetch dati Alpha Vantage per {len(tickers)} tickers.")
    data = {ticker: {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'} for ticker in tickers}; api_key = None;
    try: api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]; logger.info("Chiave API Alpha Vantage letta.");
    except KeyError: logger.error("Secret 'ALPHA_VANTAGE_API_KEY' non definito."); st.error("Secret 'ALPHA_VANTAGE_API_KEY' non definito."); return data
    except Exception as e: logger.exception("Errore lettura secrets Alpha Vantage:"); st.error(f"Errore lettura secrets: {e}"); return data
    if not api_key: logger.error("Chiave API Alpha Vantage vuota."); st.error("Chiave API Alpha Vantage vuota nei Secrets."); return data
    ts = TimeSeries(key=api_key, output_format='pandas'); calls_made = 0; max_calls_per_minute = 5; delay_between_calls = (60.0 / max_calls_per_minute) + 1.0
    for ticker_sym in tickers:
        # Non c'è un modo affidabile per tracciare il limite giornaliero qui senza stato esterno
        # if calls_made >= 25: msg = f"Limite giornaliero AV raggiunto? Stop fetch per {ticker_sym}+."; logger.warning(msg); st.warning(msg); break
        try:
            logger.info(f"Fetch AV per {ticker_sym} (Pausa {delay_between_calls:.1f}s)..."); time.sleep(delay_between_calls); quote_data, meta_data = ts.get_quote_endpoint(symbol=ticker_sym); calls_made += 1
            if not quote_data.empty:
                try: data[ticker_sym]['price'] = float(quote_data['05. price'].iloc[0])
                except (KeyError, ValueError, IndexError, TypeError): data[ticker_sym]['price'] = np.nan
                try: data[ticker_sym]['change'] = float(quote_data['09. change'].iloc[0])
                except (KeyError, ValueError, IndexError, TypeError): data[ticker_sym]['change'] = np.nan
                try: data[ticker_sym]['change_percent'] = quote_data['10. change percent'].iloc[0]
                except (KeyError, IndexError, TypeError): data[ticker_sym]['change_percent'] = 'N/A'
                logger.info(f"Dati AV per {ticker_sym} recuperati.")
            else: logger.warning(f"Risposta vuota AV per {ticker_sym}."); st.warning(f"Risposta vuota AV per {ticker_sym}."); data[ticker_sym] = {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'}
        except ValueError as ve:
             msg = f"Errore AV (ValueError) per {ticker_sym}: {ve}"; logger.warning(msg); st.warning(msg);
             if "API call frequency" in str(ve).lower() or "API key" in str(ve).lower() or "limit" in str(ve).lower() or "premium" in str(ve).lower(): logger.error("Errore chiave/limite API Alpha Vantage rilevato. Interruzione fetch."); st.error("Errore chiave/limite API Alpha Vantage rilevato."); break
             data[ticker_sym] = {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'}
        except Exception as e: msg = f"Errore generico AV per {ticker_sym}: {e}"; logger.exception(msg); st.warning(msg); data[ticker_sym] = {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'}
    logger.info("Fine fetch dati Alpha Vantage.")
    return data

# --- Funzioni Calcolo Indicatori ---
def calculate_rsi_manual(series, period=RSI_PERIOD):
    if not isinstance(series, pd.Series) or series.empty or series.isna().all(): return np.nan
    series = series.dropna();
    if len(series) < period + 1: return np.nan
    delta = series.diff(); gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean(); avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    if avg_gain.isna().all() or avg_loss.isna().all(): return np.nan
    last_avg_loss = avg_loss.iloc[-1]; last_avg_gain = avg_gain.iloc[-1]
    if last_avg_loss == 0: return 100.0 if last_avg_gain > 0 else 50.0
    rs = last_avg_gain / last_avg_loss; rsi = 100.0 - (100.0 / (1.0 + rs))
    return max(0.0, min(100.0, rsi))

def calculate_stoch_rsi(series, rsi_period=RSI_PERIOD, stoch_period=SRSI_PERIOD, k_smooth=SRSI_K, d_smooth=SRSI_D):
    if not isinstance(series, pd.Series) or series.empty or series.isna().all(): return np.nan, np.nan
    series = series.dropna();
    if len(series) < rsi_period + stoch_period: return np.nan, np.nan
    delta = series.diff(); gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean(); avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan); rsi_series = (100.0 - (100.0 / (1.0 + rs))).dropna()
    if len(rsi_series) < stoch_period: return np.nan, np.nan
    min_rsi = rsi_series.rolling(window=stoch_period).min(); max_rsi = rsi_series.rolling(window=stoch_period).max()
    range_rsi = max_rsi - min_rsi; stoch_rsi_k_raw = 100 * (rsi_series - min_rsi) / range_rsi.replace(0, np.nan)
    stoch_rsi_k_raw = stoch_rsi_k_raw.dropna()
    if len(stoch_rsi_k_raw) < k_smooth : return np.nan, np.nan
    stoch_rsi_k = stoch_rsi_k_raw.rolling(window=k_smooth).mean()
    if len(stoch_rsi_k.dropna()) < d_smooth :
         last_k_val = stoch_rsi_k.iloc[-1]; return max(0.0, min(100.0, last_k_val)) if pd.notna(last_k_val) else np.nan, np.nan
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_smooth).mean()
    last_k = stoch_rsi_k.iloc[-1]; last_d = stoch_rsi_d.iloc[-1]
    last_k = max(0.0, min(100.0, last_k)) if pd.notna(last_k) else np.nan; last_d = max(0.0, min(100.0, last_d)) if pd.notna(last_d) else np.nan
    return last_k, last_d

def calculate_macd_manual(series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    if not isinstance(series, pd.Series) or series.empty or series.isna().all(): return np.nan, np.nan, np.nan
    series = series.dropna();
    if len(series) < slow + signal -1 : return np.nan, np.nan, np.nan
    ema_fast = series.ewm(span=fast, adjust=False).mean(); ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow; signal_line = macd_line.ewm(span=signal, adjust=False).mean(); histogram = macd_line - signal_line
    last_macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else np.nan
    last_signal = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else np.nan
    last_hist = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else np.nan
    return last_macd, last_signal, last_hist

def calculate_sma_manual(series, period):
    if not isinstance(series, pd.Series) or series.empty or series.isna().all(): return np.nan
    series = series.dropna();
    if len(series) < period: return np.nan
    return series.rolling(window=period).mean().iloc[-1]

def calculate_vwap_manual(df, period=VWAP_PERIOD):
    required_cols = ['close', 'volume'];
    if not isinstance(df, pd.DataFrame) or df.empty or not all(col in df.columns for col in required_cols): return np.nan
    df_valid = df.dropna(subset=required_cols);
    if len(df_valid) < period: return np.nan
    df_period = df_valid.iloc[-period:]; total_volume = df_period['volume'].sum()
    if total_volume == 0: return df_period['close'].iloc[-1] if not df_period.empty else np.nan
    vwap = (df_period['close'] * df_period['volume']).sum() / total_volume; return vwap

def compute_all_indicators(symbol, hist_daily_df, hist_hourly_df):
    """Computes all technical indicators for a given symbol, logging warnings for insufficient data."""
    # RSI(1h) rimosso dal dizionario iniziale
    indicators = {"RSI (1d)": np.nan, "RSI (1w)": np.nan, "RSI (1mo)": np.nan,"SRSI %K (1d)": np.nan, "SRSI %D (1d)": np.nan,"MACD Line (1d)": np.nan, "MACD Signal (1d)": np.nan, "MACD Hist (1d)": np.nan,f"MA({MA_SHORT}d)": np.nan, f"MA({MA_LONG}d)": np.nan,"VWAP (1d)": np.nan,}
    min_len_rsi_base = RSI_PERIOD + 1; min_len_srsi_base = RSI_PERIOD + SRSI_PERIOD + 5; min_len_macd_base = MACD_SLOW + MACD_SIGNAL + 5; min_len_vwap_base = VWAP_PERIOD + 1
    if not hist_daily_df.empty and 'close' in hist_daily_df.columns:
        close_daily = hist_daily_df['close'].dropna(); len_daily = len(close_daily)
        # Calcola indicatori giornalieri
        if len_daily >= min_len_rsi_base: indicators["RSI (1d)"] = calculate_rsi_manual(close_daily, RSI_PERIOD)
        else: logger.warning(f"{symbol}: Dati insuff. ({len_daily}/{min_len_rsi_base}) per RSI(1d)")
        if len_daily >= min_len_srsi_base: indicators["SRSI %K (1d)"], indicators["SRSI %D (1d)"] = calculate_stoch_rsi(close_daily, RSI_PERIOD, SRSI_PERIOD, SRSI_K, SRSI_D)
        else: logger.warning(f"{symbol}: Dati insuff. ({len_daily}/{min_len_srsi_base}) per SRSI(1d)")
        if len_daily >= min_len_macd_base: macd_l, macd_s, macd_h = calculate_macd_manual(close_daily, MACD_FAST, MACD_SLOW, MACD_SIGNAL); indicators["MACD Line (1d)"] = macd_l; indicators["MACD Signal (1d)"] = macd_s; indicators["MACD Hist (1d)"] = macd_h
        else: logger.warning(f"{symbol}: Dati insuff. ({len_daily}/{min_len_macd_base}) per MACD(1d)")
        if len_daily >= MA_SHORT: indicators[f"MA({MA_SHORT}d)"] = calculate_sma_manual(close_daily, MA_SHORT)
        else: logger.warning(f"{symbol}: Dati insuff. ({len_daily}/{MA_SHORT}) per MA({MA_SHORT}d)")
        if len_daily >= MA_LONG: indicators[f"MA({MA_LONG}d)"] = calculate_sma_manual(close_daily, MA_LONG)
        else: logger.warning(f"{symbol}: Dati insuff. ({len_daily}/{MA_LONG}) per MA({MA_LONG}d)")
        if len_daily >= min_len_vwap_base: indicators["VWAP (1d)"] = calculate_vwap_manual(hist_daily_df, VWAP_PERIOD)
        else: logger.warning(f"{symbol}: Dati insuff. ({len_daily}/{min_len_vwap_base}) per VWAP(1d)")
        # Calcola indicatori Weekly/Monthly (Blocco try/except corretto)
        if len_daily > min_len_rsi_base and pd.api.types.is_datetime64_any_dtype(close_daily.index):
            try: # Weekly RSI
                df_weekly = close_daily.resample('W-MON').last()
                if len(df_weekly.dropna()) >= min_len_rsi_base: indicators["RSI (1w)"] = calculate_rsi_manual(df_weekly, RSI_PERIOD)
                else: logger.warning(f"{symbol}: Dati Weekly insuff. ({len(df_weekly.dropna())}/{min_len_rsi_base}) per RSI(1w)")
            except Exception as e: logger.exception(f"{symbol}: Errore calcolo RSI weekly:") # FIX SyntaxError
            try: # Monthly RSI
                df_monthly = close_daily.resample('ME').last()
                if len(df_monthly.dropna()) >= min_len_rsi_base: indicators["RSI (1mo)"] = calculate_rsi_manual(df_monthly, RSI_PERIOD)
                else: logger.warning(f"{symbol}: Dati Monthly insuff. ({len(df_monthly.dropna())}/{min_len_rsi_base}) per RSI(1mo)")
            except Exception as e: logger.exception(f"{symbol}: Errore calcolo RSI monthly:")
    else: logger.warning(f"{symbol}: Dati giornalieri vuoti o mancanti per calcolo indicatori.")
    # NON calcola più RSI(1h)
    # if not hist_hourly_df.empty and 'close' in hist_hourly_df.columns: ...
    return indicators

# --- Funzioni Segnale ---
def generate_gpt_signal(rsi_1d, rsi_1h, rsi_1w, macd_hist, ma_short, ma_long, srsi_k, srsi_d, current_price):
    # [Aggiornata per non usare rsi_1h]
    rsi_1h = np.nan # Imposta a NaN perché non più calcolato
    required_inputs = [rsi_1d, macd_hist, ma_short, ma_long, current_price];
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
    if pd.notna(rsi_1w):
        if rsi_1w < 40: score += 1
        elif rsi_1w > 60: score -= 1
    # Rimosso contributo rsi_1h
    if pd.notna(srsi_k) and pd.notna(srsi_d):
        if srsi_k < 20 and srsi_d < 20: score += 1
        elif srsi_k > 80 and srsi_d > 80: score -= 1
    if score >= 5: return "⚡️ Strong Buy"
    elif score >= 2: return "🟢 Buy"
    elif score <= -5: return "🚨 Strong Sell"
    elif score <= -2: return "🔴 Sell"
    elif score > 0: return "⏳ CTB" if pd.notna(rsi_1d) and rsi_1d < 55 else "🟡 Hold"
    else: return "⚠️ CTS" if pd.notna(rsi_1d) and rsi_1d > 45 else "🟡 Hold"

def generate_gemini_alert(ma_short, ma_long, macd_hist, rsi_1d):
    # [Invariata]
    if pd.isna(ma_short) or pd.isna(ma_long) or pd.isna(macd_hist) or pd.isna(rsi_1d): return "⚪️ N/D"
    is_uptrend_ma = ma_short > ma_long; is_momentum_positive = macd_hist > 0; is_not_extremely_overbought = rsi_1d < 80
    is_downtrend_ma = ma_short < ma_long; is_momentum_negative = macd_hist < 0; is_not_extremely_oversold = rsi_1d > 20
    if is_uptrend_ma and is_momentum_positive and is_not_extremely_overbought: return "⚡️ Strong Buy"
    elif is_downtrend_ma and is_momentum_negative and is_not_extremely_oversold: return "🚨 Strong Sell"
    else: return "🟡 Hold"


# --- INIZIO ESECUZIONE PRINCIPALE APP ---
logger.info("Inizio esecuzione UI principale.")
try: # Blocco try principale

    if not check_password(): st.stop()
    logger.info("Password check superato.")

    # --- TITOLO, BOTTONE, TIMESTAMP ---
    col_title, col_button_placeholder, col_button = st.columns([4, 1, 1])
    with col_title: st.title("📈 Crypto Technical Dashboard Pro")
    with col_button:
        st.write("") # Spacer
        if st.button("🔄 Aggiorna", help=f"Forza aggiornamento dati", key="refresh_button"):
            logger.info("Bottone Aggiorna cliccato.");
            if 'api_warning_shown' in st.session_state: del st.session_state['api_warning_shown']
            st.cache_data.clear(); st.query_params.clear(); st.rerun()
    last_update_placeholder = st.empty()
    # Aggiornato caption per AV riattivato ma limitato
    st.caption(f"Cache: Crypto Live ({CACHE_TTL/60:.0f}m), Storico ({CACHE_HIST_TTL/60:.0f}m), Tradizionale ({CACHE_TRAD_TTL/3600:.0f}h).")

    # --- SEZIONE MARKET OVERVIEW (AV ATTIVO CON CACHE LUNGA) ---
    st.markdown("---"); st.subheader("🌐 Market Overview")
    fear_greed_value = get_fear_greed_index()
    total_market_cap = get_global_market_data_cg(VS_CURRENCY)
    etf_flow_value = get_etf_flow()
    # ** RIATTIVATA CHIAMATA ALPHA VANTAGE **
    traditional_market_data = get_traditional_market_data_av(TRAD_TICKERS_AV)

    def format_delta(change_val, change_pct_str):
        delta_string = None;
        if pd.notna(change_val) and isinstance(change_pct_str, str) and change_pct_str not in ['N/A', '', None]:
            try: change_pct_val = float(change_pct_str.replace('%','').strip()); delta_string = f"{change_val:+.2f} ({change_pct_val:+.2f}%)"
            except (ValueError, AttributeError): delta_string = f"{change_val:+.2f} (?%)"
        elif pd.notna(change_val): delta_string = f"{change_val:+.2f}"
        return delta_string

    # RIGA 1 Overview
    overview_items_row1 = [("Fear & Greed Index", None, get_fear_greed_index, "Fonte: Alternative.me"),(f"Total Crypto M.Cap ({VS_CURRENCY.upper()})", None, lambda: f"${format_large_number(total_market_cap)}", "Fonte: CoinGecko"),("Crypto ETFs Flow (Daily)", None, get_etf_flow, "Dato N/A"),("S&P 500 (SPY)", "SPY", None, "Fonte: Alpha Vantage (ETF)"),("Nasdaq (QQQ)", "QQQ", None, "Fonte: Alpha Vantage (ETF)")]
    overview_cols_1 = st.columns(5)
    for i, (label, ticker, func, help_text) in enumerate(overview_items_row1):
        with overview_cols_1[i]:
            value_str = "N/A"; delta_txt = None; d_color = "off"; change = np.nan
            if ticker:
                trad_info = traditional_market_data.get(ticker, {}); price = trad_info.get('price', np.nan); change = trad_info.get('change', np.nan); change_pct = trad_info.get('change_percent', 'N/A')
                value_str = f"{price:,.2f}" if pd.notna(price) else "N/A"; delta_txt = format_delta(change, change_pct)
                if pd.notna(change): d_color = "normal" # Colore Delta corretto
            elif func: value_str = func()
            st.metric(label=label, value=value_str, delta=delta_txt, delta_color=d_color, help=help_text)

    # RIGA 2 Overview
    overview_items_row2 = [("Gold (GLD)", "GLD", None, "Fonte: Alpha Vantage (ETF)"), ("Silver (SLV)", "SLV", None, "Fonte: Alpha Vantage (ETF)"), ("Natural Gas (UNG)", "UNG", None, "Fonte: Alpha Vantage (ETF)"), ("UVXY (Volatility)", "UVXY", None, "Fonte: Alpha Vantage"), ("TQQQ (Nasdaq 3x)", "TQQQ", None, "Fonte: Alpha Vantage")]
    overview_cols_2 = st.columns(5)
    for i, (label, ticker, func, help_text) in enumerate(overview_items_row2):
         with overview_cols_2[i]:
            price = np.nan; change = np.nan; change_pct = 'N/A'; value_str = "N/A"; delta_txt = None; d_color = "off"
            trad_info = traditional_market_data.get(ticker, {}); price = trad_info.get('price', np.nan); change = trad_info.get('change', np.nan); change_pct = trad_info.get('change_percent', 'N/A')
            value_str = f"{price:,.2f}" if pd.notna(price) else "N/A"; delta_txt = format_delta(change, change_pct)
            if pd.notna(change): d_color = "normal" # Colore Delta corretto
            st.metric(label=label, value=value_str, delta=delta_txt, delta_color=d_color, help=help_text)

    # SEZIONE TITOLI PRINCIPALI
    st.markdown("<h6>Titoli Principali (Prezzi e Var. Giorno):</h6>", unsafe_allow_html=True)
    stock_tickers_row_av = ['NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR', 'COIN', 'MSTR']
    num_stock_cols = 5; stock_cols = st.columns(num_stock_cols)
    for idx, ticker in enumerate(stock_tickers_row_av):
        col_index = idx % num_stock_cols; current_col = stock_cols[col_index]
        with current_col:
            price = np.nan; change = np.nan; change_pct = 'N/A'; value_str = "N/A"; delta_txt = None; d_color = "off"
            trad_info = traditional_market_data.get(ticker, {}); price = trad_info.get('price', np.nan); change = trad_info.get('change', np.nan); change_pct = trad_info.get('change_percent', 'N/A')
            value_str = f"{price:,.2f}" if pd.notna(price) else "N/A"; delta_txt = format_delta(change, change_pct)
            if pd.notna(change): d_color = "normal" # Colore Delta corretto
            st.metric(label=ticker, value=value_str, delta=delta_txt, delta_color=d_color)
    st.markdown("---")

    # --- LOGICA PRINCIPALE DASHBOARD CRYPTO ---
    st.subheader(f"📊 Analisi Tecnica Crypto ({NUM_COINS} Asset)")
    logger.info("Inizio recupero dati crypto live.")
    market_data_df, last_cg_update_utc = get_coingecko_market_data(COINGECKO_IDS_LIST, VS_CURRENCY)

    # Gestione Timestamp
    if last_cg_update_utc and ZoneInfo:
        try: local_tz = ZoneInfo("Europe/Rome");
        if last_cg_update_utc.tzinfo is None: last_cg_update_utc = last_cg_update_utc.replace(tzinfo=ZoneInfo("UTC"))
        last_cg_update_local = last_cg_update_utc.astimezone(local_tz); last_update_placeholder.markdown(f"*Dati live CoinGecko aggiornati alle: **{last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S %Z')}***")
        logger.info(f"Timestamp aggiornamento: {last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        except Exception as e: logger.exception("Errore conversione timestamp:"); last_update_placeholder.markdown(f"*Errore conversione timestamp ({e}). Ora UTC approx: {last_cg_update_utc.strftime('%Y-%m-%d %H:%M:%S')}*")
    elif last_cg_update_utc: last_cg_update_rome_approx = last_cg_update_utc + timedelta(hours=2); last_update_placeholder.markdown(f"*Dati live CoinGecko aggiornati alle: **{last_cg_update_rome_approx.strftime('%Y-%m-%d %H:%M:%S')} (Ora approx. Roma)***"); logger.info(f"Timestamp aggiornamento (approx): {last_cg_update_rome_approx.strftime('%Y-%m-%d %H:%M:%S')}")
    else: logger.warning("Timestamp aggiornamento dati live CoinGecko non disponibile."); last_update_placeholder.markdown("*Timestamp aggiornamento dati live CoinGecko non disponibile.*")

    # Blocco se dati live falliscono
    if market_data_df.empty:
        msg = "Errore critico: Impossibile caricare dati live CoinGecko. Tabella non generata."
        if st.session_state.get("api_warning_shown", False): msg = "Tabella Analisi Tecnica non generata causa errore caricamento dati live (Limite API?)."
        logger.error(msg); st.error(msg); st.stop()
    logger.info(f"Dati live CoinGecko OK ({len(market_data_df)} righe), inizio ciclo elaborazione crypto.")

    # --- CICLO PROCESSING PER OGNI COIN ---
    results = []; fetch_errors_for_display = []
    process_start_time = time.time()
    logger.info(f"Inizio ciclo elaborazione per {NUM_COINS} crypto.")
    # Stima tempo aggiornata con NUM_COINS
    estimated_wait_secs = NUM_COINS * 2 * 4; estimated_wait_mins = estimated_wait_secs / 60
    with st.spinner(f"Recupero dati storici e calcolo indicatori per {NUM_COINS} crypto... (Richiede ~{estimated_wait_mins:.1f} min)"):
        coin_ids_ordered = market_data_df.index.tolist()
        logger.info(f"Lista ID CoinGecko da processare (dalla chiamata live): {coin_ids_ordered}")
        actual_processed_count = 0
        for i, coin_id in enumerate(coin_ids_ordered):
            symbol = next((sym for sym, c_id in SYMBOL_TO_ID_MAP.items() if c_id == coin_id), "N/A")
            logger.info(f"--- Elaborazione {i+1}/{len(coin_ids_ordered)}: {symbol} ({coin_id}) ---")
            try:
                if symbol == "N/A": msg = f"{coin_id}: ID non trovato nella mappa SYMBOL_TO_ID_MAP."; logger.warning(msg); fetch_errors_for_display.append(msg); continue
                live_data = market_data_df.loc[coin_id]; name = live_data.get('name', coin_id)
                rank = live_data.get('market_cap_rank', 'N/A'); current_price = live_data.get('current_price', np.nan); volume_24h = live_data.get('total_volume', np.nan)
                change_1h = live_data.get('price_change_percentage_1h_in_currency', np.nan); change_24h = live_data.get('price_change_percentage_24h_in_currency', np.nan)
                change_7d = live_data.get('price_change_percentage_7d_in_currency', np.nan); change_30d = live_data.get('price_change_percentage_30d_in_currency', np.nan)
                change_1y = live_data.get('price_change_percentage_1y_in_currency', np.nan)
                hist_daily_df, status_daily = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_DAILY, interval='daily')
                if status_daily != "Success": fetch_errors_for_display.append(f"{symbol}: Daily - {status_daily}")
                hist_hourly_df, status_hourly = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_HOURLY, interval='hourly')
                if status_hourly != "Success": fetch_errors_for_display.append(f"{symbol}: Hourly - {status_hourly}")
                indicators = compute_all_indicators(symbol, hist_daily_df, hist_hourly_df)
                gpt_signal = generate_gpt_signal(indicators.get("RSI (1d)"), None, indicators.get("RSI (1w)"), indicators.get("MACD Hist (1d)"), indicators.get(f"MA({MA_SHORT}d)"), indicators.get(f"MA({MA_LONG}d)"), indicators.get("SRSI %K (1d)"), indicators.get("SRSI %D (1d)"), current_price) # Passato None per RSI(1h)
                gemini_alert = generate_gemini_alert(indicators.get(f"MA({MA_SHORT}d)"), indicators.get(f"MA({MA_LONG}d)"), indicators.get("MACD Hist (1d)"), indicators.get("RSI (1d)"))
                coingecko_link = f"https://www.coingecko.com/en/coins/{coin_id}" # Link CoinGecko
                results.append({
                    "Rank": rank, "Symbol": symbol, "Name": name, "Gemini Alert": gemini_alert, "GPT Signal": gpt_signal,
                    f"Prezzo ({VS_CURRENCY.upper()})": current_price, "% 1h": change_1h, "% 24h": change_24h, "% 7d": change_7d, "% 30d": change_30d, "% 1y": change_1y,
                     # RSI(1h) Rimosso: "RSI (1h)": indicators.get("RSI (1h)"),
                    "RSI (1d)": indicators.get("RSI (1d)"), "RSI (1w)": indicators.get("RSI (1w)"), "RSI (1mo)": indicators.get("RSI (1mo)"),
                    "SRSI %K (1d)": indicators.get("SRSI %K (1d)"), "SRSI %D (1d)": indicators.get("SRSI %D (1d)"), "MACD Hist (1d)": indicators.get("MACD Hist (1d)"),
                    f"MA({MA_SHORT}d)": indicators.get(f"MA({MA_SHORT}d)"), f"MA({MA_LONG}d)": indicators.get(f"MA({MA_LONG}d)"), "VWAP (1d)": indicators.get("VWAP (1d)"),
                    f"Volume 24h ({VS_CURRENCY.upper()})": volume_24h, "Link": coingecko_link # Aggiunto Link
                })
                logger.info(f"--- Elaborazione {symbol} completata. ---")
                actual_processed_count += 1
            except Exception as coin_err: err_msg = f"Errore irreversibile durante elaborazione di {symbol} ({coin_id}): {coin_err}"; logger.exception(err_msg); st.warning(f"Errore elaborazione {symbol}. Controlla log sotto."); fetch_errors_for_display.append(f"{symbol}: Errore Elaborazione - Vedi Log")
    process_end_time = time.time(); total_time = process_end_time - process_start_time
    logger.info(f"Fine ciclo elaborazione crypto. Processate {actual_processed_count}/{len(coin_ids_ordered)} coin. Tempo totale: {total_time:.1f} sec")
    st.sidebar.info(f"Tempo elaborazione crypto: {total_time:.1f} sec")

    # --- CREA E VISUALIZZA DATAFRAME ---
    if results:
        logger.info(f"Creazione DataFrame con {len(results)} risultati.")
        try:
            df = pd.DataFrame(results); df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce'); df.set_index('Rank', inplace=True, drop=True); df.sort_index(inplace=True)
            # AGGIORNATO cols_order: rimosso RSI(1h), aggiunto Link
            cols_order = ["Symbol", "Name", "Gemini Alert", "GPT Signal",f"Prezzo ({VS_CURRENCY.upper()})","% 1h", "% 24h", "% 7d", "% 30d", "% 1y","RSI (1d)", "RSI (1w)", "RSI (1mo)","SRSI %K (1d)", "SRSI %D (1d)","MACD Hist (1d)", f"MA({MA_SHORT}d)", f"MA({MA_LONG}d)", "VWAP (1d)",f"Volume 24h ({VS_CURRENCY.upper()})", "Link"]
            cols_to_show = [col for col in cols_order if col in df.columns]; df_display = df[cols_to_show].copy()
            formatters = {}; currency_col = f"Prezzo ({VS_CURRENCY.upper()})"; volume_col = f"Volume 24h ({VS_CURRENCY.upper()})"; pct_cols = ["% 1h", "% 24h", "% 7d", "% 30d", "% 1y"]
            rsi_srsi_cols = [col for col in df_display.columns if ("RSI" in col or "SRSI" in col) and col != "RSI (1h)"]; macd_cols = [col for col in df_display.columns if "MACD" in col]; ma_vwap_cols = [col for col in df_display.columns if "MA" in col or "VWAP" in col]
            formatters[currency_col] = "${:,.4f}";
            for col in pct_cols: formatters[col] = "{:+.2f}%";
            formatters[volume_col] = lambda x: f"${format_large_number(x)}";
            for col in rsi_srsi_cols: formatters[col] = "{:.1f}";
            for col in macd_cols: formatters[col] = "{:.4f}";
            for col in ma_vwap_cols: formatters[col] = "{:,.2f}";
            # Non formattare la colonna Link, sarà gestita da column_config
            # if "Link" in formatters: del formatters['Link']
            styled_df = df_display.style.format(formatters, na_rep="N/A", precision=4, subset=list(formatters.keys()))

            def highlight_pct_col_style(val):
                if pd.isna(val): return ''
                color = 'green' if val > 0 else 'red' if val < 0 else '#6c757d'; return f'color: {color};'
            def highlight_signal_style(val):
                style = 'color: #6c757d;'; font_weight = 'normal'
                if isinstance(val, str):
                    if "Strong Buy" in val: style = 'color: #198754;'; font_weight = 'bold';
                    elif "Buy" in val and "Strong" not in val: style = 'color: #28a745;'; font_weight = 'normal';
                    elif "Strong Sell" in val: style = 'color: #dc3545;'; font_weight = 'bold';
                    elif "Sell" in val and "Strong" not in val: style = 'color: #fd7e14;'; font_weight = 'normal';
                    elif "CTB" in val: style = 'color: #20c997;'; font_weight = 'normal';
                    elif "CTS" in val: style = 'color: #ffc107; color: #000;'; font_weight = 'normal';
                    elif "N/D" in val: style = 'color: #adb5bd;'; font_weight = 'normal';
                return f'{style} font-weight: {font_weight};'

            # Applica stili condizionali (assicurati che le colonne esistano)
            cols_for_pct_style = [col for col in pct_cols if col in df_display.columns]
            if cols_for_pct_style: styled_df = styled_df.applymap(highlight_pct_col_style, subset=cols_for_pct_style)
            if "Gemini Alert" in df_display.columns: styled_df = styled_df.applymap(highlight_signal_style, subset=["Gemini Alert"])
            if "GPT Signal" in df_display.columns: styled_df = styled_df.applymap(highlight_signal_style, subset=["GPT Signal"])

            logger.info("Visualizzazione DataFrame principale.")
            st.dataframe(
                styled_df,
                use_container_width=True,
                column_config={
                    "Link": st.column_config.LinkColumn(
                        "CoinGecko", # Header colonna
                        help="Link alla pagina CoinGecko",
                        display_text="🔗 Link", # Testo visualizzato
                        width="small" # Rendi colonna più stretta
                    )
                }
            )
        except Exception as df_err: logger.exception("Errore durante creazione o styling DataFrame:"); st.error(f"Errore visualizzazione tabella: {df_err}")
    else: logger.warning("Nessun risultato crypto valido da visualizzare."); st.warning("Nessun risultato crypto valido da visualizzare.")

    # --- EXPANDER ERRORI/NOTE ---
    fetch_errors_unique_display = sorted(list(set(fetch_errors_for_display)))
    if fetch_errors_unique_display:
        with st.expander("ℹ️ Note Recupero Dati / Calcolo Indicatori", expanded=True):
            st.warning("Si sono verificati problemi durante recupero/calcolo (controlla ID CoinGecko se vedi 'Not Found'):")
            max_errors_to_show = 30; error_list_md = ""
            for i, error_msg in enumerate(fetch_errors_unique_display):
                if i < max_errors_to_show: error_list_md += f"- {error_msg}\n"
                elif i == max_errors_to_show: error_list_md += f"- ... e altri {len(fetch_errors_unique_display) - max_errors_to_show} errori.\n"; break
            st.markdown(error_list_md)
    else:
         with st.expander("ℹ️ Note Recupero Dati / Calcolo Indicatori", expanded=False):
              st.success("Nessun problema rilevato durante recupero dati o calcolo indicatori.")

    # --- SEZIONE NEWS RIMOSSA ---

    # --- LEGENDA ---
    st.divider()
    with st.expander("📘 Legenda Indicatori Tecnici e Segnali", expanded=False):
        # Legenda aggiornata per riflettere AV attivo (ma limitato) e assenza RSI(1h)
        st.markdown("""
        *Disclaimer: Questa dashboard è solo a scopo informativo e non costituisce consulenza finanziaria.*
        **Market Overview:**
        * **Fear & Greed Index:** Indice sentiment (Fonte: Alternative.me).
        * **Total Crypto M.Cap:** Capitalizzazione totale crypto (Fonte: CoinGecko).
        * **Crypto ETFs Flow:** Flusso netto giornaliero ETF (Dato **N/A**).
        * **S&P 500 (SPY), etc.:** Prezzi mercati tradizionali (Fonte: Alpha Vantage). **Aggiornati con ritardo (cache 4h)** causa limiti API gratuite. Variazione giornaliera (Rosso/Verde).
        * **Titoli Principali:** Prezzi azioni (Fonte: Alpha Vantage). **Aggiornati con ritardo (cache 4h).** Variazione giornaliera (Rosso/Verde).
        **Tabella Analisi Tecnica:**
        * **Variazioni Percentuali (%):** Cambiamento prezzo (Fonte: CoinGecko).
        * **Indicatori Momentum:** RSI (1d, 1w, 1mo), SRSI, MACD Hist. (RSI 1h rimosso).
        * **Indicatori Trend:** MA (20d, 50d), VWAP (14d).
        * **Segnali Combinati (Esempio):** Gemini Alert, GPT Signal. **Cautela, non consulenza.**
        * **Generale:** N/A: Dato non disponibile/errore (Vedi Log sotto). **Link:** Link alla pagina CoinGecko.
        **Note:**
        * Recupero dati storici CoinGecko rallentato (pausa 4s). **Caricamento iniziale richiede minuti.**
        * Dati mercato tradizionale (Alpha Vantage) usano **cache 4h** (limiti API gratis). Richiede chiave API nei Secrets.
        * DYOR. Performance passate non garantiscono futuro.
        """)

    # --- Footer ---
    st.divider()
    st.caption("Disclaimer: Strumento a scopo informativo/didattico. Non costituisce consulenza finanziaria. DYOR.")

# --- Blocco try...except principale ---
except Exception as main_exception:
    logger.exception("!!! ERRORE NON GESTITO NELL'ESECUZIONE PRINCIPALE DELL'APP !!!")
    st.error("Si è verificato un errore imprevisto nell'applicazione. Controlla il log qui sotto per dettagli tecnici.")

# --- VISUALIZZAZIONE LOG APPLICAZIONE (Sempre alla fine) ---
st.divider()
st.subheader("📄 Log Applicazione")
st.caption("Questa sezione mostra i log generati durante l'esecuzione. Utile per il debug.")
log_content = log_stream.getvalue()
st.text_area("Log:", value=log_content, height=400, key="log_display_area", help="Puoi selezionare tutto (Ctrl+A o Cmd+A) e copiare (Ctrl+C o Cmd+C) questo log per condividerlo.")
logger.info("--- Fine esecuzione script app.py ---")
log_stream.close() # Chiude lo stream