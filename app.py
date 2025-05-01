# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time
import math
import yfinance as yf

# --- Configurazione Globale ---
SYMBOL_TO_ID_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "RNDR": "render-token",
    "FET": "artificial-superintelligence-alliance",
    "TAO": "bittensor", "LINK": "chainlink", "AVAX": "avalanche-2",
    "HBAR": "hedera-hashgraph", "PEPE": "pepe", "UNI": "uniswap",
    "TIA": "celestia", "JUP": "jupiter-aggregator", "IMX": "immutable-x",
    "TRUMP": "maga", "NEAR": "near-protocol", "AERO": "aerodrome-finance",
    "TRON": "tron", "AERGO": "aergo", "ADA": "cardano", "MKR": "maker"
}
SYMBOLS = list(SYMBOL_TO_ID_MAP.keys())
COINGECKO_IDS_LIST = list(SYMBOL_TO_ID_MAP.values())
NUM_COINS = len(SYMBOLS)
TRAD_TICKERS = ['^GSPC', '^IXIC', 'GC=F', 'UVXY', 'TQQQ']
VS_CURRENCY = "usd"; CACHE_TTL = 1800 # 30 min
DAYS_HISTORY_DAILY = 365; DAYS_HISTORY_HOURLY = 7
RSI_PERIOD = 14; SRSI_PERIOD = 14; SRSI_K = 3; SRSI_D = 3
MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9
MA_SHORT = 20; MA_LONG = 50; VWAP_PERIOD = 14

# --- Password Protection ---
def check_password():
    if "password_correct" not in st.session_state: st.session_state.password_correct = False
    if not st.session_state.password_correct:
        password = st.text_input("🔑 Password", type="password", key="password_input")
        if password == "Leonardo":
             st.session_state.password_correct = True
             if st.query_params.get("rerun") != "false":
                  st.query_params["rerun"] = "false"; st.rerun()
        elif password: st.warning("Password errata."); st.stop()
        else: st.stop()
    return True
if not check_password(): st.stop()

# --- Funzioni API CoinGecko ---
@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento dati di mercato (CoinGecko)...")
def get_coingecko_market_data(ids_list, currency):
    ids_string = ",".join(ids_list)
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': currency, 'ids': ids_string, 'order': 'market_cap_desc',
              'per_page': str(len(ids_list)), 'page': 1, 'sparkline': False,
              'price_change_percentage': '1h,24h,7d,30d,1y'} # Aggiunto 30d, 1y
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
    time.sleep(1.5)
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

# --- Funzioni Dati Mercato Generale ---
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_fear_greed_index_cmc():
    url = "https://pro-api.coinmarketcap.com/v1/crypto/fear-and-greed" # Endpoint corretto?
    # Prova senza chiave API, potrebbe richiedere registrazione gratuita a CMC per una chiave
    # headers = {'X-CMC_PRO_API_KEY': 'TUA_CHIAVE_API_CMC'}
    try:
        response = requests.get(url, timeout=10) # senza headers
        response.raise_for_status(); data = response.json()
        if data and data.get("data") and isinstance(data["data"], list) and len(data["data"]) > 0:
             latest_data = data["data"][0]; value = latest_data.get("score"); desc = latest_data.get("rating")
             if value is not None and desc is not None: return f"{int(value)} ({desc})"
        return "N/A"
    except requests.exceptions.HTTPError as http_err:
         if http_err.response.status_code in [401, 403]: return "N/A (API Key?)"
         else: return f"N/A (Err {http_err.response.status_code})"
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

def get_etf_flow(): return "N/A" # Placeholder

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_traditional_market_data_yf(tickers):
    data = {}
    for ticker_sym in tickers:
        try:
            time.sleep(0.2); ticker_obj = yf.Ticker(ticker_sym); info = ticker_obj.info
            price = info.get('regularMarketPrice', info.get('currentPrice', info.get('previousClose', np.nan)))
            data[ticker_sym] = price
        except Exception: data[ticker_sym] = np.nan
    return data

# --- Funzioni Calcolo Indicatori (Manuali) ---
def calculate_rsi_manual(series, period=RSI_PERIOD):
    # ... (codice invariato) ...
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

def calculate_stoch_rsi(series, rsi_period=RSI_PERIOD, stoch_period=SRSI_PERIOD, k_smooth=SRSI_K, d_smooth=SRSI_D):
    # ... (codice invariato) ...
    rsi_series = pd.Series(dtype=float)
    if isinstance(series, pd.Series) and not series.isna().all():
        delta = series.diff(); gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm