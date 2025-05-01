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
    time.sleep(1.5) # Manteniamo delay alto
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
             volumes_df['timestamp'] = pd