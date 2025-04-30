# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time # Per throttling richieste
import math

# --- Configurazione Globale ---
# Lista Simboli Aggiornata
SYMBOLS = ["BTC", "ETH", "BNB", "SOL", "XRP", "RNDR", "FET", "RAY", "SUI", "ONDO", "ARB"]
NUM_COINS = len(SYMBOLS)
VS_CURRENCY = "usd" # Valuta di riferimento
CACHE_TTL = 300 # Cache di 5 minuti (300 sec) per dati API
DAYS_HISTORY_DAILY = 365 # Giorni di storico per indicatori daily/weekly
DAYS_HISTORY_HOURLY = 7 # Giorni di storico per indicatori hourly (limite comune API free)

# Mappatura Simbolo -> ID CoinGecko Aggiornata
SYMBOL_TO_ID_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "RNDR": "render-token",
    "FET": "artificial-superintelligence-alliance", # ID post-merge ASI Alliance
    "RAY": "raydium", "SUI": "sui", "ONDO": "ondo", "ARB": "arbitrum"
}
ID_TO_SYMBOL_MAP = {v: k for k, v in SYMBOL_TO_ID_MAP.items()}
COINGECKO_IDS_LIST = [SYMBOL_TO_ID_MAP[s] for s in SYMBOLS if s in SYMBOL_TO_ID_MAP]

# Periodi Indicatori
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MA_SHORT = 20
MA_LONG = 50
VWAP_PERIOD = 14

# --- Funzioni API CoinGecko ---

@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento dati di mercato (CoinGecko)...")
def get_coingecko_market_data(ids_list, currency):
    """Ottiene dati di mercato completi (rank, prezzo, vol, cap...) da CoinGecko."""
    ids_string = ",".join(ids_list)
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': currency,
        'ids': ids_string,
        'order': 'market_cap_desc', # Anche se filtriamo per ID, questo assicura un ordine base
        'per_page': str(len(ids_list)), # Richiedi esattamente quanti ID passiamo
        'page': 1,
        'sparkline': False,
        'price_change_percentage': '1h,24h,7d' # Richiedi variazioni % necessarie
    }
    timestamp = datetime.now()
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        # Converti lista di dizionari in DataFrame indicizzato per ID per accesso facile
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('id', inplace=True)
        return df, timestamp
    except requests.exceptions.RequestException as e:
        st.error(f"Errore API Mercato CoinGecko: {e}")
        return pd.DataFrame(), timestamp
    except Exception as e:
        st.error(f"Errore Processamento Dati Mercato CoinGecko: {e}")
        return pd.DataFrame(), timestamp

@st.cache_data(ttl=CACHE_TTL * 2, show_spinner=False) # Cache più lunga per storico
def get_coingecko_historical_data(coin_id, currency, days, interval='daily'):
    """Ottiene dati storici OHLCV da CoinGecko /market_chart."""
    # Aggiungi piccolo delay per rispettare il rate limit (cruciale con chiamate multiple)
    time.sleep(0.5) # Aumentato leggermente il delay
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': currency,
        'days': str(days),
        'interval': interval if interval == 'hourly' else 'daily', # API supporta solo 'daily' o inferisce per <90gg
        'precision': 'full'
    }
    try:
        response = requests.get(url, params=params, timeout=20) # Timeout più lungo per storico
        response.raise_for_status()
        data = response.json()
        if not data or 'prices' not in data or not data['prices']: return pd.DataFrame()

        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
        prices_df.set_index('timestamp', inplace=True)

        hist_df = prices_df
        if 'total_volumes' in data and data['total_volumes']:
             volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
             volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
             volumes_df.set_index('timestamp', inplace=True)
             # Usa 'outer' join e poi interpola/ffill per gestire timestamp disallineati
             hist_df = prices_df.join(volumes_df, how='outer').interpolate(method='time').ffill().bfill()
        else: hist_df['volume'] = 0

        # Aggiungi H/L/O approssimati se servissero (VWAP manuale non li usa)
        hist_df['high'] = hist_df['close'] # Stima
        hist_df['low'] = hist_df['close']  # Stima
        hist_df['open'] = hist_df['close'].shift(1) # Stima

        hist_df = hist_df[~hist_df.index.duplicated(keep='last')]
        hist_df.dropna(subset=['close'], inplace=True)
        return hist_df

    except Exception as e: return pd.DataFrame() # Fallback silenzioso per ora

# --- Funzioni Calcolo Indicatori (Manuali con Pandas) ---

def calculate_rsi_manual(series, period=RSI_PERIOD):
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
    rs = last_avg_gain / last_avg_loss; rsi =