import pandas as pd
import ta
import streamlit as st
import yfinance as yf
import requests
import json
import traceback # Manteniamo traceback per debug

# --- Costanti per parametri e soglie ---
# (Invariate)
RSI_WINDOW = 14
MA_WINDOW = 14
SRSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGN = 9
STOCH_WINDOW = 14
EMA_SHORT = 3
EMA_LONG = 30
MFI_WINDOW = 14

RSI_LOW = 30
RSI_HIGH = 70
SRSI_LOW = 0.2
SRSI_HIGH = 0.8
MACD_THRESHOLD_LOW = -0.1
MACD_THRESHOLD_HIGH = 0.1
STOCH_LOW = 20
STOCH_HIGH = 80
MFI_LOW = 20
MFI_HIGH = 80

SIGNAL_STRONG_BUY_THRESHOLD = 4
SIGNAL_BUY_THRESHOLD = 2
SIGNAL_STRONG_SELL_THRESHOLD = -4
SIGNAL_SELL_THRESHOLD = -2
# --- Fine Costanti ---

# (Funzione get_top_100_crypto_symbols invariata rispetto all'ultima versione)
@st.cache_data(ttl=600, show_spinner="Fetching crypto list...")
def get_top_100_crypto_symbols():
    """Fetches top 100 crypto symbols by market cap from CoinGecko, filtering problematic names."""
    fallback_list = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD', 'ADA-USD', 'TRX-USD', 'DOT-USD', 'MATIC-USD']
    try:
        url = 'https://api.coingecko.com/api/v3/coins/markets'
        params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 100, 'page': 1}
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        tickers = []
        for coin in data:
            if 'symbol' in coin:
                symbol_upper = coin['symbol'].upper()
                ticker = f"{symbol_upper}-USD"
                if not symbol_upper or len(symbol_upper) < 2 or "-USD" in symbol_upper or " " in ticker or ticker.count('-') > 1:
                    continue
                tickers.append(ticker)
        if not tickers:
             st.warning("Nessun ticker valido ricevuto o filtrato da CoinGecko API.")
             return fallback_list
        return tickers
    except requests.exceptions.Timeout:
        st.error("Errore API CoinGecko: Timeout durante la richiesta.")
        return fallback_list
    except requests.exceptions.HTTPError as e:
        st.error(f"Errore HTTP API CoinGecko: {e}")
        return fallback_list
    except requests.exceptions.RequestException as e:
        st.error(f"Errore generico API CoinGecko: {e}")
        return fallback_list
    except (ValueError, json.JSONDecodeError) as e:
         st.error(f"Errore parsing risposta CoinGecko: {e}")
         return fallback_list
    except Exception as e:
        st.error(f"Errore imprevisto in get_top_100_crypto_symbols: {e}\n{traceback.format_exc()}")
        return fallback_list


@st.cache_data(ttl=300, show_spinner="Calculating indicators...")
def fetch_indicators_with_signals(symbol, interval, period):
    """Fetches data using yfinance and calculates technical indicators. Returns dataframe and signals dict."""
    signals = {
        'RSI': '⚪ N/A', 'SRSI': '⚪ N/A', 'MACD': '⚪ N/A', 'MA': '⚪ N/A',
        'Doda Stoch': '⚪ N/A', 'GChannel': '⚪ N/A', 'Vol Flow': '⚪ N/A', 'VWAP': '⚪ N/A'
    }
    df = pd.DataFrame()
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume'] # Colonne essenziali

    try:
        request_period = period
        if interval == '1wk': request_period = '8mo' if period == '6mo' else period
        if interval == '1mo': request_period = '26mo' if period == '2y' else period

        df = yf.download(tickers=symbol, interval=interval, period=request_period, progress=False, ignore_tz=True)

        # --- NUOVO CONTROLLO COLONNE ESSENZIALI ---
        if df.empty or not all(col in df.columns for col in required_columns):
            # Se il df è vuoto o mancano colonne OHLCV, non possiamo procedere
            # st.warning(f"Dati yfinance mancanti o incompleti per {symbol} ({interval}, {period}). Colonne: {df.columns if not df.empty else 'Empty DF'}")
            return df, signals # Restituisce df vuoto/invalido e segnali N/A
        # --- FINE CONTROLLO ---

        # Rimuovi righe con NaN in colonne essenziali (ora sappiamo che esistono)
        df.dropna(subset=required_columns, inplace=True)

        # Controlla lunghezza minima DOPO dropna
        min_required_len = max(RSI_WINDOW, MA_WINDOW, SRSI_WINDOW, MACD_SLOW, STOCH_WINDOW, MFI_WINDOW, EMA_LONG, 2) + 5
        if df.empty or len(df) < min_required_len:
            return df, signals

        # Assicurati che l'indice sia un DatetimeIndex e gestiscilo
        if not isinstance(df.index, pd.DatetimeIndex):
           df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]

        # Conversione a numerico e gestione NaN (ora le colonne esistono)
        df['high'] = pd.to_numeric(df['High'], errors='coerce')
        df['low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df.dropna(subset=['high', 'low', 'close', 'volume'], inplace=True)
        if df.empty or len(df) < min_required_len:
             return df, signals

        # (Funzione Helper _safe_calculate invariata)
        def _safe_calculate(indicator_func, *args, **kwargs):
            try:
                indicator_series = indicator_func(*args, **kwargs)
                if indicator_series is None or not isinstance(indicator_series, pd.Series) or indicator_series.empty:
                    return float('nan')
                indicator_series = indicator_series.replace([float('inf'), -float('inf')], float('nan')).dropna