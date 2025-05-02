# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time
import math
# import yfinance as yf # Non più usato per fetch dati tradizionali
import feedparser # Per News RSS
from alpha_vantage.timeseries import TimeSeries # Per Alpha Vantage

# Import zoneinfo for timezone handling if available (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except ImportError:
    st.warning("Modulo 'zoneinfo' non trovato. Usando fallback UTC+2 per Roma. Considera aggiornamento Python o aggiunta 'tzdata' a requirements.txt")
    ZoneInfo = None # Define as None to handle conditional logic later

# --- Layout App Streamlit ---
# Set page config first
st.set_page_config(layout="wide", page_title="Crypto Technical Dashboard Pro", page_icon="📈")

# --- INIZIO: Codice CSS per ridurre font st.metric ---
# Inject custom CSS with st.markdown to adjust metric value font size
st.markdown("""
<style>
/* Seleziona l'elemento che contiene il valore principale di st.metric */
div[data-testid="stMetricValue"] {
    font-size: 14px !important; /* Imposta dimensione font a 14px come richiesto */
}
/* Opzionale: Se vuoi rimpicciolire anche l'etichetta sopra il valore */
/*
div[data-testid="stMetricLabel"] > label {
    font-size: 12px !important; /* Esempio: Rimuovi /* e */ per attivare */
}
*/
</style>
""", unsafe_allow_html=True)
# --- FINE: Codice CSS ---


# --- Configurazione Globale ---

# Mappa Simbolo -> ID CoinGecko per facile gestione
# VERIFICARE QUESTI ID!
SYMBOL_TO_ID_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "RNDR": "render-token",
    "FET": "fetch-ai", # VERIFICARE ID!
    "RAY": "raydium", "SUI": "sui", "ONDO": "ondo-finance", # VERIFICARE ID ONDO!
    "ARB": "arbitrum", "TAO": "bittensor", "LINK": "chainlink",
    "AVAX": "avalanche-2", "HBAR": "hedera-hashgraph", "PEPE": "pepe",
    "UNI": "uniswap", "TIA": "celestia", "JUP": "jupiter-aggregator", # VERIFICARE ID!
    "IMX": "immutable-x", "TRUMP": "maga", # VERIFICARE ID!
    "NEAR": "near", # VERIFICARE ID!
    "AERO": "aerodrome-finance", "TRON": "tron", "AERGO": "aergo",
    "ADA": "cardano", "MKR": "maker"
}

# Deriva dinamicamente la lista di simboli e ID dalla mappa
SYMBOLS = list(SYMBOL_TO_ID_MAP.keys())
COINGECKO_IDS_LIST = list(SYMBOL_TO_ID_MAP.values())
NUM_COINS = len(SYMBOLS)

# Ticker per mercati tradizionali (ora per Alpha Vantage)
# Nota: Sostituiti indici/futures con ETF comuni o simboli AV noti.
# GLD per Oro, SPY per S&P 500, QQQ per Nasdaq 100. UVXY/TQQQ sono ETF.
TRAD_TICKERS_AV = ['SPY', 'QQQ', 'GLD', 'UVXY', 'TQQQ',
                   'NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR']

VS_CURRENCY = "usd" # Valuta di riferimento
CACHE_TTL = 1800 # Cache CoinGecko Live (30 min)
CACHE_HIST_TTL = CACHE_TTL * 2 # Cache CoinGecko Storico (60 min)
CACHE_TRAD_TTL = 14400 # Cache Alpha Vantage (4 ore = 14400 sec) <-- Cache lunga!

DAYS_HISTORY_DAILY = 365; DAYS_HISTORY_HOURLY = 7
RSI_PERIOD = 14; SRSI_PERIOD = 14; SRSI_K = 3; SRSI_D = 3
MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9
MA_SHORT = 20; MA_LONG = 50; VWAP_PERIOD = 14

# URL Feed RSS per notizie (Coingelegraph)
NEWS_FEED_URL = "https://cointelegraph.com/rss"
NUM_NEWS_ITEMS = 5 # Numero di notizie da mostrare

# --- Password Protection ---
def check_password():
    if "password_correct" not in st.session_state: st.session_state.password_correct = False
    if not st.session_state.password_correct:
        password = st.text_input("🔑 Password", type="password", key="password_input")
        # Legge la password dai secrets, se non c'è usa "Leonardo" come default (sconsigliato per produzione)
        correct_password = st.secrets.get("APP_PASSWORD", "Leonardo")
        if not correct_password:
             st.error("Password APP_PASSWORD non configurata nei secrets!")
             st.stop()
        if password == correct_password:
            st.session_state.password_correct = True
            if st.query_params.get("logged_in") != "true":
                st.query_params["logged_in"] = "true"
                st.rerun()
        elif password:
            st.warning("Password errata.")
            st.stop()
        else: st.stop()
    return True
if not check_password(): st.stop()

# --- Funzione Helper Formattazione Numeri Grandi ---
def format_large_number(num):
    if pd.isna(num) or not isinstance(num, (int, float)): return "N/A"
    if abs(num) < 1_000_000: return f"{num:,.0f}"
    elif abs(num) < 1_000_000_000: return f"{num / 1_000_000:.1f}M"
    elif abs(num) < 1_000_000_000_000: return f"{num / 1_000_000_000:.1f}B"
    else: return f"{num / 1_000_000_000_000:.2f}T"

# --- Funzioni API CoinGecko ---
@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento dati di mercato (CoinGecko)...")
def get_coingecko_market_data(ids_list, currency):
    ids_string = ",".join(ids_list)
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': currency, 'ids': ids_string, 'order': 'market_cap_desc',
        'per_page': str(len(ids_list)), 'page': 1, 'sparkline': False,
        'price_change_percentage': '1h,24h,7d,30d,1y', 'precision': 'full'
    }
    timestamp_utc = datetime.now(ZoneInfo("UTC") if ZoneInfo else None)
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not data:
             st.warning("API CoinGecko ha restituito dati vuoti per il mercato live.")
             return pd.DataFrame(), timestamp_utc
        df = pd.DataFrame(data)
        if not df.empty: df.set_index('id', inplace=True)
        return df, timestamp_utc
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 429:
             # Usa st.warning per non bloccare, ma segnalare
             st.warning("Attenzione API CoinGecko (Live): Limite richieste (429) raggiunto.")
        else: st.error(f"Errore HTTP API Mercato CoinGecko (Status: {http_err.response.status_code}): {http_err}")
        return pd.DataFrame(), timestamp_utc
    except requests.exceptions.RequestException as req_ex:
        st.error(f"Errore Richiesta API Mercato CoinGecko: {req_ex}")
        return pd.DataFrame(), timestamp_utc
    except Exception as e:
        st.error(f"Errore Processamento Dati Mercato CoinGecko: {e}")
        return pd.DataFrame(), timestamp_utc

@st.cache_data(ttl=CACHE_HIST_TTL, show_spinner=False) # Cache storico 60 min
def get_coingecko_historical_data(coin_id, currency, days, interval='daily'):
    # PAUSA AUMENTATA PER MITIGARE RATE LIMITING
    time.sleep(4.0) # <-- Pausa di 4 secondi!
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': currency, 'days': str(days),
        'interval': interval if interval == 'hourly' else 'daily', 'precision': 'full'
    }
    try:
        response = requests.get(url, params=params, timeout=25)
        response.raise_for_status()
        data = response.json()
        if not data or 'prices' not in data or not data['prices']:
             return pd.DataFrame(), f"No Prices Data ({coin_id})"
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms', utc=True)
        prices_df.set_index('timestamp', inplace=True)
        hist_df = prices_df
        if 'total_volumes' in data and data['total_volumes']:
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms', utc=True)
            volumes_df.set_index('timestamp', inplace=True)
            hist_df = prices_df.join(volumes_df, how='outer')
        else: hist_df['volume'] = 0.0
        hist_df = hist_df.interpolate(method='time').ffill().bfill()
        hist_df['high'] = hist_df['close']; hist_df['low'] = hist_df['close']
        hist_df['open'] = hist_df['close'].shift(1)
        hist_df['open'].fillna(hist_df['close'], inplace=True)
        hist_df = hist_df[~hist_df.index.duplicated(keep='last')].sort_index()
        hist_df.dropna(subset=['close'], inplace=True)
        if hist_df.empty: return pd.DataFrame(), f"Processed Empty ({coin_id})"
        return hist_df, "Success"
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        if status_code == 429: return pd.DataFrame(), f"Rate Limited (429) ({coin_id})"
        elif status_code == 404: return pd.DataFrame(), f"Not Found (404) ({coin_id})"
        else: return pd.DataFrame(), f"HTTP Error {status_code} ({coin_id})"
    except requests.exceptions.RequestException as req_ex:
        return pd.DataFrame(), f"Request Error ({req_ex}) ({coin_id})"
    except Exception as e:
        return pd.DataFrame(), f"Generic Error ({type(e).__name__}) ({coin_id})"

# --- Funzioni Dati Mercato Generale ---
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_fear_greed_index():
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status(); data = response.json()
        if data and data.get("data") and isinstance(data["data"], list) and len(data["data"]) > 0:
             latest_data = data["data"][0]; value = latest_data.get("value"); desc = latest_data.get("value_classification")
             if value is not None and desc is not None: return f"{int(value)} ({desc})"
        return "N/A"
    except requests.exceptions.RequestException as req_ex:
        status_code = req_ex.response.status_code if req_ex.response is not None else "N/A"; st.warning(f"Errore F&G Index (Alt.me Status: {status_code}): {req_ex}"); return "N/A"
    except Exception as e: st.warning(f"Errore Processamento F&G Index (Alt.me): {e}"); return "N/A"

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_global_market_data_cg(currency):
    url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status(); data = response.json().get('data', {})
        total_mcap = data.get('total_market_cap', {}).get(currency.lower(), np.nan)
        return total_mcap
    except requests.exceptions.RequestException as req_ex: st.warning(f"Errore API Global CoinGecko: {req_ex}"); return np.nan
    except Exception as e: st.warning(f"Errore Processamento Global CoinGecko: {e}"); return np.nan

def get_etf_flow(): return "N/A" # Placeholder

# --- NUOVA Funzione per Dati Tradizionali usando Alpha Vantage ---
@st.cache_data(ttl=CACHE_TRAD_TTL, show_spinner="Caricamento dati mercato tradizionale (Alpha Vantage)...") # Cache 4 ore
def get_traditional_market_data_av(tickers):
    """ Recupera l'ultimo prezzo per i ticker usando Alpha Vantage GLOBAL_QUOTE. """
    data = {ticker: np.nan for ticker in tickers} # Inizializza dizionario risultati

    # Leggi la chiave API dai secrets di Streamlit in modo sicuro
    api_key = None
    try:
        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
        if not api_key:
            st.error("Chiave API Alpha Vantage (ALPHA_VANTAGE_API_KEY) non trovata/vuota nei Secrets.")
            return data # Ritorna dati vuoti se chiave non impostata
    except KeyError:
         st.error("Secret 'ALPHA_VANTAGE_API_KEY' non definito nelle impostazioni dell'app.")
         return data
    except Exception as e:
         st.error(f"Errore imprevisto nel leggere la chiave API dai secrets: {e}")
         return data

    # Procedi solo se la chiave API è stata letta con successo
    if not api_key:
        return data # Dovrebbe essere già gestito sopra, ma per sicurezza

    ts = TimeSeries(key=api_key, output_format='pandas')

    # Limiti piano gratuito: ~5 chiamate/minuto, ~25/giorno (verificare!)
    calls_made = 0
    max_calls_per_minute = 5
    delay_between_calls = 60.0 / max_calls_per_minute + 1.0 # ~13 secondi

    # Cicla sui ticker richiesti
    for ticker_sym in tickers:
        # Gestione Limite Chiamate (molto basilare, potrebbe non essere perfetto)
        if calls_made >= 25: # Limite giornaliero approssimativo
            st.warning(f"Limite giornaliero Alpha Vantage (gratuito, ~25) probabilmente raggiunto. Stop recupero dati tradizionali.")
            break # Interrompi il ciclo se si sospetta limite raggiunto

        try:
            # Pausa per rispettare il limite al minuto
            # st.write(f"Attesa {delay_between_calls:.1f}s prima di chiamare {ticker_sym}...") # Debug
            time.sleep(delay_between_calls)

            # Ottieni i dati dall'endpoint GLOBAL_QUOTE
            # st.write(f"Chiamata Alpha Vantage per {ticker_sym}...") # Debug
            quote_data, meta_data = ts.get_quote_endpoint(symbol=ticker_sym)
            calls_made += 1
            # st.write(f"Risposta AV per {ticker_sym}: {quote_data}") # Debug

            # Estrai il prezzo più recente ('05. price')
            if not quote_data.empty and '05. price' in quote_data.columns:
                try:
                    # La risposta è solitamente una stringa, converti in float
                    price = float(quote_data['05. price'].iloc[0])
                    data[ticker_sym] = price
                except (ValueError, IndexError, TypeError) as conv_err:
                    st.warning(f"Formato prezzo non valido o errore conversione per {ticker_sym} da Alpha Vantage: {conv_err}")
                    data[ticker_sym] = np.nan
            else:
                # Messaggio più specifico se la risposta è vuota o manca la colonna prezzo
                if quote_data.empty:
                     st.warning(f"Risposta vuota da Alpha Vantage per {ticker_sym}. Simbolo non supportato o limite API?")
                else:
                     st.warning(f"Colonna '05. price' non trovata per {ticker_sym} da Alpha Vantage.")
                data[ticker_sym] = np.nan # Mantiene NaN se il prezzo non è trovato

        except ValueError as ve:
             # La libreria alpha_vantage solleva ValueError per molti errori API
             st.warning(f"Errore Alpha Vantage (ValueError) per {ticker_sym}: {ve}. Limite API o simbolo errato?")
             # Se l'errore indica limite API, potremmo interrompere il ciclo
             if "API call frequency" in str(ve) or "API key" in str(ve):
                 st.error(f"Errore chiave/limite API Alpha Vantage. Interruzione recupero dati tradizionali.")
                 break # Interrompi se l'errore è chiaramente legato a chiave/limiti
             data[ticker_sym] = np.nan
        except Exception as e:
            # Errore generico durante la chiamata API per un ticker
            st.warning(f"Errore generico recupero dati Alpha Vantage per {ticker_sym}: {type(e).__name__} - {e}")
            data[ticker_sym] = np.nan

    return data

# --- Funzione News RSS ---
@st.cache_data(ttl=900, show_spinner="Caricamento notizie...") # Cache 15 min
def get_crypto_news(feed_url, num_items=NUM_NEWS_ITEMS):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        feed = feedparser.parse(feed_url, request_headers=headers, timeout=15)
        if feed.bozo: exc = feed.get('bozo_exception', Exception('Unknown feedparser error')); st.warning(f"Possibile errore/warning nel parsing del feed RSS ({feed_url}): {exc}")
        if not feed.entries: st.info(f"Nessuna notizia trovata nel feed RSS: {feed_url}"); return []
        return feed.entries[:num_items]
    except Exception as e: st.error(f"Errore grave durante il recupero/parsing del feed RSS ({feed_url}): {e}"); return []

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

def compute_all_indicators(symbol, hist_daily_df, hist_hourly_df, fetch_errors_list):
    """Computes all technical indicators for a given symbol, adding errors to list."""
    indicators = {"RSI (1h)": np.nan, "RSI (1d)": np.nan, "RSI (1w)": np.nan, "RSI (1mo)": np.nan,"SRSI %K (1d)": np.nan, "SRSI %D (1d)": np.nan,"MACD Line (1d)": np.nan, "MACD Signal (1d)": np.nan, "MACD Hist (1d)": np.nan,f"MA({MA_SHORT}d)": np.nan, f"MA({MA_LONG}d)": np.nan,"VWAP (1d)": np.nan,}
    min_len_rsi_base = RSI_PERIOD + 1; min_len_srsi_base = RSI_PERIOD + SRSI_PERIOD + 5; min_len_macd_base = MACD_SLOW + MACD_SIGNAL + 5; min_len_vwap_base = VWAP_PERIOD + 1

    # --- Daily Indicators ---
    if not hist_daily_df.empty and 'close' in hist_daily_df.columns:
        close_daily = hist_daily_df['close'].dropna(); len_daily = len(close_daily)

        # Calculate indicators individually checking length requirements
        if len_daily >= min_len_rsi_base: indicators["RSI (1d)"] = calculate_rsi_manual(close_daily, RSI_PERIOD)
        else: fetch_errors_list.append(f"{symbol}: Dati insuff. ({len_daily}/{min_len_rsi_base}) per RSI(1d)")

        if len_daily >= min_len_srsi_base: indicators["SRSI %K (1d)"], indicators["SRSI %D (1d)"] = calculate_stoch_rsi(close_daily, RSI_PERIOD, SRSI_PERIOD, SRSI_K, SRSI_D)
        else: fetch_errors_list.append(f"{symbol}: Dati insuff. ({len_daily}/{min_len_srsi_base}) per SRSI(1d)")

        if len_daily >= min_len_macd_base:
             macd_l, macd_s, macd_h = calculate_macd_manual(close_daily, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
             indicators["MACD Line (1d)"] = macd_l; indicators["MACD Signal (1d)"] = macd_s; indicators["MACD Hist (1d)"] = macd_h
        else: fetch_errors_list.append(f"{symbol}: Dati insuff. ({len_daily}/{min_len_macd_base}) per MACD(1d)")

        if len_daily >= MA_SHORT: indicators[f"MA({MA_SHORT}d)"] = calculate_sma_manual(close_daily, MA_SHORT)
        else: fetch_errors_list.append(f"{symbol}: Dati insuff. ({len_daily}/{MA_SHORT}) per MA({MA_SHORT}d)")

        if len_daily >= MA_LONG: indicators[f"MA({MA_LONG}d)"] = calculate_sma_manual(close_daily, MA_LONG)
        else: fetch_errors_list.append(f"{symbol}: Dati insuff. ({len_daily}/{MA_LONG}) per MA({MA_LONG}d)")

        if len_daily >= min_len_vwap_base: indicators["VWAP (1d)"] = calculate_vwap_manual(hist_daily_df, VWAP_PERIOD)
        else: fetch_errors_list.append(f"{symbol}: Dati insuff. ({len_daily}/{min_len_vwap_base}) per VWAP(1d)")

        # --- Weekly & Monthly RSI from Daily Data ---
        if len_daily > min_len_rsi_base and pd.api.types.is_datetime64_any_dtype(close_daily.index):
            # Weekly RSI Calculation with Error Handling
            try:
                df_weekly = close_daily.resample('W-MON').last()
                if len(df_weekly.dropna()) >= min_len_rsi_base:
                    indicators["RSI (1w)"] = calculate_rsi_manual(df_weekly, RSI_PERIOD)
                else:
                     # Log insufficient data specifically for weekly
                     fetch_errors_list.append(f"{symbol}: Dati Weekly insuff. ({len(df_weekly.dropna())}/{min_len_rsi_base}) per RSI(1w)")
            except Exception as e: # Catch errors during weekly calc/resample
                 fetch_errors_list.append(f"{symbol}: Errore calcolo RSI weekly: {e}") # <-- FIX APPLIED HERE

            # Monthly RSI Calculation with Error Handling
            try:
                df_monthly = close_daily.resample('ME').last()
                if len(df_monthly.dropna()) >= min_len_rsi_base:
                    indicators["RSI (1mo)"] = calculate_rsi_manual(df_monthly, RSI_PERIOD)
                else:
                     # Log insufficient data specifically for monthly
                     fetch_errors_list.append(f"{symbol}: Dati Monthly insuff. ({len(df_monthly.dropna())}/{min_len_rsi_base}) per RSI(1mo)")
            except Exception as e: # Catch errors during monthly calc/resample
                fetch_errors_list.append(f"{symbol}: Errore calcolo RSI monthly: {e}")
        # else: # Optional: Log if daily data is too short for W/M RSI
             # if len_daily <= min_len_rsi_base:
             #    fetch_errors_list.append(f"{symbol}: Dati Daily insuff. per calcolare RSI Weekly/Monthly.")

    # --- Hourly RSI ---
    if not hist_hourly_df.empty and 'close' in hist_hourly_df.columns:
        close_hourly = hist_hourly_df['close'].dropna(); len_hourly = len(close_hourly)
        if len_hourly >= min_len_rsi_base: indicators["RSI (1h)"] = calculate_rsi_manual(close_hourly, RSI_PERIOD)
        else: fetch_errors_list.append(f"{symbol}: Dati Hourly insuff. ({len_hourly}/{min_len_rsi_base}) per RSI(1h)")

    return indicators

# --- Funzioni Segnale ---
def generate_gpt_signal(rsi_1d, rsi_1h, rsi_1w, macd_hist, ma_short, ma_long, srsi_k, srsi_d, current_price):
    # [Funzione invariata]
    required_inputs = [rsi_1d, macd_hist, ma_short, ma_long, current_price];
    if any(pd.isna(x) for x in required_inputs): return "⚪️ N/D"
    score = 0;
    if current_price > ma_long: score += 1; else: score -= 1
    if ma_short > ma_long: score += 2; else: score -= 2
    if macd_hist > 0: score += 2; else: score -= 2
    if rsi_1d < 30: score += 2; elif rsi_1d < 40: score += 1; elif rsi_1d > 70: score -= 2; elif rsi_1d > 60: score -= 1
    if pd.notna(rsi_1w):
        if rsi_1w < 40: score += 1; elif rsi_1w > 60: score -= 1
    if pd.notna(rsi_1h):
        if rsi_1h < 30: score += 1; elif rsi_1h > 70: score -= 1
    if pd.notna(srsi_k) and pd.notna(srsi_d):
        if srsi_k < 20 and srsi_d < 20: score += 1; elif srsi_k > 80 and srsi_d > 80: score -= 1
    if score >= 5: return "⚡️ Strong Buy"; elif score >= 2: return "🟢 Buy"; elif score <= -5: return "🚨 Strong Sell"; elif score <= -2: return "🔴 Sell"
    elif score > 0: return "⏳ CTB" if pd.notna(rsi_1d) and rsi_1d < 55 else "🟡 Hold"
    else: return "⚠️ CTS" if pd.notna(rsi_1d) and rsi_1d > 45 else "🟡 Hold"

def generate_gemini_alert(ma_short, ma_long, macd_hist, rsi_1d):
    # [Funzione invariata]
    if pd.isna(ma_short) or pd.isna(ma_long) or pd.isna(macd_hist) or pd.isna(rsi_1d): return "⚪️ N/D"
    is_uptrend_ma = ma_short > ma_long; is_momentum_positive = macd_hist > 0; is_not_extremely_overbought = rsi_1d < 80
    is_downtrend_ma = ma_short < ma_long; is_momentum_negative = macd_hist < 0; is_not_extremely_oversold = rsi_1d > 20
    if is_uptrend_ma and is_momentum_positive and is_not_extremely_overbought: return "⚡️ Strong Buy"
    elif is_downtrend_ma and is_momentum_negative and is_not_extremely_oversold: return "🚨 Strong Sell"
    else: return "🟡 Hold"

# --- TITOLO, BOTTONE, TIMESTAMP ---
col_title, col_button_placeholder, col_button = st.columns([4, 1, 1])
with col_title: st.title("📈 Crypto Technical Dashboard Pro")
with col_button:
    st.write("") # Spacer
    if st.button("🔄 Aggiorna", help=f"Forza aggiornamento dati", key="refresh_button"):
        st.cache_data.clear(); st.query_params.clear(); st.rerun()
last_update_placeholder = st.empty()
st.caption(f"Cache: Crypto Live ({CACHE_TTL/60:.0f}m), Crypto Storico ({CACHE_HIST_TTL/60:.0f}m), Tradizionale ({CACHE_TRAD_TTL/3600:.0f}h), Notizie (15m).")

# --- SEZIONE MARKET OVERVIEW ---
st.markdown("---")
st.subheader("🌐 Market Overview")
fear_greed_value = get_fear_greed_index()
total_market_cap = get_global_market_data_cg(VS_CURRENCY)
etf_flow_value = get_etf_flow()
# ** USA LA NUOVA FUNZIONE PER ALPHA VANTAGE **
traditional_market_data = get_traditional_market_data_av(TRAD_TICKERS_AV)
# Righe Market Overview con st.metric (font piccolo applicato via CSS)
mkt_col1, mkt_col2, mkt_col3, mkt_col4, mkt_col5 = st.columns(5)
with mkt_col1: st.metric(label="Fear & Greed Index", value=fear_greed_value, help="Fonte: Alternative.me")
with mkt_col2: st.metric(label=f"Total Crypto M.Cap ({VS_CURRENCY.upper()})", value=f"${format_large_number(total_market_cap)}", help="Fonte: CoinGecko")
with mkt_col3: st.metric(label="Crypto ETFs Flow (Daily)", value=etf_flow_value, help="Dato N/A")
with mkt_col4: st.metric(label="S&P 500 (SPY)", value=f"{traditional_market_data.get('SPY', np.nan):,.2f}" if pd.notna(traditional_market_data.get('SPY')) else "N/A")
with mkt_col5: st.metric(label="Nasdaq (QQQ)", value=f"{traditional_market_data.get('QQQ', np.nan):,.2f}" if pd.notna(traditional_market_data.get('QQQ')) else "N/A")
mkt_col6, mkt_col7, mkt_col8 = st.columns(3)
with mkt_col6: st.metric(label="Gold (GLD)", value=f"{traditional_market_data.get('GLD', np.nan):,.2f}" if pd.notna(traditional_market_data.get('GLD')) else "N/A", help="Usando ETF GLD")
with mkt_col7: st.metric(label="UVXY (Volatility)", value=f"{traditional_market_data.get('UVXY', np.nan):,.2f}" if pd.notna(traditional_market_data.get('UVXY')) else "N/A")
with mkt_col8: st.metric(label="TQQQ (Nasdaq 3x)", value=f"{traditional_market_data.get('TQQQ', np.nan):,.2f}" if pd.notna(traditional_market_data.get('TQQQ')) else "N/A")
st.markdown("<h6>Titoli Principali (Prezzi):</h6>", unsafe_allow_html=True)
stock_col1, stock_col2, stock_col3, stock_col4 = st.columns(4)
stock_tickers_row_av = ['NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR']
cols_stock = [stock_col1, stock_col2, stock_col3, stock_col4]
for idx, ticker in enumerate(stock_tickers_row_av):
    col_index = idx % 4; current_col = cols_stock[col_index]
    with current_col: price = traditional_market_data.get(ticker, np.nan); st.metric(label=ticker, value=f"{price:,.2f}" if pd.notna(price) else "N/A")
st.markdown("---")

# --- LOGICA PRINCIPALE DASHBOARD CRYPTO ---
st.subheader(f"📊 Analisi Tecnica Crypto ({NUM_COINS} Asset)")
market_data_df, last_cg_update_utc = get_coingecko_market_data(COINGECKO_IDS_LIST, VS_CURRENCY)
# Gestione Timestamp
if last_cg_update_utc and ZoneInfo:
    try: local_tz = ZoneInfo("Europe/Rome");
    if last_cg_update_utc.tzinfo is None: last_cg_update_utc = last_cg_update_utc.replace(tzinfo=ZoneInfo("UTC"))
    last_cg_update_local = last_cg_update_utc.astimezone(local_tz); last_update_placeholder.markdown(f"*Dati live CoinGecko aggiornati alle: **{last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S %Z')}***")
    except Exception as e: last_update_placeholder.markdown(f"*Errore conversione timestamp CoinGecko: {e}*")
elif last_cg_update_utc: last_cg_update_rome_approx = last_cg_update_utc + timedelta(hours=2); last_update_placeholder.markdown(f"*Dati live CoinGecko aggiornati alle: **{last_cg_update_rome_approx.strftime('%Y-%m-%d %H:%M:%S')} (Ora approx. Roma)***")
else: last_update_placeholder.markdown("*Timestamp aggiornamento dati live CoinGecko non disponibile.*")

# Blocco se dati live falliscono
if market_data_df.empty:
    st.error("Errore critico: Impossibile caricare dati live CoinGecko. La tabella non può essere generata.")
    st.stop()

# --- CICLO PROCESSING PER OGNI COIN ---
results = []; fetch_errors = []
# Aggiornato spinner per indicare potenziale lentezza
with st.spinner(f"Recupero dati storici e calcolo indicatori per {NUM_COINS} crypto... (Richiede tempo causa pause API CoinGecko)"):
    coin_ids_ordered = market_data_df.index.tolist()
    for i, coin_id in enumerate(coin_ids_ordered):
        # Stampa di Debug (rimuovere in produzione)
        # st.write(f"Processing {i+1}/{NUM_COINS}: {coin_id}")
        if coin_id not in market_data_df.index: fetch_errors.append(f"{coin_id}: Dati live non trovati."); continue
        live_data = market_data_df.loc[coin_id]; symbol = live_data.get('symbol', coin_id).upper(); name = live_data.get('name', coin_id)
        rank = live_data.get('market_cap_rank', 'N/A'); current_price = live_data.get('current_price', np.nan); volume_24h = live_data.get('total_volume', np.nan)
        change_1h = live_data.get('price_change_percentage_1h_in_currency', np.nan); change_24h = live_data.get('price_change_percentage_24h_in_currency', np.nan)
        change_7d = live_data.get('price_change_percentage_7d_in_currency', np.nan); change_30d = live_data.get('price_change_percentage_30d_in_currency', np.nan)
        change_1y = live_data.get('price_change_percentage_1y_in_currency', np.nan)
        # Fetch dati storici (ora con pausa maggiorata dentro la funzione)
        hist_daily_df, status_daily = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_DAILY, interval='daily')
        hist_hourly_df, status_hourly = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_HOURLY, interval='hourly')
        if status_daily != "Success": fetch_errors.append(f"{symbol}: Daily - {status_daily}")
        if status_hourly != "Success": fetch_errors.append(f"{symbol}: Hourly - {status_hourly}")
        # Calcola indicatori
        indicators = compute_all_indicators(symbol, hist_daily_df, hist_hourly_df, fetch_errors)
        # Genera segnali
        gpt_signal = generate_gpt_signal(indicators.get("RSI (1d)"), indicators.get("RSI (1h)"), indicators.get("RSI (1w)"), indicators.get("MACD Hist (1d)"), indicators.get(f"MA({MA_SHORT}d)"), indicators.get(f"MA({MA_LONG}d)"), indicators.get("SRSI %K (1d)"), indicators.get("SRSI %D (1d)"), current_price)
        gemini_alert = generate_gemini_alert(indicators.get(f"MA({MA_SHORT}d)"), indicators.get(f"MA({MA_LONG}d)"), indicators.get("MACD Hist (1d)"), indicators.get("RSI (1d)"))
        # Aggiungi risultati
        results.append({"Rank": rank, "Symbol": symbol, "Name": name,"Gemini Alert": gemini_alert, "GPT Signal": gpt_signal,f"Prezzo ({VS_CURRENCY.upper()})": current_price,"% 1h": change_1h, "% 24h": change_24h, "% 7d": change_7d, "% 30d": change_30d, "% 1y": change_1y,"RSI (1h)": indicators.get("RSI (1h)"), "RSI (1d)": indicators.get("RSI (1d)"),"RSI (1w)": indicators.get("RSI (1w)"), "RSI (1mo)": indicators.get("RSI (1mo)"),"SRSI %K (1d)": indicators.get("SRSI %K (1d)"), "SRSI %D (1d)": indicators.get("SRSI %D (1d)"),"MACD Hist (1d)": indicators.get("MACD Hist (1d)"),f"MA({MA_SHORT}d)": indicators.get(f"MA({MA_SHORT}d)"), f"MA({MA_LONG}d)": indicators.get(f"MA({MA_LONG}d)"),"VWAP (1d)": indicators.get("VWAP (1d)"),f"Volume 24h ({VS_CURRENCY.upper()})": volume_24h,})

# --- CREA E VISUALIZZA DATAFRAME ---
if results:
    df = pd.DataFrame(results)
    try: df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce'); df.set_index('Rank', inplace=True, drop=True); df.sort_index(inplace=True)
    except Exception as e: st.warning(f"Errore impostando/ordinando per Rank: {e}. Mostrando in ordine API.")
    cols_order = ["Symbol", "Name", "Gemini Alert", "GPT Signal",f"Prezzo ({VS_CURRENCY.upper()})","% 1h", "% 24h", "% 7d", "% 30d", "% 1y","RSI (1h)", "RSI (1d)", "RSI (1w)", "RSI (1mo)","SRSI %K (1d)", "SRSI %D (1d)","MACD Hist (1d)", f"MA({MA_SHORT}d)", f"MA({MA_LONG}d)", "VWAP (1d)",f"Volume 24h ({VS_CURRENCY.upper()})"]
    cols_to_show = [col for col in cols_order if col in df.columns]; df_display = df[cols_to_show].copy()
    formatters = {}; currency_col = f"Prezzo ({VS_CURRENCY.upper()})"; volume_col = f"Volume 24h ({VS_CURRENCY.upper()})"; pct_cols = ["% 1h", "% 24h", "% 7d", "% 30d", "% 1y"]
    rsi_srsi_cols = [col for col in df_display.columns if "RSI" in col or "SRSI" in col]; macd_cols = [col for col in df_display.columns if "MACD" in col]; ma_vwap_cols = [col for col in df_display.columns if "MA" in col or "VWAP" in col]
    formatters[currency_col] = "${:,.4f}";
    for col in pct_cols: formatters[col] = "{:+.2f}%";
    formatters[volume_col] = lambda x: f"${format_large_number(x)}";
    for col in rsi_srsi_cols: formatters[col] = "{:.1f}";
    for col in macd_cols: formatters[col] = "{:.4f}";
    for col in ma_vwap_cols: formatters[col] = "{:,.2f}";
    styled_df = df_display.style.format(formatters, na_rep="N/A", precision=4)
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
    for col in pct_cols:
        if col in df_display.columns: styled_df = styled_df.applymap(highlight_pct_col_style, subset=[col])
    if "Gemini Alert" in df_display.columns: styled_df = styled_df.applymap(highlight_signal_style, subset=["Gemini Alert"])
    if "GPT Signal" in df_display.columns: styled_df = styled_df.applymap(highlight_signal_style, subset=["GPT Signal"])
    st.dataframe(styled_df, use_container_width=True)
else: st.warning("Nessun risultato crypto valido da visualizzare dopo l'elaborazione.")

# --- EXPANDER ERRORI/NOTE ---
if fetch_errors:
    with st.expander("ℹ️ Note Recupero Dati / Calcolo Indicatori", expanded=True): # Default a Espanso se ci sono errori
        unique_errors = sorted(list(set(fetch_errors))); st.warning("Si sono verificati problemi durante recupero/calcolo (controlla ID CoinGecko se vedi 'Not Found'):")
        max_errors_to_show = 25; error_list_md = "" # Mostra più errori
        for i, error_msg in enumerate(unique_errors):
            if i < max_errors_to_show: error_list_md += f"- {error_msg}\n"
            elif i == max_errors_to_show: error_list_md += f"- ... e altri {len(unique_errors) - max_errors_to_show} errori.\n"; break
        st.markdown(error_list_md)

# --- SEZIONE NEWS ---
st.markdown("---"); st.subheader("📰 Ultime Notizie Crypto (Cointelegraph Feed)"); news_items = get_crypto_news(NEWS_FEED_URL)
if news_items:
    for item in news_items:
        title = item.get('title', 'Titolo non disponibile'); link = item.get('link', '#'); pub_date_str = ""
        if hasattr(item, 'published_parsed') and item.published_parsed:
            try: pub_dt_utc = datetime.fromtimestamp(time.mktime(item.published_parsed));
            if ZoneInfo: local_tz = ZoneInfo("Europe/Rome"); pub_dt_local = pub_dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(local_tz); pub_date_str = f" - *{pub_dt_local.strftime('%d %b, %H:%M %Z')}*"
            else: pub_dt_local_approx = pub_dt_utc + timedelta(hours=2); pub_date_str = f" - *{pub_dt_local_approx.strftime('%d %b, %H:%M')} (approx)*"
            except Exception: pass
        st.markdown(f"- [{title}]({link}){pub_date_str}")
else: st.warning("Impossibile caricare le notizie dal feed RSS o nessuna notizia trovata.")

# --- LEGENDA ---
st.divider()
with st.expander("📘 Legenda Indicatori Tecnici e Segnali", expanded=False): # Collassata di default
    st.markdown("""
    *Disclaimer: Questa dashboard è solo a scopo informativo e non costituisce consulenza finanziaria.*

    **Market Overview:**
    * **Fear & Greed Index:** Indice sentiment da Alternative.me (0=Paura Estrema, 100=Euforia Estrema). Misura il sentimento generale del mercato crypto.
    * **Total Crypto M.Cap:** Capitalizzazione totale mercato crypto (Fonte: CoinGecko). Valore approssimativo dell'intero mercato.
    * **Crypto ETFs Flow:** Flusso netto giornaliero ETF crypto spot (Dato **N/A**). Mostrerebbe l'interesse istituzionale tramite ETF.
    * **S&P 500 (SPY), Nasdaq (QQQ), Gold (GLD), UVXY, TQQQ:** Prezzi indicativi dei principali mercati tradizionali per contesto (Fonte: Alpha Vantage via ETF/Ticker comuni). **Aggiornati con ritardo (cache lunga 4h)** a causa dei limiti API gratuite.
    * **Titoli Principali:** Prezzi indicativi di azioni selezionate rilevanti per tech/mercato (Fonte: Alpha Vantage). **Aggiornati con ritardo (cache lunga 4h).**

    **Tabella Analisi Tecnica:**
    * **Variazioni Percentuali (%):** Cambiamento di prezzo rispetto a 1 ora, 24 ore, 7 giorni, 30 giorni, 1 anno (Fonte: CoinGecko).
    * **Indicatori Momentum (Misurano la velocità e la forza del movimento dei prezzi):**
        * **RSI (1h/1d/1w/1mo):** Relative Strength Index (0-100). Indica se un asset è potenzialmente ipercomprato (`>70`) o ipervenduto (`<30`). Le versioni 1w/1mo sono calcolate dai dati giornalieri e danno una visione a più lungo termine.
        * **SRSI %K / %D (1d):** Stochastic RSI (0-100). Versione più sensibile dell'RSI, utile per identificare cicli a breve termine. `>80` Ipercomprato, `<20` Ipervenduto. %K è la linea veloce, %D è la media mobile di %K.
        * **MACD Hist (1d):** Moving Average Convergence Divergence Histogram. Rappresenta la differenza tra la linea MACD (EMA veloce - EMA lenta) e la Signal Line (EMA della linea MACD). Barre sopra lo zero (`>0`) indicano momentum rialzista; barre sotto lo zero (`<0`) indicano momentum ribassista. L'altezza della barra indica la forza del momentum.
    * **Indicatori Trend (Aiutano a identificare la direzione generale del mercato):**
        * **MA (20d, 50d):** Simple Moving Average (Media Mobile Semplice) del prezzo di chiusura. Linee più lisce del prezzo, usate per identificare trend e potenziali livelli di supporto/resistenza. L'incrocio tra MA a breve (20d) e lungo termine (50d) può segnalare cambi di trend (es. Golden Cross: MA20 sopra MA50; Death Cross: MA20 sotto MA50).
        * **VWAP (1d):** Volume-Weighted Average Price (Prezzo Medio Ponderato per Volumi). Calcolato sugli ultimi 14 giorni di dati giornalieri. Considerato da alcuni trader come un indicatore del valore "equo" basato sui volumi scambiati.
    * **Segnali Combinati (Esemplificativi - NON CONSULENZA FINANZIARIA):**
        * **Gemini Alert:** Logica semplice basata su confluenza di segnali **Daily**: `⚡️ Strong Buy` (MA20>MA50 & MACD Hist>0 & RSI<80). `🚨 Strong Sell` (MA20<MA50 & MACD Hist<0 & RSI>20). `🟡 Hold` altrimenti. Utile per una rapida valutazione della condizione attuale del trend giornaliero.
        * **GPT Signal:** Logica più complessa che assegna un punteggio basato su molteplici indicatori (MAs, MACD, RSIs Daily/Weekly/Hourly, SRSI) per dare una valutazione generale. L'interpretazione è: `⚡️ Strong Buy` (Score >= 5), `🟢 Buy` (2-4), `⏳ CTB` (Close To Buy - Score > 0 & RSI<55), `🟡 Hold` (Neutro), `⚠️ CTS` (Close To Sell - Score < 0 & RSI>45), `🔴 Sell` (-4 to -2), `🚨 Strong Sell` (Score <= -5). **Da usare con estrema cautela**, è solo un esempio di combinazione.
    * **Generale:**
        * **N/A:** Dato non disponibile (es. da API), non applicabile, o errore nel calcolo (spesso per mancanza di dati storici sufficienti o limiti API).

    **Note:**
    * I calcoli degli indicatori Weekly/Monthly dipendono dalla disponibilità e qualità dei dati Daily.
    * Il recupero dati storici CoinGecko è rallentato (**pausa 4s**) per cercare di rispettare limiti API gratuiti. Il caricamento iniziale potrebbe richiedere alcuni minuti.
    * I dati mercato tradizionale (Alpha Vantage) usano cache lunga (**4h**) per rispettare limiti API gratuiti.
    * Le performance passate non sono indicative di risultati futuri.
    """)

# --- Footer ---
st.divider()
st.caption("Disclaimer: Strumento a scopo informativo/didattico. Non costituisce consulenza finanziaria. DYOR (Do Your Own Research).")