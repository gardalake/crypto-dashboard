# Versione: v17.2 - Password hardcoded (non da secrets)
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time
# import math # Non sembra utilizzato
# import yfinance as yf
# import feedparser
from alpha_vantage.timeseries import TimeSeries # Per Alpha Vantage
import logging
import io

# --- INIZIO: Configurazione Logging in UI ---
log_stream = io.StringIO()
logging.basicConfig(
    stream=log_stream,
    level=logging.DEBUG, # Livello DEBUG
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)
logger.info("Logging configurato per UI (Livello DEBUG).")
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
# MAPPA FINALE CON 15 COIN FORNITE DALL'UTENTE
SYMBOL_TO_ID_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "RNDR": "render-token",
    "RAY": "raydium",
    "SUI": "sui",
    "ONDO": "ondo-finance",
    "ARB": "arbitrum",
    "TAO": "bittensor",
    "LINK": "chainlink",
    "HBAR": "hedera-hashgraph",
    "IMX": "immutable-x",
    "TRUMP": "official-trump",
    "AERO": "aerodrome-finance",
    "MKR": "maker",
}

SYMBOLS = list(SYMBOL_TO_ID_MAP.keys())
COINGECKO_IDS_LIST = list(SYMBOL_TO_ID_MAP.values())
NUM_COINS = len(SYMBOLS)
logger.info(f"Numero coins configurate: {NUM_COINS}") # Ora 15
TRAD_TICKERS_AV = [
    'SPY', 'QQQ', 'GLD', 'SLV', 'UNG', 'UVXY', 'TQQQ', 'NVDA', 'GOOGL', 'AAPL',
    'META', 'TSLA', 'MSFT', 'TSM', 'PLTR', 'COIN', 'MSTR'
]
logger.info(f"Tickers tradizionali configurati (Alpha Vantage): {TRAD_TICKERS_AV}")
VS_CURRENCY = "usd"
CACHE_TTL = 1800  # 30 min
CACHE_HIST_TTL = CACHE_TTL * 2 # 60 min
CACHE_TRAD_TTL = 14400 # 4h (per Alpha Vantage)
DAYS_HISTORY_DAILY = 365
DAYS_HISTORY_HOURLY = 7 # Mantenuto anche se non usato per indicatori, potenzialmente utile in futuro
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
logger.info("Fine configurazione globale.")

# --- DEFINIZIONI FUNZIONI ---

def check_password():
    """Verifica la password inserita dall'utente (password hardcoded)."""
    logger.debug("Esecuzione check_password.")
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        pwd_col, btn_col = st.columns([3, 1])
        with pwd_col:
            password = st.text_input("🔑 Password", type="password", key="password_input_field")
        with btn_col:
            # Allinea verticalmente il bottone con il campo password
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            login_button_pressed = st.button("Accedi", key="login_button")

        # Controlla solo se il bottone è premuto o se c'è testo nel campo password (per invio con Enter)
        should_check = login_button_pressed or (password and password != "") # Modificato per checkare solo se c'è input

        if not should_check:
            logger.debug("In attesa di input password o click bottone.")
            st.stop() # Ferma l'esecuzione se non c'è né bottone premuto né input
        else:
            # --- MODIFICA APPLICATA QUI ---
            # Imposta la password direttamente nel codice invece che leggerla dai secrets
            # !!! ATTENZIONE: Questa password sarà visibile nel codice sorgente su GitHub !!!
            # !!! SOSTITUISCI "Leonardo" CON LA TUA PASSWORD DESIDERATA !!!
            correct_password = "Leonardo"

            # Il controllo 'if not correct_password:' è stato rimosso perché non più necessario

            if password == correct_password:
                logger.info("Password corretta.")
                st.session_state.password_correct = True
                # Pulisci i query params se presenti per evitare loop o stati indesiderati
                if st.query_params.get("logged_in") != "true":
                     # Aggiorna il query param per indicare il login (opzionale, ma può essere utile)
                    st.query_params["logged_in"] = "true"
                    st.rerun() # Ricarica l'app per mostrare il contenuto protetto
            else:
                logger.warning("Password errata inserita.")
                st.warning("Password errata.")
                st.stop() # Ferma l'esecuzione se la password è errata

    logger.debug("Check password superato.")
    return True

def format_large_number(num):
    """Formatta numeri grandi in formato leggibile (M, B, T)."""
    if pd.isna(num) or not isinstance(num, (int, float)):
        return "N/A"
    num_abs = abs(num)
    sign = "-" if num < 0 else ""
    if num_abs < 1_000_000:
        return f"{sign}{num_abs:,.0f}"
    elif num_abs < 1_000_000_000:
        return f"{sign}{num_abs / 1_000_000:.1f}M"
    elif num_abs < 1_000_000_000_000:
        return f"{sign}{num_abs / 1_000_000_000:.1f}B"
    else:
        return f"{sign}{num_abs / 1_000_000_000_000:.2f}T"

@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento dati di mercato (CoinGecko)...")
def get_coingecko_market_data(ids_list, currency):
    """Recupera i dati di mercato live da CoinGecko."""
    logger.info(f"Tentativo fetch dati live CoinGecko per {len(ids_list)} IDs.")
    ids_string = ",".join(ids_list)
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': currency,
        'ids': ids_string,
        'order': 'market_cap_desc',
        'per_page': str(len(ids_list)),
        'page': 1,
        'sparkline': False,
        'price_change_percentage': '1h,24h,7d,30d,1y',
        'precision': 'full'
    }
    timestamp_utc = datetime.now(ZoneInfo("UTC") if ZoneInfo else None)
    # Inizializza lo stato se non esiste
    if 'api_warning_shown' not in st.session_state:
        st.session_state.api_warning_shown = False

    try:
        logger.debug(f"Requesting URL: {url} with params: {params}")
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status() # Solleva eccezione per status code 4xx/5xx
        data = response.json()

        if not data:
            logger.warning("API CoinGecko live: Dati vuoti ricevuti.")
            st.warning("API CoinGecko live: Dati vuoti ricevuti.")
            return pd.DataFrame(), timestamp_utc

        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('id', inplace=True)

        st.session_state["api_warning_shown"] = False # Reset warning se la chiamata ha successo
        logger.info(f"Dati live CoinGecko recuperati per {len(df)} coins.")
        return df, timestamp_utc

    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        logger.warning(f"Errore HTTP API Mercato CoinGecko (Status: {status_code}): {http_err}")
        if status_code == 429:
            # Mostra warning solo se non già mostrato per evitare ripetizioni
            if not st.session_state.get("api_warning_shown", False):
                st.warning("Attenzione API CoinGecko (Live): Limite richieste (429) raggiunto. I dati potrebbero non essere aggiornati.")
                st.session_state["api_warning_shown"] = True
        else:
            st.error(f"Errore HTTP API Mercato CoinGecko (Status: {status_code}): {http_err}")
        return pd.DataFrame(), timestamp_utc # Ritorna DF vuoto in caso di errore
    except requests.exceptions.RequestException as req_ex:
        logger.error(f"Errore Richiesta API Mercato CoinGecko: {req_ex}")
        st.error(f"Errore Richiesta API Mercato CoinGecko: {req_ex}")
        return pd.DataFrame(), timestamp_utc
    except Exception as e:
        logger.exception("Errore Processamento Dati Mercato CoinGecko:")
        st.error(f"Errore Processamento Dati Mercato CoinGecko: {e}")
        return pd.DataFrame(), timestamp_utc

@st.cache_data(ttl=CACHE_HIST_TTL, show_spinner=False)
def get_coingecko_historical_data(coin_id, currency, days, interval='daily'):
    """Recupera i dati storici da CoinGecko con pausa per rate limiting."""
    # Pausa di 6 secondi per rispettare i limiti API di CoinGecko
    logger.debug(f"Inizio fetch storico per {coin_id} ({interval}), pausa 6s...")
    time.sleep(6.0)
    logger.debug(f"Fine pausa per {coin_id} ({interval}), inizio chiamata API.")

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': currency,
        'days': str(days),
        'interval': interval if interval == 'hourly' else 'daily', # Usa 'hourly' solo se specificato
        'precision': 'full'
    }
    status_msg = f"Errore Sconosciuto ({coin_id}, {interval})" # Messaggio di default

    try:
        logger.debug(f"Requesting URL: {url} with params: {params}")
        response = requests.get(url, params=params, timeout=25)
        response.raise_for_status()
        data = response.json()

        # Controlla se 'prices' esiste e non è vuoto
        if not data or 'prices' not in data or not data['prices']:
            status_msg = f"No Prices Data ({coin_id}, {interval})"
            logger.warning(status_msg)
            return pd.DataFrame(), status_msg

        # Crea DataFrame prezzi
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms', utc=True)
        prices_df.set_index('timestamp', inplace=True)
        hist_df = prices_df # Inizia con i prezzi

        # Aggiungi volumi se disponibili
        if 'total_volumes' in data and data['total_volumes']:
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms', utc=True)
            volumes_df.set_index('timestamp', inplace=True)
            # Unisci prezzi e volumi
            hist_df = prices_df.join(volumes_df, how='outer')
        else:
            # Se i volumi non sono presenti, aggiungi colonna con 0.0
            hist_df['volume'] = 0.0

        # Gestione Dati Mancanti e Creazione OHLC approssimato
        hist_df = hist_df.interpolate(method='time').ffill().bfill() # Interpola e riempi NaN
        hist_df['high'] = hist_df['close'] # Approssima High/Low/Open con Close
        hist_df['low'] = hist_df['close']
        hist_df['open'] = hist_df['close'].shift(1)
        # Usa .loc per evitare SettingWithCopyWarning e riempi il primo Open
        hist_df.loc[hist_df.index[0], 'open'] = hist_df['close'].iloc[0]

        # Pulizia finale
        hist_df = hist_df[~hist_df.index.duplicated(keep='last')].sort_index() # Rimuovi duplicati e ordina
        hist_df.dropna(subset=['close'], inplace=True) # Rimuovi righe senza prezzo

        if hist_df.empty:
            status_msg = f"Processed Empty ({coin_id}, {interval})"
            logger.warning(status_msg)
            return pd.DataFrame(), status_msg

        status_msg = "Success"
        logger.info(f"Dati storici CoinGecko per {coin_id} ({interval}) recuperati, {len(hist_df)} righe.")
        return hist_df, status_msg

    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        if status_code == 429:
            status_msg = f"Rate Limited (429) ({coin_id}, {interval})"
        elif status_code == 404:
            status_msg = f"Not Found (404) ({coin_id}, {interval})" # ID errato!
        else:
            status_msg = f"HTTP Error {status_code} ({coin_id}, {interval})"
        logger.warning(f"Errore HTTP API Storico CoinGecko: {status_msg}")
        return pd.DataFrame(), status_msg
    except requests.exceptions.RequestException as req_ex:
        status_msg = f"Request Error ({req_ex}) ({coin_id}, {interval})"
        logger.error(f"Errore Richiesta API Storico CoinGecko: {status_msg}")
        return pd.DataFrame(), status_msg
    except Exception as e:
        status_msg = f"Generic Error ({type(e).__name__}) ({coin_id}, {interval})"
        logger.exception(f"Errore Processamento API Storico CoinGecko per {coin_id} ({interval}):")
        return pd.DataFrame(), status_msg

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_fear_greed_index():
    """Recupera l'indice Fear & Greed da Alternative.me."""
    logger.info("Tentativo fetch Fear & Greed Index.")
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Estrazione sicura dei dati
        if data and isinstance(data.get("data"), list) and len(data["data"]) > 0:
             latest_data = data["data"][0]
             value = latest_data.get("value")
             desc = latest_data.get("value_classification")
             if value is not None and desc is not None:
                 logger.info(f"F&G Index: {value} ({desc}).")
                 return f"{int(value)} ({desc})" # Ritorna stringa formattata
        logger.warning("Formato dati F&G Index inatteso ricevuto da API.")
        return "N/A"
    except requests.exceptions.RequestException as req_ex:
        status_code = req_ex.response.status_code if req_ex.response is not None else "N/A"
        msg = f"Errore API F&G Index (Alternative.me Status: {status_code}): {req_ex}"
        logger.warning(msg)
        st.warning(msg) # Mostra warning in UI per errori API F&G
        return "N/A"
    except Exception as e:
        msg = f"Errore Processamento F&G Index (Alternative.me): {e}"
        logger.exception(msg)
        st.warning(msg) # Mostra warning in UI per errori generici F&G
        return "N/A"

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_global_market_data_cg(currency):
    """Recupera i dati globali (es. Total Market Cap) da CoinGecko."""
    logger.info("Tentativo fetch dati Global CoinGecko.")
    url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get('data', {})
        # Estrazione sicura del market cap per la valuta specificata
        total_mcap = data.get('total_market_cap', {}).get(currency.lower(), np.nan)
        if pd.notna(total_mcap):
             logger.info(f"Global CoinGecko M.Cap ({currency.upper()}): {total_mcap}.")
        else:
             logger.warning(f"Global CoinGecko M.Cap ({currency.upper()}) non trovato nella risposta.")
        return total_mcap
    except requests.exceptions.RequestException as req_ex:
        msg = f"Errore API Global CoinGecko: {req_ex}"
        logger.warning(msg)
        st.warning(msg) # Mostra warning in UI
        return np.nan
    except Exception as e:
        msg = f"Errore Processamento Global CoinGecko: {e}"
        logger.exception(msg)
        st.warning(msg) # Mostra warning in UI
        return np.nan

def get_etf_flow():
    """Placeholder per la funzione di recupero dati ETF Flow."""
    logger.debug("get_etf_flow chiamato (placeholder).")
    return "N/A" # Attualmente non implementato

@st.cache_data(ttl=CACHE_TRAD_TTL, show_spinner="Caricamento dati mercato tradizionale (Alpha Vantage)...")
def get_traditional_market_data_av(tickers):
    """Recupera dati quote da Alpha Vantage per i ticker tradizionali."""
    logger.info(f"Tentativo fetch dati Alpha Vantage per {len(tickers)} tickers.")
    data = {ticker: {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'} for ticker in tickers}
    api_key = None

    # Recupero sicuro API Key dai secrets (MANTENUTO ANCHE SE NON USATO PER PASSWORD APP)
    # È buona pratica mantenere la chiave AV nei secrets
    try:
        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
        logger.info("Chiave API Alpha Vantage letta dai secrets.")
    except KeyError:
        logger.error("Secret 'ALPHA_VANTAGE_API_KEY' non definito nei secrets di Streamlit.")
        st.error("Errore Configurazione: Chiave API Alpha Vantage non trovata nei secrets. I dati dei mercati tradizionali non saranno disponibili.")
        return data # Ritorna dati vuoti se la chiave non è configurata
    except Exception as e:
        logger.exception("Errore imprevisto durante lettura secrets Alpha Vantage:")
        st.error(f"Errore lettura secrets Alpha Vantage: {e}")
        return data

    if not api_key:
        logger.error("Chiave API Alpha Vantage trovata ma è vuota.")
        st.error("Errore Configurazione: Chiave API Alpha Vantage vuota nei Secrets. I dati dei mercati tradizionali non saranno disponibili.")
        return data

    ts = TimeSeries(key=api_key, output_format='pandas')
    calls_made = 0
    # Limiti API free tier AV: 5 chiamate/minuto, ~25 (o più) chiamate/giorno.
    max_calls_per_minute = 5
    max_calls_this_run = 25 # Limite prudenziale per una singola esecuzione
    delay_between_calls = (60.0 / max_calls_per_minute) + 1.0 # Aggiunge 1s di margine

    for ticker_sym in tickers:
        # Controllo limite chiamate per questa esecuzione
        if calls_made >= max_calls_this_run:
            msg = f"Limite chiamate AV per questa esecuzione ({max_calls_this_run}) raggiunto. Stop fetch per {ticker_sym} e successivi."
            logger.warning(msg)
            st.warning(msg) # Informa l'utente in UI
            break # Interrompe il ciclo per non superare il limite

        try:
            # Log e Pausa prima della chiamata
            logger.info(f"Fetch AV per {ticker_sym} (Call #{calls_made+1}/{max_calls_this_run}, Pausa {delay_between_calls:.1f}s)...")
            time.sleep(delay_between_calls)

            # Chiamata API
            quote_data, meta_data = ts.get_quote_endpoint(symbol=ticker_sym)
            calls_made += 1
            logger.debug(f"Risposta AV per {ticker_sym}: Head:\n{quote_data.head()}")

            # Processa i dati se la risposta non è vuota
            if not quote_data.empty:
                # Estrazione sicura dei valori dal DataFrame della risposta
                try: data[ticker_sym]['price'] = float(quote_data['05. price'].iloc[0])
                except (KeyError, ValueError, IndexError, TypeError): data[ticker_sym]['price'] = np.nan
                try: data[ticker_sym]['change'] = float(quote_data['09. change'].iloc[0])
                except (KeyError, ValueError, IndexError, TypeError): data[ticker_sym]['change'] = np.nan
                try: data[ticker_sym]['change_percent'] = quote_data['10. change percent'].iloc[0]
                except (KeyError, IndexError, TypeError): data[ticker_sym]['change_percent'] = 'N/A'
                logger.info(f"Dati AV per {ticker_sym} recuperati OK.")
            else:
                logger.warning(f"Risposta vuota da Alpha Vantage per {ticker_sym}.")
                st.warning(f"Risposta vuota da Alpha Vantage per {ticker_sym}.")
                # Assicura che i dati siano NaN/N/A se la risposta è vuota
                data[ticker_sym] = {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'}

        except ValueError as ve:
             # Errore specifico di AV, spesso legato a limiti API o chiave errata
             msg = f"Errore Alpha Vantage (ValueError) per {ticker_sym}: {ve}"
             logger.warning(msg)
             st.warning(msg) # Mostra l'errore in UI
             # Controlla se l'errore indica problemi di rate limit o API key
             ve_str = str(ve).lower()
             if "call frequency" in ve_str or "api key" in ve_str or "limit" in ve_str or "premium" in ve_str:
                 logger.error("Errore chiave/limite API Alpha Vantage rilevato. Interruzione fetch per mercati tradizionali.")
                 st.error("Errore chiave/limite API Alpha Vantage rilevato. Interruzione fetch.")
                 break # Interrompe il ciclo in caso di errore API critico
             # Se è un ValueError generico, imposta i dati a N/A e continua con il prossimo ticker
             data[ticker_sym] = {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'}
        except Exception as e:
             # Errore generico durante la chiamata AV
             msg = f"Errore generico fetch Alpha Vantage per {ticker_sym}: {e}"
             logger.exception(msg) # Logga l'eccezione completa
             st.warning(msg) # Mostra un warning generico in UI
             data[ticker_sym] = {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'}

    logger.info(f"Fine fetch dati Alpha Vantage. Effettuate {calls_made} chiamate.")
    return data

# --- Funzioni Calcolo Indicatori ---

def calculate_rsi_manual(series: pd.Series, period: int = RSI_PERIOD) -> float:
    """Calcola l'ultimo valore RSI manually."""
    if not isinstance(series, pd.Series) or series.empty or series.isna().all():
        return np.nan
    series = series.dropna()
    if len(series) < period + 1:
        return np.nan

    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Usa EWM per calcolare la media mobile esponenziale del guadagno/perdita
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Evita divisione per zero e gestisci NaN
    if avg_gain.isna().all() or avg_loss.isna().all():
        return np.nan
    last_avg_loss = avg_loss.iloc[-1]
    last_avg_gain = avg_gain.iloc[-1]

    if pd.isna(last_avg_loss) or pd.isna(last_avg_gain):
        return np.nan

    if last_avg_loss == 0:
        # RSI è 100 se solo guadagni, 50 se nessun cambiamento, 0 se solo perdite (ma il caso 0 è gestito sopra)
        return 100.0 if last_avg_gain > 0 else 50.0

    rs = last_avg_gain / last_avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Assicura che RSI sia tra 0 e 100
    return max(0.0, min(100.0, rsi))


def calculate_stoch_rsi(series: pd.Series, rsi_period: int = RSI_PERIOD, stoch_period: int = SRSI_PERIOD, k_smooth: int = SRSI_K, d_smooth: int = SRSI_D) -> tuple[float, float]:
    """Calcola gli ultimi valori %K e %D dello StochRSI."""
    if not isinstance(series, pd.Series) or series.empty or series.isna().all():
        return np.nan, np.nan
    series = series.dropna()
    # Verifica lunghezza sufficiente per tutti i calcoli
    if len(series) < rsi_period + stoch_period + max(k_smooth, d_smooth) -1 : # Stima approssimativa, meglio abbondare
         return np.nan, np.nan

    # 1. Calcola RSI
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan) # Evita divisione per zero
    rsi_series = (100.0 - (100.0 / (1.0 + rs))).dropna()

    if len(rsi_series) < stoch_period:
        return np.nan, np.nan

    # 2. Calcola StochRSI (%K Raw)
    min_rsi = rsi_series.rolling(window=stoch_period, min_periods=stoch_period).min()
    max_rsi = rsi_series.rolling(window=stoch_period, min_periods=stoch_period).max()
    range_rsi = max_rsi - min_rsi
    # Evita divisione per zero se il range è 0
    stoch_rsi_k_raw = 100 * (rsi_series - min_rsi) / range_rsi.replace(0, np.nan)
    stoch_rsi_k_raw = stoch_rsi_k_raw.dropna() # Rimuovi NaN iniziali

    if len(stoch_rsi_k_raw) < k_smooth:
        return np.nan, np.nan

    # 3. Calcola %K (Smooth)
    stoch_rsi_k = stoch_rsi_k_raw.rolling(window=k_smooth, min_periods=k_smooth).mean()

    # Se non ci sono abbastanza dati per %D, ritorna solo %K (se disponibile)
    if len(stoch_rsi_k.dropna()) < d_smooth:
         last_k_val = stoch_rsi_k.iloc[-1]
         # Clamp K tra 0 e 100
         k_final = max(0.0, min(100.0, last_k_val)) if pd.notna(last_k_val) else np.nan
         return k_final, np.nan

    # 4. Calcola %D (Smooth di %K)
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_smooth, min_periods=d_smooth).mean()

    # Estrai gli ultimi valori validi
    last_k = stoch_rsi_k.iloc[-1]
    last_d = stoch_rsi_d.iloc[-1]

    # Clamp finale tra 0 e 100 e gestione NaN
    k_final = max(0.0, min(100.0, last_k)) if pd.notna(last_k) else np.nan
    d_final = max(0.0, min(100.0, last_d)) if pd.notna(last_d) else np.nan

    return k_final, d_final


def calculate_macd_manual(series: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> tuple[float, float, float]:
    """Calcola gli ultimi valori di MACD Line, Signal Line e Histogram."""
    if not isinstance(series, pd.Series) or series.empty or series.isna().all():
        return np.nan, np.nan, np.nan
    series = series.dropna()
    # Verifica lunghezza sufficiente
    if len(series) < slow + signal - 1: # Lunghezza minima per avere un valore di signal line
        return np.nan, np.nan, np.nan

    # Calcola EMA Veloce e Lenta
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()

    # Calcola Linea MACD
    macd_line = ema_fast - ema_slow

    # Calcola Linea Segnale (EMA della linea MACD)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()

    # Calcola Istogramma MACD
    histogram = macd_line - signal_line

    # Estrai gli ultimi valori, gestendo i NaN che possono verificarsi all'inizio
    last_macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else np.nan
    last_signal = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else np.nan
    last_hist = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else np.nan

    return last_macd, last_signal, last_hist


def calculate_sma_manual(series: pd.Series, period: int) -> float:
    """Calcola l'ultimo valore della Media Mobile Semplice (SMA)."""
    if not isinstance(series, pd.Series) or series.empty or series.isna().all():
        return np.nan
    series = series.dropna()
    if len(series) < period:
        return np.nan
    # Calcola la media mobile e prendi l'ultimo valore
    return series.rolling(window=period, min_periods=period).mean().iloc[-1]


def calculate_vwap_manual(df: pd.DataFrame, period: int = VWAP_PERIOD) -> float:
    """Calcola l'ultimo valore del VWAP (Volume Weighted Average Price)."""
    required_cols = ['close', 'volume']
    if not isinstance(df, pd.DataFrame) or df.empty or not all(col in df.columns for col in required_cols):
        return np.nan

    # Lavora solo con righe che hanno sia prezzo che volume validi
    df_valid = df[required_cols].dropna()
    if len(df_valid) < period:
        return np.nan

    # Seleziona l'ultimo periodo
    df_period = df_valid.iloc[-period:]
    # Calcola (Prezzo * Volume) per il periodo
    pv = df_period['close'] * df_period['volume']
    # Calcola VWAP: Sum(PV) / Sum(Volume)
    total_volume = df_period['volume'].sum()

    # Gestisci il caso in cui il volume totale sia zero
    if total_volume == 0:
        # Ritorna l'ultimo prezzo di chiusura se il volume è zero
        return df_period['close'].iloc[-1] if not df_period.empty else np.nan

    vwap = pv.sum() / total_volume
    return vwap

def compute_all_indicators(symbol: str, hist_daily_df: pd.DataFrame) -> dict:
    """Calcola tutti gli indicatori tecnici richiesti per una coin."""
    # Rimosso hist_hourly_df perché non più usato
    indicators = {
        "RSI (1d)": np.nan, "RSI (1w)": np.nan, "RSI (1mo)": np.nan,
        "SRSI %K (1d)": np.nan, "SRSI %D (1d)": np.nan,
        "MACD Line (1d)": np.nan, "MACD Signal (1d)": np.nan, "MACD Hist (1d)": np.nan,
        f"MA({MA_SHORT}d)": np.nan, f"MA({MA_LONG}d)": np.nan,
        "VWAP (1d)": np.nan,
    }
    # Lunghezze minime stimate per avere un valore valido (considerando periodi + smoothing)
    min_len_rsi_base = RSI_PERIOD + 1
    min_len_srsi_base = RSI_PERIOD + SRSI_PERIOD + max(SRSI_K, SRSI_D) + 5 # Più conservativo
    min_len_macd_base = MACD_SLOW + MACD_SIGNAL + 5 # Più conservativo
    min_len_sma_short = MA_SHORT
    min_len_sma_long = MA_LONG
    min_len_vwap_base = VWAP_PERIOD

    if not hist_daily_df.empty and 'close' in hist_daily_df.columns:
        close_daily = hist_daily_df['close'].dropna()
        len_daily = len(close_daily)

        # Calcola indicatori giornalieri se ci sono abbastanza dati
        if len_daily >= min_len_rsi_base:
            indicators["RSI (1d)"] = calculate_rsi_manual(close_daily, RSI_PERIOD)
        else: logger.warning(f"{symbol}: Dati giornalieri insuff. ({len_daily}/{min_len_rsi_base}) per RSI(1d)")

        if len_daily >= min_len_srsi_base:
            k, d = calculate_stoch_rsi(close_daily, RSI_PERIOD, SRSI_PERIOD, SRSI_K, SRSI_D)
            indicators["SRSI %K (1d)"] = k
            indicators["SRSI %D (1d)"] = d
        else: logger.warning(f"{symbol}: Dati giornalieri insuff. ({len_daily}/{min_len_srsi_base}) per SRSI(1d)")

        if len_daily >= min_len_macd_base:
            macd_l, macd_s, macd_h = calculate_macd_manual(close_daily, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            indicators["MACD Line (1d)"] = macd_l
            indicators["MACD Signal (1d)"] = macd_s
            indicators["MACD Hist (1d)"] = macd_h
        else: logger.warning(f"{symbol}: Dati giornalieri insuff. ({len_daily}/{min_len_macd_base}) per MACD(1d)")

        if len_daily >= min_len_sma_short:
            indicators[f"MA({MA_SHORT}d)"] = calculate_sma_manual(close_daily, MA_SHORT)
        else: logger.warning(f"{symbol}: Dati giornalieri insuff. ({len_daily}/{min_len_sma_short}) per MA({MA_SHORT}d)")

        if len_daily >= min_len_sma_long:
            indicators[f"MA({MA_LONG}d)"] = calculate_sma_manual(close_daily, MA_LONG)
        else: logger.warning(f"{symbol}: Dati giornalieri insuff. ({len_daily}/{min_len_sma_long}) per MA({MA_LONG}d)")

        # VWAP richiede il DataFrame con 'close' e 'volume'
        if len_daily >= min_len_vwap_base and 'volume' in hist_daily_df.columns:
             indicators["VWAP (1d)"] = calculate_vwap_manual(hist_daily_df.iloc[-len_daily:], VWAP_PERIOD) # Passa solo i dati validi usati per close_daily
        elif 'volume' not in hist_daily_df.columns:
             logger.warning(f"{symbol}: Colonna 'volume' mancante per VWAP(1d)")
        else: logger.warning(f"{symbol}: Dati giornalieri insuff. ({len_daily}/{min_len_vwap_base}) per VWAP(1d)")


        # Calcola indicatori Weekly/Monthly se ci sono abbastanza dati giornalieri e l'indice è datetime
        if len_daily > min_len_rsi_base and pd.api.types.is_datetime64_any_dtype(close_daily.index):
            try: # Weekly RSI
                # Resample a settimanale (inizio settimana Lunedì), prendendo l'ultimo valore della settimana
                df_weekly = close_daily.resample('W-MON').last()
                # Verifica se ci sono abbastanza settimane per calcolare l'RSI
                if len(df_weekly.dropna()) >= min_len_rsi_base:
                    indicators["RSI (1w)"] = calculate_rsi_manual(df_weekly, RSI_PERIOD)
                else: logger.warning(f"{symbol}: Dati Weekly insuff. ({len(df_weekly.dropna())}/{min_len_rsi_base}) per RSI(1w)")
            except Exception as e:
                logger.exception(f"{symbol}: Errore durante calcolo RSI weekly:")

            try: # Monthly RSI
                # Resample a mensile (fine mese), prendendo l'ultimo valore del mese
                df_monthly = close_daily.resample('ME').last()
                # Verifica se ci sono abbastanza mesi per calcolare l'RSI
                if len(df_monthly.dropna()) >= min_len_rsi_base:
                    indicators["RSI (1mo)"] = calculate_rsi_manual(df_monthly, RSI_PERIOD)
                else: logger.warning(f"{symbol}: Dati Monthly insuff. ({len(df_monthly.dropna())}/{min_len_rsi_base}) per RSI(1mo)")
            except Exception as e:
                logger.exception(f"{symbol}: Errore durante calcolo RSI monthly:")
    else:
        logger.warning(f"{symbol}: Dati giornalieri (hist_daily_df) vuoti o mancanti per calcolo indicatori.")

    # RSI 1h rimosso
    return indicators

# --- Funzioni Segnale ---

def generate_gpt_signal(rsi_1d, rsi_1w, macd_hist, ma_short, ma_long, srsi_k, srsi_d, current_price):
    """Genera un segnale basato su una combinazione di indicatori (stile 'GPT')."""
    # RSI 1h non è più un input richiesto
    required_inputs = [rsi_1d, macd_hist, ma_short, ma_long, current_price]
    # Se uno qualsiasi degli input *fondamentali* è NaN, ritorna N/D
    if any(pd.isna(x) for x in required_inputs):
        return "⚪️ N/D"

    score = 0
    # Trend (MA Crossover e Prezzo vs MA Lunga)
    if current_price > ma_long: score += 1
    else: score -= 1
    if ma_short > ma_long: score += 2 # Golden cross implicito
    else: score -= 2 # Death cross implicito

    # Momentum (MACD Histogram)
    if macd_hist > 0: score += 2 # Momentum positivo
    else: score -= 2 # Momentum negativo

    # Oscillatore (RSI Daily)
    if rsi_1d < 30: score += 2 # Ipervenduto forte
    elif rsi_1d < 40: score += 1 # Tendente all'ipervenduto
    elif rsi_1d > 70: score -= 2 # Ipercomprato forte
    elif rsi_1d > 60: score -= 1 # Tendente all'ipercomprato

    # Oscillatore (RSI Weekly - se disponibile)
    if pd.notna(rsi_1w):
        if rsi_1w < 40: score += 1 # Debolezza anche su TF settimanale
        elif rsi_1w > 60: score -= 1 # Forza anche su TF settimanale

    # Oscillatore (StochRSI Daily - se disponibile)
    if pd.notna(srsi_k) and pd.notna(srsi_d):
        if srsi_k < 20 and srsi_d < 20: score += 1 # Ipervenduto SRSI
        elif srsi_k > 80 and srsi_d > 80: score -= 1 # Ipercomprato SRSI

    # Mappatura Punteggio -> Segnale
    if score >= 5: return "⚡️ Strong Buy"
    elif score >= 2: return "🟢 Buy"
    elif score <= -5: return "🚨 Strong Sell"
    elif score <= -2: return "🔴 Sell"
    # Condizioni per Consider To Buy/Sell (CTB/CTS)
    elif score > 0: # Tendenza leggermente rialzista o neutra-rialzista
        # Consider To Buy se RSI non è troppo alto
        return "⏳ CTB" if pd.notna(rsi_1d) and rsi_1d < 55 else "🟡 Hold"
    else: # Tendenza leggermente ribassista o neutra-ribassista
        # Consider To Sell se RSI non è troppo basso
        return "⚠️ CTS" if pd.notna(rsi_1d) and rsi_1d > 45 else "🟡 Hold"


def generate_gemini_alert(ma_short, ma_long, macd_hist, rsi_1d):
    """Genera un alert basato su MA Crossover, MACD e RSI (stile 'Gemini')."""
    # Controlla se tutti gli input necessari sono validi
    if pd.isna(ma_short) or pd.isna(ma_long) or pd.isna(macd_hist) or pd.isna(rsi_1d):
        return "⚪️ N/D" # Non disponibile se manca qualche indicatore

    # Condizioni per Strong Buy
    is_uptrend_ma = ma_short > ma_long       # Golden cross MA
    is_momentum_positive = macd_hist > 0     # Momentum rialzista
    is_not_extremely_overbought = rsi_1d < 80 # Non eccessivamente ipercomprato

    # Condizioni per Strong Sell
    is_downtrend_ma = ma_short < ma_long       # Death cross MA
    is_momentum_negative = macd_hist < 0     # Momentum ribassista
    is_not_extremely_oversold = rsi_1d > 20  # Non eccessivamente ipervenduto

    # Logica Alert
    if is_uptrend_ma and is_momentum_positive and is_not_extremely_overbought:
        return "⚡️ Strong Buy"
    elif is_downtrend_ma and is_momentum_negative and is_not_extremely_oversold:
        return "🚨 Strong Sell"
    else:
        # In tutti gli altri casi (condizioni miste o neutre)
        return "🟡 Hold"


# --- INIZIO ESECUZIONE PRINCIPALE APP ---
logger.info("Inizio esecuzione UI principale.")
try: # Blocco try principale per catturare errori imprevisti nell'app

    # 1. Controllo Password (ora usa password hardcoded)
    if not check_password():
        st.stop() # Ferma l'esecuzione se la password non è corretta
    logger.info("Password check superato.")

    # --- TITOLO, BOTTONE REFRESH, TIMESTAMP ---
    col_title, col_button_placeholder, col_button = st.columns([4, 1, 1])
    with col_title:
        st.title("📈 Crypto Technical Dashboard Pro")
    with col_button:
        st.write("") # Spacer per allineare il bottone
        if st.button("🔄 Aggiorna", help="Forza l'aggiornamento di tutti i dati (cancella la cache)", key="refresh_button"):
            logger.info("Bottone Aggiorna cliccato. Pulizia cache e rerun.")
            # Pulisci lo stato di warning API e la cache dati
            if 'api_warning_shown' in st.session_state:
                del st.session_state['api_warning_shown']
            st.cache_data.clear()
            # Pulisci i query params per un refresh pulito
            st.query_params.clear()
            st.rerun()

    last_update_placeholder = st.empty() # Placeholder per il timestamp
    st.caption(f"Cache: Crypto Live ({CACHE_TTL/60:.0f}m), Storico ({CACHE_HIST_TTL/60:.0f}m), Tradizionale ({CACHE_TRAD_TTL/3600:.0f}h).")

    # --- SEZIONE MARKET OVERVIEW ---
    st.markdown("---")
    st.subheader("🌐 Market Overview")

    # Recupero dati per l'overview (usano cache)
    fear_greed_value = get_fear_greed_index()
    total_market_cap = get_global_market_data_cg(VS_CURRENCY)
    etf_flow_value = get_etf_flow() # Placeholder
    # Recupera dati Alpha Vantage (con cache lunga)
    traditional_market_data = get_traditional_market_data_av(TRAD_TICKERS_AV)

    def format_delta(change_val, change_pct_str):
        """Formatta il delta per st.metric, includendo valore e percentuale."""
        delta_string = None # Default a None se non calcolabile
        if pd.notna(change_val) and isinstance(change_pct_str, str) and change_pct_str not in ['N/A', '', None]:
            try:
                # Pulisce la stringa percentuale e converte in float
                change_pct_val = float(change_pct_str.replace('%','').strip())
                delta_string = f"{change_val:+.2f} ({change_pct_val:+.2f}%)"
            except (ValueError, AttributeError):
                # Fallback se la conversione percentuale fallisce
                delta_string = f"{change_val:+.2f} (?%)"
        elif pd.notna(change_val):
            # Mostra solo il valore se la percentuale non è disponibile
            delta_string = f"{change_val:+.2f}"
        return delta_string

    # Funzione helper per renderizzare le metriche (sia AV che altre)
    def render_metric(column, label, value_func=None, ticker=None, data_dict=None, help_text=None):
        """Renderizza una metrica in una colonna specificata."""
        value_str = "N/A"
        delta_txt = None
        d_color = "off" # Colore delta 'off' per default (grigio)

        if ticker and data_dict: # Caso Ticker Tradizionale (Alpha Vantage)
            trad_info = data_dict.get(ticker, {})
            price = trad_info.get('price', np.nan)
            change = trad_info.get('change', np.nan)
            change_pct = trad_info.get('change_percent', 'N/A')

            value_str = f"${price:,.2f}" if pd.notna(price) else "N/A" # Aggiunge $ ai prezzi AV
            delta_txt = format_delta(change, change_pct)
            # Imposta colore normale (verde/rosso) solo se il 'change' è un numero valido
            if pd.notna(change):
                d_color = "normal"
        elif value_func: # Caso valore calcolato da funzione (F&G, MCap, ETF Flow)
            try:
                value_str = value_func()
                # Assicura che value_str sia una stringa o gestibile da st.metric
                if value_str is None: value_str = "N/A"
                else: value_str = str(value_str) # Converte a stringa per sicurezza
            except Exception as e:
                logger.error(f"Errore in value_func per metrica '{label}': {e}")
                value_str = "Errore"

            # Questi non hanno un delta associato nel formato attuale
            delta_txt = None
            d_color = "off"
        else: # Fallback se né ticker né funzione sono forniti
             value_str = "N/A"

        column.metric(label=label, value=value_str, delta=delta_txt, delta_color=d_color, help=help_text)

    # RIGA 1 Overview
    overview_items_row1 = [
        ("Fear & Greed Index", None, get_fear_greed_index, "Fonte: Alternative.me"),
        (f"Total Crypto M.Cap ({VS_CURRENCY.upper()})", None, lambda: f"${format_large_number(total_market_cap)}", "Fonte: CoinGecko"),
        ("Crypto ETFs Flow (Daily)", None, get_etf_flow, "Dato N/A (Placeholder)"),
        ("S&P 500 (SPY)", "SPY", None, "Fonte: Alpha Vantage (ETF)"),
        ("Nasdaq (QQQ)", "QQQ", None, "Fonte: Alpha Vantage (ETF)")
    ]
    overview_cols_1 = st.columns(len(overview_items_row1))
    for i, (label, ticker, func, help_text) in enumerate(overview_items_row1):
        render_metric(overview_cols_1[i], label, value_func=func, ticker=ticker, data_dict=traditional_market_data, help_text=help_text)

    # RIGA 2 Overview
    overview_items_row2 = [
        ("Gold (GLD)", "GLD", None, "Fonte: Alpha Vantage (ETF)"),
        ("Silver (SLV)", "SLV", None, "Fonte: Alpha Vantage (ETF)"),
        ("Natural Gas (UNG)", "UNG", None, "Fonte: Alpha Vantage (ETF)"),
        ("UVXY (Volatility)", "UVXY", None, "Fonte: Alpha Vantage"),
        ("TQQQ (Nasdaq 3x)", "TQQQ", None, "Fonte: Alpha Vantage")
    ]
    overview_cols_2 = st.columns(len(overview_items_row2))
    for i, (label, ticker, func, help_text) in enumerate(overview_items_row2):
         render_metric(overview_cols_2[i], label, value_func=func, ticker=ticker, data_dict=traditional_market_data, help_text=help_text)


    # SEZIONE TITOLI PRINCIPALI (ALPHA VANTAGE)
    st.markdown("<h6>Titoli Principali (Prezzi e Var. Giorno - Fonte: Alpha Vantage, Cache 4h):</h6>", unsafe_allow_html=True)
    stock_tickers_row_av = ['NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR', 'COIN', 'MSTR']
    num_stock_cols = 5 # Numero di colonne per i titoli
    stock_cols = st.columns(num_stock_cols)
    for idx, ticker in enumerate(stock_tickers_row_av):
        col_index = idx % num_stock_cols # Distribuisce i ticker nelle colonne
        current_col = stock_cols[col_index]
        # Usa la stessa funzione helper render_metric
        render_metric(current_col, label=ticker, ticker=ticker, data_dict=traditional_market_data, help_text=f"Ticker: {ticker}") # Aggiunto help text con nome ticker
    st.markdown("---")

    # --- LOGICA PRINCIPALE DASHBOARD CRYPTO ---
    st.subheader(f"📊 Analisi Tecnica Crypto ({NUM_COINS} Asset)")
    logger.info("Inizio recupero dati crypto live.")
    # Recupero dati live CoinGecko (usa cache)
    market_data_df, last_cg_update_utc = get_coingecko_market_data(COINGECKO_IDS_LIST, VS_CURRENCY)

    # Gestione e Visualizzazione Timestamp Aggiornamento
    if last_cg_update_utc:
        timestamp_display_str = "*Timestamp aggiornamento dati live CoinGecko non disponibile.*"
        try:
            if ZoneInfo: # Se zoneinfo è disponibile, usa fuso orario preciso
                local_tz = ZoneInfo("Europe/Rome")
                # Assicura che il timestamp sia timezone-aware (UTC) prima di convertire
                if last_cg_update_utc.tzinfo is None:
                    last_cg_update_utc = last_cg_update_utc.replace(tzinfo=ZoneInfo("UTC"))
                last_cg_update_local = last_cg_update_utc.astimezone(local_tz)
                timestamp_display_str = f"*Dati live CoinGecko aggiornati alle: **{last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S %Z')}***"
                logger.info(f"Timestamp visualizzato: {last_cg_update_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            else: # Fallback se zoneinfo non è disponibile
                offset_hours = 2 # Assumi +2 per Roma (approssimativo)
                last_cg_update_rome_approx = last_cg_update_utc + timedelta(hours=offset_hours)
                timestamp_display_str = f"*Dati live CoinGecko aggiornati alle: **{last_cg_update_rome_approx.strftime('%Y-%m-%d %H:%M:%S')} (Ora approx. Roma)***"
                logger.info(f"Timestamp visualizzato (approx): {last_cg_update_rome_approx.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            logger.exception("Errore durante formattazione/conversione timestamp:")
            # Mostra errore e timestamp UTC grezzo se la conversione fallisce
            timestamp_display_str = f"*Errore conversione timestamp ({e}). Ora UTC: {last_cg_update_utc.strftime('%Y-%m-%d %H:%M:%S')}*"

        last_update_placeholder.markdown(timestamp_display_str)
    else:
        logger.warning("Timestamp aggiornamento dati live CoinGecko non disponibile (last_cg_update_utc è None).")
        last_update_placeholder.markdown("*Timestamp aggiornamento dati live CoinGecko non disponibile.*")

    # Blocco Esecuzione se i dati live CoinGecko falliscono (critico)
    if market_data_df.empty:
        msg = "Errore critico: Impossibile caricare dati live CoinGecko. La tabella di analisi non può essere generata."
        # Personalizza messaggio se il problema è noto (es. rate limit 429)
        if st.session_state.get("api_warning_shown", False):
            msg = "Tabella Analisi Tecnica non generata: errore caricamento dati live (possibile limite API CoinGecko raggiunto)."
        logger.error(msg)
        st.error(msg)
        st.stop() # Ferma l'esecuzione dell'app
    logger.info(f"Dati live CoinGecko OK ({len(market_data_df)} righe), inizio ciclo elaborazione crypto.")

    # --- CICLO PROCESSING PER OGNI COIN ---
    results = [] # Lista per memorizzare i risultati di ogni coin
    fetch_errors_for_display = [] # Lista per aggregare errori specifici delle coin da mostrare all'utente
    process_start_time = time.time()

    # Usa il numero effettivo di coin restituite dalla chiamata live
    effective_num_coins = len(market_data_df.index)
    if effective_num_coins != NUM_COINS:
        logger.warning(f"Numero coin ricevute da API ({effective_num_coins}) diverso da configurate ({NUM_COINS}). Processando {effective_num_coins}.")

    # Stima tempo basata sulla pausa di 6s per chiamata storica (2 chiamate per coin: daily + hourly)
    estimated_wait_secs = effective_num_coins * 2 * 6.0 # 2 chiamate (daily, hourly) * 6s pausa
    estimated_wait_mins = estimated_wait_secs / 60
    spinner_msg = f"Recupero dati storici e calcolo indicatori per {effective_num_coins} crypto... (Richiede ~{estimated_wait_mins:.1f} min)"

    with st.spinner(spinner_msg):
        # Ordina gli ID coin in base all'ordine restituito da CoinGecko (di solito market cap)
        coin_ids_ordered = market_data_df.index.tolist()
        logger.info(f"Lista ID CoinGecko da processare (ordine da API live): {coin_ids_ordered}")
        actual_processed_count = 0 # Contatore coin processate con successo

        for i, coin_id in enumerate(coin_ids_ordered):
            # Trova il simbolo corrispondente all'ID CoinGecko
            symbol = next((sym for sym, c_id in SYMBOL_TO_ID_MAP.items() if c_id == coin_id), "N/A")
            logger.info(f"--- Elaborazione {i+1}/{effective_num_coins}: {symbol} ({coin_id}) ---")

            try:
                # Salta se l'ID non è nella nostra mappa (non dovrebbe succedere se SYMBOL_TO_ID_MAP è aggiornato)
                if symbol == "N/A":
                    msg = f"{coin_id}: ID CoinGecko non trovato nella mappa SYMBOL_TO_ID_MAP locale. Saltato."
                    logger.warning(msg)
                    fetch_errors_for_display.append(msg)
                    continue

                # Estrai dati live per la coin corrente
                live_data = market_data_df.loc[coin_id]
                name = live_data.get('name', coin_id) # Usa ID se manca nome
                rank = live_data.get('market_cap_rank', 'N/A')
                current_price = live_data.get('current_price', np.nan)
                volume_24h = live_data.get('total_volume', np.nan)
                # Estrai variazioni percentuali (gestisci NaN)
                change_1h = live_data.get('price_change_percentage_1h_in_currency', np.nan)
                change_24h = live_data.get('price_change_percentage_24h_in_currency', np.nan)
                change_7d = live_data.get('price_change_percentage_7d_in_currency', np.nan)
                change_30d = live_data.get('price_change_percentage_30d_in_currency', np.nan)
                change_1y = live_data.get('price_change_percentage_1y_in_currency', np.nan)

                # Recupera dati storici (Daily & Hourly) - con gestione errori e status
                # Nota: L'hourly viene fetchato ma non usato in compute_all_indicators v17.1
                hist_daily_df, status_daily = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_DAILY, interval='daily')
                if status_daily != "Success":
                    fetch_errors_for_display.append(f"{symbol}: Storico Daily - {status_daily}") # Logga errore per UI

                hist_hourly_df, status_hourly = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_HOURLY, interval='hourly')
                if status_hourly != "Success":
                    fetch_errors_for_display.append(f"{symbol}: Storico Hourly - {status_hourly}") # Logga errore per UI

                # Calcola indicatori (passa solo il daily df)
                indicators = compute_all_indicators(symbol, hist_daily_df)

                # Genera segnali
                gpt_signal = generate_gpt_signal(
                    indicators.get("RSI (1d)"), indicators.get("RSI (1w)"),
                    indicators.get("MACD Hist (1d)"), indicators.get(f"MA({MA_SHORT}d)"),
                    indicators.get(f"MA({MA_LONG}d)"), indicators.get("SRSI %K (1d)"),
                    indicators.get("SRSI %D (1d)"), current_price
                )
                gemini_alert = generate_gemini_alert(
                    indicators.get(f"MA({MA_SHORT}d)"), indicators.get(f"MA({MA_LONG}d)"),
                    indicators.get("MACD Hist (1d)"), indicators.get("RSI (1d)")
                )

                # Link a CoinGecko
                coingecko_link = f"https://www.coingecko.com/en/coins/{coin_id}"

                # Aggiungi risultati alla lista
                results.append({
                    "Rank": rank, "Symbol": symbol, "Name": name,
                    "Gemini Alert": gemini_alert, "GPT Signal": gpt_signal,
                    f"Prezzo ({VS_CURRENCY.upper()})": current_price,
                    "% 1h": change_1h, "% 24h": change_24h, "% 7d": change_7d, "% 30d": change_30d, "% 1y": change_1y,
                    "RSI (1d)": indicators.get("RSI (1d)"),
                    "RSI (1w)": indicators.get("RSI (1w)"),
                    "RSI (1mo)": indicators.get("RSI (1mo)"),
                    "SRSI %K (1d)": indicators.get("SRSI %K (1d)"),
                    "SRSI %D (1d)": indicators.get("SRSI %D (1d)"),
                    "MACD Hist (1d)": indicators.get("MACD Hist (1d)"),
                    f"MA({MA_SHORT}d)": indicators.get(f"MA({MA_SHORT}d)"),
                    f"MA({MA_LONG}d)": indicators.get(f"MA({MA_LONG}d)"),
                    "VWAP (1d)": indicators.get("VWAP (1d)"),
                    f"Volume 24h ({VS_CURRENCY.upper()})": volume_24h,
                    "Link": coingecko_link
                })
                logger.info(f"--- Elaborazione {symbol} completata con successo. ---")
                actual_processed_count += 1

            except Exception as coin_err:
                # Cattura eccezioni non previste durante l'elaborazione di una singola coin
                err_msg = f"Errore irreversibile durante elaborazione di {symbol} ({coin_id}): {coin_err}"
                logger.exception(err_msg) # Logga l'errore completo
                # Aggiungi un messaggio generico all'elenco degli errori per l'utente
                fetch_errors_for_display.append(f"{symbol}: Errore Elaborazione Grave - Vedi Log Applicazione per dettagli")
                # Non incrementare actual_processed_count e continua con la prossima coin

    process_end_time = time.time()
    total_time = process_end_time - process_start_time
    logger.info(f"Fine ciclo elaborazione crypto. Processate {actual_processed_count}/{effective_num_coins} coin. Tempo totale: {total_time:.1f} sec")
    # Mostra tempo elaborazione nella sidebar (opzionale)
    st.sidebar.info(f"Tempo elaborazione crypto: {total_time:.1f} sec")

    # --- CREA E VISUALIZZA DATAFRAME FINALE ---
    if results:
        logger.info(f"Creazione DataFrame finale con {len(results)} risultati.")
        try:
            df = pd.DataFrame(results)
            # Converti Rank in numerico per ordinamento, gestendo 'N/A' o errori
            df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
            # Imposta Rank come indice e ordina
            df.set_index('Rank', inplace=True, drop=True)
            df.sort_index(inplace=True)

            # Definisci l'ordine desiderato delle colonne
            cols_order = [
                "Symbol", "Name", "Gemini Alert", "GPT Signal",
                f"Prezzo ({VS_CURRENCY.upper()})",
                "% 1h", "% 24h", "% 7d", "% 30d", "% 1y",
                "RSI (1d)", "RSI (1w)", "RSI (1mo)",
                "SRSI %K (1d)", "SRSI %D (1d)",
                "MACD Hist (1d)",
                f"MA({MA_SHORT}d)", f"MA({MA_LONG}d)", "VWAP (1d)",
                f"Volume 24h ({VS_CURRENCY.upper()})",
                "Link"
            ]
            # Seleziona solo le colonne esistenti nell'ordine definito
            cols_to_show = [col for col in cols_order if col in df.columns]
            df_display = df[cols_to_show].copy() # Crea copia per styling

            # Definizione formattatori per le colonne numeriche
            formatters = {}
            currency_col = f"Prezzo ({VS_CURRENCY.upper()})"
            volume_col = f"Volume 24h ({VS_CURRENCY.upper()})"
            pct_cols = ["% 1h", "% 24h", "% 7d", "% 30d", "% 1y"]
            rsi_srsi_cols = [col for col in df_display.columns if ("RSI" in col or "SRSI" in col)]
            macd_cols = [col for col in df_display.columns if "MACD" in col]
            ma_vwap_cols = [col for col in df_display.columns if "MA" in col or "VWAP" in col]

            if currency_col in df_display.columns: formatters[currency_col] = "${:,.4f}" # Prezzo con 4 decimali
            if volume_col in df_display.columns: formatters[volume_col] = lambda x: f"${format_large_number(x)}" # Volume formattato
            for col in pct_cols:
                if col in df_display.columns: formatters[col] = "{:+.2f}%" # Percentuali con segno e 2 decimali
            for col in rsi_srsi_cols:
                if col in df_display.columns: formatters[col] = "{:.1f}" # RSI/SRSI con 1 decimale
            for col in macd_cols:
                if col in df_display.columns: formatters[col] = "{:.4f}" # MACD con 4 decimali
            for col in ma_vwap_cols:
                if col in df_display.columns: formatters[col] = "{:,.2f}" # MA/VWAP con 2 decimali

            # Applica la formattazione usando Styler
            styled_df = df_display.style.format(formatters, na_rep="N/A", precision=4, subset=list(formatters.keys()))

            # Funzioni di Stile Condizionale
            def highlight_pct_col_style(val):
                """Colora testo verde/rosso per variazioni percentuali."""
                if pd.isna(val) or not isinstance(val, (int, float)): return ''
                color = 'green' if val > 0 else 'red' if val < 0 else '#6c757d' # Grigio per 0%
                return f'color: {color};'

            def highlight_signal_style(val):
                """Colora e formatta testo per segnali/alert."""
                style = 'color: #6c757d;' # Grigio default
                font_weight = 'normal'
                if isinstance(val, str):
                    if "Strong Buy" in val: style = 'color: #198754;'; font_weight = 'bold' # Verde scuro grassetto
                    elif "Buy" in val and "Strong" not in val: style = 'color: #28a745;' # Verde normale
                    elif "Strong Sell" in val: style = 'color: #dc3545;'; font_weight = 'bold' # Rosso scuro grassetto
                    elif "Sell" in val and "Strong" not in val: style = 'color: #fd7e14;' # Arancione normale
                    elif "CTB" in val: style = 'color: #20c997;' # Ciano/Teal
                    elif "CTS" in val: style = 'color: #ffc107; color: #000;' # Giallo scuro (con testo nero per leggibilità)
                    elif "Hold" in val: style = 'color: #6c757d;' # Grigio
                    elif "N/D" in val: style = 'color: #adb5bd;' # Grigio chiaro
                return f'{style} font-weight: {font_weight};'

            # Applica stili condizionali alle colonne appropriate
            cols_for_pct_style = [col for col in pct_cols if col in df_display.columns]
            if cols_for_pct_style:
                styled_df = styled_df.applymap(highlight_pct_col_style, subset=cols_for_pct_style)
            if "Gemini Alert" in df_display.columns:
                styled_df = styled_df.applymap(highlight_signal_style, subset=["Gemini Alert"])
            if "GPT Signal" in df_display.columns:
                styled_df = styled_df.applymap(highlight_signal_style, subset=["GPT Signal"])

            # Visualizza il DataFrame stilizzato
            logger.info("Visualizzazione DataFrame principale stilizzato.")
            st.dataframe(
                styled_df,
                use_container_width=True,
                column_config={
                    "Link": st.column_config.LinkColumn(
                        "CoinGecko",
                        help="Link alla pagina CoinGecko della coin",
                        display_text="🔗 Link", # Testo visualizzato nel link
                        width="small" # Larghezza colonna
                    )
                }
            )
        except Exception as df_err:
            logger.exception("Errore durante creazione o styling del DataFrame principale:")
            st.error(f"Errore durante la visualizzazione della tabella principale: {df_err}")
    else:
        # Messaggio se non ci sono risultati validi da mostrare
        logger.warning("Nessun risultato crypto valido da visualizzare nel DataFrame.")
        st.warning("Nessun risultato crypto valido da visualizzare. Controllare errori elaborazione coin.")

    # --- EXPANDER ERRORI ELABORAZIONE COIN (Mostra solo se ci sono errori) ---
    # Rimuovi duplicati dagli errori e ordina
    fetch_errors_unique_display = sorted(list(set(fetch_errors_for_display)))
    if fetch_errors_unique_display:
        with st.expander("❗️ Errori Durante Elaborazione Coin Specifiche", expanded=True):
            st.warning("Alcune coin non hanno potuto essere elaborate correttamente. Possibili cause: errori API (es. 429 Rate Limit, 404 Not Found), dati storici mancanti o problemi interni.")
            max_errors_to_show = 30 # Limita numero errori mostrati per non affollare UI
            error_list_md = ""
            for i, error_msg in enumerate(fetch_errors_unique_display):
                if i < max_errors_to_show:
                    error_list_md += f"- {error_msg}\n"
                elif i == max_errors_to_show:
                    error_list_md += f"- ... e altri {len(fetch_errors_unique_display) - max_errors_to_show} errori.\n"
                    break
            st.markdown(error_list_md)
            st.info("Controlla il Log Applicazione completo (in fondo alla pagina) per i traceback dettagliati degli errori 'Gravi'.")

    # --- LEGENDA ---
    st.divider()
    with st.expander("📘 Legenda Indicatori Tecnici e Segnali", expanded=False):
        # Markdown multiline per leggibilità
        st.markdown("""
        *Disclaimer: Questa dashboard è fornita solo a scopo informativo e didattico e non costituisce in alcun modo consulenza finanziaria.*

        **Market Overview:**
        *   **Fear & Greed Index:** Indice del sentiment di mercato crypto (Fonte: Alternative.me). Valori bassi indicano paura (potenziale opportunità di acquisto), valori alti indicano avidità (potenziale rischio).
        *   **Total Crypto M.Cap:** Capitalizzazione di mercato totale di tutte le criptovalute (Fonte: CoinGecko). Indica la dimensione generale del mercato.
        *   **Crypto ETFs Flow:** Flusso netto giornaliero di capitale negli ETF crypto spot (es. Bitcoin). **Dato attualmente N/A (Non Applicabile/Non Disponibile) in questa versione.**
        *   **S&P 500 (SPY), Nasdaq (QQQ), Gold (GLD), etc.:** Prezzi di riferimento dei mercati tradizionali (Azioni, Materie Prime, Volatilità). Fonte: Alpha Vantage. **Aggiornati con ritardo significativo (cache 4h)** a causa dei limiti dell'API gratuita. La variazione mostrata è quella giornaliera (Rosso/Verde).
        *   **Titoli Principali:** Prezzi di alcune azioni rilevanti (Tech, Crypto-related). Fonte: Alpha Vantage. **Aggiornati con ritardo (cache 4h).** Variazione giornaliera (Rosso/Verde).

        **Tabella Analisi Tecnica Crypto:**
        *   **Rank:** Posizione della coin per capitalizzazione di mercato (Fonte: CoinGecko).
        *   **Gemini Alert / GPT Signal:** Segnali **esemplificativi e sperimentali** generati automaticamente sulla base di indicatori tecnici. **NON sono raccomandazioni di trading.** Usare con estrema cautela e fare sempre le proprie ricerche (DYOR).
            *   ⚡️ Strong Buy / 🟢 Buy: Condizioni tecniche potenzialmente rialziste.
            *   🚨 Strong Sell / 🔴 Sell: Condizioni tecniche potenzialmente ribassiste.
            *   🟡 Hold: Condizioni neutre o miste.
            *   ⏳ CTB (Consider To Buy): Tendenza potenzialmente rialzista, ma da monitorare.
            *   ⚠️ CTS (Consider To Sell): Tendenza potenzialmente ribassista, ma da monitorare.
            *   ⚪️ N/D: Segnale non disponibile (dati insufficienti).
        *   **Prezzo:** Prezzo corrente della coin nella valuta selezionata (USD) (Fonte: CoinGecko).
        *   **% 1h, 24h, 7d, 30d, 1y:** Variazioni percentuali del prezzo su diversi intervalli temporali (Fonte: CoinGecko).
        *   **Indicatori Momentum (Oscillatori):**
            *   **RSI (1d, 1w, 1mo):** Relative Strength Index su timeframe giornaliero, settimanale, mensile. Misura la velocità e il cambiamento dei movimenti di prezzo. Valori < 30 indicano ipervenduto, > 70 ipercomprato (ma i livelli possono variare).
            *   **SRSI %K / %D (1d):** Stochastic RSI. Oscillatore applicato all'RSI per identificare condizioni di ipercomprato/ipervenduto più sensibili. Valori < 20 ipervenduto, > 80 ipercomprato.
            *   **MACD Hist (1d):** Moving Average Convergence Divergence Histogram. Differenza tra la linea MACD e la linea segnale. Barre sopra lo zero indicano momentum rialzista, sotto lo zero momentum ribassista.
        *   **Indicatori Trend:**
            *   **MA(20d) / MA(50d):** Medie Mobili Semplici (Simple Moving Averages) a 20 e 50 giorni. Aiutano a identificare la direzione del trend. Incroci (es. MA20 > MA50) possono generare segnali.
            *   **VWAP (1d):** Volume Weighted Average Price (calcolato su 14 periodi giornalieri). Prezzo medio ponderato per il volume. Usato spesso come riferimento intraday, qui calcolato su base giornaliera.
        *   **Volume 24h:** Volume totale scambiato nelle ultime 24 ore (Fonte: CoinGecko).
        *   **Link:** Collegamento diretto alla pagina della coin su CoinGecko.
        *   **N/A:** Dato non disponibile (es. dati storici insufficienti, errore API, indicatore non calcolabile).

        **Note Importanti:**
        *   Il recupero dei dati storici da CoinGecko è volutamente rallentato (**pausa di 6 secondi** tra le richieste per rispettare i limiti API gratuiti). Il **caricamento iniziale dell'app può richiedere diversi minuti**, specialmente con molte coin.
        *   I dati dei mercati tradizionali (Alpha Vantage) utilizzano una **cache di 4 ore** a causa dei limiti stringenti dell'API gratuita (5 chiamate/minuto, ~25/giorno). Richiedono una chiave API valida configurata nei `secrets` di Streamlit per funzionare.
        *   **DYOR (Do Your Own Research):** Fai sempre le tue ricerche approfondite prima di prendere decisioni di investimento. Le performance passate non sono indicative dei risultati futuri.
        """)

    # --- Footer ---
    st.divider()
    st.caption("Disclaimer: Strumento fornito a solo scopo informativo e didattico. Non costituisce consulenza finanziaria. Fai sempre le tue ricerche (DYOR).")

# --- Blocco try...except principale ---
except Exception as main_exception:
    # Cattura qualsiasi eccezione non gestita nel blocco principale dell'app
    logger.exception("!!! ERRORE NON GESTITO NELL'ESECUZIONE PRINCIPALE DELL'APP !!!")
    st.error(f"""
    Si è verificato un errore imprevisto nell'applicazione.
    Dettagli dell'errore: {main_exception}
    Controlla il log qui sotto per informazioni tecniche più dettagliate (traceback).
    Prova ad aggiornare la pagina o a premere il bottone 'Aggiorna'.
    """)
    # Non usare st.stop() qui per permettere la visualizzazione del log sottostante

# --- VISUALIZZAZIONE LOG APPLICAZIONE (Sempre alla fine) ---
st.divider()
st.subheader("📄 Log Applicazione")
st.caption("Questa sezione mostra i messaggi di log generati durante l'ultima esecuzione dell'app (Livello DEBUG). Utile per il debug e per capire cosa è successo.")
# Recupera il contenuto dello stream di log
log_content = log_stream.getvalue()
st.text_area(
    "Log:",
    value=log_content,
    height=400,
    key="log_display_area",
    help="Puoi selezionare tutto (Ctrl+A o Cmd+A) e copiare (Ctrl+C o Cmd+C) questo log per analizzarlo o condividerlo."
)
logger.info("--- Fine esecuzione script Streamlit app.py ---")
# Chiudi lo stream di log in memoria (buona pratica)
log_stream.close()
