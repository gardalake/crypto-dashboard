# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time
import math
# import yfinance as yf # Non più usato
# import feedparser # Non più usato
from alpha_vantage.timeseries import TimeSeries # Per Alpha Vantage
import logging # Per logging
import io # Per logging in memoria

# --- INIZIO: Configurazione Logging in UI ---
log_stream = io.StringIO() # Buffer in memoria per i log
logging.basicConfig(
    stream=log_stream,
    level=logging.INFO, # Cattura INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True # Forza la riconfigurazione se già chiamato
)
logger = logging.getLogger(__name__) # Usa un logger specifico per questo modulo
logger.info("Logging configurato per UI.")
# --- FINE: Configurazione Logging in UI ---

# Import zoneinfo for timezone handling if available (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
    logger.info("Modulo 'zoneinfo' importato con successo.")
except ImportError:
    logger.warning("Modulo 'zoneinfo' non trovato. Usando fallback UTC+2 per Roma.")
    st.warning("Modulo 'zoneinfo' non trovato. Usando fallback UTC+2 per Roma. Considera aggiornamento Python o aggiunta 'tzdata' a requirements.txt")
    ZoneInfo = None

# --- Layout App Streamlit ---
st.set_page_config(layout="wide", page_title="Crypto Technical Dashboard Pro", page_icon="📈")

# --- INIZIO: Codice CSS per ridurre font st.metric ---
st.markdown("""
<style>
div[data-testid="stMetricValue"] {
    font-size: 14px !important; /* Font size 14px */
}
</style>
""", unsafe_allow_html=True)
logger.info("CSS per font metriche applicato.")
# --- FINE: Codice CSS ---


# --- Configurazione Globale ---
logger.info("Inizio configurazione globale.")
SYMBOL_TO_ID_MAP = { "BTC": "bitcoin", "ETH": "ethereum" } # LISTA RIDOTTA PER TEST
SYMBOLS = list(SYMBOL_TO_ID_MAP.keys()); COINGECKO_IDS_LIST = list(SYMBOL_TO_ID_MAP.values())
NUM_COINS = len(SYMBOLS); logger.info(f"Numero coins configurate: {NUM_COINS}")
TRAD_TICKERS_AV = ['SPY', 'QQQ', 'GLD', 'NVDA', 'COIN'] # LISTA RIDOTTA PER TEST
logger.info(f"Tickers tradizionali configurati: {TRAD_TICKERS_AV}")
VS_CURRENCY = "usd"; CACHE_TTL = 1800; CACHE_HIST_TTL = CACHE_TTL * 2; CACHE_TRAD_TTL = 14400
DAYS_HISTORY_DAILY = 365; DAYS_HISTORY_HOURLY = 7
RSI_PERIOD = 14; SRSI_PERIOD = 14; SRSI_K = 3; SRSI_D = 3; MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9
MA_SHORT = 20; MA_LONG = 50; VWAP_PERIOD = 14
logger.info("Fine configurazione globale.")

# --- DEFINIZIONI FUNZIONI (Password, Format, API, Indicatori - SENZA IMPLEMENTAZIONE INDICATORI/SEGNALI) ---

def check_password():
    # [Codice invariato]
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
    # [Codice invariato]
    if pd.isna(num) or not isinstance(num, (int, float)): return "N/A"
    if abs(num) < 1_000_000: return f"{num:,.0f}"
    elif abs(num) < 1_000_000_000: return f"{num / 1_000_000:.1f}M"
    elif abs(num) < 1_000_000_000_000: return f"{num / 1_000_000_000:.1f}B"
    else: return f"{num / 1_000_000_000_000:.2f}T"

@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento dati di mercato (CoinGecko)...")
def get_coingecko_market_data(ids_list, currency):
    # [Codice invariato con logging]
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
        st.session_state["api_warning_shown"] = False; logger.info("Dati live CoinGecko recuperati."); return df, timestamp_utc
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 429: logger.warning("API CoinGecko (Live): Limite richieste (429)."); st.warning("Attenzione API CoinGecko (Live): Limite richieste (429) raggiunto."); st.session_state["api_warning_shown"] = True
        else: logger.error(f"Errore HTTP API Mercato CoinGecko (Status: {http_err.response.status_code}): {http_err}"); st.error(f"Errore HTTP API Mercato CoinGecko (Status: {http_err.response.status_code}): {http_err}")
        return pd.DataFrame(), timestamp_utc
    except requests.exceptions.RequestException as req_ex: logger.error(f"Errore Richiesta API Mercato CoinGecko: {req_ex}"); st.error(f"Errore Richiesta API Mercato CoinGecko: {req_ex}"); return pd.DataFrame(), timestamp_utc
    except Exception as e: logger.exception("Errore Processamento Dati Mercato CoinGecko:"); st.error(f"Errore Processamento Dati Mercato CoinGecko: {e}"); return pd.DataFrame(), timestamp_utc

@st.cache_data(ttl=CACHE_TRAD_TTL, show_spinner="Caricamento dati mercato tradizionale (Alpha Vantage)...")
def get_traditional_market_data_av(tickers):
    # [Codice invariato con logging]
    logger.info(f"Tentativo fetch dati Alpha Vantage per {len(tickers)} tickers.")
    data = {ticker: {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'} for ticker in tickers}; api_key = None;
    try: api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]; logger.info("Chiave API Alpha Vantage letta.");
    except KeyError: logger.error("Secret 'ALPHA_VANTAGE_API_KEY' non definito."); st.error("Secret 'ALPHA_VANTAGE_API_KEY' non definito."); return data
    except Exception as e: logger.exception("Errore lettura secrets Alpha Vantage:"); st.error(f"Errore lettura secrets: {e}"); return data
    if not api_key: logger.error("Chiave API Alpha Vantage vuota."); st.error("Chiave API Alpha Vantage vuota nei Secrets."); return data
    ts = TimeSeries(key=api_key, output_format='pandas'); calls_made = 0; max_calls_per_minute = 5; delay_between_calls = (60.0 / max_calls_per_minute) + 1.0
    for ticker_sym in tickers:
        if calls_made >= 25: msg = f"Limite giornaliero AV raggiunto. Stop fetch per {ticker_sym}+."; logger.warning(msg); st.warning(msg); break
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
             if "API call frequency" in str(ve) or "API key" in str(ve) or "limit" in str(ve).lower(): logger.error("Errore chiave/limite API AV."); st.error("Errore chiave/limite API AV."); break
             data[ticker_sym] = {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'}
        except Exception as e: msg = f"Errore generico AV per {ticker_sym}: {e}"; logger.exception(msg); st.warning(msg); data[ticker_sym] = {'price': np.nan, 'change': np.nan, 'change_percent': 'N/A'}
    logger.info("Fine fetch dati Alpha Vantage.")
    return data

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_fear_greed_index():
    # [Codice con sintassi corretta e logging]
    logger.info("Tentativo fetch Fear & Greed Index.")
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data and data.get("data") and isinstance(data["data"], list) and len(data["data"]) > 0:
             latest_data = data["data"][0]; value = latest_data.get("value"); desc = latest_data.get("value_classification")
             if value is not None and desc is not None: logger.info(f"Fear & Greed Index recuperato: {value} ({desc})."); return f"{int(value)} ({desc})"
        logger.warning("Formato dati F&G Index inatteso."); return "N/A"
    except requests.exceptions.RequestException as req_ex:
        status_code = req_ex.response.status_code if req_ex.response is not None else "N/A"; msg = f"Errore F&G Index (Alt.me Status: {status_code}): {req_ex}"; logger.warning(msg); st.warning(msg); return "N/A"
    except Exception as e: msg = f"Errore Processamento F&G Index (Alt.me): {e}"; logger.exception(msg); st.warning(msg); return "N/A"

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_global_market_data_cg(currency):
    # [Codice con sintassi corretta e logging]
    logger.info("Tentativo fetch dati Global CoinGecko.")
    url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json().get('data', {})
        total_mcap = data.get('total_market_cap', {}).get(currency.lower(), np.nan); logger.info(f"Dati Global CoinGecko recuperati (M.Cap: {total_mcap})."); return total_mcap
    except requests.exceptions.RequestException as req_ex: msg = f"Errore API Global CoinGecko: {req_ex}"; logger.warning(msg); st.warning(msg); return np.nan
    except Exception as e: msg = f"Errore Processamento Global CoinGecko: {e}"; logger.exception(msg); st.warning(msg); return np.nan

def get_etf_flow(): return "N/A"

# --- FUNZIONI INDICATORI E SEGNALI (SVUOTATE PER TEST) ---
def compute_all_indicators(symbol, hist_daily_df, hist_hourly_df):
    logger.info(f"Skipping indicator calculation for {symbol} (Test Mode).")
    return {} # Ritorna dizionario vuoto
def generate_gpt_signal(*args, **kwargs): return "N/D (Test)" # Ritorna N/D
def generate_gemini_alert(*args, **kwargs): return "N/D (Test)" # Ritorna N/D


# --- INIZIO ESECUZIONE PRINCIPALE APP ---
logger.info("Inizio esecuzione UI principale.")
try: # Blocco try principale per catturare errori runtime

    # --- Password Protection ---
    if not check_password(): st.stop()
    logger.info("Password check superato.")

    # --- TITOLO, BOTTONE, TIMESTAMP ---
    col_title, col_button_placeholder, col_button = st.columns([4, 1, 1])
    with col_title: st.title("📈 Crypto Dashboard (TEST COMPILAZIONE)") # Titolo modificato per test
    with col_button:
        st.write("")
        if st.button("🔄 Aggiorna", help=f"Forza aggiornamento dati", key="refresh_button"):
            logger.info("Bottone Aggiorna cliccato."); st.cache_data.clear(); st.query_params.clear(); st.rerun()
    last_update_placeholder = st.empty()
    st.caption(f"Cache: Crypto Live ({CACHE_TTL/60:.0f}m), Storico ({CACHE_HIST_TTL/60:.0f}m), Tradizionale ({CACHE_TRAD_TTL/3600:.0f}h).")

    # --- SEZIONE MARKET OVERVIEW ---
    st.markdown("---"); st.subheader("🌐 Market Overview (Test)")
    fear_greed_value = get_fear_greed_index()
    total_market_cap = get_global_market_data_cg(VS_CURRENCY)
    etf_flow_value = get_etf_flow()
    traditional_market_data = get_traditional_market_data_av(TRAD_TICKERS_AV) # Chiama AV

    def format_delta(change_val, change_pct_str):
        delta_string = None;
        if pd.notna(change_val) and isinstance(change_pct_str, str) and change_pct_str not in ['N/A', '', None]:
            try: change_pct_val = float(change_pct_str.replace('%','').strip()); delta_string = f"{change_val:+.2f} ({change_pct_val:+.2f}%)"
            except (ValueError, AttributeError): delta_string = f"{change_val:+.2f} (?%)"
        elif pd.notna(change_val): delta_string = f"{change_val:+.2f}"
        return delta_string

    # Layout Market Overview (2x5)
    overview_items_row1 = [ ("Fear & Greed Index", None, get_fear_greed_index, "Fonte: Alternative.me"), (f"Total Crypto M.Cap ({VS_CURRENCY.upper()})", None, lambda: f"${format_large_number(total_market_cap)}", "Fonte: CoinGecko"), ("Crypto ETFs Flow (Daily)", None, get_etf_flow, "Dato N/A"), ("S&P 500 (SPY)", "SPY", None, "Fonte: Alpha Vantage"), ("Nasdaq (QQQ)", "QQQ", None, "Fonte: Alpha Vantage")]
    overview_items_row2 = [ ("Gold (GLD)", "GLD", None, "Fonte: Alpha Vantage"), ("Silver (SLV)", "SLV", None, "Fonte: Alpha Vantage"), ("Natural Gas (UNG)", "UNG", None, "Fonte: Alpha Vantage"), ("UVXY (Volatility)", "UVXY", None, "Fonte: Alpha Vantage"), ("TQQQ (Nasdaq 3x)", "TQQQ", None, "Fonte: Alpha Vantage")]
    overview_cols_1 = st.columns(5)
    for i, (label, ticker, func, help_text) in enumerate(overview_items_row1):
        with overview_cols_1[i]:
            if ticker: trad_info = traditional_market_data.get(ticker, {}); price = trad_info.get('price', np.nan); change = trad_info.get('change', np.nan); change_pct = trad_info.get('change_percent', 'N/A'); value_str = f"{price:,.2f}" if pd.notna(price) else "N/A"; delta_txt = format_delta(change, change_pct); d_color = "off";
            else: value_str = func() if func else "N/A"; delta_txt = None; d_color = "off"
            if pd.notna(change): d_color = "normal"
            st.metric(label=label, value=value_str, delta=delta_txt, delta_color=d_color, help=help_text)
    overview_cols_2 = st.columns(5)
    for i, (label, ticker, func, help_text) in enumerate(overview_items_row2):
         with overview_cols_2[i]:
            trad_info = traditional_market_data.get(ticker, {}); price = trad_info.get('price', np.nan); change = trad_info.get('change', np.nan); change_pct = trad_info.get('change_percent', 'N/A'); value_str = f"{price:,.2f}" if pd.notna(price) else "N/A"; delta_txt = format_delta(change, change_pct); d_color = "off";
            if pd.notna(change): d_color = "normal"
            st.metric(label=label, value=value_str, delta=delta_txt, delta_color=d_color, help=help_text)

    # Titoli Principali
    st.markdown("<h6>Titoli Principali (Prezzi e Var. Giorno - Test):</h6>", unsafe_allow_html=True)
    stock_tickers_row_av = ['NVDA', 'GOOGL', 'AAPL', 'META', 'TSLA', 'MSFT', 'TSM', 'PLTR', 'COIN', 'MSTR']
    num_stock_cols = 5; stock_cols = st.columns(num_stock_cols)
    for idx, ticker in enumerate(stock_tickers_row_av):
        col_index = idx % num_stock_cols; current_col = stock_cols[col_index]
        with current_col:
            trad_info = traditional_market_data.get(ticker, {}); price = trad_info.get('price', np.nan); change = trad_info.get('change', np.nan); change_pct = trad_info.get('change_percent', 'N/A'); value_str = f"{price:,.2f}" if pd.notna(price) else "N/A"; delta_txt = format_delta(change, change_pct); d_color = "off";
            if pd.notna(change): d_color = "normal"
            st.metric(label=ticker, value=value_str, delta=delta_txt, delta_color=d_color)
    st.markdown("---")

    # --- LOGICA PRINCIPALE DASHBOARD CRYPTO (COMMENTATA) ---
    st.subheader(f"📊 Analisi Tecnica Crypto ({NUM_COINS} Asset) - Sezione Disabilitata per Test")
    logger.info("Skipping main crypto processing loop for compilation test.")
    st.info("La tabella di analisi crypto è disabilitata in questa modalità di test.")
    # market_data_df, last_cg_update_utc = get_coingecko_market_data(COINGECKO_IDS_LIST, VS_CURRENCY)
    # ... (Gestione Timestamp commentata) ...
    # if market_data_df.empty: ... st.stop() ...
    # ... (CICLO PROCESSING PER OGNI COIN COMMENTATO) ...
    # ... (CREA E VISUALIZZA DATAFRAME COMMENTATO) ...
    # ... (EXPANDER ERRORI/NOTE COMMENTATO) ...


    # --- SEZIONE NEWS (RIMOSSA) ---


    # --- LEGENDA ---
    st.divider()
    with st.expander("📘 Legenda Indicatori Tecnici e Segnali", expanded=False):
        st.markdown("""*(Legenda standard...)*""") # Legenda ridotta per test

    # --- Footer ---
    st.divider()
    st.caption("Disclaimer: Strumento a scopo informativo/didattico...")


except Exception as main_exception:
    logger.exception("!!! ERRORE NON GESTITO NELL'ESECUZIONE PRINCIPALE DELL'APP !!!")
    st.error("Si è verificato un errore imprevisto nell'applicazione. Controlla il log qui sotto per dettagli tecnici.")


# --- VISUALIZZAZIONE LOG APPLICAZIONE (Sempre alla fine) ---
st.divider()
st.subheader("📄 Log Applicazione")
st.caption("Questa sezione mostra i log generati durante l'esecuzione. Utile per il debug.")
log_content = log_stream.getvalue()
st.text_area("Log:", value=log_content, height=400, key="log_display_area", help="Puoi selezionare tutto (Ctrl+A o Cmd+A) e copiare (Ctrl+C o Cmd+C) questo log per condividerlo.")
logger.info("--- Fine esecuzione script app.py (Versione Semplificata) ---")
log_stream.close()