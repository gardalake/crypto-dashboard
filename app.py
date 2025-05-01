# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time
import math
import yfinance as yf # Aggiunto per dati mercato tradizionale

# --- Configurazione Globale ---

# Mappa Simbolo -> ID CoinGecko per facile gestione
# AGGIUNGI/RIMUOVI coppie qui per cambiare le coin monitorate
SYMBOL_TO_ID_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "RNDR": "render-token",
    "FET": "artificial-superintelligence-alliance", # ID post-merge ASI Alliance
    "TAO": "bittensor", "LINK": "chainlink", "AVAX": "avalanche-2",
    "HBAR": "hedera-hashgraph", "PEPE": "pepe", "UNI": "uniswap",
    "TIA": "celestia", "JUP": "jupiter-aggregator", "IMX": "immutable-x",
    "TRUMP": "maga", "NEAR": "near-protocol", "AERO": "aerodrome-finance",
    "TRON": "tron", "AERGO": "aergo", "ADA": "cardano", "MKR": "maker"
}

# Deriva dinamicamente la lista di simboli e ID dalla mappa
SYMBOLS = list(SYMBOL_TO_ID_MAP.keys())
COINGECKO_IDS_LIST = list(SYMBOL_TO_ID_MAP.values())
NUM_COINS = len(SYMBOLS)

# Ticker per mercati tradizionali (yfinance)
TRAD_TICKERS = ['^GSPC', '^IXIC', 'GC=F', 'UVXY', 'TQQQ'] # S&P500, Nasdaq, Gold, UVXY, TQQQ

VS_CURRENCY = "usd" # Valuta di riferimento
CACHE_TTL = 1800 # NUOVO: Cache di 30 minuti (1800 sec) per dati API
DAYS_HISTORY_DAILY = 365
DAYS_HISTORY_HOURLY = 7

# Periodi Indicatori
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
             # Usa st.rerun() solo se la password è CORRETTA la prima volta
             # per evitare loop infiniti se l'utente sbaglia e poi corregge
             if st.query_params.get("rerun") != "false":
                  st.query_params["rerun"] = "false" # Evita rerun multipli
                  st.rerun()
        elif password: st.warning("Password errata."); st.stop()
        else: st.stop()
    return True
if not check_password(): st.stop()

# --- Funzioni API e Calcolo Indicatori (Manuali) ---
# (calculate_rsi_manual, calculate_stoch_rsi, calculate_macd_manual,
#  calculate_sma_manual, calculate_vwap_manual - invariate dalla versione precedente)
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
    rs = last_avg_gain / last_avg_loss; rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_stoch_rsi(series, rsi_period=RSI_PERIOD, stoch_period=SRSI_PERIOD, k_smooth=SRSI_K, d_smooth=SRSI_D):
    rsi_series = pd.Series(dtype=float) # Inizializza come Series vuota
    if isinstance(series, pd.Series) and not series.isna().all():
        delta = series.diff(); gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
        avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi_series_calc = 100.0 - (100.0 / (1.0 + rs))
        rsi_series = rsi_series_calc.dropna()

    if len(rsi_series) < stoch_period: return np.nan, np.nan
    min_rsi = rsi_series.rolling(window=stoch_period).min()
    max_rsi = rsi_series.rolling(window=stoch_period).max()
    range_rsi = max_rsi - min_rsi; range_rsi[range_rsi == 0] = np.nan
    stoch_rsi_k = 100 * (rsi_series - min_rsi) / range_rsi
    stoch_rsi_k = stoch_rsi_k.dropna()
    if len(stoch_rsi_k) < k_smooth : return np.nan, np.nan
    stoch_rsi_k_smoothed = stoch_rsi_k.rolling(window=k_smooth).mean()
    if len(stoch_rsi_k_smoothed.dropna()) < d_smooth : return stoch_rsi_k_smoothed.iloc[-1], np.nan
    stoch_rsi_d = stoch_rsi_k_smoothed.rolling(window=d_smooth).mean()
    last_k = stoch_rsi_k_smoothed.iloc[-1] if not pd.isna(stoch_rsi_k_smoothed.iloc[-1]) else np.nan
    last_d = stoch_rsi_d.iloc[-1] if not pd.isna(stoch_rsi_d.iloc[-1]) else np.nan
    return last_k, last_d

def calculate_macd_manual(series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan, np.nan, np.nan
    series = series.dropna(); len_series = len(series)
    if len_series < slow : return np.nan, np.nan, np.nan
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    if len(macd_line.dropna()) < signal: return macd_line.iloc[-1], np.nan, np.nan
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    last_macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else np.nan
    last_signal = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else np.nan
    last_hist = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else np.nan
    return last_macd, last_signal, last_hist

def calculate_sma_manual(series, period):
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan
    series = series.dropna()
    if len(series) < period: return np.nan
    return series.rolling(window=period).mean().iloc[-1]

def calculate_vwap_manual(df, period=VWAP_PERIOD):
    if not all(col in df.columns for col in ['close', 'volume']) or df['close'].isna().all() or df['volume'].isna().all(): return np.nan
    df_period = df.iloc[-period:].copy(); df_period.dropna(subset=['close', 'volume'], inplace=True)
    if df_period.empty or df_period['volume'].sum() == 0: return np.nan
    vwap = (df_period['close'] * df_period['volume']).sum() / df_period['volume'].sum()
    return vwap

# --- Funzioni Segnale (Gemini Alert modificato per stile) ---
def generate_gpt_signal(rsi_1d, rsi_1h, rsi_1w, macd_hist, ma_short, ma_long, srsi_k, srsi_d, current_price):
    required_inputs = [rsi_1d, macd_hist, ma_short, ma_long, current_price]
    if any(pd.isna(x) for x in required_inputs): return "⚪️ N/D"
    score = 0
    # Logica Punteggio (invariata, include SRSI ora)
    if current_price > ma_long: score += 1; else: score -= 1
    if ma_short > ma_long: score += 2; else: score -= 2
    if macd_hist > 0: score += 2; else: score -= 2
    if rsi_1d < 30: score += 2; elif rsi_1d < 40: score += 1; elif rsi_1d > 70: score -= 2; elif rsi_1d > 60: score -= 1
    if not pd.isna(rsi_1w):
        if rsi_1w < 30: score += 1; elif rsi_1w > 70: score -= 1
    if not pd.isna(rsi_1h):
        if rsi_1h > 60: score += 1; elif rsi_1h < 40: score -= 1
    if not pd.isna(srsi_k) and not pd.isna(srsi_d):
        if srsi_k < 20 and srsi_d < 20: score += 1
        elif srsi_k > 80 and srsi_d > 80: score -= 1
    # Mappatura Score (con CTB/CTS)
    if score >= 5: return "⚡️ Strong Buy"; elif score >= 2: return "🟢 Buy"
    elif score <= -5: return "🚨 Strong Sell"; elif score <= -2: return "🔴 Sell"
    elif score >= 0: return "⏳ CTB" if not pd.isna(rsi_1d) and rsi_1d < 45 and rsi_1d > 30 else "🟡 Hold"
    else: return "⚠️ CTS" if not pd.isna(rsi_1d) and rsi_1d > 55 and rsi_1d < 70 else "🟡 Hold"

def generate_gemini_alert(ma_short, ma_long, macd_hist, rsi_1d):
    """Genera un alert specifico basato su forte confluenza DAILY (restituisce emoji)."""
    if pd.isna(ma_short) or pd.isna(ma_long) or pd.isna(macd_hist) or pd.isna(rsi_1d): return "⏳" # Dati insuff.
    is_uptrend = ma_short > ma_long; is_momentum_positive = macd_hist > 0; is_not_overbought = rsi_1d < 70
    if is_uptrend and is_momentum_positive and is_not_overbought: return "✅" # BUY Emoji
    is_downtrend = ma_short < ma_long; is_momentum_negative = macd_hist < 0; is_not_oversold = rsi_1d > 30
    if is_downtrend and is_momentum_negative and is_not_oversold: return "❌" # SELL Emoji
    return "➖" # Neutro Emoji

# --- Funzioni Fetch Dati Mercato Generale ---

@st.cache_data(ttl=CACHE_TTL, show_spinner=False) # Cache 30 min
def get_fear_greed_index_cmc():
    """Tenta di ottenere il Fear & Greed Index da CoinMarketCap."""
    # Nota: CMC potrebbe richiedere chiave API per questo endpoint in futuro o bloccarlo.
    # Il piano gratuito CMC potrebbe o non potrebbe includerlo. Testiamo senza chiave.
    url = "https://pro-api.coinmarketcap.com/v1/crypto/fear-and-greed"
    # Headers potrebbero essere necessari se serve API Key
    # headers = {'X-CMC_PRO_API_KEY': 'TUA_CHIAVE_API_CMC'}
    try:
        # response = requests.get(url, headers=headers, timeout=10) # Se usi chiave API
        response = requests.get(url, timeout=10) # Prova senza chiave
        response.raise_for_status()
        data = response.json()
        # Estrai il valore più recente
        if data and data.get("data") and isinstance(data["data"], list) and len(data["data"]) > 0:
             latest_data = data["data"][0]
             value = latest_data.get("score")
             desc = latest_data.get("rating")
             # timestamp = latest_data.get("timestamp") # Potrebbe essere utile
             if value is not None and desc is not None:
                  return f"{int(value)} ({desc})"
        return "N/A"
    except requests.exceptions.HTTPError as http_err:
         if http_err.response.status_code == 401 or http_err.response.status_code == 403:
              return "N/A (API Key?)" # Probabile richiesta chiave API
         else:
              return f"N/A (Err {http_err.response.status_code})"
    except Exception:
        return "N/A (Errore)"

@st.cache_data(ttl=CACHE_TTL, show_spinner=False) # Cache 30 min
def get_global_market_data_cg(currency):
    """Ottiene dati globali (Total Market Cap) da CoinGecko."""
    url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get('data', {})
        total_mcap = data.get('total_market_cap', {}).get(currency.lower(), np.nan)
        # Altri dati disponibili: total_volume, market_cap_percentage, etc.
        return total_mcap
    except Exception:
        return np.nan

def get_etf_flow():
    """Placeholder per ETF Flow - Dato difficile da ottenere via API gratuita."""
    return "N/A"

@st.cache_data(ttl=CACHE_TTL, show_spinner=False) # Cache 30 min
def get_traditional_market_data_yf(tickers):
    """Ottiene dati (prezzo approx) per ticker tradizionali via yfinance."""
    data = {}
    for ticker_sym in tickers:
        try:
            time.sleep(0.2) # Piccolo delay anche per yfinance
            ticker_obj = yf.Ticker(ticker_sym)
            info = ticker_obj.info # .info può essere lento e fallire
            # Cerca diversi campi prezzo per robustezza
            price = info.get('regularMarketPrice', info.get('currentPrice', info.get('previousClose', np.nan)))
            data[ticker_sym] = price
        except Exception:
            data[ticker_sym] = np.nan # Fallback a NaN
    return data

# --- NUOVA Funzione Raggruppata per Calcolare TUTTI gli Indicatori ---
def compute_all_indicators(symbol, hist_daily_df, hist_hourly_df, fetch_errors_list):
    """Calcola tutti gli indicatori per una coin e gestisce errori."""
    indicators = { # Inizializza con NaN
        "RSI (1h)": np.nan, "RSI (1d)": np.nan, "RSI (1w)": np.nan, "RSI (1mo)": np.nan,
        "SRSI %K (1d)": np.nan, "SRSI %D (1d)": np.nan,
        "MACD Line (1d)": np.nan, "MACD Signal (1d)": np.nan, "MACD Hist (1d)": np.nan,
        f"MA({MA_SHORT}d)": np.nan, f"MA({MA_LONG}d)": np.nan, "VWAP (1d)": np.nan,
        "Doda Stoch": "N/A", "GChannel": "N/A", "Volume Flow": "N/A", # Placeholder
    }
    min_len_daily_full = max(RSI_PERIOD + 1, SRSI_PERIOD + RSI_PERIOD + 1, MACD_SLOW + MACD_SIGNAL, MA_LONG)
    min_len_rsi_base = RSI_PERIOD + 1

    # Calcoli su Dati Daily
    if not hist_daily_df.empty:
        close_daily = hist_daily_df['close']
        len_daily = len(close_daily.dropna()) # Usa lunghezza dati validi

        if len_daily >= min_len_daily_full:
            indicators["RSI (1d)"] = calculate_rsi_manual(close_daily)
            indicators["SRSI %K (1d)"], indicators["SRSI %D (1d)"] = calculate_stoch_rsi(close_daily)
            macd_l, macd_s, macd_h = calculate_macd_manual(close_daily)
            indicators["MACD Line (1d)"] = macd_l; indicators["MACD Signal (1d)"] = macd_s; indicators["MACD Hist (1d)"] = macd_h
            indicators[f"MA({MA_SHORT}d)"] = calculate_sma_manual(close_daily, MA_SHORT)
            indicators[f"MA({MA_LONG}d)"] = calculate_sma_manual(close_daily, MA_LONG)
            indicators["VWAP (1d)"] = calculate_vwap_manual(hist_daily_df)
        else: fetch_errors_list.append(f"Dati Daily insuff. ({len_daily}/{min_len_daily_full}) per ind. base per {symbol}.")

        # Weekly (richiede meno storico)
        try:
            df_weekly = close_daily.resample('W-MON').last()
            if len(df_weekly.dropna()) >= min_len_rsi_base: indicators["RSI (1w)"] = calculate_rsi_manual(df_weekly)
            else: fetch_errors_list.append(f"Dati insuff. ({len(df_weekly.dropna())}/{min_len_rsi_base} sett.) per RSI 1w per {symbol}.")
        except Exception as e: fetch_errors_list.append(f"Errore resampling weekly per {symbol}: {e}")

        # Monthly
        try:
            df_monthly = close_daily.resample('ME').last()
            if len(df_monthly.dropna()) >= min_len_rsi_base: indicators["RSI (1mo)"] = calculate_rsi_manual(df_monthly)
            else: fetch_errors_list.append(f"Dati insuff. ({len(df_monthly.dropna())}/{min_len_rsi_base} mesi) per RSI 1mo per {symbol}.")
        except Exception as e: fetch_errors_list.append(f"Errore resampling monthly per {symbol}: {e}")

    # Calcoli su Dati Hourly
    if not hist_hourly_df.empty:
        len_hourly = len(hist_hourly_df['close'].dropna())
        if len_hourly >= min_len_rsi_base: indicators["RSI (1h)"] = calculate_rsi_manual(hist_hourly_df['close'])
        else: fetch_errors_list.append(f"Dati Hourly insuff. ({len_hourly}/{min_len_rsi_base} ore) per RSI 1h per {symbol}.")

    return indicators

# --- Configurazione Pagina Streamlit ---
st.set_page_config(layout="wide", page_title="Crypto Technical Dashboard Pro", page_icon="📈")

# --- Layout Titolo e Bottone Aggiorna ---
col_title, col_button = st.columns([4, 1]) # Dà più spazio al titolo
with col_title:
    st.title("📈 Crypto Technical Dashboard Pro") # Titolo aggiornato
with col_button:
    st.write("") # Spazio vuoto per allineare verticalmente il bottone
    st.write("")
    if st.button("🔄 Aggiorna", help="Forza l'aggiornamento dei dati"):
        st.cache_data.clear(); st.rerun()

# --- Timestamp Aggiornamento Live ---
last_cg_update_placeholder = st.empty()
# Caption generale spostata qui
st.caption(f"Dati live/storici CoinGecko (cache {CACHE_TTL/60:.0f}min live / {CACHE_TTL*2/60:.0f}min storico). Dati mercato trad. yfinance (cache {CACHE_TTL/60:.0f}min).")

# --- NUOVA SEZIONE: Market Overview ---
st.markdown("---") # Separatore
st.subheader("🌐 Market Overview")

# Fetch dati mercato generale (con cache)
fear_greed = get_fear_greed_index_cmc()
total_mcap = get_global_market_data_cg(VS_CURRENCY)
etf_flow = get_etf_flow() # Placeholder N/A
trad_data = get_traditional_market_data_yf(TRAD_TICKERS)

# Visualizzazione in colonne con st.metric
mkt_col1, mkt_col2, mkt_col3, mkt_col4 = st.columns(4)
with mkt_col1:
    st.metric(label="Fear & Greed Index (CMC)", value=fear_greed)
    st.metric(label="S&P 500 (^GSPC)",
              value=f"{trad_data.get('^GSPC', 0):,.2f}" if not pd.isna(trad_data.get('^GSPC')) else "N/A")
with mkt_col2:
    st.metric(label=f"Total Crypto Market Cap ({VS_CURRENCY.upper()})",
              value=f"<span class="math-inline">\{total\_mcap\:,\.0f\}" if not pd\.isna\(total\_mcap\) else "N/A"\)
st\.metric\(label\="Nasdaq \(^IXIC\)",
value\=f"\{trad\_data\.get\('^IXIC', 0\)\:,\.2f\}" if not pd\.isna\(trad\_data\.get\('^IXIC'\)\) else "N/A"\)
with mkt\_col3\:
st\.metric\(label\="Crypto ETFs Net Flow \(Daily\)", value\=etf\_flow,
help\="Dato non disponibile tramite API gratuite affidabili al momento\."\)
st\.metric\(label\="Gold \(GC\=F\)",
value\=f"\{trad\_data\.get\('GC\=F', 0\)\:,\.2f\}" if not pd\.isna\(trad\_data\.get\('GC\=F'\)\) else "N/A"\)
with mkt\_col4\:
st\.metric\(label\="UVXY \(Volatility\)",
value\=f"\{trad\_data\.get\('UVXY', 0\)\:,\.2f\}" if not pd\.isna\(trad\_data\.get\('UVXY'\)\) else "N/A"\)
st\.metric\(label\="TQQQ \(Nasdaq 3x\)",
value\=f"\{trad\_data\.get\('TQQQ', 0\)\:,\.2f\}" if not pd\.isna\(trad\_data\.get\('TQQQ'\)\) else "N/A"\)
st\.markdown\("\-\-\-"\) \# Separatore
\# \-\-\- Logica Principale Dashboard Crypto \-\-\-
market\_data\_df, last\_cg\_update\_utc \= get\_coingecko\_market\_data\(COINGECKO\_IDS\_LIST, VS\_CURRENCY\)
last\_cg\_update\_rome \= last\_cg\_update\_utc \+ timedelta\(hours\=2\)
last\_cg\_update\_placeholder\.markdown\(f"\*Prezzi Live aggiornati alle\: \*\*\{last\_cg\_update\_rome\.strftime\('%Y\-%m\-%d %H\:%M\:%S'\)\} \(Ora di Roma\)\*\*\*"\)
if market\_data\_df\.empty\: st\.error\("Errore caricamento dati crypto\. Impossibile continuare\."\); st\.stop\(\)
results \= \[\]; fetch\_errors \= \[\]
progress\_bar \= st\.progress\(0, text\="Analisi Criptovalute\.\.\."\)
coin\_ids\_ordered \= market\_data\_df\.index\.tolist\(\)
for i, coin\_id in enumerate\(coin\_ids\_ordered\)\:
live\_data \= market\_data\_df\.loc\[coin\_id\]
symbol \= live\_data\.get\('symbol', coin\_id\)\.upper\(\); name \= live\_data\.get\('name', coin\_id\)
rank \= live\_data\.get\('market\_cap\_rank', 'N/A'\); current\_price \= live\_data\.get\('current\_price', np\.nan\)
\# \-\-\- AGGIUNTA NUOVE VARIAZIONI % \-\-\-
change\_1h \= live\_data\.get\('price\_change\_percentage\_1h\_in\_currency', np\.nan\)
change\_24h \= live\_data\.get\('price\_change\_percentage\_24h\_in\_currency', np\.nan\)
change\_7d \= live\_data\.get\('price\_change\_percentage\_7d\_in\_currency', np\.nan\)
change\_30d \= live\_data\.get\('price\_change\_percentage\_30d\_in\_currency', np\.nan\) \# Da richiedere nell'API
change\_1y \= live\_data\.get\('price\_change\_percentage\_1y\_in\_currency', np\.nan\)   \# Da richiedere nell'API
volume\_24h \= live\_data\.get\('total\_volume', np\.nan\)
hist\_daily\_df, status\_daily \= get\_coingecko\_historical\_data\(coin\_id, VS\_CURRENCY, DAYS\_HISTORY\_DAILY, interval\='daily'\)
hist\_hourly\_df, status\_hourly \= get\_coingecko\_historical\_data\(coin\_id, VS\_CURRENCY, DAYS\_HISTORY\_HOURLY, interval\='hourly'\)
if status\_daily \!\= "Success"\: fetch\_errors\.append\(f"\{symbol\}\: Dati Daily \- \{status\_daily\}"\)
if status\_hourly \!\= "Success"\: fetch\_errors\.append\(f"\{symbol\}\: Dati Hourly \- \{status\_hourly\}"\)
\# Calcola tutti gli indicatori
indicators \= compute\_all\_indicators\(symbol, hist\_daily\_df, hist\_hourly\_df, fetch\_errors\)
\# Calcola i segnali
gpt\_signal \= generate\_gpt\_signal\(
indicators\.get\("RSI \(1d\)"\), indicators\.get\("RSI \(1h\)"\), indicators\.get\("RSI \(1w\)"\),
indicators\.get\("MACD Hist \(1d\)"\), indicators\.get\(f"MA\(\{MA\_SHORT\}d\)"\),
indicators\.get\(f"MA\(\{MA\_LONG\}d\)"\), indicators\.get\("SRSI %K \(1d\)"\), \# Aggiunto SRSI K
indicators\.get\("SRSI %D \(1d\)"\), current\_price \# Aggiunto SRSI D
\)
gemini\_alert \= generate\_gemini\_alert\(
indicators\.get\(f"MA\(\{MA\_SHORT\}d\)"\), indicators\.get\(f"MA\(\{MA\_LONG\}d\)"\),
indicators\.get\("MACD Hist \(1d\)"\), indicators\.get\("RSI \(1d\)"\)
\)
\# \-\-\- Assembla Risultati riga \(con nuove colonne %\) \-\-\-
row\_data \= \{
"Rank"\: rank, "Symbol"\: symbol, "Name"\: name, "Gemini Alert"\: gemini\_alert, "GPT Signal"\: gpt\_signal,
f"Prezzo \(\{VS\_CURRENCY\.upper\(\)\}\)"\: current\_price,
"% 1h"\: change\_1h, "% 24h"\: change\_24h, "% 7d"\: change\_7d, "% 30d"\: change\_30d, "% 1y"\: change\_1y, \# Aggiunte % mese/anno
"RSI \(1h\)"\: indicators\.get\("RSI \(1h\)"\), "RSI \(1d\)"\: indicators\.get\("RSI \(1d\)"\),
"RSI \(1w\)"\: indicators\.get\("RSI \(1w\)"\), "RSI \(1mo\)"\: indicators\.get\("RSI \(1mo\)"\),
"SRSI %K \(1d\)"\: indicators\.get\("SRSI %K \(1d\)"\), "SRSI %D \(1d\)"\: indicators\.get\("SRSI %D \(1d\)"\),
"MACD Hist \(1d\)"\: indicators\.get\("MACD Hist \(1d\)"\),
f"MA\(\{MA\_SHORT\}d\)"\: indicators\.get\(f"MA\(\{MA\_SHORT\}d\)"\), f"MA\(\{MA\_LONG\}d\)"\: indicators\.get\(f"MA\(\{MA\_LONG\}d\)"\),
"VWAP \(1d\)"\: indicators\.get\("VWAP \(1d\)"\), f"Volume 24h \(\{VS\_CURRENCY\.upper\(\)\}\)"\: volume\_24h,
"Doda Stoch"\: "N/A", "GChannel"\: "N/A", "Volume Flow"\: "N/A", \# Placeholder
\}
results\.append\(row\_data\)
progress\_bar\.progress\(\(i \+ 1\) / len\(coin\_ids\_ordered\), text\=f"Analisi Criptovalute\.\.\. \(\{symbol\}\)"\)
progress\_bar\.empty\(\)
if fetch\_errors\:
with st\.expander\("ℹ️ Note Recupero Dati / Calcolo Indicatori", expanded\=False\)\:
unique\_errors \= sorted\(list\(set\(fetch\_errors\)\)\);
for error\_msg in unique\_errors\: st\.info\(error\_msg\)
if results\:
df \= pd\.DataFrame\(results\); df\.set\_index\('Rank', inplace\=True\)
\# Ordine colonne aggiornato
cols\_order \= \[
"Symbol", "Name", "Gemini Alert", "GPT Signal",
f"Prezzo \(\{VS\_CURRENCY\.upper\(\)\}\)",
"% 1h", "% 24h", "% 7d", "% 30d", "% 1y", \# Aggiunte % mese/anno
"RSI \(1h\)", "RSI \(1d\)", "RSI \(1w\)", "RSI \(1mo\)",
"SRSI %K \(1d\)", "SRSI %D \(1d\)",
"MACD Hist \(1d\)", f"MA\(\{MA\_SHORT\}d\)", f"MA\(\{MA\_LONG\}d\)", "VWAP \(1d\)",
f"Volume 24h \(\{VS\_CURRENCY\.upper\(\)\}\)"
\]
cols\_to\_show \= \[col for col in cols\_order if col in df\.columns\]
df\_display \= df\[cols\_to\_show\]\.copy\(\)
\# Formattazione manuale \(include nuove colonne %\)
formatters \= \{\}; currency\_col \= f"Prezzo \(\{VS\_CURRENCY\.upper\(\)\}\)"
volume\_col \= f"Volume 24h \(\{VS\_CURRENCY\.upper\(\)\}\)";
pct\_cols \= \["% 1h", "% 24h", "% 7d", "% 30d", "% 1y"\] \# Aggiornata lista %
for col in df\_display\.columns\:
if pd\.api\.types\.is\_numeric\_dtype\(df\_display\[col\]\.infer\_objects\(\)\.dtype\)\:
if col \=\= currency\_col\: formatters\[col\] \= lambda x\: f"</span>{x:,.4f}" if pd.notna(x) else "N/A"
            elif col in pct_cols: formatters[col] = lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
            elif col == volume_col: formatters[col] = lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
            elif "RSI" in col or "SRSI" in col: formatters[col] = lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
            elif "MACD" in col: formatters[col] = lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            elif "MA" in col or "VWAP" in col: formatters[col] = lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
            else: formatters[col] = lambda x: str(x) if pd.notna(x) else "N/A"
            df_display[col] = df_display[col].apply(formatters[col])
        elif df_display[col].dtype == 'object': df_display[col].fillna("N/A", inplace=True)
    df_display.fillna("N/A", inplace=True)

    # Stili (Gemini Alert modificato)
    def highlight_pct_col(col): # Funzione stile %
        colors = [''] * len(col)
        for i, val in enumerate(col):
            if isinstance(val, str) and val.endswith('%') and val != 'N/A':
                try: num = float(val.replace('%','')); colors[i] = 'color: green' if num > 0 else 'color: red' if num