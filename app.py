# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import numpy as np
from datetime import datetime
import time # Per throttling richieste yfinance (anche se la cache aiuta)

# --- Configurazione Globale ---
SYMBOLS_YF = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"] # Tickers per yFinance
COINGECKO_IDS = "bitcoin,ethereum,binancecoin,solana,ripple" # IDs per CoinGecko API
VS_CURRENCY = "usd"
CACHE_TTL = 300 # Cache di 5 minuti (300 sec)
YF_HISTORY_PERIOD = "6mo" # Aumentato periodo storico per robustezza indicatori
YF_HISTORY_INTERVAL = "1d"

# Periodi Indicatori
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MA_SHORT = 20
MA_LONG = 50

# --- Funzioni Fetch Dati ---

@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento prezzi live (CoinGecko)...")
def get_current_prices_coingecko(ids, currency):
    """Ottiene i prezzi correnti da CoinGecko."""
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ids,
        "vs_currencies": currency
    }
    prices = {}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        for coin_id, price_data in data.items():
            prices[coin_id] = price_data.get(currency, np.nan)
        return prices, datetime.now() # Ritorna anche timestamp
    except requests.exceptions.RequestException as e:
        st.error(f"Errore API CoinGecko: {e}")
        return {}, datetime.now()
    except Exception as e:
        st.error(f"Errore processamento dati CoinGecko: {e}")
        return {}, datetime.now()

@st.cache_data(ttl=CACHE_TTL * 2, show_spinner=False) # Cache più lunga per dati storici
def get_price_history_yf(symbol, period=YF_HISTORY_PERIOD, interval=YF_HISTORY_INTERVAL):
    """Ottiene dati storici da yFinance."""
    # time.sleep(0.1) # Piccolo delay opzionale
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval, auto_adjust=True)
        data.rename(columns={"Close": "price"}, inplace=True, errors='ignore') # Rinomina se 'Close' esiste
        # Assicurati che la colonna 'price' esista
        if 'price' not in data.columns:
             # Fallback se auto_adjust=False o per altre ragioni 'Close' non c'è
             if 'Close' in data.columns:
                  data['price'] = data['Close']
             else:
                  # st.warning(f"Colonna 'price' o 'Close' non trovata per {symbol}")
                  return pd.DataFrame() # Ritorna vuoto se non c'è prezzo

        # Rimuovi eventuali righe con NaN nel prezzo (potrebbero invalidare i calcoli)
        data.dropna(subset=['price'], inplace=True)

        if data.empty:
             # st.info(f"Debug: Nessun dato storico valido trovato da yFinance per {symbol} ({period}/{interval})")
             return pd.DataFrame()

        return data[['price']] # Ritorna solo la colonna prezzo valida
    except Exception as e:
        # st.warning(f"Debug: Errore yfinance per {symbol}: {e}")
        return pd.DataFrame()

# --- Funzioni Calcolo Indicatori ---

def calculate_rsi(series, period=RSI_PERIOD):
    """Calcola l'RSI in modo robusto."""
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan
    series = series.dropna() # Lavora solo sui dati validi
    if len(series) < period + 1: return np.nan # Non ci sono abbastanza dati

    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0) # Assicura che loss sia sempre >= 0

    # Usa EWM (Exponential Moving Average) per calcolare media di gain/loss - più stabile di SMA per RSI
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Recupera l'ultimo valore calcolato
    last_avg_gain = avg_gain.iloc[-1]
    last_avg_loss = avg_loss.iloc[-1]

    if pd.isna(last_avg_gain) or pd.isna(last_avg_loss): return np.nan # Se EWM non ha abbastanza dati

    if last_avg_loss == 0:
        return 100.0 if last_avg_gain > 0 else 50.0
    
    rs = last_avg_gain / last_avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd(series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """Calcola MACD, Signal Line, e Histogram."""
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan, np.nan, np.nan
    series = series.dropna()
    if len(series) < slow + signal: return np.nan, np.nan, np.nan # Dati insufficienti

    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    # Ritorna gli ultimi valori
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

def calculate_ma(series, period):
    """Calcola la Media Mobile Semplice (SMA)."""
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan
    series = series.dropna()
    if len(series) < period: return np.nan
    return series.rolling(window=period).mean().iloc[-1]

def calculate_change_pct(series):
    """Calcola la variazione percentuale tra gli ultimi due punti."""
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan
    series = series.dropna()
    if len(series) < 2: return np.nan

    last_price = series.iloc[-1]
    prev_price = series.iloc[-2]
    if pd.isna(last_price) or pd.isna(prev_price) or prev_price == 0: return np.nan
    
    change = ((last_price - prev_price) / prev_price) * 100
    return change

# --- Configurazione Pagina Streamlit ---
st.set_page_config(layout="wide")
st.title("📊 Crypto Dashboard LIVE – Indicatori Tecnici + GPT Signal")
st.caption(f"Dati live da CoinGecko, Indicatori da yFinance. Cache: {CACHE_TTL}s. Periodo Storico Usato: {YF_HISTORY_PERIOD}")

# --- Logica Principale ---
ticker_to_id_map = {"BTC-USD": "bitcoin", "ETH-USD": "ethereum", "BNB-USD": "binancecoin", "SOL-USD": "solana", "XRP-USD": "ripple"}
id_to_ticker_map = {v: k for k, v in ticker_to_id_map.items()}

cg_prices, last_cg_update = get_current_prices_coingecko(COINGECKO_IDS, VS_CURRENCY)

results = []
yf_fetch_errors = [] # Lista per tracciare errori yfinance

progress_bar = st.progress(0, text="Analisi Criptovalute...")

for i, yf_symbol in enumerate(SYMBOLS_YF):
    coingecko_id = ticker_to_id_map.get(yf_symbol)
    if not coingecko_id: continue # Salta se non c'è mapping

    current_price = cg_prices.get(coingecko_id, np.nan)

    # Fetch storico (usa la cache)
    hist = get_price_history_yf(yf_symbol)

    # Variabili per indicatori (default NaN)
    change_24h = np.nan
    rsi1d = np.nan
    macd_line, macd_signal_val, macd_hist = np.nan, np.nan, np.nan
    ma_short, ma_long = np.nan, np.nan

    # Controlla se abbiamo dati storici PRIMA di calcolare
    if not hist.empty:
        # Stampa di Debug (opzionale)
        # st.write(f"Debug: {yf_symbol} - Dati Storici: {len(hist)} righe")

        change_24h = calculate_change_pct(hist['price'])

        # Calcola indicatori solo se ci sono abbastanza dati
        if len(hist) >= RSI_PERIOD + 1:
            rsi1d = calculate_rsi(hist['price'])
        else:
            # Registra info per l'utente
             yf_fetch_errors.append(f"Dati insufficienti ({len(hist)} punti) per RSI 1d per {yf_symbol}.")

        if len(hist) >= MACD_SLOW + MACD_SIGNAL:
             macd_line, macd_signal_val, macd_hist = calculate_macd(hist['price'])
        else:
             yf_fetch_errors.append(f"Dati insufficienti ({len(hist)} punti) per MACD per {yf_symbol}.")
        
        if len(hist) >= MA_SHORT:
             ma_short = calculate_ma(hist['price'], MA_SHORT)
        else:
             yf_fetch_errors.append(f"Dati insufficienti ({len(hist)} punti) per MA({MA_SHORT}d) per {yf_symbol}.")

        if len(hist) >= MA_LONG:
             ma_long = calculate_ma(hist['price'], MA_LONG)
        else:
             yf_fetch_errors.append(f"Dati insufficienti ({len(hist)} punti) per MA({MA_LONG}d) per {yf_symbol}.")

    else:
        # Registra errore se yfinance non ha restituito dati
        yf_fetch_errors.append(f"Nessun dato storico restituito da yFinance per {yf_symbol}.")

    # --- Visualizzazione Prezzo ---
    price_display = f"${current_price:,.2f}" if not pd.isna(current_price) else "N/A"
    change_display = f" ({change_24h:+.2f}%)" if not pd.isna(change_24h) else ""
    full_price_display = f"{price_display}{change_display}" if price_display != "N/A" else "N/A"

    # --- GPT Signal (basato su % change e RSI ora) ---
    gpt = "🟡 Hold" # Default
    if not pd.isna(change_24h) and not pd.isna(rsi1d):
        if change_24h >= 1.5 and rsi1d < 65: gpt = "🔶 Strong Buy"
        elif change_24h >= 0.5 and rsi1d < 60: gpt = "🟢 Buy"
        elif change_24h <= -1.5 and rsi1d > 35: gpt = "🔻 Strong Sell"
        elif change_24h <= -0.5 and rsi1d > 40: gpt = "🔴 Sell"
    elif not pd.isna(change_24h): # Fallback solo su % change se RSI manca
        if change_24h >= 2: gpt = "🔶 Strong Buy"
        elif change_24h >= 1: gpt = "🟢 Buy"
        elif change_24h <= -2: gpt = "🔻 Strong Sell"
        elif change_24h <= -1: gpt = "🔴 Sell"


    # --- Assembla Risultati ---
    results.append({
        "Crypto": yf_symbol,
        f"Prezzo ({VS_CURRENCY.upper()})": full_price_display,
        "GPT Signal": gpt,
        "RSI 1h": "N/A",
        "RSI 1d": f"{rsi1d:.2f}" if not pd.isna(rsi1d) else "N/A", # Mostra RSI calcolato
        "RSI 1w": "N/A",
        "RSI 1mo": "N/A",
        "MACD Hist": f"{macd_hist:.2f}" if not pd.isna(macd_hist) else "N/A",
        "MA Short": f"{ma_short:.2f}" if not pd.isna(ma_short) else "N/A",
        "MA Long": f"{ma_long:.2f}" if not pd.isna(ma_long) else "N/A",
        # Placeholder non implementati
        "SRSI": "N/A", "Doda Stoch": "N/A", "GChannel": "N/A", "Volume Flow": "N/A", "VWAP": "N/A"
    })

    progress_bar.progress((i + 1) / len(SYMBOLS_YF), text=f"Analisi Criptovalute... ({yf_symbol})")

progress_bar.empty()

# --- Mostra Errori/Info Raccolti ---
if yf_fetch_errors:
    with st.expander("ℹ️ Note sul Calcolo Indicatori", expanded=False):
        for error_msg in set(yf_fetch_errors): # Usa set per evitare duplicati
            st.info(error_msg)

# --- Crea e Visualizza DataFrame ---
if results:
    df = pd.DataFrame(results)

    # Funzione per colorare la percentuale
    def highlight_pct(val):
        if isinstance(val, str) and '%' in val and '(' in val:
            try:
                num_str = val[val.rfind('(')+1 : val.rfind('%')]
                num = float(num_str)
                if num > 0: return 'color: green'
                elif num < 0: return 'color: red'
            except ValueError: return ''
        return ''

    # Applica lo stile (usa .map per compatibilità futura)
    st.dataframe(
        df.style.map(highlight_pct, subset=[f"Prezzo ({VS_CURRENCY.upper()})"]),
        use_container_width=True,
        hide_index=True,
         # Configurazione colonne per allineamento e formati se necessario
         # Esempio:
        # column_config={
        #     "RSI 1d": st.column_config.NumberColumn(format="%.2f"),
        #     "MACD Hist": st.column_config.NumberColumn(format="%.2f"),
        #      ...etc
        # }
        )
else:
    st.warning("Nessun risultato da visualizzare.")

# --- Legenda ---
st.markdown("### ℹ️ Indicatori Tecnici - Legenda")
st.markdown(f"""
- **RSI (Relative Strength Index)**: misura la forza di un trend (0–100), sopra 70 è ipercomprato, sotto 30 è ipervenduto. (Calcolato su dati giornalieri, periodo {RSI_PERIOD})
- **MACD (Moving Average Convergence Divergence)**: confronto di medie mobili (EMA {MACD_FAST}, {MACD_SLOW}, Signal {MACD_SIGNAL}), segnala cambi di tendenza. L'Istogramma (Hist) mostra il momentum. (Calcolato su dati giornalieri)
- **MA (Simple Moving Average)**: Media mobile semplice. (Calcolata su dati giornalieri, periodi {MA_SHORT} e {MA_LONG})
- **GPT Signal**: decisione AI *semplificata* basata sulla variazione % giornaliera e RSI. **Non è consulenza finanziaria.**
- *SRSI, Doda Stoch, GChannel, Volume Flow, VWAP, RSI (1h, 1w, 1mo)*: Non implementati in questa versione (N/A).
""")

st.markdown(f"⏱️ Prezzi live aggiornati da CoinGecko circa alle: **{last_cg_update.strftime('%Y-%m-%d %H:%M:%S')}** (Cache: {CACHE_TTL}s)")