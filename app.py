# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
import pandas_ta as ta # Import pandas-ta
from datetime import datetime, timedelta
import time # Per throttling richieste (anche se la cache aiuta)
import math # Per calcoli VWAP

# --- Configurazione Globale ---
# Lista delle crypto da monitorare (Simboli comuni)
# Assicurati che questi simboli siano mappabili agli ID di CoinGecko sotto
SYMBOLS = ["BTC", "ETH", "BNB", "SOL", "XRP"]
VS_CURRENCY = "usd" # Valuta di riferimento
CACHE_TTL = 300 # Cache di 5 minuti (300 sec) per dati API
DAYS_HISTORY = 365 # Giorni di storico da scaricare per indicatori daily/weekly/monthly

# Mappatura (può essere espansa o caricata dinamicamente)
# Chiave: Simbolo (usato per visualizzazione), Valore: ID CoinGecko (usato per API)
# Ottenuta da API CoinGecko /coins/list o nota
SYMBOL_TO_ID_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "XRP": "ripple",
    # Aggiungi altre coppie se necessario
}
# Crea mappa inversa per lookup facile
ID_TO_SYMBOL_MAP = {v: k for k, v in SYMBOL_TO_ID_MAP.items()}
COINGECKO_IDS_LIST = list(SYMBOL_TO_ID_MAP.values())

# Periodi Indicatori standard (usati da pandas-ta)
RSI_PERIOD = 14
SRSI_PERIOD = 14
SRSI_K = 3
SRSI_D = 3
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MA_SHORT = 20
MA_LONG = 50
VWAP_PERIOD = 14 # Giorni per VWAP basato su dati giornalieri

# --- Funzioni API CoinGecko ---

@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento prezzi live (CoinGecko)...")
def get_coingecko_current_prices(ids_list, currency):
    """Ottiene i prezzi correnti e la variazione 24h da CoinGecko."""
    ids_string = ",".join(ids_list)
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ids_string,
        "vs_currencies": currency,
        "include_24hr_change": "true"
    }
    prices = {}
    changes = {}
    timestamp = datetime.now()
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        for coin_id, price_data in data.items():
            prices[coin_id] = price_data.get(currency, np.nan)
            changes[coin_id] = price_data.get(f"{currency}_24h_change", np.nan)
        return prices, changes, timestamp
    except requests.exceptions.RequestException as e:
        st.error(f"Errore API Prezzi CoinGecko: {e}")
        return {}, {}, timestamp
    except Exception as e:
        st.error(f"Errore Processamento Prezzi CoinGecko: {e}")
        return {}, {}, timestamp

# NOTA: La cache qui usa l'ID della coin per distinguere le chiamate
@st.cache_data(ttl=CACHE_TTL * 2, show_spinner=False) # Cache più lunga per dati storici
def get_coingecko_historical_data(coin_id, currency, days):
    """Ottiene dati storici (prezzi, volumi) da CoinGecko /market_chart."""
    time.sleep(0.3) # Delay per rispettare rate limit API gratuita CoinGecko
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': currency,
        'days': str(days), # 'max' per tutto lo storico, o numero giorni
        'interval': 'daily' if days > 90 else '', # Richiedi daily se > 90gg, altrimenti lascia auto
        'precision': 'full' # Chiedi massima precisione
    }
    try:
        response = requests.get(url, params=params, timeout=20) # Timeout più lungo per storico
        response.raise_for_status()
        data = response.json()

        if not data or 'prices' not in data or not data['prices']:
             return pd.DataFrame() # Ritorna vuoto se non ci sono dati prezzo

        # Converti prezzi in DataFrame (timestamp, prezzo)
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
        prices_df.set_index('timestamp', inplace=True)

        # Converti volumi in DataFrame (timestamp, volume) - SE ESISTONO
        volumes_df = None
        if 'total_volumes' in data and data['total_volumes']:
             volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
             volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
             volumes_df.set_index('timestamp', inplace=True)

             # Unisci prezzi e volumi basandoti sul timestamp (indice)
             hist_df = prices_df.join(volumes_df, how='inner') # 'inner' per avere solo timestamp con entrambi
        else:
            # Se non ci sono volumi, ritorna solo i prezzi e metti volume a NaN o 0
            hist_df = prices_df
            hist_df['volume'] = 0 # O np.nan, a seconda di come pandas-ta gestisce NaN in VWAP

        # Aggiungi colonne High, Low, Open (approssimate dai dati giornalieri/orari se non presenti)
        # Per /market_chart, abbiamo solo 'price' (che è il Close).
        # pandas-ta può funzionare spesso anche solo con 'close', ma alcuni indicatori richiedono OHLC.
        # Per ora lavoriamo solo con 'price' (Close) e 'volume'.
        hist_df.rename(columns={'price': 'close'}, inplace=True) # Rinomina per pandas-ta

        # Rimuovi eventuali duplicati di indice (timestamp) mantenendo l'ultimo
        hist_df = hist_df[~hist_df.index.duplicated(keep='last')]

        return hist_df

    except requests.exceptions.RequestException as e:
        # Non mostrare errore qui direttamente per non affollare, verrà gestito dopo
        # st.warning(f"API Error fetching history for {coin_id}: {e}")
        return pd.DataFrame() # Ritorna DataFrame vuoto in caso di errore
    except Exception as e:
        # st.warning(f"Processing Error fetching history for {coin_id}: {e}")
        return pd.DataFrame()

# --- Funzione Calcolo Indicatori con Pandas TA ---
def calculate_indicators(df_hist):
    """Calcola indicatori tecnici usando pandas-ta."""
    indicators = {}
    if df_hist.empty or len(df_hist) < 10: # Minimo arbitrario per qualsiasi calcolo
        return indicators # Ritorna dizionario vuoto se non ci sono dati

    # Calcola indicatori richiesti (aggiungi strategia se vuoi usare .strategy)
    try:
        # RSI (Daily)
        df_hist.ta.rsi(length=RSI_PERIOD, append=True) # Appende colonna 'RSI_14'
        indicators['RSI_1d'] = df_hist['RSI_14'].iloc[-1] if 'RSI_14' in df_hist.columns and not pd.isna(df_hist['RSI_14'].iloc[-1]) else np.nan

        # SRSI (Daily)
        df_hist.ta.stochrsi(length=RSI_PERIOD, rsi_length=RSI_PERIOD, k=SRSI_K, d=SRSI_D, append=True)
        # pandas-ta appende 'STOCHRSIk_14_14_3_3' e 'STOCHRSId_14_14_3_3'
        srsi_k_col = f'STOCHRSIk_{RSI_PERIOD}_{RSI_PERIOD}_{SRSI_K}_{SRSI_D}'
        srsi_d_col = f'STOCHRSId_{RSI_PERIOD}_{RSI_PERIOD}_{SRSI_K}_{SRSI_D}'
        indicators['SRSI_k'] = df_hist[srsi_k_col].iloc[-1] if srsi_k_col in df_hist.columns and not pd.isna(df_hist[srsi_k_col].iloc[-1]) else np.nan
        indicators['SRSI_d'] = df_hist[srsi_d_col].iloc[-1] if srsi_d_col in df_hist.columns and not pd.isna(df_hist[srsi_d_col].iloc[-1]) else np.nan

        # MACD (Daily)
        df_hist.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL, append=True)
        # Appende MACD_12_26_9, MACDh_12_26_9 (histogram), MACDs_12_26_9 (signal)
        macd_hist_col = f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}'
        indicators['MACD_Hist'] = df_hist[macd_hist_col].iloc[-1] if macd_hist_col in df_hist.columns and not pd.isna(df_hist[macd_hist_col].iloc[-1]) else np.nan

        # SMA (Daily)
        df_hist.ta.sma(length=MA_SHORT, append=True) # Appende 'SMA_20'
        indicators['MA_Short'] = df_hist[f'SMA_{MA_SHORT}'].iloc[-1] if f'SMA_{MA_SHORT}' in df_hist.columns and not pd.isna(df_hist[f'SMA_{MA_SHORT}'].iloc[-1]) else np.nan
        df_hist.ta.sma(length=MA_LONG, append=True) # Appende 'SMA_50'
        indicators['MA_Long'] = df_hist[f'SMA_{MA_LONG}'].iloc[-1] if f'SMA_{MA_LONG}' in df_hist.columns and not pd.isna(df_hist[f'SMA_{MA_LONG}'].iloc[-1]) else np.nan

        # VWAP (Daily) - Richiede High, Low, Close, Volume. /market_chart non li ha tutti.
        # pandas-ta calcola VWAP se trova 'high', 'low', 'close', 'volume'.
        # Tentiamo di creare HLC approssimati dal close giornaliero (non ideale!)
        # Se 'volume' non è stato aggiunto prima, VWAP fallirà silenziosamente.
        if 'volume' in df_hist.columns and df_hist['volume'].notna().any():
             # Crea HLC fittizi per permettere il calcolo (MOLTO APPROSSIMATO!)
             # Se si vogliono HLC reali, serve un'altra fonte API
             df_hist['high'] = df_hist['close'] # Stima
             df_hist['low'] = df_hist['close']  # Stima
             df_hist['open'] = df_hist['close'].shift(1) # Stima
             df_hist.ta.vwap(length=VWAP_PERIOD, append=True) # Richiede HLCV
             vwap_col = f'VWAP_{VWAP_PERIOD}'
             indicators['VWAP'] = df_hist[vwap_col].iloc[-1] if vwap_col in df_hist.columns and not pd.isna(df_hist[vwap_col].iloc[-1]) else np.nan
        else:
             indicators['VWAP'] = np.nan # Volume mancante

    except Exception as e:
        # st.warning(f"Errore durante calcolo indicatori: {e}")
        # Ritorna NaN per gli indicatori se c'è un errore nel calcolo
        keys = ['RSI_1d', 'SRSI_k', 'SRSI_d', 'MACD_Hist', 'MA_Short', 'MA_Long', 'VWAP']
        for k in keys: indicators.setdefault(k, np.nan) # Imposta a NaN se non già presente

    return indicators

# --- Funzione Segnale GPT Semplificato ---
def generate_signal(rsi_val, macd_hist_val, ma_short_val, ma_long_val):
    """Genera un segnale esemplificativo basato su alcuni indicatori."""
    if pd.isna(rsi_val) or pd.isna(macd_hist_val) or pd.isna(ma_short_val) or pd.isna(ma_long_val):
        return "⚪️ N/D" # Non disponibile se mancano dati

    score = 0
    # RSI
    if rsi_val < 30: score += 2
    elif rsi_val < 40: score += 1
    elif rsi_val > 70: score -= 2
    elif rsi_val > 60: score -= 1

    # MACD Histogram
    if macd_hist_val > 0: score += 1
    else: score -= 1

    # MA Crossover
    if ma_short_val > ma_long_val: score += 1
    else: score -= 1

    # Mappa score
    if score >= 3: return "🔶 Strong Buy"
    elif score >= 1: return "🟢 Buy"
    elif score <= -3: return "🔻 Strong Sell"
    elif score <= -1: return "🔴 Sell"
    else: return "🟡 Hold"

# --- Configurazione Pagina Streamlit ---
st.set_page_config(layout="wide", page_title="Crypto Tech Dashboard (CoinGecko)", page_icon="🦎")
st.title("🦎 Crypto Technical Dashboard (CoinGecko API)")
st.caption(f"Dati live e storici da CoinGecko API. Cache: {CACHE_TTL}s.")

# --- Logica Principale ---
cg_prices_dict, cg_changes_dict, last_cg_update = get_coingecko_current_prices(COINGECKO_IDS_LIST, VS_CURRENCY)

results = []
fetch_errors = [] # Traccia errori fetch storico

progress_bar = st.progress(0, text="Analisi Criptovalute...")

for i, coin_id in enumerate(COINGECKO_IDS_LIST):
    symbol = ID_TO_SYMBOL_MAP.get(coin_id, coin_id.capitalize()) # Usa ID se simbolo non trovato

    current_price = cg_prices_dict.get(coin_id, np.nan)
    change_24h = cg_changes_dict.get(coin_id, np.nan)

    # Fetch storico
    hist_df = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY)

    # Calcola indicatori
    if not hist_df.empty:
        indicators = calculate_indicators(hist_df.copy()) # Passa una copia per evitare side effects
    else:
        indicators = {} # Dizionario vuoto se non ci sono dati storici
        fetch_errors.append(f"Dati storici non disponibili o errore API per {symbol} ({coin_id}). Indicatori non calcolati.")

    # --- Assembla Risultati ---
    results.append({
        "Symbol": symbol,
        "ID": coin_id, # Tieni ID per riferimento interno se serve
        f"Prezzo ({VS_CURRENCY.upper()})": current_price,
        "% 24h": change_24h,
        "RSI (1d)": indicators.get('RSI_1d', np.nan),
        "SRSI k (1d)": indicators.get('SRSI_k', np.nan),
        #"SRSI d (1d)": indicators.get('SRSI_d', np.nan), # Spesso ridondante, mostriamo solo k
        "MACD Hist (1d)": indicators.get('MACD_Hist', np.nan),
        f"MA({MA_SHORT}d)": indicators.get('MA_Short', np.nan),
        f"MA({MA_LONG}d)": indicators.get('MA_Long', np.nan),
        "VWAP (1d)": indicators.get('VWAP', np.nan),
        # Placeholder non implementati / non specificati
        "RSI (1h)": "N/A", "RSI (1w)": "N/A", "RSI (1mo)": "N/A", # Da affinare
        "Doda Stoch": "N/A", "GChannel": "N/A", "Volume Flow": "N/A",
    })

    progress_bar.progress((i + 1) / len(COINGECKO_IDS_LIST), text=f"Analisi Criptovalute... ({symbol})")

progress_bar.empty()

# --- Mostra Errori/Info Raccolti ---
if fetch_errors:
    with st.expander("ℹ️ Note sul Recupero Dati Storici (CoinGecko)", expanded=False):
        unique_errors = sorted(list(set(fetch_errors)))
        for error_msg in unique_errors:
            st.info(error_msg)

# --- Crea e Visualizza DataFrame ---
if results:
    df = pd.DataFrame(results)

    # Calcola segnale GPT dopo aver assemblato il DataFrame completo
    df['GPT Signal'] = df.apply(lambda row: generate_signal(
                                    row.get('RSI (1d)'),
                                    row.get('MACD Hist (1d)'),
                                    row.get(f'MA({MA_SHORT}d)'),
                                    row.get(f'MA({MA_LONG}d)')
                                ), axis=1)


    # Seleziona e ordina colonne per la visualizzazione
    cols_order = [
        "Symbol", f"Prezzo ({VS_CURRENCY.upper()})", "% 24h", "GPT Signal",
        "RSI (1d)", "SRSI k (1d)", "MACD Hist (1d)", f"MA({MA_SHORT}d)", f"MA({MA_LONG}d)", "VWAP (1d)",
        # Aggiungere qui le altre colonne N/A se si vogliono mostrare
        # "RSI (1h)", "RSI (1w)", "RSI (1mo)", "Doda Stoch", "GChannel", "Volume Flow"
    ]
    cols_to_show = [col for col in cols_order if col in df.columns]
    df_display = df[cols_to_show].copy() # Crea copia per evitare SettingWithCopyWarning

    # Formattazione numerica e N/A
    numeric_cols = df_display.select_dtypes(include=np.number).columns
    formatters = {}
    currency_col = f"Prezzo ({VS_CURRENCY.upper()})"
    pct_col = "% 24h"

    for col in numeric_cols:
        if col == currency_col:
            formatters[col] = lambda x: f"${x:,.4f}" if pd.notna(x) else "N/A" # 4 decimali per prezzo
        elif col == pct_col:
             formatters[col] = lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A" # 2 decimali con segno per %
        elif "RSI" in col or "SRSI" in col:
             formatters[col] = lambda x: f"{x:.1f}" if pd.notna(x) else "N/A" # 1 decimale per RSI/SRSI
        elif "MACD" in col:
            formatters[col] = lambda x: f"{x:.4f}" if pd.notna(x) else "N/A" # 4 decimali per MACD
        elif "MA" in col or "VWAP" in col:
             formatters[col] = lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A" # 2 decimali per MA/VWAP

    # Applica formattazione manuale (st.dataframe ha problemi con NaNs a volte)
    for col, func in formatters.items():
        if col in df_display.columns:
             df_display[col] = df_display[col].apply(func)

    # Sostituisci eventuali NaN rimasti (non numerici?) con "N/A"
    df_display.fillna("N/A", inplace=True)

    # Funzione per colorare la percentuale
    def highlight_pct_val(val):
        if isinstance(val, str) and val.endswith('%'):
            try:
                num = float(val.replace('%',''))
                if num > 0: return 'color: green'
                elif num < 0: return 'color: red'
            except ValueError: return ''
        return ''

    st.dataframe(
        df_display.style.map(highlight_pct_val, subset=[pct_col]), # Usa .map sulla colonna %
        use_container_width=True,
        hide_index=True,
        # Si potrebbe usare column_config ma la formattazione manuale sopra è più robusta con N/A
    )
else:
    st.warning("Nessun risultato da visualizzare. Controlla le note sul recupero dati.")

# --- Legenda ---
st.markdown("### ℹ️ Indicatori Tecnici - Legenda")
st.markdown(f"""
* **RSI (Relative Strength Index):** Indicatore di momentum (0-100). >70 Ipercomprato, <30 Ipervenduto. (Calcolato su dati giornalieri, periodo {RSI_PERIOD})
* **SRSI k (Stochastic RSI %K):** Versione stocastica dell'RSI per identificare cicli brevi. (Calcolato su dati giornalieri)
* **MACD Hist (Moving Average Convergence Divergence Histogram):** Differenza tra linea MACD e linea segnale. Indica momentum. (Calcolato su dati giornalieri)
* **MA (Simple Moving Average):** Media mobile semplice del prezzo. (Calcolata su dati giornalieri, periodi {MA_SHORT} e {MA_LONG})
* **VWAP (Volume Weighted Average Price):** Prezzo medio ponderato per volume. (Calcolato su dati giornalieri - *approssimato data la mancanza di HLC*)
* **GPT Signal:** Segnale *esemplificativo* basato su RSI, MACD Hist, MA Crossover. **Non è consulenza finanziaria.**
* *RSI (1h, 1w, 1mo), Doda Stoch, GChannel, Volume Flow:* Non implementati o richiedono dati/logica specifica (N/A).
""")

st.markdown(f"⏱️ Prezzi live aggiornati da CoinGecko circa alle: **{last_cg_update.strftime('%Y-%m-%d %H:%M:%S')}** (Cache: {CACHE_TTL}s)")