# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊 Crypto Dashboard LIVE – Indicatori Tecnici + GPT Signal")

# --- Configurazione ---
SYMBOLS_YF = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"] # Tickers per yFinance
COINGECKO_IDS = "bitcoin,ethereum,binancecoin,solana,ripple" # IDs per CoinGecko API
VS_CURRENCY = "usd"
CACHE_TTL = 300 # Cache di 5 minuti (300 sec)
YF_HISTORY_PERIOD = "3mo" # Periodo storico per calcoli giornalieri (RSI, % change)
YF_HISTORY_INTERVAL = "1d"
RSI_PERIOD = 14

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
        # Mappa gli ID di CoinGecko ai prezzi
        for coin_id, price_data in data.items():
            prices[coin_id] = price_data.get(currency, np.nan) # Usa NaN se manca il prezzo
        return prices
    except requests.exceptions.RequestException as e:
        st.error(f"Errore API CoinGecko: {e}")
        return {}
    except Exception as e:
        st.error(f"Errore processamento dati CoinGecko: {e}")
        return {}

@st.cache_data(ttl=CACHE_TTL * 2, show_spinner=False) # Cache più lunga per dati storici
def get_price_history_yf(symbol, period=YF_HISTORY_PERIOD, interval=YF_HISTORY_INTERVAL):
    """Ottiene dati storici da yFinance."""
    try:
        # Usare yf.Ticker per gestire meglio errori per singolo ticker
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval, auto_adjust=True) # auto_adjust=True semplifica (usa solo 'Close')
        if data.empty:
            # st.warning(f"Nessun dato storico da yFinance per {symbol} ({period}/{interval})")
            return pd.DataFrame()
        # Rinomina 'Close' in 'price' per coerenza (auto_adjust rimuove le altre colonne OHLCV)
        data.rename(columns={"Close": "price"}, inplace=True)
        return data[['price']] # Ritorna solo la colonna prezzo
    except Exception as e:
        # st.warning(f"Errore durante il fetch da yFinance per {symbol}: {e}")
        return pd.DataFrame()

# --- Funzioni Calcolo Indicatori ---

def calculate_rsi(series, period=RSI_PERIOD):
    """Calcola l'RSI."""
    if not isinstance(series, pd.Series) or series.empty or len(series) < period + 1:
        return np.nan
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    # Evita divisione per zero
    if loss.iloc[-1] == 0:
        return 100.0 if gain.iloc[-1] > 0 else 50.0 # RSI è 100 se solo guadagni, 50 se nessun cambio
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] # Ritorna l'ultimo valore RSI numerico

def calculate_change_pct(series):
    """Calcola la variazione percentuale tra gli ultimi due punti."""
    if not isinstance(series, pd.Series) or len(series) < 2:
        return np.nan
    last_price = series.iloc[-1]
    prev_price = series.iloc[-2]
    if pd.isna(last_price) or pd.isna(prev_price) or prev_price == 0:
        return np.nan
    change = ((last_price - prev_price) / prev_price) * 100
    return change

# --- Logica Principale ---

# Mappa Ticker yFinance a ID CoinGecko (necessario per unire i dati)
# Potrebbe essere necessario aggiustarla se i ticker/id cambiano
ticker_to_id_map = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "BNB-USD": "binancecoin",
    "SOL-USD": "solana",
    "XRP-USD": "ripple",
}
# Inverti mappa per lookup facile
id_to_ticker_map = {v: k for k, v in ticker_to_id_map.items()}

# Fetch prezzi live
cg_prices = get_current_prices_coingecko(COINGECKO_IDS, VS_CURRENCY)

results = []
progress_bar = st.progress(0, text="Analisi Criptovalute...")

for i, yf_symbol in enumerate(SYMBOLS_YF):
    coingecko_id = ticker_to_id_map.get(yf_symbol)
    if not coingecko_id:
        st.warning(f"ID CoinGecko non trovato per {yf_symbol}, crypto saltata.")
        continue

    current_price = cg_prices.get(coingecko_id, np.nan) # Prendi prezzo live

    # Fetch storico per calcoli
    hist = get_price_history_yf(yf_symbol)

    # Calcola % Change (se possibile)
    change_24h = calculate_change_pct(hist['price']) if not hist.empty else np.nan

    # Calcola RSI 1d (se possibile)
    rsi1d = calculate_rsi(hist['price']) if not hist.empty else np.nan

    # --- Visualizzazione Prezzo ---
    price_display = f"${current_price:,.2f}" if not pd.isna(current_price) else "N/A"
    change_display = f" ({change_24h:+.2f}%)" if not pd.isna(change_24h) else ""
    full_price_display = f"{price_display}{change_display}" if price_display != "N/A" else "N/A"


    # --- GPT Signal (basato solo sulla % change giornaliera in questa versione) ---
    gpt = "🟡 Hold" # Default
    if not pd.isna(change_24h):
        if change_24h >= 2: gpt = "🔶 Strong Buy"
        elif change_24h >= 1: gpt = "🟢 Buy"
        elif change_24h <= -2: gpt = "🔻 Strong Sell"
        elif change_24h <= -1: gpt = "🔴 Sell"


    # --- Assembla Risultati ---
    # Solo gli indicatori implementati avranno valori, gli altri rimangono N/A
    results.append({
        "Crypto": yf_symbol,
        f"Prezzo ({VS_CURRENCY.upper()})": full_price_display, # Colonna prezzo aggiornata
        "GPT Signal": gpt,
        "RSI 1h": "N/A", # Non implementato
        "RSI 1d": f"{rsi1d:.2f}" if not pd.isna(rsi1d) else "N/A", # Mostra RSI calcolato
        "RSI 1w": "N/A", # Non implementato
        "RSI 1mo": "N/A", # Non implementato
        "SRSI": "N/A",
        "MACD": "N/A",
        "MA": "N/A",
        "Doda Stoch": "N/A",
        "GChannel": "N/A",
        "Volume Flow": "N/A",
        "VWAP": "N/A"
    })

    progress_bar.progress((i + 1) / len(SYMBOLS_YF), text=f"Analisi Criptovalute... ({yf_symbol})")

progress_bar.empty() # Rimuovi barra progresso

# --- Crea e Visualizza DataFrame ---
if results:
    df = pd.DataFrame(results)

    # Funzione per colorare la percentuale (leggermente modificata)
    def highlight_pct(val):
        if isinstance(val, str) and '%' in val and '(' in val:
            try:
                # Estrae il numero dalla parentesi
                num_str = val[val.rfind('(')+1 : val.rfind('%')]
                num = float(num_str)
                if num > 0: return 'color: green'
                elif num < 0: return 'color: red'
            except ValueError: return '' # Ignora se non è un numero valido
        return ''

    # Applica lo stile solo alla colonna prezzo che contiene la %
    st.dataframe(
        df.style.applymap(highlight_pct, subset=[f"Prezzo ({VS_CURRENCY.upper()})"]),
        use_container_width=True,
        hide_index=True # Nasconde l'indice numerico di default
        )
else:
    st.warning("Nessun risultato da visualizzare.")


# --- Legenda ---
st.markdown("### ℹ️ Indicatori Tecnici - Legenda")
# (Legenda invariata rispetto al file originale)
st.markdown("""
- **RSI (Relative Strength Index)**: misura la forza di un trend (0–100), sopra 70 è ipercomprato, sotto 30 è ipervenduto. **(Implementato solo 1d)**
- **SRSI**: stocastico dell'RSI. (N/A)
- **MACD**: confronto di medie mobili, segnala cambi di tendenza. (N/A)
- **MA**: media mobile semplice. (N/A)
- **Doda Stoch**: versione personalizzata dello stocastico. (N/A)
- **GChannel**: simile a Donchian/Gann Channel. (N/A)
- **Volume Flow**: flusso di capitale basato sul volume. (N/A)
- **VWAP**: prezzo medio ponderato per il volume. (N/A)
- **GPT Signal**: decisione finale AI semplificata basata sulla variazione % giornaliera.
""")


st.markdown(f"⏱️ Dati aggiornati circa alle: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}** (Cache: {CACHE_TTL}s)")