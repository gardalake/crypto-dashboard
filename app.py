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
VS_CURRENCY = "usd"
CACHE_TTL = 300
DAYS_HISTORY_DAILY = 365
DAYS_HISTORY_HOURLY = 7

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
    ids_string = ",".join(ids_list)
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': currency, 'ids': ids_string, 'order': 'market_cap_desc',
        'per_page': str(len(ids_list)), 'page': 1, 'sparkline': False,
        'price_change_percentage': '1h,24h,7d'
    }
    timestamp = datetime.now()
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status(); data = response.json()
        df = pd.DataFrame(data);
        if not df.empty: df.set_index('id', inplace=True)
        return df, timestamp
    except Exception as e:
        st.error(f"Errore API Mercato CoinGecko: {e}")
        return pd.DataFrame(), timestamp

@st.cache_data(ttl=CACHE_TTL * 2, show_spinner=False)
def get_coingecko_historical_data(coin_id, currency, days, interval='daily'):
    time.sleep(0.5)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': currency, 'days': str(days),
              'interval': interval if interval == 'hourly' else 'daily', 'precision': 'full'}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status(); data = response.json()
        if not data or 'prices' not in data or not data['prices']: return pd.DataFrame()

        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
        prices_df.set_index('timestamp', inplace=True)

        hist_df = prices_df
        if 'total_volumes' in data and data['total_volumes']:
             volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
             volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
             volumes_df.set_index('timestamp', inplace=True)
             hist_df = prices_df.join(volumes_df, how='outer').interpolate(method='time').ffill().bfill()
        else: hist_df['volume'] = 0

        hist_df['high'] = hist_df['close']; hist_df['low'] = hist_df['close']; hist_df['open'] = hist_df['close'].shift(1)
        hist_df = hist_df[~hist_df.index.duplicated(keep='last')]
        hist_df.dropna(subset=['close'], inplace=True)
        return hist_df
    except Exception as e: return pd.DataFrame()

# --- Funzioni Calcolo Indicatori (Manuali - Invariate) ---

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

def calculate_macd_manual(series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan, np.nan, np.nan
    series = series.dropna()
    if len(series) < slow : return np.nan, np.nan, np.nan
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    if len(macd_line.dropna()) < signal: return macd_line.iloc[-1], np.nan, np.nan
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    # Ritorna l'ultimo valore valido di ciascuno
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

# --- Funzione Segnale GPT Semplificato (Invariata nella logica base, ma usata per GPT Signal) ---
def generate_gpt_signal(rsi_1d, rsi_1h, rsi_1w, macd_hist, ma_short, ma_long, current_price):
    """Genera un segnale più articolato (ma ancora basato su regole)."""
    # Controlla se tutti gli input necessari sono validi (non NaN)
    required_inputs = [rsi_1d, macd_hist, ma_short, ma_long, current_price] # RSI 1h/1w opzionali
    if any(pd.isna(x) for x in required_inputs):
        return "⚪️ N/D" # Non disponibile se mancano dati chiave

    score = 0; reasons = []
    # 1. Trend di fondo (MA Long vs Prezzo) - Peso 1
    if current_price > ma_long: score += 1; reasons.append("P>MA50")
    else: score -= 1; reasons.append("P<MA50")
    # 2. Trend Medio Termine (MA Crossover) - Peso 2
    if ma_short > ma_long: score += 2; reasons.append("MA20>MA50")
    else: score -= 2; reasons.append("MA20<MA50")
    # 3. Momentum (MACD Histogram) - Peso 2
    if macd_hist > 0: score += 2; reasons.append("MACD+")
    else: score -= 2; reasons.append("MACD-")
    # 4. Ipercomprato/venduto Daily (RSI 1d) - Peso 2
    if rsi_1d < 30: score += 2; reasons.append("RSI1d<30")
    elif rsi_1d < 40: score += 1; reasons.append("RSI1d<40")
    elif rsi_1d > 70: score -= 2; reasons.append("RSI1d>70")
    elif rsi_1d > 60: score -= 1; reasons.append("RSI1d>60")
    # 5. Ipercomprato/venduto Weekly (RSI 1w) - Peso 1 (meno reattivo)
    if not pd.isna(rsi_1w):
        if rsi_1w < 30: score += 1; reasons.append("RSI1w<30")
        elif rsi_1w > 70: score -= 1; reasons.append("RSI1w>70")
    # 6. Forza a Breve (RSI 1h) - Peso 1
    if not pd.isna(rsi_1h):
        if rsi_1h > 60: score += 1; reasons.append("RSI1h>60")
        elif rsi_1h < 40: score -= 1; reasons.append("RSI1h<40")

    # Mappatura Score
    if score >= 5: return "⚡️ Strong Buy"
    elif score >= 2: return "🟢 Buy"
    elif score <= -5: return "🚨 Strong Sell"
    elif score <= -2: return "🔴 Sell"
    else: return "🟡 Hold"

# --- NUOVA Funzione per "Gemini Alert" (basata su confluenza più stringente) ---
def generate_gemini_alert(ma_short, ma_long, macd_hist, rsi_1d):
    """Genera un alert specifico basato su forte confluenza di segnali DAILY."""
    # Condizione base: abbiamo bisogno di tutti questi indicatori
    if pd.isna(ma_short) or pd.isna(ma_long) or pd.isna(macd_hist) or pd.isna(rsi_1d):
        return "⏳" # Dati insuff.

    # Condizioni per Alert BUY
    is_uptrend = ma_short > ma_long
    is_momentum_positive = macd_hist > 0
    is_not_overbought = rsi_1d < 70 # Soglia standard ipercomprato

    if is_uptrend and is_momentum_positive and is_not_overbought:
        # Aggiungere un trigger più specifico? Es. MACD appena crossato? RSI in salita?
        # Per ora, la semplice confluenza
        return "✅ BUY Signal"

    # Condizioni per Alert SELL
    is_downtrend = ma_short < ma_long
    is_momentum_negative = macd_hist < 0
    is_not_oversold = rsi_1d > 30 # Soglia standard ipervenduto

    if is_downtrend and is_momentum_negative and is_not_oversold:
        # Aggiungere trigger più specifico?
        return "❌ SELL Signal"

    # Nessun alert specifico
    return "➖" # Segnale neutro o trattino

# --- Configurazione Pagina Streamlit ---
st.set_page_config(layout="wide", page_title="Crypto Tech Dashboard Pro", page_icon="📈")
st.title("📈 Crypto Technical Dashboard Pro (CoinGecko API)")
st.caption(f"Dati live e storici da CoinGecko API. Cache: {CACHE_TTL}s.")

# --- Logica Principale ---
market_data_df, last_cg_update = get_coingecko_market_data(COINGECKO_IDS_LIST, VS_CURRENCY)

if market_data_df.empty:
     st.error("Errore nel caricamento dei dati di mercato live da CoinGecko. Riprova più tardi.")
     st.stop()

results = []; fetch_errors = []
progress_bar = st.progress(0, text="Analisi Criptovalute...")
coin_ids_ordered = market_data_df.index.tolist() # Mantieni ordine market cap

for i, coin_id in enumerate(coin_ids_ordered):
    live_data = market_data_df.loc[coin_id]
    symbol = live_data.get('symbol', coin_id).upper()
    name = live_data.get('name', coin_id)
    rank = live_data.get('market_cap_rank', 'N/A')
    current_price = live_data.get('current_price', np.nan)
    change_1h = live_data.get('price_change_percentage_1h_in_currency', np.nan)
    change_24h = live_data.get('price_change_percentage_24h_in_currency', np.nan)
    change_7d = live_data.get('price_change_percentage_7d_in_currency', np.nan)
    volume_24h = live_data.get('total_volume', np.nan)
    market_cap = live_data.get('market_cap', np.nan)

    hist_daily_df = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_DAILY, interval='daily')
    hist_hourly_df = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_HOURLY, interval='hourly')

    rsi1d, rsi1h, rsi1w = np.nan, np.nan, np.nan
    macd_line, macd_signal, macd_hist = np.nan, np.nan, np.nan
    ma_short, ma_long = np.nan, np.nan
    vwap = np.nan

    if not hist_daily_df.empty:
        rsi1d = calculate_rsi_manual(hist_daily_df['close'])
        macd_line, macd_signal, macd_hist = calculate_macd_manual(hist_daily_df['close'])
        ma_short = calculate_sma_manual(hist_daily_df['close'], MA_SHORT)
        ma_long = calculate_sma_manual(hist_daily_df['close'], MA_LONG)
        vwap = calculate_vwap_manual(hist_daily_df)
        try:
            df_weekly = hist_daily_df['close'].resample('W-MON').last()
            if len(df_weekly) >= RSI_PERIOD + 1: rsi1w = calculate_rsi_manual(df_weekly)
            else: fetch_errors.append(f"Dati insuff. ({len(df_weekly)} sett.) per RSI 1w per {symbol}.")
        except Exception as e: fetch_errors.append(f"Errore resampling settimanale per {symbol}: {e}")
        min_len_needed = max(RSI_PERIOD + 1, MACD_SLOW + MACD_SIGNAL, MA_LONG)
        if len(hist_daily_df) < min_len_needed: fetch_errors.append(f"Dati Daily insuff. ({len(hist_daily_df)}/{min_len_needed}) per indicatori per {symbol}.")
    else: fetch_errors.append(f"Nessun dato storico Daily da CoinGecko per {symbol}.")

    if not hist_hourly_df.empty:
        if len(hist_hourly_df) >= RSI_PERIOD + 1: rsi1h = calculate_rsi_manual(hist_hourly_df['close'])
        else: fetch_errors.append(f"Dati Hourly insuff. ({len(hist_hourly_df)} ore) per RSI 1h per {symbol}.")
    else: fetch_errors.append(f"Nessun dato storico Hourly da CoinGecko per {symbol}.")

    # Calcola i segnali
    gpt_signal = generate_gpt_signal(rsi1d, rsi1h, rsi1w, macd_hist, ma_short, ma_long, current_price)
    gemini_alert = generate_gemini_alert(ma_short, ma_long, macd_hist, rsi1d) # Usa nuova funzione alert

    # --- Assembla Risultati ---
    results.append({
        "Rank": rank,
        "Symbol": symbol,
        "Name": name,
        "Gemini Alert": gemini_alert, # NUOVA COLONNA ALERT
        f"Prezzo ({VS_CURRENCY.upper()})": current_price,
        "% 1h": change_1h,
        "% 24h": change_24h,
        "% 7d": change_7d,
        "GPT Signal": gpt_signal, # Segnale combinato meno stringente
        "RSI (1h)": rsi1h,
        "RSI (1d)": rsi1d,
        "RSI (1w)": rsi1w,
        "MACD Hist (1d)": macd_hist,
        f"MA({MA_SHORT}d)": ma_short,
        f"MA({MA_LONG}d)": ma_long,
        "VWAP (1d)": vwap,
        f"Volume 24h ({VS_CURRENCY.upper()})": volume_24h,
        f"Market Cap ({VS_CURRENCY.upper()})": market_cap,
    })
    progress_bar.progress((i + 1) / len(coin_ids_ordered), text=f"Analisi Criptovalute... ({symbol})")

progress_bar.empty()

# --- Mostra Errori/Info Raccolti ---
if fetch_errors:
    with st.expander("ℹ️ Note sul Recupero Dati Storici/Calcolo Indicatori", expanded=False):
        unique_errors = sorted(list(set(fetch_errors)))
        for error_msg in unique_errors: st.info(error_msg)

# --- Crea e Visualizza DataFrame ---
if results:
    df = pd.DataFrame(results)
    df.set_index('Rank', inplace=True)

    # Ricalcola GPT signal qui se necessario o assicurati sia già nel df
    # df['GPT Signal'] = df.apply(...) # Già fatto sopra nell'append

    # Ordine colonne aggiornato con Gemini Alert
    cols_order = [
        "Symbol", "Name",
        "Gemini Alert", # Posizionata dopo il nome
        f"Prezzo ({VS_CURRENCY.upper()})",
        "% 1h", "% 24h", "% 7d", "GPT Signal",
        "RSI (1h)", "RSI (1d)", "RSI (1w)", "MACD Hist (1d)",
        f"MA({MA_SHORT}d)", f"MA({MA_LONG}d)", "VWAP (1d)",
        f"Volume 24h ({VS_CURRENCY.upper()})", f"Market Cap ({VS_CURRENCY.upper()})"
    ]
    cols_to_show = [col for col in cols_order if col in df.columns]
    df_display = df[cols_to_show].copy()

    # Formattazione manuale (aggiornata per gestire NaN)
    numeric_cols = df_display.select_dtypes(include=np.number).columns
    formatters = {}
    currency_col = f"Prezzo ({VS_CURRENCY.upper()})"
    market_cap_col = f"Market Cap ({VS_CURRENCY.upper()})"
    volume_col = f"Volume 24h ({VS_CURRENCY.upper()})"
    pct_cols = ["% 1h", "% 24h", "% 7d"]

    for col in df_display.columns: # Itera su tutte le colonne per applicare formattazione
        if col in numeric_cols:
            if col == currency_col: formatters[col] = lambda x: f"${x:,.4f}" if pd.notna(x) else "N/A"
            elif col in pct_cols: formatters[col] = lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
            elif col == market_cap_col: formatters[col] = lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
            elif col == volume_col: formatters[col] = lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
            elif "RSI" in col: formatters[col] = lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
            elif "MACD" in col: formatters[col] = lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            elif "MA" in col or "VWAP" in col: formatters[col] = lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
            else: formatters[col] = lambda x: str(x) if pd.notna(x) else "N/A" # Fallback per altri numeri
        elif col not in ["Symbol", "Name", "GPT Signal", "Gemini Alert"]: # Colonne stringa non formattate
             formatters[col] = lambda x: str(x) if pd.notna(x) else "N/A"

    for col, func in formatters.items():
        if col in df_display.columns:
            # Applica solo se la colonna esiste e non è già stringa (evita errori su Symbol/Name)
            if df_display[col].dtype != 'object':
                 df_display[col] = df_display[col].apply(func)

    # Sostituisci eventuali NaN/None rimasti (es. da colonne stringa originali)
    df_display.fillna("N/A", inplace=True)

    # Funzione per colorare % change
    def highlight_pct_col(col):
        colors = []
        for val in col:
            color = ''
            if isinstance(val, str) and val.endswith('%'):
                try: num = float(val.replace('%','')); color = 'color: green' if num > 0 else 'color: red' if num < 0 else ''
                except ValueError: pass
            colors.append(color)
        return colors

    # Funzione per colorare Gemini Alert
    def highlight_gemini_alert(col):
        colors = []
        for val in col:
            if val == "✅ BUY Signal": color = 'background-color: #28a745; color: white;' # Verde
            elif val == "❌ SELL Signal": color = 'background-color: #dc3545; color: white;' # Rosso
            else: color = ''
            colors.append(color)
        return colors


    # Applica stili
    styled_df = df_display.style
    for col_name in pct_cols:
        if col_name in df_display.columns:
             styled_df = styled_df.apply(highlight_pct_col, subset=[col_name], axis=0) # Usa apply con axis=0
    if "Gemini Alert" in df_display.columns:
        styled_df = styled_df.apply(highlight_gemini_alert, subset=["Gemini Alert"], axis=0)

    # Mostra DataFrame
    st.dataframe(styled_df, use_container_width=True)
else:
    st.warning("Nessun risultato da visualizzare.")

# --- Legenda Approfondita e Aggiornata ---
st.divider()
with st.expander("📘 Legenda Indicatori Tecnici e Segnali", expanded=True): # Mostra espanso di default
    st.markdown("""
    Questa sezione spiega gli indicatori e i segnali visualizzati nella tabella. Ricorda che l'analisi tecnica è uno strumento e non una garanzia di risultati futuri. **Questa dashboard è solo a scopo informativo e non costituisce consulenza finanziaria.**

    **Indicatori di Momentum:**
    * **RSI (Relative Strength Index - 1h, 1d, 1w):**
        * *Cos'è:* Oscillatore (0-100) che misura la velocità e l'entità delle recenti variazioni di prezzo. Aiuta a identificare condizioni di ipercomprato/ipervenduto e la forza del momentum.
        * *Interpretazione:*
            * `> 70`: **Ipercomprato**. Suggerisce che il prezzo è salito rapidamente e potrebbe essere dovuto per una correzione o consolidamento. Non è necessariamente un segnale di vendita immediato in un trend forte.
            * `< 30`: **Ipervenduto**. Suggerisce che il prezzo è sceso rapidamente e potrebbe essere vicino a un minimo temporaneo o pronto per un rimbalzo. Non è necessariamente un segnale di acquisto immediato in un trend debole.
            * *Divergenze:* Una divergenza tra il prezzo (es. nuovi massimi) e l'RSI (mancati nuovi massimi) può segnalare un indebolimento del trend e una potenziale inversione.
            * *Timeframe:* L'RSI **1h** indica il momentum a brevissimo termine (scalping, intraday). L'RSI **1d** è utile per lo swing trading a breve-medio termine. L'RSI **1w** mostra il quadro a medio-lungo termine. (Nota: Dati 1h/1w dipendono dalla disponibilità API).
    * **MACD Hist (Moving Average Convergence Divergence Histogram - 1d):**
        * *Cos'è:* Visualizza la differenza tra la linea MACD (EMA 12 - EMA 26) e la Signal Line (EMA 9 della linea MACD). Misura il momentum del trend.
        * *Interpretazione:*
            * *Istogramma > 0 (Positivo):* Momentum rialzista in atto (linea MACD sopra Signal). Barre crescenti indicano accelerazione rialzista.
            * *Istogramma < 0 (Negativo):* Momentum ribassista in atto (linea MACD sotto Signal). Barre decrescenti (più negative) indicano accelerazione ribassista.
            * *Crossover dello Zero:* Il passaggio da negativo a positivo è un segnale di acquisto potenziale; da positivo a negativo è un segnale di vendita potenziale.
            * *Divergenze:* Come per l'RSI, divergenze tra prezzo e istogramma possono preannunciare inversioni.

    **Indicatori di Trend:**
    * **MA (Simple Moving Average - 20d, 50d):**
        * *Cos'è:* Media mobile semplice (aritmetica) dei prezzi di chiusura degli ultimi N giorni. Usata per smussare il prezzo e identificare la direzione generale del trend.
        * *Interpretazione:*
            * *Posizione Prezzo:* Prezzo sopra la MA = tendenza rialzista; Prezzo sotto la MA = tendenza ribassista. Le MA agiscono spesso come livelli di supporto (se il prezzo è sopra) o resistenza (se il prezzo è sotto) dinamici.
            * *Pendenza MA:* Una MA crescente indica un trend rialzista, una decrescente un trend ribassista.
            * *Crossover MA (Golden/Death Cross):* MA 20d che incrocia *sopra* MA 50d ("Golden Cross") è un segnale rialzista di medio termine. MA 20d che incrocia *sotto* MA 50d ("Death Cross") è un segnale ribassista di medio termine.
    * **VWAP (Volume Weighted Average Price - 1d):**
        * *Cos'è:* Prezzo medio ponderato per il volume in un dato periodo. Indica il prezzo "medio" a cui la maggior parte del volume è stata scambiata.
        * *Interpretazione:* Spesso usato come benchmark intraday dai trader istituzionali. Se il prezzo è sopra il VWAP, i rialzisti sono considerati in controllo; se sotto, i ribassisti. Può agire da supporto/resistenza. *(Nota: Il calcolo qui usa dati giornalieri e HLC approssimati, rendendolo meno preciso di un VWAP intraday reale).*

    **Segnali Combinati (Esemplificativi):**
    * **GPT Signal:**
        * *Cos'è:* Un segnale direzionale generale (da Strong Buy a Strong Sell) basato su una **combinazione pesata** di diversi indicatori (MA Crossover, MACD Hist, RSI 1d, 1w, 1h). Cerca di dare un quadro d'insieme del sentiment tecnico.
        * *Logica:* Assegna punteggi basati sullo stato di ciascun indicatore (trend, momentum, ipercomprato/venduto su vari timeframe) e li somma per ottenere un giudizio complessivo.
        * *Disclaimer:* **Esemplificativo, non è consulenza finanziaria.** Utile per avere una sintesi rapida, ma va interpretato nel contesto e non seguito ciecamente.
    * **Gemini Alert:**
        * *Cos'è:* Un alert più **specifico e stringente** che si attiva solo quando c'è una **forte confluenza** di segnali rialzisti o ribassisti basati sugli indicatori **giornalieri** principali (MA Crossover, MACD Hist positivo/negativo, RSI non in estremo).
        * *Logica:*
            * `✅ BUY Signal`: Appare solo se MA20 > MA50 *E* MACD Hist > 0 *E* RSI < 70.
            * `❌ SELL Signal`: Appare solo se MA20 < MA50 *E* MACD Hist < 0 *E* RSI > 30.
            * `➖` / `⏳`: Nessun alert forte o dati insufficienti.
        * *Disclaimer:* Anche questo è un segnale **esemplificativo e NON è consulenza finanziaria.** Rappresenta una specifica strategia basata su confluenza, ma non garantisce risultati.

    **Generale:**
    * **N/A:** Indica che un dato o indicatore non è disponibile o non è stato possibile calcolarlo (es. crypto nuova, dati storici/orari insufficienti da CoinGecko, errore API).
    """)

st.markdown(f"⏱️ Prezzi live aggiornati da CoinGecko circa alle: **{last_cg_update.strftime('%Y-%m-%d %H:%M:%S')}** (Cache: {CACHE_TTL}s)")