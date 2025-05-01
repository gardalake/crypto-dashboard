# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time
import math
import yfinance as yf

# --- Configurazione Globale ---
SYMBOL_TO_ID_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "RNDR": "render-token",
    "FET": "artificial-superintelligence-alliance",
    "TAO": "bittensor", "LINK": "chainlink", "AVAX": "avalanche-2",
    "HBAR": "hedera-hashgraph", "PEPE": "pepe", "UNI": "uniswap",
    "TIA": "celestia", "JUP": "jupiter-aggregator", "IMX": "immutable-x",
    "TRUMP": "maga", "NEAR": "near-protocol", "AERO": "aerodrome-finance",
    "TRON": "tron", "AERGO": "aergo", "ADA": "cardano", "MKR": "maker"
}
SYMBOLS = list(SYMBOL_TO_ID_MAP.keys())
COINGECKO_IDS_LIST = list(SYMBOL_TO_ID_MAP.values())
NUM_COINS = len(SYMBOLS)
TRAD_TICKERS = ['^GSPC', '^IXIC', 'GC=F', 'UVXY', 'TQQQ']
VS_CURRENCY = "usd"; CACHE_TTL = 1800 # 30 min
DAYS_HISTORY_DAILY = 365; DAYS_HISTORY_HOURLY = 7
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
             if st.query_params.get("rerun") != "false":
                  st.query_params["rerun"] = "false"; st.rerun()
        elif password: st.warning("Password errata."); st.stop()
        else: st.stop()
    return True
if not check_password(): st.stop()

# --- Funzioni API CoinGecko ---
@st.cache_data(ttl=CACHE_TTL, show_spinner="Caricamento dati di mercato (CoinGecko)...")
def get_coingecko_market_data(ids_list, currency):
    ids_string = ",".join(ids_list)
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': currency, 'ids': ids_string, 'order': 'market_cap_desc',
              'per_page': str(len(ids_list)), 'page': 1, 'sparkline': False,
              'price_change_percentage': '1h,24h,7d,30d,1y'} # Aggiunto 30d, 1y
    timestamp = datetime.now()
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status(); data = response.json()
        df = pd.DataFrame(data);
        if not df.empty: df.set_index('id', inplace=True)
        return df, timestamp
    except requests.exceptions.RequestException as req_ex:
        status_code = req_ex.response.status_code if req_ex.response is not None else "N/A"
        st.error(f"Errore API Mercato CoinGecko (Status: {status_code}): {req_ex}")
        return pd.DataFrame(), timestamp
    except Exception as e: st.error(f"Errore Processamento Dati Mercato CoinGecko: {e}"); return pd.DataFrame(), timestamp

@st.cache_data(ttl=CACHE_TTL * 2, show_spinner=False)
def get_coingecko_historical_data(coin_id, currency, days, interval='daily'):
    time.sleep(1.5)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': currency, 'days': str(days),
              'interval': interval if interval == 'hourly' else 'daily', 'precision': 'full'}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status(); data = response.json()
        if not data or 'prices' not in data or not data['prices']: return pd.DataFrame(), "No Prices Data"
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
        if hist_df.empty: return pd.DataFrame(), "Processed Empty"
        return hist_df, "Success"
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 429: return pd.DataFrame(), "Rate Limited (429)"
        else: return pd.DataFrame(), f"HTTP Error {http_err.response.status_code}"
    except Exception as e: return pd.DataFrame(), f"Generic Error: {type(e).__name__}"

# --- Funzioni Dati Mercato Generale ---
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_fear_greed_index_cmc():
    url = "https://pro-api.coinmarketcap.com/v1/crypto/fear-and-greed" # Endpoint corretto?
    # Prova senza chiave API, potrebbe richiedere registrazione gratuita a CMC per una chiave
    # headers = {'X-CMC_PRO_API_KEY': 'TUA_CHIAVE_API_CMC'}
    try:
        response = requests.get(url, timeout=10) # senza headers
        response.raise_for_status(); data = response.json()
        if data and data.get("data") and isinstance(data["data"], list) and len(data["data"]) > 0:
             latest_data = data["data"][0]; value = latest_data.get("score"); desc = latest_data.get("rating")
             if value is not None and desc is not None: return f"{int(value)} ({desc})"
        return "N/A"
    except requests.exceptions.HTTPError as http_err:
         if http_err.response.status_code in [401, 403]: return "N/A (API Key?)"
         else: return f"N/A (Err {http_err.response.status_code})"
    except Exception: return "N/A (Errore)"

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_global_market_data_cg(currency):
    url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status(); data = response.json().get('data', {})
        total_mcap = data.get('total_market_cap', {}).get(currency.lower(), np.nan)
        return total_mcap
    except Exception: return np.nan

def get_etf_flow(): return "N/A" # Placeholder

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_traditional_market_data_yf(tickers):
    data = {}
    for ticker_sym in tickers:
        try:
            time.sleep(0.2); ticker_obj = yf.Ticker(ticker_sym); info = ticker_obj.info
            price = info.get('regularMarketPrice', info.get('currentPrice', info.get('previousClose', np.nan)))
            data[ticker_sym] = price
        except Exception: data[ticker_sym] = np.nan
    return data

# --- Funzioni Calcolo Indicatori (Manuali) ---
def calculate_rsi_manual(series, period=RSI_PERIOD):
    # ... (codice invariato) ...
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
    # ... (codice invariato) ...
    rsi_series = pd.Series(dtype=float)
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
    # ... (codice invariato) ...
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
    # ... (codice invariato) ...
    if not isinstance(series, pd.Series) or series.isna().all(): return np.nan
    series = series.dropna()
    if len(series) < period: return np.nan
    return series.rolling(window=period).mean().iloc[-1]

def calculate_vwap_manual(df, period=VWAP_PERIOD):
    # ... (codice invariato) ...
    if not all(col in df.columns for col in ['close', 'volume']) or df['close'].isna().all() or df['volume'].isna().all(): return np.nan
    df_period = df.iloc[-period:].copy(); df_period.dropna(subset=['close', 'volume'], inplace=True)
    if df_period.empty or df_period['volume'].sum() == 0: return np.nan
    vwap = (df_period['close'] * df_period['volume']).sum() / df_period['volume'].sum()
    return vwap

# --- Funzione Raggruppata Indicatori ---
def compute_all_indicators(symbol, hist_daily_df, hist_hourly_df, fetch_errors_list):
    # ... (codice invariato, calcola e ritorna dict 'indicators') ...
    indicators = { "RSI (1h)": np.nan, "RSI (1d)": np.nan, "RSI (1w)": np.nan, "RSI (1mo)": np.nan,
                   "SRSI %K (1d)": np.nan, "SRSI %D (1d)": np.nan, "MACD Line (1d)": np.nan,
                   "MACD Signal (1d)": np.nan, "MACD Hist (1d)": np.nan, f"MA({MA_SHORT}d)": np.nan,
                   f"MA({MA_LONG}d)": np.nan, "VWAP (1d)": np.nan, "Doda Stoch": "N/A",
                   "GChannel": "N/A", "Volume Flow": "N/A"}
    min_len_daily_full = max(RSI_PERIOD + 1, SRSI_PERIOD + RSI_PERIOD + 1, MACD_SLOW + MACD_SIGNAL, MA_LONG)
    min_len_rsi_base = RSI_PERIOD + 1
    if not hist_daily_df.empty:
        close_daily = hist_daily_df['close']; len_daily = len(close_daily.dropna())
        if len_daily >= min_len_daily_full:
            indicators["RSI (1d)"] = calculate_rsi_manual(close_daily)
            indicators["SRSI %K (1d)"], indicators["SRSI %D (1d)"] = calculate_stoch_rsi(close_daily)
            macd_l, macd_s, macd_h = calculate_macd_manual(close_daily)
            indicators["MACD Line (1d)"] = macd_l; indicators["MACD Signal (1d)"] = macd_s; indicators["MACD Hist (1d)"] = macd_h
            indicators[f"MA({MA_SHORT}d)"] = calculate_sma_manual(close_daily, MA_SHORT)
            indicators[f"MA({MA_LONG}d)"] = calculate_sma_manual(close_daily, MA_LONG)
            indicators["VWAP (1d)"] = calculate_vwap_manual(hist_daily_df)
        else: fetch_errors_list.append(f"Dati Daily insuff. ({len_daily}/{min_len_daily_full}) per ind. base per {symbol}.")
        try: # Weekly
            df_weekly = close_daily.resample('W-MON').last()
            if len(df_weekly.dropna()) >= min_len_rsi_base: indicators["RSI (1w)"] = calculate_rsi_manual(df_weekly)
            else: fetch_errors_list.append(f"Dati insuff. ({len(df_weekly.dropna())}/{min_len_rsi_base} sett.) per RSI 1w per {symbol}.")
        except Exception as e: fetch_errors_list.append(f"Errore resampling weekly per {symbol}: {e}")
        try: # Monthly
            df_monthly = close_daily.resample('ME').last()
            if len(df_monthly.dropna()) >= min_len_rsi_base: indicators["RSI (1mo)"] = calculate_rsi_manual(df_monthly)
            else: fetch_errors_list.append(f"Dati insuff. ({len(df_monthly.dropna())}/{min_len_rsi_base} mesi) per RSI 1mo per {symbol}.")
        except Exception as e: fetch_errors_list.append(f"Errore resampling monthly per {symbol}: {e}")
    if not hist_hourly_df.empty:
        len_hourly = len(hist_hourly_df['close'].dropna())
        if len_hourly >= min_len_rsi_base: indicators["RSI (1h)"] = calculate_rsi_manual(hist_hourly_df['close'])
        else: fetch_errors_list.append(f"Dati Hourly insuff. ({len_hourly}/{min_len_rsi_base} ore) per RSI 1h per {symbol}.")
    return indicators

# --- Funzioni Segnale (SINTASSI CORRETTA) ---
def generate_gpt_signal(rsi_1d, rsi_1h, rsi_1w, macd_hist, ma_short, ma_long, srsi_k, srsi_d, current_price):
    required_inputs = [rsi_1d, macd_hist, ma_short, ma_long, current_price]
    if any(pd.isna(x) for x in required_inputs): return "⚪️ N/D"
    score = 0
    # 1. Trend di fondo (MA Long vs Prezzo) - Corretto
    if current_price > ma_long:
        score += 1
    else:
        score -= 1
    # 2. Trend Medio Termine (MA Crossover) - Corretto
    if ma_short > ma_long:
        score += 2
    else:
        score -= 2
    # 3. Momentum (MACD Histogram) - Corretto
    if macd_hist > 0:
        score += 2
    else:
        score -= 2
    # 4. Ipercomprato/venduto Daily (RSI 1d)
    if rsi_1d < 30: score += 2
    elif rsi_1d < 40: score += 1
    elif rsi_1d > 70: score -= 2
    elif rsi_1d > 60: score -= 1
    # 5. Ipercomprato/venduto Weekly (RSI 1w)
    if not pd.isna(rsi_1w):
        if rsi_1w < 30: score += 1
        elif rsi_1w > 70: score -= 1
    # 6. Forza a Breve (RSI 1h)
    if not pd.isna(rsi_1h):
        if rsi_1h > 60: score += 1
        elif rsi_1h < 40: score -= 1
    # 7. SRSI (StochRSI Daily)
    if not pd.isna(srsi_k) and not pd.isna(srsi_d):
        if srsi_k < 20 and srsi_d < 20: score += 1
        elif srsi_k > 80 and srsi_d > 80: score -= 1
    # Mappatura Score
    if score >= 5: return "⚡️ Strong Buy"
    elif score >= 2: return "🟢 Buy"
    elif score <= -5: return "🚨 Strong Sell"
    elif score <= -2: return "🔴 Sell"
    elif score >= 0: return "⏳ CTB" if not pd.isna(rsi_1d) and rsi_1d < 45 and rsi_1d > 30 else "🟡 Hold"
    else: return "⚠️ CTS" if not pd.isna(rsi_1d) and rsi_1d > 55 and rsi_1d < 70 else "🟡 Hold"

def generate_gemini_alert(ma_short, ma_long, macd_hist, rsi_1d):
    # ... (codice invariato, già corretto) ...
    if pd.isna(ma_short) or pd.isna(ma_long) or pd.isna(macd_hist) or pd.isna(rsi_1d): return "⏳"
    is_uptrend = ma_short > ma_long; is_momentum_positive = macd_hist > 0; is_not_overbought = rsi_1d < 70
    if is_uptrend and is_momentum_positive and is_not_overbought: return "✅" # BUY Emoji
    is_downtrend = ma_short < ma_long; is_momentum_negative = macd_hist < 0; is_not_oversold = rsi_1d > 30
    if is_downtrend and is_momentum_negative and is_not_oversold: return "❌" # SELL Emoji
    return "➖" # Neutro Emoji

# --- Layout App Streamlit ---
st.set_page_config(layout="wide", page_title="Crypto Tech Dashboard Pro", page_icon="📈")

# --- Titolo e Bottone Aggiorna (Spostato) ---
col_title, col_button_placeholder, col_button = st.columns([4, 1, 1]) # Aggiusta spazi se serve
with col_title:
    st.title("📈 Crypto Technical Dashboard Pro") # Titolo aggiornato
with col_button:
    st.write("") # Spazio per allineare
    if st.button("🔄 Aggiorna", help="Forza l'aggiornamento dei dati"):
        st.cache_data.clear(); st.rerun()

# Timestamp
last_cg_update_placeholder = st.empty()
st.caption(f"Dati CoinGecko (cache {CACHE_TTL/60:.0f}min live / {CACHE_TTL*2/60:.0f}min storico). Dati Mercato Tradizionale (cache {CACHE_TTL/60:.0f}min).")


# --- Sezione Market Overview ---
st.markdown("---")
st.subheader("🌐 Market Overview")
fear_greed = get_fear_greed_index_cmc()
total_mcap = get_global_market_data_cg(VS_CURRENCY)
etf_flow = get_etf_flow() # Placeholder N/A
trad_data = get_traditional_market_data_yf(TRAD_TICKERS)
mkt_col1, mkt_col2, mkt_col3, mkt_col4 = st.columns(4)
with mkt_col1:
    st.metric(label="Fear & Greed (CMC)", value=fear_greed, help="Potrebbe richiedere API Key CMC gratuita.")
    st.metric(label="S&P 500 (^GSPC)", value=f"{trad_data.get('^GSPC', 0):,.2f}" if not pd.isna(trad_data.get('^GSPC')) else "N/A")
with mkt_col2:
    st.metric(label=f"Total Crypto M.Cap ({VS_CURRENCY.upper()})", value=f"${total_mcap:,.0f}" if not pd.isna(total_mcap) else "N/A")
    st.metric(label="Nasdaq (^IXIC)", value=f"{trad_data.get('^IXIC', 0):,.2f}" if not pd.isna(trad_data.get('^IXIC')) else "N/A")
with mkt_col3:
    st.metric(label="Crypto ETFs Flow (Daily)", value=etf_flow, help="Dato non disponibile via API gratuita.")
    st.metric(label="Gold (GC=F)", value=f"{trad_data.get('GC=F', 0):,.2f}" if not pd.isna(trad_data.get('GC=F')) else "N/A")
with mkt_col4:
    st.metric(label="UVXY (Volatility)", value=f"{trad_data.get('UVXY', 0):,.2f}" if not pd.isna(trad_data.get('UVXY')) else "N/A")
    st.metric(label="TQQQ (Nasdaq 3x)", value=f"{trad_data.get('TQQQ', 0):,.2f}" if not pd.isna(trad_data.get('TQQQ')) else "N/A")
st.markdown("---")

# --- Logica Principale Dashboard Crypto ---
market_data_df, last_cg_update_utc = get_coingecko_market_data(COINGECKO_IDS_LIST, VS_CURRENCY)
last_cg_update_rome = last_cg_update_utc + timedelta(hours=2)
last_cg_update_placeholder.markdown(f"*Prezzi Live aggiornati alle: **{last_cg_update_rome.strftime('%Y-%m-%d %H:%M:%S')} (Ora di Roma)***")

if market_data_df.empty: st.error("Errore caricamento dati crypto CoinGecko."); st.stop()

results = []; fetch_errors = []
progress_bar = st.progress(0, text=f"Analisi {NUM_COINS} Criptovalute...")
coin_ids_ordered = market_data_df.index.tolist() # Usa ordine da API /markets

for i, coin_id in enumerate(coin_ids_ordered):
    if coin_id not in market_data_df.index: continue # Salta se ID non trovato nei dati live
    live_data = market_data_df.loc[coin_id]
    symbol = live_data.get('symbol', coin_id).upper(); name = live_data.get('name', coin_id)
    rank = live_data.get('market_cap_rank', 'N/A'); current_price = live_data.get('current_price', np.nan)
    change_1h = live_data.get('price_change_percentage_1h_in_currency', np.nan)
    change_24h = live_data.get('price_change_percentage_24h_in_currency', np.nan)
    change_7d = live_data.get('price_change_percentage_7d_in_currency', np.nan)
    change_30d = live_data.get('price_change_percentage_30d_in_currency', np.nan) # Nuovo
    change_1y = live_data.get('price_change_percentage_1y_in_currency', np.nan)   # Nuovo
    volume_24h = live_data.get('total_volume', np.nan)

    hist_daily_df, status_daily = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_DAILY, interval='daily')
    hist_hourly_df, status_hourly = get_coingecko_historical_data(coin_id, VS_CURRENCY, DAYS_HISTORY_HOURLY, interval='hourly')

    if status_daily != "Success": fetch_errors.append(f"{symbol}: Dati Daily - {status_daily}")
    if status_hourly != "Success": fetch_errors.append(f"{symbol}: Dati Hourly - {status_hourly}")

    indicators = compute_all_indicators(symbol, hist_daily_df, hist_hourly_df, fetch_errors)
    gpt_signal = generate_gpt_signal( indicators.get("RSI (1d)"), indicators.get("RSI (1h)"), indicators.get("RSI (1w)"), indicators.get("MACD Hist (1d)"), indicators.get(f"MA({MA_SHORT}d)"), indicators.get(f"MA({MA_LONG}d)"), indicators.get("SRSI %K (1d)"), indicators.get("SRSI %D (1d)"), current_price)
    gemini_alert = generate_gemini_alert( indicators.get(f"MA({MA_SHORT}d)"), indicators.get(f"MA({MA_LONG}d)"), indicators.get("MACD Hist (1d)"), indicators.get("RSI (1d)"))

    results.append({
        "Rank": rank, "Symbol": symbol, "Name": name, "Gemini Alert": gemini_alert, "GPT Signal": gpt_signal,
        f"Prezzo ({VS_CURRENCY.upper()})": current_price, "% 4h": "N/A", # Aggiunto N/A per 4h
        "% 1h": change_1h, "% 24h": change_24h, "% 7d": change_7d, "% 30d": change_30d, "% 1y": change_1y,
        "RSI (1h)": indicators.get("RSI (1h)"), "RSI (1d)": indicators.get("RSI (1d)"),
        "RSI (1w)": indicators.get("RSI (1w)"), "RSI (1mo)": indicators.get("RSI (1mo)"),
        "SRSI %K (1d)": indicators.get("SRSI %K (1d)"), "SRSI %D (1d)": indicators.get("SRSI %D (1d)"),
        "MACD Hist (1d)": indicators.get("MACD Hist (1d)"),
        f"MA({MA_SHORT}d)": indicators.get(f"MA({MA_SHORT}d)"), f"MA({MA_LONG}d)": indicators.get(f"MA({MA_LONG}d)"),
        "VWAP (1d)": indicators.get("VWAP (1d)"), f"Volume 24h ({VS_CURRENCY.upper()})": volume_24h,
        "Doda Stoch": "N/A", "GChannel": "N/A", "Volume Flow": "N/A",
    })
    progress_bar.progress((i + 1) / len(coin_ids_ordered), text=f"Analisi Criptovalute... ({symbol})")

progress_bar.empty()

if fetch_errors:
    with st.expander("ℹ️ Note Recupero Dati / Calcolo Indicatori", expanded=False):
        unique_errors = sorted(list(set(fetch_errors)));
        for error_msg in unique_errors: st.info(error_msg)

if results:
    df = pd.DataFrame(results); df.set_index('Rank', inplace=True)

    # Ordine colonne aggiornato
    cols_order = [
        "Symbol", "Name", "Gemini Alert", "GPT Signal",
        f"Prezzo ({VS_CURRENCY.upper()})",
        "% 4h", "% 1h", "% 24h", "% 7d", "% 30d", "% 1y", # Aggiornato ordine %
        "RSI (1h)", "RSI (1d)", "RSI (1w)", "RSI (1mo)",
        "SRSI %K (1d)", "SRSI %D (1d)",
        "MACD Hist (1d)", f"MA({MA_SHORT}d)", f"MA({MA_LONG}d)", "VWAP (1d)",
        f"Volume 24h ({VS_CURRENCY.upper()})"
    ]
    cols_to_show = [col for col in cols_order if col in df.columns]
    df_display = df[cols_to_show].copy()

    # Formattazione manuale
    formatters = {}; currency_col = f"Prezzo ({VS_CURRENCY.upper()})"; volume_col = f"Volume 24h ({VS_CURRENCY.upper()})";
    pct_cols = ["% 4h", "% 1h", "% 24h", "% 7d", "% 30d", "% 1y"]
    for col in df_display.columns:
        # Applica solo a colonne non-oggetto o che non sono già formattate come N/A
        if col in formatters or df_display[col].dtype == 'object': continue # Salta se già formattato o stringa

        if pd.api.types.is_numeric_dtype(df_display[col].dtype):
             if col == currency_col: formatters[col] = lambda x: f"${x:,.4f}" if pd.notna(x) else "N/A"
             elif col in pct_cols: formatters[col] = lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
             elif col == volume_col: formatters[col] = lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
             elif "RSI" in col or "SRSI" in col: formatters[col] = lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
             elif "MACD" in col: formatters[col] = lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
             elif "MA" in col or "VWAP" in col: formatters[col] = lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
             else: formatters[col] = lambda x: str(x) if pd.notna(x) else "N/A" # Fallback
             df_display[col] = df_display[col].apply(formatters[col])

    df_display.fillna("N/A", inplace=True) # Applica fillna finale

    # Stili (invariati)
    def highlight_pct_col(col):#... (come prima)
        colors = [''] * len(col)
        for i, val in enumerate(col):
            if isinstance(val, str) and val.endswith('%') and val != 'N/A':
                try: num = float(val.replace('%','')); colors[i] = 'color: green' if num > 0 else 'color: red' if num < 0 else ''
                except ValueError: pass
        return colors
    def highlight_gemini_alert(col): # Stile Alert Emoji aggiornato
        colors = [''] * len(col)
        for i, val in enumerate(col):
            if val == "✅": colors[i] = 'color: #28a745; font-weight: bold;' # Verde
            elif val == "❌": colors[i] = 'color: #dc3545; font-weight: bold;' # Rosso
            elif val == "⏳": colors[i] = 'color: #ffc107;' # Giallo
        return colors
    def highlight_gpt_signal(col): #... (come prima)
        colors = [''] * len(col)
        for i, val in enumerate(col):
            if isinstance(val, str):
                 if "Buy" in val: colors[i] = 'color: #198754;'
                 elif "Sell" in val: colors[i] = 'color: #dc3545;'
                 elif "CTB" in val: colors[i] = 'color: #20c997;'
                 elif "CTS" in val: colors[i] = 'color: #fd7e14;'
                 elif "Hold" in val: colors[i] = 'color: #6c757d;'
        return colors

    styled_df = df_display.style
    for col_name in pct_cols:
        if col_name in df_display.columns: styled_df = styled_df.apply(highlight_pct_col, subset=[col_name], axis=0)
    if "Gemini Alert" in df_display.columns: styled_df = styled_df.apply(highlight_gemini_alert, subset=["Gemini Alert"], axis=0)
    if "GPT Signal" in df_display.columns: styled_df = styled_df.apply(highlight_gpt_signal, subset=["GPT Signal"], axis=0)

    st.dataframe(styled_df, use_container_width=True)
else: st.warning("Nessun risultato crypto da visualizzare.")

# --- Legenda Aggiornata ---
st.divider()
with st.expander("📘 Legenda Indicatori Tecnici e Segnali", expanded=False):
    st.markdown(f"""
    *Disclaimer: Solo a scopo informativo, non costituisce consulenza finanziaria.*

    **Market Overview:**
    * **Fear & Greed (CMC):** Indice di sentiment (0-100) da CoinMarketCap (potrebbe richiedere API Key).
    * **Total Crypto M.Cap:** Capitalizzazione totale mercato crypto (da CoinGecko).
    * **Crypto ETFs Flow:** Flusso netto giornaliero negli ETF crypto spot (Dato N/A - fonte non disponibile).
    * **S&P 500, Nasdaq, Gold, UVXY, TQQQ:** Prezzi indicativi indici/asset tradizionali (da yfinance, potrebbero essere N/A).

    **Tabella Crypto:**
    * **Variazioni Percentuali:** % 1h, 24h, 7d, 30d, 1y (da CoinGecko). *(% 4h/15m non disponibili).*
    * **RSI (1h, 1d, 1w, 1mo):** Momentum (0-100). `>70` Overbought, `<30` Oversold. *(Dati H/W/M dipendono da API).*
    * **SRSI %K / %D (1d):** Stocastico dell'RSI (0-100). `>80` Overbought, `<20` Oversold. Identifica cicli brevi.
    * **MACD Hist (1d):** Momentum trend. `>0` Buy, `<0` Sell.
    * **MA (SMA - 20d, 50d):** Medie Mobili Semplici. Trend e Supporto/Resistenza.
    * **VWAP (1d):** Prezzo medio ponderato volume. *(Calc. approx).*
    * **Gemini Alert:** `✅` BUY (MA20>MA50 & MACD Hist>0 & RSI<70). `❌` SELL (MA20<MA50 & MACD Hist<0 & RSI>30). `➖` Neutro. `⏳` Dati insuff. **(NON CONSULENZA FINANZIARIA)**.
    * **GPT Signal:** Sintesi generale pesata (MAs, MACD, RSI 1d/1w/1h, SRSI). Include `⏳ CTB` (Close to Buy), `⚠️ CTS` (Close to Sell). **(NON CONSULENZA FINANZIARIA)**.
    * **N/A:** Dato/Indicatore non disponibile o non calcolabile.
    """)

# --- Footer ---
st.divider()
st.caption(f"Disclaimer: Questa dashboard è solo a scopo informativo e non costituisce consulenza finanziaria.")