import pandas as pd
import ta
import streamlit as st
import yfinance as yf
import requests
import json
import traceback

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

# --- MODIFICATA PER TOP 10 ---
@st.cache_data(ttl=600, show_spinner="Fetching top 10 crypto list...") # Messaggio spinner aggiornato
def get_top_10_crypto_symbols(): # Nome funzione aggiornato
    """Fetches top 10 crypto symbols by market cap from CoinGecko."""
    # Lista fallback con 10 simboli comuni
    fallback_list = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'USDC-USD', 'DOGE-USD', 'ADA-USD', 'SHIB-USD']
    try:
        url = 'https://api.coingecko.com/api/v3/coins/markets'
        # --- MODIFICATO: per_page=10 ---
        params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 10, 'page': 1}
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        tickers = []
        for coin in data:
            # Filtro nomi strani (invariato)
            if 'symbol' in coin:
                symbol_upper = coin['symbol'].upper()
                ticker = f"{symbol_upper}-USD"
                if not symbol_upper or len(symbol_upper) < 2 or "-USD" in symbol_upper or " " in ticker or ticker.count('-') > 1:
                    continue
                tickers.append(ticker)

        if not tickers:
             st.warning("Nessun ticker valido ricevuto o filtrato da CoinGecko API (Top 10).")
             return fallback_list
        # Assicura che non ci siano più di 10 simboli (dovrebbe essere già così)
        return tickers[:10]
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
        st.error(f"Errore imprevisto in get_top_10_crypto_symbols: {e}\n{traceback.format_exc()}")
        return fallback_list
# --- FINE MODIFICA ---


@st.cache_data(ttl=300, show_spinner="Calculating indicators...")
def fetch_indicators_with_signals(symbol, interval, period):
    """Fetches data using yfinance and calculates technical indicators. Returns dataframe and signals dict."""
    signals = {
        'RSI': '⚪ N/A', 'SRSI': '⚪ N/A', 'MACD': '⚪ N/A', 'MA': '⚪ N/A',
        'Doda Stoch': '⚪ N/A', 'GChannel': '⚪ N/A', 'Vol Flow': '⚪ N/A', 'VWAP': '⚪ N/A'
    }
    df = pd.DataFrame()
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    try:
        request_period = period
        if interval == '1wk': request_period = '8mo' if period == '6mo' else period
        if interval == '1mo': request_period = '26mo' if period == '2y' else period

        df = yf.download(tickers=symbol, interval=interval, period=request_period, progress=False, ignore_tz=True)

        if df.empty or not all(col in df.columns for col in required_columns):
            return df, signals

        df.dropna(subset=required_columns, inplace=True)

        min_required_len = max(RSI_WINDOW, MA_WINDOW, SRSI_WINDOW, MACD_SLOW, STOCH_WINDOW, MFI_WINDOW, EMA_LONG, 2) + 5
        if df.empty or len(df) < min_required_len:
            return df, signals

        if not isinstance(df.index, pd.DatetimeIndex):
           df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]

        df['high'] = pd.to_numeric(df['High'], errors='coerce')
        df['low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df.dropna(subset=['high', 'low', 'close', 'volume'], inplace=True)
        if df.empty or len(df) < min_required_len:
             return df, signals

        # --- Funzione Helper CORRETTA per Calcolo Sicuro Indicatore ---
        def _safe_calculate(indicator_func, *args, **kwargs):
            try:
                indicator_series = indicator_func(*args, **kwargs) # Chiamata alla funzione 'ta'
                if indicator_series is None or not isinstance(indicator_series, pd.Series) or indicator_series.empty:
                    return float('nan')
                indicator_series = indicator_series.replace([float('inf'), -float('inf')], float('nan')).dropna()
                if indicator_series.empty:
                     return float('nan')
                last_val = indicator_series.iloc[-1]
                return last_val if pd.notna(last_val) else float('nan')
            except Exception:
                return float('nan')
        # --- Fine Funzione Helper CORRETTA ---


        # (Calcolo indicatori usando helper invariato)
        rsi_val = _safe_calculate(ta.momentum.RSIIndicator(df['close'], window=RSI_WINDOW).rsi)
        srsi_val = _safe_calculate(ta.momentum.StochRSIIndicator(df['close'], window=SRSI_WINDOW).stochrsi)
        macd_obj = ta.trend.MACD(df['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGN)
        macd_val = _safe_calculate(macd_obj.macd)
        ma_val = _safe_calculate(df['close'].rolling(window=MA_WINDOW).mean)
        doda_val = _safe_calculate(ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=STOCH_WINDOW).stoch)
        ema_short_series = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
        ema_long_series = df['close'].ewm(span=EMA_LONG, adjust=False).mean()
        gchannel_series = ema_short_series - ema_long_series
        gchannel_val = _safe_calculate(lambda s: s, gchannel_series)
        vfi_val = _safe_calculate(ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=MFI_WINDOW).money_flow_index)
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            volume_cumsum = df['volume'].cumsum()
            vwap_series = ((typical_price * df['volume']).cumsum()) / volume_cumsum.replace(0, float('nan'))
            vwap_series = vwap_series.replace([float('inf'), -float('inf')], float('nan')).dropna()
            vwap_val = vwap_series.iloc[-1] if not vwap_series.empty and pd.notna(vwap_series.iloc[-1]) else float('nan')
        except Exception: vwap_val = float('nan')

        # (Funzione signal_emoji invariata)
        def signal_emoji(value, low, high, strong_factor=0.33):
            if pd.isna(value): return '⚪ N/A'
            low_strong = low * (1 - strong_factor)
            high_strong = high * (1 + strong_factor)
            if value < low_strong: return '🔶'
            elif value < low: return '🟢'
            elif value > high_strong: return '🔻'
            elif value > high: return '🔴'
            else: return '🟡'

        # (Assegnazione segnali invariata)
        signals['RSI'] = f"{rsi_val:.2f} {signal_emoji(rsi_val, RSI_LOW, RSI_HIGH)}" if pd.notna(rsi_val) else '⚪ N/A'
        signals['SRSI'] = f"{srsi_val:.2f} {signal_emoji(srsi_val, SRSI_LOW, SRSI_HIGH)}" if pd.notna(srsi_val) else '⚪ N/A'
        signals['MACD'] = f"{macd_val:.2f} {signal_emoji(macd_val, MACD_THRESHOLD_LOW, MACD_THRESHOLD_HIGH, strong_factor=0.5)}" if pd.notna(macd_val) else '⚪ N/A'
        signals['MA'] = f"{ma_val:.2f}" if pd.notna(ma_val) else '⚪ N/A'
        signals['Doda Stoch'] = f"{doda_val:.2f} {signal_emoji(doda_val, STOCH_LOW, STOCH_HIGH)}" if pd.notna(doda_val) else '⚪ N/A'
        if pd.notna(gchannel_val):
            gc_signal = '🟢' if gchannel_val > 0 else ('🔴' if gchannel_val < 0 else '🟡')
            signals['GChannel'] = f"{gchannel_val:.2f} {gc_signal}"
        else: signals['GChannel'] = '⚪ N/A'
        signals['Vol Flow'] = f"{vfi_val:.2f} {signal_emoji(vfi_val, MFI_LOW, MFI_HIGH)}" if pd.notna(vfi_val) else '⚪ N/A'
        last_close = df['close'].iloc[-1] if not df['close'].empty else float('nan')
        if pd.notna(vwap_val) and pd.notna(last_close):
            vwap_signal = '🟢' if last_close > vwap_val else ('🔴' if last_close < vwap_val else '🟡')
            signals['VWAP'] = f"{vwap_val:.2f} {vwap_signal}"
        elif pd.notna(vwap_val): signals['VWAP'] = f"{vwap_val:.2f}"
        else: signals['VWAP'] = '⚪ N/A'

        return df, signals

    # Gestione eccezioni *esterne* al calcolo degli indicatori (es. errore download)
    except Exception as e:
        # print(f"Debug: Errore Exception generico in fetch_indicators_with_signals per {symbol} ({interval}): {e}")
        # print(traceback.format_exc()) # Utile per debug profondo se necessario
        return df, signals # Restituisce df vuoto e segnali N/A


# (Funzione calculate_price_change invariata)
def calculate_price_change(daily_df):
    if daily_df is None or not isinstance(daily_df, pd.DataFrame) or daily_df.empty or 'Close' not in daily_df.columns:
        return "N/A", 0.0
    try:
        if not isinstance(daily_df.index, pd.DatetimeIndex):
           daily_df.index = pd.to_datetime(daily_df.index)
        daily_df = daily_df.sort_index()
        daily_df = daily_df[~daily_df.index.duplicated(keep='first')]
    except Exception:
         return "N/A", 0.0
    valid_closes = pd.to_numeric(daily_df['Close'], errors='coerce').dropna()
    if len(valid_closes) < 2:
        last_valid_price = valid_closes.iloc[-1] if len(valid_closes) == 1 and pd.notna(valid_closes.iloc[-1]) else None
        return f"${last_valid_price:.2f} (?%)" if last_valid_price is not None else "N/A", 0.0
    latest_price = valid_closes.iloc[-1]
    prev_price = valid_closes.iloc[-2]
    if pd.isna(latest_price) or pd.isna(prev_price) or prev_price == 0:
        return f"${latest_price:.2f} (?%)" if pd.notna(latest_price) else "N/A", 0.0
    pct_change = ((latest_price - prev_price) / prev_price) * 100
    price_info = f"${latest_price:.2f} ({pct_change:+.2f}%)"
    return price_info, pct_change

# (Funzione calculate_signal_score invariata)
def calculate_signal_score(hourly_signals, daily_signals):
    decision_score = 0
    indicators_for_score = [
        hourly_signals.get('RSI', ''), daily_signals.get('RSI', ''),
        daily_signals.get('MACD', ''), hourly_signals.get('SRSI', '') ]
    for indicator_signal in indicators_for_score:
        if isinstance(indicator_signal, str):
            if '🔶' in indicator_signal: decision_score += 2
            elif '🟢' in indicator_signal: decision_score += 1
            elif '🟡' in indicator_signal: decision_score += 0
            elif '🔴' in indicator_signal: decision_score -= 1
            elif '🔻' in indicator_signal: decision_score -= 2
    if decision_score >= SIGNAL_STRONG_BUY_THRESHOLD: return '**🔶 Strong Buy**'
    elif decision_score >= SIGNAL_BUY_THRESHOLD: return '**🟢 Buy**'
    elif decision_score <= SIGNAL_STRONG_SELL_THRESHOLD: return '**🔻 Strong Sell**'
    elif decision_score <= SIGNAL_SELL_THRESHOLD: return '**🔴 Sell**'
    else: return '**🟡 Hold**'


def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    st.title("📈 Live Crypto Technical Dashboard (Top 10)") # Titolo aggiornato

    # (Expanders invariati)
    with st.expander("ℹ️ Signal Score Legend"): st.markdown(f"""...""") # Contenuto omesso
    with st.expander("📚 Indicators Description"): st.markdown("""...""") # Contenuto omesso

    # --- MODIFICATA chiamata funzione ---
    crypto_symbols = get_top_10_crypto_symbols()
    # --- FINE MODIFICA ---

    if not crypto_symbols:
        st.error("Impossibile recuperare la lista delle Top 10 criptovalute. Riprova più tardi.")
        return

    results = []
    symbols_processed_count = 0
    symbols_valid_count = 0
    total_symbols = len(crypto_symbols) # Ora sarà 10 o meno
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, symbol in enumerate(crypto_symbols):
        symbols_processed_count += 1
        status_text.text(f"Processing {symbol} ({symbols_processed_count}/{total_symbols})...")
        try:
            hourly_df, hourly_signals = fetch_indicators_with_signals(symbol, '60m', '7d')
            daily_df, daily_signals = fetch_indicators_with_signals(symbol, '1d', '3mo')
            weekly_df, weekly_signals = fetch_indicators_with_signals(symbol, '1wk', '8mo')
            monthly_df, monthly_signals = fetch_indicators_with_signals(symbol, '1mo', '26mo')

            # --- PUNTO CHIAVE: Assicurati che daily_df sia valido prima di usarlo ---
            # calculate_price_change gestisce già df vuoto o senza colonne, ma aggiungiamo chiarezza
            if daily_df is None or daily_df.empty:
                 # st.warning(f"Skipping {symbol} due to empty daily data for price calculation.") # Log opzionale
                 continue

            price_info, pct_change = calculate_price_change(daily_df)
            if price_info == "N/A":
                 # st.warning(f"Skipping {symbol} due to N/A price info.") # Log opzionale
                 continue

            signal_decision = calculate_signal_score(hourly_signals, daily_signals)

            combined = {
                'Crypto': symbol, 'Price (1d %)': price_info, 'Signal Score': signal_decision,
                'RSI (1h)': hourly_signals.get('RSI', '⚪ N/A'), 'RSI (1d)': daily_signals.get('RSI', '⚪ N/A'),
                'RSI (1w)': weekly_signals.get('RSI', '⚪ N/A'), 'RSI (1mo)': monthly_signals.get('RSI', '⚪ N/A'),
                'SRSI (1h)': hourly_signals.get('SRSI', '⚪ N/A'), 'MACD (1d)': daily_signals.get('MACD', '⚪ N/A'),
                'MA (1h)': hourly_signals.get('MA', '⚪ N/A'), 'Doda Stoch (1h)': hourly_signals.get('Doda Stoch', '⚪ N/A'),
                'GChannel (1h)': hourly_signals.get('GChannel', '⚪ N/A'), 'Vol Flow (1h)': hourly_signals.get('Vol Flow', '⚪ N/A'),
                'VWAP (1h)': hourly_signals.get('VWAP', '⚪ N/A')
            }
            results.append(combined)
            symbols_valid_count += 1

        except Exception as e:
            st.error(f"Errore GRAVE non gestito nel ciclo principale per {symbol}: {e}", icon="🚨")
            # Stampa il traceback completo nei log di Streamlit Cloud
            st.error(f"Traceback: {traceback.format_exc()}")

        progress_bar.progress((i + 1) / total_symbols)

    final_message = f"Processing complete. {symbols_valid_count} symbols loaded with valid data out of {symbols_processed_count} processed."
    if symbols_valid_count == 0 and symbols_processed_count > 0:
         final_message += " Nessun dato valido trovato. Controllare la fonte dati (yfinance) o eventuali errori persistenti."
    status_text.text(final_message)
    progress_bar.empty()

    # (Visualizzazione Tabella invariata)
    if results:
        df_results = pd.DataFrame(results)
        def highlight_price_change(val):
            color = ''
            if isinstance(val, str) and '(' in val and '%' in val:
                try:
                    percent_str = val[val.find("(")+1:val.find("%)")]
                    percent = float(percent_str)
                    if percent > 1: color = 'lightgreen'
                    elif percent < -1: color = 'lightcoral'
                    elif percent > 0: color = 'honeydew'
                    elif percent < 0: color = 'mistyrose'
                except ValueError: pass
            return f'background-color: {color}' if color else ''
        def align_center(x): return ['text-align: center'] * len(x)

        styled_df = df_results.style.applymap(highlight_price_change, subset=['Price (1d %)'])\
                                   .set_properties(**{'text-align': 'center'}, subset=['Signal Score', 'RSI (1h)', 'RSI (1d)', 'RSI (1w)', 'RSI (1mo)', 'SRSI (1h)', 'MACD (1d)', 'Doda Stoch (1h)'])
        st.dataframe(styled_df, use_container_width=True, height=None)
    else:
        st.warning("⚠️ Nessun dato disponibile per le Top 10 criptovalute al momento. Controlla i log di errore sopra o riprova più tardi.") # Messaggio warning aggiornato

if __name__ == '__main__':
    # Aggiungiamo un try-except globale per catturare errori all'avvio
    try:
        main()
    except Exception as e:
        st.error(f"Errore critico all'avvio dell'applicazione: {e}", icon="💥")
        st.error(f"Traceback: {traceback.format_exc()}")