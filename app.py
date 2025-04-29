import pandas as pd
import ta
import streamlit as st
import yfinance as yf
import requests
import json
import traceback # Importa traceback per debug dettagliato

# --- Costanti per parametri e soglie ---
# (Costanti invariate rispetto a prima)
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

@st.cache_data(ttl=600, show_spinner="Fetching crypto list...")
def get_top_100_crypto_symbols():
    """Fetches top 100 crypto symbols by market cap from CoinGecko, filtering problematic names."""
    fallback_list = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD', 'ADA-USD', 'TRX-USD', 'DOT-USD', 'MATIC-USD']
    try:
        url = 'https://api.coingecko.com/api/v3/coins/markets'
        params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 100, 'page': 1}
        response = requests.get(url, params=params, timeout=15) # Aumentato leggermente timeout
        response.raise_for_status()
        data = response.json()

        tickers = []
        for coin in data:
            if 'symbol' in coin:
                symbol_upper = coin['symbol'].upper()
                ticker = f"{symbol_upper}-USD"
                # --- Filtro Simboli ---
                # Salta simboli vuoti, troppo corti, o palesemente strani
                if not symbol_upper or len(symbol_upper) < 2 or "-USD" in symbol_upper or " " in ticker or ticker.count('-') > 1:
                    # st.warning(f"Skipping potentially problematic ticker: {ticker}") # Log opzionale
                    continue
                tickers.append(ticker)
                # --- Fine Filtro ---

        if not tickers:
             st.warning("Nessun ticker valido ricevuto o filtrato da CoinGecko API.")
             return fallback_list
        return tickers
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
        st.error(f"Errore imprevisto in get_top_100_crypto_symbols: {e}\n{traceback.format_exc()}") # Log traceback
        return fallback_list


@st.cache_data(ttl=300, show_spinner="Calculating indicators...")
def fetch_indicators_with_signals(symbol, interval, period):
    """Fetches data using yfinance and calculates technical indicators. Returns dataframe and signals dict."""
    signals = {
        'RSI': '⚪ N/A', 'SRSI': '⚪ N/A', 'MACD': '⚪ N/A', 'MA': '⚪ N/A',
        'Doda Stoch': '⚪ N/A', 'GChannel': '⚪ N/A', 'Vol Flow': '⚪ N/A', 'VWAP': '⚪ N/A'
    }
    df = pd.DataFrame()

    try:
        # Aumenta leggermente il periodo richiesto per avere più margine per i calcoli
        # Nota: yfinance potrebbe comunque restituire meno dati se la storia è limitata
        request_period = period
        if interval == '1wk': request_period = '8mo' if period == '6mo' else period # Più margine per weekly
        if interval == '1mo': request_period = '26mo' if period == '2y' else period # Più margine per monthly

        df = yf.download(tickers=symbol, interval=interval, period=request_period, progress=False, ignore_tz=True) # Ignora timezone può aiutare

        # Rimuovi righe con NaN in colonne essenziali se create da yfinance
        df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

        # Controlla se il dataframe è valido DOPO il download e la pulizia
        min_required_len = max(RSI_WINDOW, MA_WINDOW, SRSI_WINDOW, MACD_SLOW, STOCH_WINDOW, MFI_WINDOW, EMA_LONG, 2) + 5 # Aggiungi un piccolo buffer
        if df.empty or len(df) < min_required_len:
            # Non mostrare warning qui per non intasare l'output, verrà gestito nel main loop se il prezzo non è disponibile
            # st.warning(f"Dati insufficienti per {symbol} ({interval}, {period}) dopo download/pulizia. Len: {len(df)}")
            return df, signals

        # Assicurati che l'indice sia un DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
           df.index = pd.to_datetime(df.index)
        # Assicurati che sia ordinato
        df.sort_index(inplace=True)
        # Rimuovi eventuali duplicati nell'indice che possono creare problemi
        df = df[~df.index.duplicated(keep='first')]

        # Rinomina colonne se necessario (yfinance di solito usa Maiuscole)
        # e assicurati che siano numeriche
        df['high'] = pd.to_numeric(df['High'], errors='coerce')
        df['low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['Volume'], errors='coerce')

        # Ricontrolla NaN dopo conversione
        df.dropna(subset=['high', 'low', 'close', 'volume'], inplace=True)
        if df.empty or len(df) < min_required_len:
             # st.warning(f"Dati insufficienti per {symbol} ({interval}, {period}) dopo conversione/pulizia NaN. Len: {len(df)}")
             return df, signals


        # --- Funzione Helper per Calcolo Sicuro Indicatore ---
        def _safe_calculate(indicator_func, *args, **kwargs):
            try:
                # Chiamata alla funzione indicatore di 'ta'
                indicator_series = indicator_func(*args, **kwargs)

                # Controllo robusto del risultato
                if indicator_series is None or not isinstance(indicator_series, pd.Series) or indicator_series.empty:
                    return float('nan')

                # Pulisci NaN/inf nel risultato prima di iloc
                indicator_series = indicator_series.replace([float('inf'), -float('inf')], float('nan')).dropna()

                if indicator_series.empty:
                     return float('nan')

                # Estrai l'ultimo valore valido
                last_val = indicator_series.iloc[-1]
                return last_val if pd.notna(last_val) else float('nan')

            except Exception: # Cattura qualsiasi errore durante il calcolo
                # print(f"Debug: Errore calcolo indicatore per {symbol} ({interval}): {e}") # Debug opzionale
                return float('nan')

        # --- Calcolo Indicatori usando l'Helper ---
        rsi_val = _safe_calculate(ta.momentum.RSIIndicator(df['close'], window=RSI_WINDOW).rsi)
        srsi_val = _safe_calculate(ta.momentum.StochRSIIndicator(df['close'], window=SRSI_WINDOW).stochrsi)

        macd_obj = ta.trend.MACD(df['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGN)
        macd_val = _safe_calculate(macd_obj.macd)
        # macd_signal_val = _safe_calculate(macd_obj.macd_signal) # Non usato
        # macd_hist_val = _safe_calculate(macd_obj.macd_diff) # Non usato

        ma_val = _safe_calculate(df['close'].rolling(window=MA_WINDOW).mean)

        doda_val = _safe_calculate(ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=STOCH_WINDOW).stoch)

        ema_short_series = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
        ema_long_series = df['close'].ewm(span=EMA_LONG, adjust=False).mean()
        gchannel_series = ema_short_series - ema_long_series
        gchannel_val = _safe_calculate(lambda s: s, gchannel_series) # Passa la serie calcolata all'helper

        vfi_val = _safe_calculate(ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=MFI_WINDOW).money_flow_index)

        try: # VWAP ha calcolo custom
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            # Evita divisione per zero se il volume cumulativo è zero all'inizio
            volume_cumsum = df['volume'].cumsum()
            vwap_series = ((typical_price * df['volume']).cumsum()) / volume_cumsum.replace(0, float('nan'))
            vwap_series = vwap_series.replace([float('inf'), -float('inf')], float('nan')).dropna()
            vwap_val = vwap_series.iloc[-1] if not vwap_series.empty and pd.notna(vwap_series.iloc[-1]) else float('nan')
        except Exception:
            vwap_val = float('nan')

        # --- Generazione Segnali Emoji ---
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

        # --- Assegnazione Segnali ---
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
        elif pd.notna(vwap_val):
             signals['VWAP'] = f"{vwap_val:.2f}"
        else: signals['VWAP'] = '⚪ N/A'

        return df, signals

    except yf.utils.YFNotImplementedError as e:
         st.warning(f"Errore yfinance (NotImplemented) per {symbol} ({interval}, {period}): {e}", icon="⚠️")
         return df, signals
    except Exception as e:
        # Non mostrare warning per ogni piccolo errore qui, troppi log
        # Stampa solo se è un errore veramente inatteso nel fetch/calcolo generale
        # print(f"Debug: Errore generale fetch/calcolo per {symbol} ({interval}, {period}): {e}")
        # print(traceback.format_exc()) # Debug opzionale
        return df, signals # Restituisce df vuoto e segnali N/A


def calculate_price_change(daily_df):
    """Calculates latest price and 1-day percentage change from daily dataframe."""
    if daily_df is None or not isinstance(daily_df, pd.DataFrame) or daily_df.empty or 'Close' not in daily_df.columns:
        return "N/A", 0.0

    # Assicurati che l'indice sia ordinato temporalmente e rimuovi duplicati
    try:
        if not isinstance(daily_df.index, pd.DatetimeIndex):
           daily_df.index = pd.to_datetime(daily_df.index)
        daily_df = daily_df.sort_index()
        daily_df = daily_df[~daily_df.index.duplicated(keep='first')]
    except Exception:
         # Fallback se l'indice non è gestibile
         return "N/A", 0.0

    # Prendi gli ultimi due prezzi validi dalla colonna 'Close' numerica
    valid_closes = pd.to_numeric(daily_df['Close'], errors='coerce').dropna()

    if len(valid_closes) < 2:
        last_valid_price = valid_closes.iloc[-1] if len(valid_closes) == 1 and pd.notna(valid_closes.iloc[-1]) else None
        return f"${last_valid_price:.2f} (?%)" if last_valid_price is not None else "N/A", 0.0

    latest_price = valid_closes.iloc[-1]
    prev_price = valid_closes.iloc[-2]

    # Aggiunto controllo per NaN anche qui per sicurezza
    if pd.isna(latest_price) or pd.isna(prev_price) or prev_price == 0:
        return f"${latest_price:.2f} (?%)" if pd.notna(latest_price) else "N/A", 0.0

    pct_change = ((latest_price - prev_price) / prev_price) * 100
    price_info = f"${latest_price:.2f} ({pct_change:+.2f}%)"
    return price_info, pct_change

# (Funzione calculate_signal_score invariata)
def calculate_signal_score(hourly_signals, daily_signals):
    """Calculates a composite signal score based on selected indicators."""
    decision_score = 0
    indicators_for_score = [
        hourly_signals.get('RSI', ''),
        daily_signals.get('RSI', ''),
        daily_signals.get('MACD', ''),
        hourly_signals.get('SRSI', '')
    ]
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
    st.title("📈 Live Crypto Technical Dashboard")

    # (Expanders invariati)
    with st.expander("ℹ️ Signal Score Legend"):
        st.markdown(f"""...""") # Contenuto omesso per brevità
    with st.expander("📚 Indicators Description"):
        st.markdown("""...""") # Contenuto omesso per brevità

    crypto_symbols = get_top_100_crypto_symbols()
    if not crypto_symbols:
        st.error("Impossibile recuperare la lista delle criptovalute. Riprova più tardi.")
        return

    results = []
    total_symbols = len(crypto_symbols)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, symbol in enumerate(crypto_symbols):
        status_text.text(f"Processing {symbol} ({i+1}/{total_symbols})...")
        try:
            # Fetch data & indicators for different timeframes
            # Usiamo periodi più lunghi nella richiesta per dare margine a yf/ta
            hourly_df, hourly_signals = fetch_indicators_with_signals(symbol, '60m', '7d')
            daily_df, daily_signals = fetch_indicators_with_signals(symbol, '1d', '3mo')
            weekly_df, weekly_signals = fetch_indicators_with_signals(symbol, '1wk', '8mo') # Periodo aumentato
            monthly_df, monthly_signals = fetch_indicators_with_signals(symbol, '1mo', '26mo') # Periodo aumentato

            # Calcola prezzo e variazione % dal dataframe giornaliero (daily_df)
            price_info, pct_change = calculate_price_change(daily_df) # Passa il df giornaliero valido
            if price_info == "N/A":
                 # Non mostrare warning per ogni simbolo senza prezzo, solo alla fine se results è vuoto
                 # st.warning(f"Prezzo non disponibile per {symbol}. Simbolo saltato.")
                 continue # Salta il simbolo se non abbiamo prezzo valido

            # Calcola lo score composito dei segnali
            signal_decision = calculate_signal_score(hourly_signals, daily_signals)

            # Assembla i risultati per la tabella
            combined = {
                'Crypto': symbol,
                'Price (1d %)': price_info,
                'Signal Score': signal_decision,
                'RSI (1h)': hourly_signals.get('RSI', '⚪ N/A'),
                'RSI (1d)': daily_signals.get('RSI', '⚪ N/A'),
                'RSI (1w)': weekly_signals.get('RSI', '⚪ N/A'),
                'RSI (1mo)': monthly_signals.get('RSI', '⚪ N/A'),
                'SRSI (1h)': hourly_signals.get('SRSI', '⚪ N/A'),
                'MACD (1d)': daily_signals.get('MACD', '⚪ N/A'),
                'MA (1h)': hourly_signals.get('MA', '⚪ N/A'),
                'Doda Stoch (1h)': hourly_signals.get('Doda Stoch', '⚪ N/A'),
                'GChannel (1h)': hourly_signals.get('GChannel', '⚪ N/A'),
                'Vol Flow (1h)': hourly_signals.get('Vol Flow', '⚪ N/A'),
                'VWAP (1h)': hourly_signals.get('VWAP', '⚪ N/A')
            }
            results.append(combined)

        except Exception as e:
            # Stampa errore completo con traceback per il debug
            st.error(f"Errore GRAVE non gestito nel ciclo principale per {symbol}: {e}", icon="🚨")
            st.error(f"Traceback: {traceback.format_exc()}") # Stampa il traceback completo
            # Considera 'continue' se vuoi che l'app provi gli altri simboli nonostante l'errore grave
            # continue

        # Aggiorna la progress bar
        progress_bar.progress((i + 1) / total_symbols)

    # Messaggio finale e rimozione progress bar
    final_message = f"Processing complete. {len(results)} symbols loaded out of {total_symbols} processed."
    if len(results) == 0 and total_symbols > 0:
         final_message += " Nessun dato valido trovato per la visualizzazione."
    status_text.text(final_message)
    progress_bar.empty()

    # --- Visualizzazione Tabella ---
    if results:
        df_results = pd.DataFrame(results)
        # (Funzioni di stile highlight_price_change e align_center invariate)
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
        def align_center(x):
            return ['text-align: center'] * len(x)

        styled_df = df_results.style.applymap(highlight_price_change, subset=['Price (1d %)'])\
                                   .set_properties(**{'text-align': 'center'}, subset=['Signal Score', 'RSI (1h)', 'RSI (1d)', 'RSI (1w)', 'RSI (1mo)', 'SRSI (1h)', 'MACD (1d)', 'Doda Stoch (1h)'])
        st.dataframe(styled_df, use_container_width=True, height=None) # Rimuovi altezza fissa
    else:
        st.warning("⚠️ Nessun dato disponibile per le criptovalute al momento. Controlla i log di errore sopra o riprova più tardi.")

if __name__ == '__main__':
    main()