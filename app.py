import pandas as pd
import ta
import streamlit as st
import yfinance as yf
import requests
import json  # Aggiunto per JSONDecodeError

# --- Costanti per parametri e soglie ---
RSI_WINDOW = 14
MA_WINDOW = 14
SRSI_WINDOW = 14 # ta usa 14 di default
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGN = 9
STOCH_WINDOW = 14
EMA_SHORT = 3
EMA_LONG = 30
MFI_WINDOW = 14 # ta usa 14 di default

RSI_LOW = 30
RSI_HIGH = 70
SRSI_LOW = 0.2
SRSI_HIGH = 0.8
MACD_THRESHOLD_LOW = -0.1 # Soglia esempio per MACD (aggiustare se necessario)
MACD_THRESHOLD_HIGH = 0.1 # Soglia esempio per MACD (aggiustare se necessario)
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
    """Fetches top 100 crypto symbols by market cap from CoinGecko."""
    try:
        url = 'https://api.coingecko.com/api/v3/coins/markets'
        params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 100, 'page': 1} # Corretto per 100
        response = requests.get(url, params=params, timeout=10) # Aggiunto timeout
        response.raise_for_status() # Controlla errori HTTP (4xx, 5xx)
        data = response.json()
        # Assicura che 'symbol' esista prima di creare il ticker
        tickers = [f"{coin['symbol'].upper()}-USD" for coin in data if 'symbol' in coin]
        if not tickers:
             st.warning("Nessun ticker valido ricevuto da CoinGecko API.")
             # Fallback più significativo o gestito nell'app principale
             return ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD', 'TRX-USD', 'DOT-USD', 'MATIC-USD'] # Fallback leggermente esteso
        return tickers
    except requests.exceptions.Timeout:
        st.error("Errore API CoinGecko: Timeout durante la richiesta.")
        return ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD'] # Fallback
    except requests.exceptions.HTTPError as e:
        st.error(f"Errore HTTP API CoinGecko: {e}")
        return ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD'] # Fallback
    except requests.exceptions.RequestException as e:
        st.error(f"Errore generico API CoinGecko: {e}")
        return ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD'] # Fallback
    except (ValueError, json.JSONDecodeError) as e:
         st.error(f"Errore parsing risposta CoinGecko: {e}")
         return ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD'] # Fallback
    except Exception as e: # Catchall per errori imprevisti
        st.error(f"Errore imprevisto in get_top_100_crypto_symbols: {e}")
        return ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD'] # Fallback


@st.cache_data(ttl=300, show_spinner="Calculating indicators...")
def fetch_indicators_with_signals(symbol, interval, period):
    """Fetches data using yfinance and calculates technical indicators. Returns dataframe and signals dict."""
    signals = {
        'RSI': 'N/A', 'SRSI': 'N/A', 'MACD': 'N/A', 'MA': 'N/A',
        'Doda Stoch': 'N/A', 'GChannel': 'N/A', 'Vol Flow': 'N/A', 'VWAP': 'N/A'
    }
    df = pd.DataFrame() # Inizializza df vuoto

    try:
        df = yf.download(tickers=symbol, interval=interval, period=period, progress=False) # Disabilita progress bar di yf

        # Controlla se il dataframe è vuoto o troppo corto per gli indicatori
        if df.empty or len(df) < max(RSI_WINDOW, MA_WINDOW, SRSI_WINDOW, MACD_SLOW, STOCH_WINDOW, MFI_WINDOW, EMA_LONG, 2): # Usa 2 come minimo assoluto
            st.warning(f"Dati insufficienti per {symbol} ({interval}, {period}). Indicatori non calcolati.")
            return df, signals # Restituisce df vuoto e segnali N/A

        # Rinomina colonne se necessario (yfinance di solito usa Maiuscole)
        df['close'] = df['Close']
        df['high'] = df['High']
        df['low'] = df['Low']
        df['volume'] = df['Volume']

        # --- Calcolo Indicatori (con gestione errori individuali opzionale) ---
        try:
            rsi_val = ta.momentum.RSIIndicator(df['close'], window=RSI_WINDOW).rsi().iloc[-1]
        except Exception: rsi_val = float('nan')

        try:
            srsi_val = ta.momentum.StochRSIIndicator(df['close'], window=SRSI_WINDOW).stochrsi().iloc[-1]
        except Exception: srsi_val = float('nan')

        try:
            macd_line = ta.trend.MACD(df['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGN)
            macd_val = macd_line.macd().iloc[-1]
            # macd_signal_val = macd_line.macd_signal().iloc[-1] # Non usato attualmente
            # macd_hist_val = macd_line.macd_diff().iloc[-1] # Non usato attualmente
        except Exception: macd_val = float('nan')

        try:
            ma_val = df['close'].rolling(window=MA_WINDOW).mean().iloc[-1]
        except Exception: ma_val = float('nan')

        try:
            doda_val = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=STOCH_WINDOW).stoch().iloc[-1]
        except Exception: doda_val = float('nan')

        try:
            ema_short = df['close'].ewm(span=EMA_SHORT, adjust=False).mean().iloc[-1]
            ema_long = df['close'].ewm(span=EMA_LONG, adjust=False).mean().iloc[-1]
            gchannel_val = ema_short - ema_long
        except Exception: gchannel_val = float('nan')

        try:
            vfi_val = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=MFI_WINDOW).money_flow_index().iloc[-1]
        except Exception: vfi_val = float('nan')

        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap_series = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            vwap_val = vwap_series.iloc[-1]
        except Exception: vwap_val = float('nan')


        # --- Generazione Segnali Emoji ---
        def signal_emoji(value, low, high, strong_factor=0.33):
            if pd.isna(value): return '⚪ N/A' # Emoji per N/A
            low_strong = low * (1 - strong_factor)
            high_strong = high * (1 + strong_factor)

            if value < low_strong: return '🔶' # Strong Buy/Oversold
            elif value < low: return '🟢' # Buy/Oversold
            elif value > high_strong: return '🔻' # Strong Sell/Overbought
            elif value > high: return '🔴' # Sell/Overbought
            else: return '🟡' # Hold/Neutral

        # --- Assegnazione Segnali ---
        signals['RSI'] = f"{rsi_val:.2f} {signal_emoji(rsi_val, RSI_LOW, RSI_HIGH)}" if not pd.isna(rsi_val) else '⚪ N/A'
        signals['SRSI'] = f"{srsi_val:.2f} {signal_emoji(srsi_val, SRSI_LOW, SRSI_HIGH)}" if not pd.isna(srsi_val) else '⚪ N/A'
        # Per MACD, il segnale è spesso basato su cross con linea segnale o zero. Usiamo soglie semplici per ora.
        signals['MACD'] = f"{macd_val:.2f} {signal_emoji(macd_val, MACD_THRESHOLD_LOW, MACD_THRESHOLD_HIGH, strong_factor=0.5)}" if not pd.isna(macd_val) else '⚪ N/A' # Ajustato strong_factor
        signals['MA'] = f"{ma_val:.2f}" if not pd.isna(ma_val) else '⚪ N/A' # MA è valore, non segnale qui
        signals['Doda Stoch'] = f"{doda_val:.2f} {signal_emoji(doda_val, STOCH_LOW, STOCH_HIGH)}" if not pd.isna(doda_val) else '⚪ N/A'
        # GChannel può essere interpretato come segnale (positivo=bullish, negativo=bearish)
        if not pd.isna(gchannel_val):
            gc_signal = '🟢' if gchannel_val > 0 else ('🔴' if gchannel_val < 0 else '🟡')
            signals['GChannel'] = f"{gchannel_val:.2f} {gc_signal}"
        else:
            signals['GChannel'] = '⚪ N/A'
        signals['Vol Flow'] = f"{vfi_val:.2f} {signal_emoji(vfi_val, MFI_LOW, MFI_HIGH)}" if not pd.isna(vfi_val) else '⚪ N/A'
        # VWAP di per sè è un prezzo medio, confronto con prezzo attuale dà segnale
        last_close = df['close'].iloc[-1]
        if not pd.isna(vwap_val) and not pd.isna(last_close):
            vwap_signal = '🟢' if last_close > vwap_val else ('🔴' if last_close < vwap_val else '🟡')
            signals['VWAP'] = f"{vwap_val:.2f} {vwap_signal}"
        else:
             signals['VWAP'] = f"{vwap_val:.2f}" if not pd.isna(vwap_val) else '⚪ N/A'


        return df, signals # Restituisce sia il DataFrame che i segnali

    except yf.utils.YFNotImplementedError as e:
         st.warning(f"Errore yfinance (NotImplemented) per {symbol} ({interval}, {period}): {e}")
         return df, signals # Restituisce df vuoto e segnali N/A
    except Exception as e:
        st.warning(f"Errore durante il fetch/calcolo indicatori per {symbol} ({interval}, {period}): {e}")
        # Potrebbe essere utile loggare l'errore completo per debug: print(f"Error details for {symbol}: {traceback.format_exc()}")
        return df, signals # Restituisce df vuoto e segnali N/A

def calculate_price_change(daily_df):
    """Calculates latest price and 1-day percentage change from daily dataframe."""
    if daily_df.empty or len(daily_df) < 2:
        return "N/A", 0.0

    # Assicurati che l'indice sia ordinato temporalmente
    daily_df = daily_df.sort_index()

    # Prendi gli ultimi due prezzi validi
    valid_closes = daily_df['Close'].dropna()
    if len(valid_closes) < 2:
        return f"${valid_closes.iloc[-1]:.2f} (?%)" if len(valid_closes) == 1 else "N/A", 0.0

    latest_price = valid_closes.iloc[-1]
    prev_price = valid_closes.iloc[-2]

    if pd.isna(latest_price) or pd.isna(prev_price) or prev_price == 0:
        return f"${latest_price:.2f} (?%)" if not pd.isna(latest_price) else "N/A", 0.0

    pct_change = ((latest_price - prev_price) / prev_price) * 100
    price_info = f"${latest_price:.2f} ({pct_change:+.2f}%)"
    return price_info, pct_change

def calculate_signal_score(hourly_signals, daily_signals):
    """Calculates a composite signal score based on selected indicators."""
    decision_score = 0
    # Indicatori usati per lo score: RSI (1h), RSI (1d), MACD (daily), SRSI (hourly)
    indicators_for_score = [
        hourly_signals.get('RSI', ''),
        daily_signals.get('RSI', ''),
        daily_signals.get('MACD', ''), # Usiamo MACD giornaliero per lo score
        hourly_signals.get('SRSI', '')
    ]

    for indicator_signal in indicators_for_score:
        if isinstance(indicator_signal, str):
            if '🔶' in indicator_signal: decision_score += 2
            elif '🟢' in indicator_signal: decision_score += 1
            elif '🟡' in indicator_signal: decision_score += 0
            elif '🔴' in indicator_signal: decision_score -= 1
            elif '🔻' in indicator_signal: decision_score -= 2
            # Ignora '⚪ N/A'

    if decision_score >= SIGNAL_STRONG_BUY_THRESHOLD: return '**🔶 Strong Buy**'
    elif decision_score >= SIGNAL_BUY_THRESHOLD: return '**🟢 Buy**'
    elif decision_score <= SIGNAL_STRONG_SELL_THRESHOLD: return '**🔻 Strong Sell**'
    elif decision_score <= SIGNAL_SELL_THRESHOLD: return '**🔴 Sell**'
    else: return '**🟡 Hold**'

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    st.title("📈 Live Crypto Technical Dashboard")
    # st.write("Auto-refresh every 5 minutes. Use your mouse scroll wheel or trackpad to move up/down.") # Streamlit gestisce auto-refresh con @st.cache_data

    with st.expander("ℹ️ Signal Score Legend"): # Rinominato da GPT a Signal Score
        st.markdown(f"""
        **Come viene calcolato il Punteggio Segnali:**

        - 🔶 **Strong Buy**: Segnali positivi molto forti (+{SIGNAL_STRONG_BUY_THRESHOLD} punti o più)
        - 🟢 **Buy**: Segnali positivi moderati (da +{SIGNAL_BUY_THRESHOLD} a +{SIGNAL_STRONG_BUY_THRESHOLD-1} punti)
        - 🟡 **Hold**: Segnali neutrali o incerti (da {SIGNAL_SELL_THRESHOLD+1} a +{SIGNAL_BUY_THRESHOLD-1} punti)
        - 🔴 **Sell**: Segnali negativi moderati (da {SIGNAL_STRONG_SELL_THRESHOLD+1} a {SIGNAL_SELL_THRESHOLD} punti)
        - 🔻 **Strong Sell**: Segnali negativi molto forti ({SIGNAL_STRONG_SELL_THRESHOLD} punti o meno)

        **Indicatori usati per il punteggio:** RSI (1h), RSI (1d), MACD (1d), SRSI (1h)
        """)

    with st.expander("📚 Indicators Description"):
        st.markdown("""
        **RSI (Relative Strength Index)**: Misura la velocità e il cambiamento dei movimenti di prezzo. Utile per identificare condizioni di ipercomprato (>70) / ipervenduto (<30).
        **Stochastic RSI (SRSI)**: Un indicatore di momentum basato sull'RSI. Più sensibile ai recenti movimenti di prezzo, utile per segnali a breve termine (0.8 ipercomprato, 0.2 ipervenduto).
        **MACD (Moving Average Convergence Divergence)**: Mostra la relazione tra due medie mobili esponenziali dei prezzi. Incroci della linea MACD con la sua linea segnale o con lo zero possono indicare cambiamenti di trend.
        **MA (Moving Average)**: Media mobile semplice del prezzo di chiusura. Aiuta a identificare la direzione del trend generale.
        **Doda Stochastic Oscillator**: Un altro indicatore di momentum che confronta il prezzo di chiusura con il suo range di prezzo in un dato periodo (80 ipercomprato, 20 ipervenduto).
        **GChannel (Guppy Channel)**: Differenza tra EMA a breve (3) e lungo (30) periodo. Positivo suggerisce trend rialzista, negativo ribassista.
        **Vol Flow (Money Flow Index - MFI)**: Indicatore di volume pesato simile all'RSI, ma incorpora il volume (80 ipercomprato, 20 ipervenduto).
        **VWAP (Volume Weighted Average Price)**: Prezzo medio ponderato per il volume. Spesso usato come benchmark. Segnale: Prezzo sopra VWAP (🟢), Prezzo sotto VWAP (🔴).
        **Emoji Segnali**: 🔶 Forte segnale positivo/ipervenduto, 🟢 Segnale positivo/ipervenduto, 🟡 Neutrale, 🔴 Segnale negativo/ipercomprato, 🔻 Forte segnale negativo/ipercomprato, ⚪ Dati non disponibili.
        """)

    crypto_symbols = get_top_100_crypto_symbols()
    if not crypto_symbols:
        st.error("Impossibile recuperare la lista delle criptovalute. Riprova più tardi.")
        return # Esce se non ci sono simboli

    results = []
    total_symbols = len(crypto_symbols)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, symbol in enumerate(crypto_symbols):
        status_text.text(f"Processing {symbol} ({i+1}/{total_symbols})...")
        try:
            # Fetch data & indicators for different timeframes
            hourly_df, hourly_signals = fetch_indicators_with_signals(symbol, '60m', '7d') # Periodo leggermente aumentato
            daily_df, daily_signals = fetch_indicators_with_signals(symbol, '1d', '3mo') # Periodo leggermente aumentato
            weekly_df, weekly_signals = fetch_indicators_with_signals(symbol, '1wk', '6mo') # Periodo leggermente aumentato
            monthly_df, monthly_signals = fetch_indicators_with_signals(symbol, '1mo', '2y') # Periodo leggermente aumentato

            # Calcola prezzo e variazione % dal dataframe giornaliero
            price_info, pct_change = calculate_price_change(daily_df)
            if price_info == "N/A":
                st.warning(f"Prezzo non disponibile per {symbol}. Simbolo saltato.")
                continue # Salta il simbolo se non abbiamo prezzo

            # Calcola lo score composito dei segnali
            signal_decision = calculate_signal_score(hourly_signals, daily_signals)

            # Assembla i risultati per la tabella
            combined = {
                'Crypto': symbol,
                'Price (1d %)': price_info,
                'Signal Score': signal_decision, # Nome colonna aggiornato
                'RSI (1h)': hourly_signals.get('RSI', '⚪ N/A'),
                'RSI (1d)': daily_signals.get('RSI', '⚪ N/A'),
                'RSI (1w)': weekly_signals.get('RSI', '⚪ N/A'),
                'RSI (1mo)': monthly_signals.get('RSI', '⚪ N/A'),
                'SRSI (1h)': hourly_signals.get('SRSI', '⚪ N/A'),
                'MACD (1d)': daily_signals.get('MACD', '⚪ N/A'), # Mostra MACD giornaliero per coerenza con score
                'MA (1h)': hourly_signals.get('MA', '⚪ N/A'),
                'Doda Stoch (1h)': hourly_signals.get('Doda Stoch', '⚪ N/A'),
                'GChannel (1h)': hourly_signals.get('GChannel', '⚪ N/A'),
                'Vol Flow (1h)': hourly_signals.get('Vol Flow', '⚪ N/A'),
                'VWAP (1h)': hourly_signals.get('VWAP', '⚪ N/A')
            }
            results.append(combined)

        except Exception as e:
            # Questo catch è una sicurezza extra, gli errori specifici dovrebbero essere gestiti dentro le funzioni
            st.error(f"Errore inatteso nel ciclo principale per {symbol}: {e}")
            # Considera di aggiungere 'continue' qui se vuoi saltare il simbolo in caso di errore grave nel loop

        # Aggiorna la progress bar
        progress_bar.progress((i + 1) / total_symbols)

    status_text.text(f"Processing complete. {len(results)} symbols loaded.")
    progress_bar.empty() # Rimuove la progress bar una volta finito


    # --- Visualizzazione Tabella ---
    if results:
        df_results = pd.DataFrame(results)

        # Funzione per colorare la percentuale di prezzo
        def highlight_price_change(val):
            color = ''
            if isinstance(val, str) and '(' in val and '%' in val:
                try:
                    percent_str = val[val.find("(")+1:val.find("%)")]
                    percent = float(percent_str)
                    if percent > 1: color = 'lightgreen' # Verde più chiaro per leggibilità
                    elif percent < -1: color = 'lightcoral' # Rosso più chiaro
                    elif percent > 0: color = 'honeydew'
                    elif percent < 0: color = 'mistyrose'
                except ValueError:
                    pass # Ignora se non può convertire
            return f'background-color: {color}' if color else ''

        # Funzione per allineare testo (opzionale, ma migliora leggibilità)
        def align_center(x):
            return ['text-align: center'] * len(x)

        # Applica stili
        styled_df = df_results.style.applymap(highlight_price_change, subset=['Price (1d %)'])\
                                   .set_properties(**{'text-align': 'center'}, subset=['Signal Score', 'RSI (1h)', 'RSI (1d)', 'RSI (1w)', 'RSI (1mo)', 'SRSI (1h)', 'MACD (1d)', 'Doda Stoch (1h)'])

        st.dataframe(styled_df, use_container_width=True, height=800) # Altezza potrebbe essere dinamica o rimossa
    else:
        st.warning("⚠️ Nessun dato disponibile per le criptovalute al momento. Controlla i log di errore sopra o riprova più tardi.")

if __name__ == '__main__':
    main()