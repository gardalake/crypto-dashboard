import pandas as pd
import ta
import streamlit as st
# import yfinance as yf # Non più usato
from pycoingecko import CoinGeckoAPI # NUOVA LIBRERIA
import requests
import json
import traceback
import time # Per gestire timestamp UNIX

# --- Costanti (invariate) ---
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

# --- Inizializza client CoinGecko ---
cg = CoinGeckoAPI()

# --- MODIFICATA: Restituisce solo BTC, ETH, SOL ---
# Rimuoviamo il caching dato che la lista è fissa e non richiede API call
# @st.cache_data(ttl=600, show_spinner="Fetching top 10 crypto list (CoinGecko IDs)...")
def get_specific_crypto_ids_and_symbols(): # Nome funzione aggiornato
    """Returns a fixed list containing Bitcoin, Ethereum, and Solana."""
    # Lista fissa con ID e Simboli richiesti
    crypto_list = [
        {'id': 'bitcoin', 'symbol': 'BTC-USD'},
        {'id': 'ethereum', 'symbol': 'ETH-USD'},
        {'id': 'solana', 'symbol': 'SOL-USD'}
    ]
    # Non servono più chiamate API o fallback qui
    return crypto_list
# --- FINE MODIFICA ---


@st.cache_data(ttl=300, show_spinner="Calculating indicators (CoinGecko)...")
def fetch_indicators_with_signals_cg(coin_id, symbol_display, days_required):
    """Fetches CoinGecko OHLC data and calculates technical indicators."""
    signals = {
        'RSI': '⚪ N/A', 'SRSI': '⚪ N/A', 'MACD': '⚪ N/A', 'MA': '⚪ N/A',
        'Doda Stoch': '⚪ N/A', 'GChannel': '⚪ N/A', 'Vol Flow': '⚪ N/A', 'VWAP': '⚪ N/A'
    }
    df = pd.DataFrame()
    required_columns_ohlc = ['Open', 'High', 'Low', 'Close'] # Volume verrà aggiunto dopo
    required_columns_final = ['open', 'high', 'low', 'close', 'volume']

    try:
        ohlc_data = cg.get_coin_ohlc_by_id(id=coin_id, vs_currency='usd', days=str(days_required))

        if not ohlc_data:
            return df, signals

        df = pd.DataFrame(ohlc_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Timestamp', inplace=True)

        # Aggiungi volume (come prima)
        try:
             chart_data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=str(days_required))
             if chart_data and 'total_volumes' in chart_data:
                 volumes = pd.DataFrame(chart_data['total_volumes'], columns=['Timestamp', 'Volume'])
                 volumes['Timestamp'] = pd.to_datetime(volumes['Timestamp'], unit='ms')
                 volumes.set_index('Timestamp', inplace=True)
                 df = df.join(volumes, how='left')
                 df['Volume'] = df['Volume'].ffill().fillna(0)
             else:
                 df['Volume'] = 0
        except Exception:
             df['Volume'] = 0

        # Verifica colonne OHLCV esistono prima di procedere
        if df.empty or not all(col in df.columns for col in required_columns_ohlc + ['Volume']):
             return df, signals

        # Conversione a numerico e minuscolo
        df['open'] = pd.to_numeric(df['Open'], errors='coerce')
        df['high'] = pd.to_numeric(df['High'], errors='coerce')
        df['low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['Volume'], errors='coerce')

        # Rimuovi righe con NaN dopo conversione e verifica lunghezza
        df.dropna(subset=required_columns_final, inplace=True)
        min_required_len = max(RSI_WINDOW, MA_WINDOW, SRSI_WINDOW, MACD_SLOW, STOCH_WINDOW, MFI_WINDOW, EMA_LONG, 2) + 5
        if df.empty or len(df) < min_required_len:
             return df, signals

        # (Helper _safe_calculate invariato)
        def _safe_calculate(indicator_func, *args, **kwargs):
            try:
                indicator_series = indicator_func(*args, **kwargs)
                if indicator_series is None or not isinstance(indicator_series, pd.Series) or indicator_series.empty: return float('nan')
                indicator_series = indicator_series.replace([float('inf'), -float('inf')], float('nan')).dropna()
                if indicator_series.empty: return float('nan')
                last_val = indicator_series.iloc[-1]
                return last_val if pd.notna(last_val) else float('nan')
            except Exception: return float('nan')

        # (Calcolo Indicatori invariato)
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
        vfi_val = _safe_calculate(ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=MFI_WINDOW).money_flow_index) if df['volume'].sum() > 0 else float('nan')
        try:
            if df['volume'].sum() > 0:
                 typical_price = (df['high'] + df['low'] + df['close']) / 3
                 volume_cumsum = df['volume'].cumsum()
                 vwap_series = ((typical_price * df['volume']).cumsum()) / volume_cumsum.replace(0, float('nan'))
                 vwap_series = vwap_series.replace([float('inf'), -float('inf')], float('nan')).dropna()
                 vwap_val = vwap_series.iloc[-1] if not vwap_series.empty and pd.notna(vwap_series.iloc[-1]) else float('nan')
            else: vwap_val = float('nan')
        except Exception: vwap_val = float('nan')

        # (Funzione signal_emoji invariata)
        def signal_emoji(value, low, high, strong_factor=0.33):
            if pd.isna(value): return '⚪ N/A'
            low_strong = low * (1 - strong_factor); high_strong = high * (1 + strong_factor)
            if value < low_strong: return '🔶'
            elif value < low: return '🟢'
            elif value > high_strong: return '🔻'
            elif value > high: return '🔴'
            else: return '🟡'

        # (Assegnazione Segnali invariata)
        signals['RSI'] = f"{rsi_val:.2f} {signal_emoji(rsi_val, RSI_LOW, RSI_HIGH)}" if pd.notna(rsi_val) else '⚪ N/A'
        signals['SRSI'] = f"{srsi_val:.2f} {signal_emoji(srsi_val, SRSI_LOW, SRSI_HIGH)}" if pd.notna(srsi_val) else '⚪ N/A'
        signals['MACD'] = f"{macd_val:.2f} {signal_emoji(macd_val, MACD_THRESHOLD_LOW, MACD_THRESHOLD_HIGH, strong_factor=0.5)}" if pd.notna(macd_val) else '⚪ N/A'
        signals['MA'] = f"{ma_val:.2f}" if pd.notna(ma_val) else '⚪ N/A'
        signals['Doda Stoch'] = f"{doda_val:.2f} {signal_emoji(doda_val, STOCH_LOW, STOCH_HIGH)}" if pd.notna(doda_val) else '⚪ N/A'
        if pd.notna(gchannel_val):
            gc_signal = '🟢' if gchannel_val > 0 else ('🔴' if gchannel_val < 0 else '🟡'); signals['GChannel'] = f"{gchannel_val:.2f} {gc_signal}"
        else: signals['GChannel'] = '⚪ N/A'
        signals['Vol Flow'] = f"{vfi_val:.2f} {signal_emoji(vfi_val, MFI_LOW, MFI_HIGH)}" if pd.notna(vfi_val) else '⚪ N/A'
        last_close = df['close'].iloc[-1] if not df['close'].empty else float('nan')
        if pd.notna(vwap_val) and pd.notna(last_close):
            vwap_signal = '🟢' if last_close > vwap_val else ('🔴' if last_close < vwap_val else '🟡'); signals['VWAP'] = f"{vwap_val:.2f} {vwap_signal}"
        elif pd.notna(vwap_val): signals['VWAP'] = f"{vwap_val:.2f}"
        else: signals['VWAP'] = '⚪ N/A'

        return df, signals

    except Exception as e:
        return df, signals


# (Funzione calculate_price_change invariata)
def calculate_price_change(daily_df):
    if daily_df is None or not isinstance(daily_df, pd.DataFrame) or daily_df.empty or 'close' not in daily_df.columns:
        return "N/A", 0.0
    try:
        if not isinstance(daily_df.index, pd.DatetimeIndex): daily_df.index = pd.to_datetime(daily_df.index)
        daily_df = daily_df.sort_index()
        daily_df = daily_df[~daily_df.index.duplicated(keep='first')]
    except Exception: return "N/A", 0.0
    valid_closes = pd.to_numeric(daily_df['close'], errors='coerce').dropna()
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
    # Titolo aggiornato per riflettere il contenuto
    st.title("📈 Live BTC/ETH/SOL Technical Dashboard (CoinGecko Data)")

    with st.expander("ℹ️ Signal Score Legend"): st.markdown(f"""...""") # Omesso
    with st.expander("📚 Indicators Description"): st.markdown("""...""") # Omesso

    # --- MODIFICATA chiamata funzione ---
    crypto_list = get_specific_crypto_ids_and_symbols() # Usa la nuova funzione
    # --- FINE MODIFICA ---

    if not crypto_list: # Controllo comunque utile se la lista fosse vuota per errore
        st.error("Lista crypto specifica vuota.")
        return

    results = []
    symbols_processed_count = 0
    symbols_valid_count = 0
    total_symbols = len(crypto_list) # Ora sarà 3
    progress_bar = st.progress(0)
    status_text = st.empty()

    days_map = {'hourly': 7, 'daily': 90, 'weekly': 240, 'monthly': 730}

    for i, crypto_info in enumerate(crypto_list):
        coin_id = crypto_info['id']
        symbol_display = crypto_info['symbol']
        symbols_processed_count += 1
        status_text.text(f"Processing {symbol_display} ({symbols_processed_count}/{total_symbols})...")

        try:
            hourly_df, hourly_signals = fetch_indicators_with_signals_cg(coin_id, symbol_display, days_map['hourly'])
            daily_df, daily_signals = fetch_indicators_with_signals_cg(coin_id, symbol_display, days_map['daily'])
            weekly_df, weekly_signals = fetch_indicators_with_signals_cg(coin_id, symbol_display, days_map['weekly'])
            monthly_df, monthly_signals = fetch_indicators_with_signals_cg(coin_id, symbol_display, days_map['monthly'])

            if daily_df is None or daily_df.empty:
                 continue
            price_info, pct_change = calculate_price_change(daily_df)
            if price_info == "N/A":
                 continue

            signal_decision = calculate_signal_score(hourly_signals, daily_signals)

            combined = {
                'Crypto': symbol_display,
                'Price (1d %)': price_info,
                'Signal Score': signal_decision,
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
            st.error(f"Errore GRAVE non gestito nel ciclo principale per {symbol_display}: {e}", icon="🚨")
            st.error(f"Traceback: {traceback.format_exc()}")

        progress_bar.progress((i + 1) / total_symbols)

    final_message = f"Processing complete (CoinGecko Data). {symbols_valid_count} symbols loaded out of {symbols_processed_count} processed."
    if symbols_valid_count == 0 and symbols_processed_count > 0:
         final_message += " Nessun dato valido trovato. Controllare la fonte dati (CoinGecko) o eventuali errori."
    status_text.text(final_message)
    progress_bar.empty()

    if results:
        df_results = pd.DataFrame(results)
        # (Funzioni stile tabella invariate)
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
        # Messaggio warning aggiornato
        st.warning("⚠️ Nessun dato disponibile per BTC/ETH/SOL (CoinGecko). Controlla i log o riprova.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"Errore critico all'avvio dell'applicazione: {e}", icon="💥")
        st.error(f"Traceback: {traceback.format_exc()}")