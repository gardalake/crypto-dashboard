import pandas as pd
import ta
import streamlit as st
import yfinance as yf
import requests

@st.cache_data(ttl=600, show_spinner=False)
def get_top_100_crypto_symbols():
    try:
        url = 'https://api.coingecko.com/api/v3/coins/markets'
        params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 10, 'page': 1}
        response = requests.get(url, params=params)
        data = response.json()
        tickers = [f"{coin['symbol'].upper()}-USD" for coin in data]
        return tickers
    except:
        return ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']

@st.cache_data(ttl=300, show_spinner=False)
def fetch_indicators_with_signals(symbol, interval, period, label):
    try:
        df = yf.download(tickers=symbol, interval=interval, period=period)
        if df.empty or len(df) < 20:
            return {f'RSI {label}': 'N/A', 'SRSI': 'N/A', 'MACD': 'N/A', 'MA': 'N/A',
                    'Doda Stoch': 'N/A', 'GChannel': 'N/A', 'Vol Flow': 'N/A', 'VWAP': 'N/A'}

        df['close'] = df['Close']
        df['high'] = df['High']
        df['low'] = df['Low']
        df['volume'] = df['Volume']

        rsi_val = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
        srsi_val = ta.momentum.StochRSIIndicator(df['close']).stochrsi().iloc[-1]
        macd_val = ta.trend.MACD(df['close']).macd().iloc[-1]
        ma = df['close'].rolling(window=14).mean().iloc[-1]
        doda = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch().iloc[-1]
        ema3 = df['close'].ewm(span=3).mean().iloc[-1]
        ema30 = df['close'].ewm(span=30).mean().iloc[-1]
        gchannel = ema3 - ema30
        vfi = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index().iloc[-1]
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        vwap_val = vwap.iloc[-1]

        def signal_emoji(value, low, high):
            if isinstance(value, str):
                return 'N/A'
            if value < low * 0.67:
                return '🔶'
            elif value < low:
                return '🟢'
            elif value > high * 1.33:
                return '🔻'
            elif value > high:
                return '🔴'
            else:
                return '🟡'

        return {
            f'RSI {label}': f"{round(rsi_val, 2)} {signal_emoji(rsi_val, 30, 70)}",
            'SRSI': f"{round(srsi_val, 2)} {signal_emoji(srsi_val, 0.2, 0.8)}",
            'MACD': f"{round(macd_val, 2)} {signal_emoji(macd_val, -0.5, 0.5)}",
            'MA': round(ma, 2),
            'Doda Stoch': round(doda, 2),
            'GChannel': round(gchannel, 2),
            'Vol Flow': round(vfi, 2),
            'VWAP': round(vwap_val, 2)
        }
    except Exception:
        return {f'RSI {label}': 'N/A', 'SRSI': 'N/A', 'MACD': 'N/A', 'MA': 'N/A',
                'Doda Stoch': 'N/A', 'GChannel': 'N/A', 'Vol Flow': 'N/A', 'VWAP': 'N/A'}

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    st.title("📈 Live Crypto Technical Dashboard")
    st.write("Auto-refresh every 5 minutes. Use your mouse scroll wheel or trackpad to move up/down.")

    with st.expander("ℹ️ GPT Signal Legend"):
        st.markdown("""
        **How the GPT score is calculated:**

        - 🔶 **Strong Buy**: Very strong positive signals (+4 points or more)
        - 🟢 **Buy**: Moderate positive signals (+2 to +3 points)
        - 🟡 **Hold**: Neutral or uncertain signals (-1 to +1 points)
        - 🔴 **Sell**: Moderate negative signals (-2 to -3 points)
        - 🔻 **Strong Sell**: Very strong negative signals (-4 points or less)

        **Indicators used:** RSI (1h), RSI (1d), MACD (daily), SRSI (hourly)
        """)

    with st.expander("📚 Indicators Description"):
        st.markdown("""
        **RSI (Relative Strength Index)**: Detects momentum, useful to identify overbought/oversold conditions in crypto.

        **Stochastic RSI (SRSI)**: A faster RSI. Detects short-term momentum changes better.

        **MACD (Moving Average Convergence Divergence)**: Captures trend strength and direction. Useful to confirm bullish/bearish trends.

        **MA (Moving Average)**: A simple smoothed average price. Helps follow trend direction.

        **Doda Stochastic Oscillator**: Quick overbought/oversold detector. Suitable for volatile cryptos.

        **GChannel (Guppy Channel)**: Detects trend reversals based on short/long EMAs.

        **Vol Flow (Volume Flow Indicator)**: Measures buying/selling pressure based on volume.

        **VWAP (Volume Weighted Average Price)**: Average price based on volume. Used by institutional traders.
        """)

    crypto_symbols = get_top_100_crypto_symbols()
    results = []
    for symbol in crypto_symbols:
        try:
            hourly = fetch_indicators_with_signals(symbol, '60m', '5d', '1h')
            daily = fetch_indicators_with_signals(symbol, '1d', '1mo', '1d')
            weekly = fetch_indicators_with_signals(symbol, '1wk', '2mo', '1w')
            monthly = fetch_indicators_with_signals(symbol, '1mo', '1y', '1mo')
        except Exception as e:
            st.warning(f"Data fetch error for {symbol}: {e}")
            continue

        price_df = yf.download(tickers=symbol, interval='1d', period='2d')
        if price_df.empty or len(price_df) < 2:
            continue
        else:
            latest_price = price_df['Close'].iloc[-1]
            prev_price = price_df['Close'].iloc[-2]
            pct_change = ((latest_price - prev_price) / prev_price) * 100
            price_info = f"${latest_price:.2f} ({pct_change:+.2f}%)"

        decision_score = 0
        indicators = [hourly['RSI 1h'], daily['RSI 1d'], daily['MACD'], hourly['SRSI']]
        for indicator in indicators:
            if isinstance(indicator, str):
                if '🔶' in indicator:
                    decision_score += 2
                elif '🟢' in indicator:
                    decision_score += 1
                elif '🟡' in indicator:
                    decision_score += 0
                elif '🔴' in indicator:
                    decision_score -= 1
                elif '🔻' in indicator:
                    decision_score -= 2

        if decision_score >= 4:
            gpt_decision = '**🔶 Strong Buy**'
        elif decision_score >= 2:
            gpt_decision = '**🟢 Buy**'
        elif decision_score <= -4:
            gpt_decision = '**🔻 Strong Sell**'
        elif decision_score <= -2:
            gpt_decision = '**🔴 Sell**'
        else:
            gpt_decision = '**🟡 Hold**'

        combined = {
            'Crypto': symbol,
            'Price (1d %)': price_info,
            'GPT': gpt_decision,
            'RSI (1h)': hourly['RSI 1h'],
            'RSI (1d)': daily['RSI 1d'],
            'RSI (1w)': weekly['RSI 1w'],
            'RSI (1mo)': monthly['RSI 1mo'],
            'SRSI': hourly['SRSI'],
            'MACD': hourly['MACD'],
            'MA': hourly['MA'],
            'Doda Stoch': hourly['Doda Stoch'],
            'GChannel': hourly['GChannel'],
            'Vol Flow': hourly['Vol Flow'],
            'VWAP': hourly['VWAP']
        }
        results.append(combined)

    df = pd.DataFrame(results)

    def highlight_price(val):
        if isinstance(val, str) and '(' in val and '%' in val:
            try:
                percent = float(val.split('(')[-1].replace('%', '').replace(')', ''))
                if percent > 0:
                    return 'color: green'
                elif percent < 0:
                    return 'color: red'
            except:
                return ''
        return ''

    styled_df = df.style.applymap(highlight_price, subset=['Price (1d %)'])
    st.dataframe(styled_df, use_container_width=True, height=800)

if __name__ == '__main__':
    main()
