
import pandas as pd
import ta
import streamlit as st
import yfinance as yf

# List of crypto tickers for yfinance
crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']

@st.cache_data(ttl=180)
def fetch_indicators_with_signals(symbol, interval, period, label):
    df = yf.download(tickers=symbol, interval=interval, period=period)

    if df.empty or len(df) < 30:
        raise ValueError("Insufficient data to compute indicators")

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
        if value < low:
            return '🟢'
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

def main():
    st.title("📈 Live Crypto Technical Dashboard")
    st.write("Auto-refresh every 3 minutes")

    results = []
    for symbol in crypto_symbols:
        try:
            hourly = fetch_indicators_with_signals(symbol, '60m', '7d', '1h')
            daily = fetch_indicators_with_signals(symbol, '1d', '1mo', '1d')
            weekly = fetch_indicators_with_signals(symbol, '1wk', '3mo', '1w')
            monthly = fetch_indicators_with_signals(symbol, '1mo', '1y', '1mo')

            combined = {
                'Crypto': symbol,
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

        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")

    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

if __name__ == '__main__':
    main()
