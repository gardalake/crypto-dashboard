
import pandas as pd
import ta
import streamlit as st
import yfinance as yf
import requests

@st.cache_data(ttl=600, show_spinner=False)
def get_top_100_crypto_symbols():
    try:
        url = 'https://api.coingecko.com/api/v3/coins/markets'
        params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 100, 'page': 1}
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

# main function and the rest would continue similarly
