
import streamlit as st
import pandas as pd
import requests

st.set_page_config(layout="wide")
st.title("📊 Crypto Dashboard LIVE – Binance API + RSI")

# Funzione per ottenere i dati da Binance (OHLC)
def get_binance_klines(symbol="BTCUSDT", interval="1h", limit=100):
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params)
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["close"] = pd.to_numeric(df["close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df[["timestamp", "close"]]

# Calcolo RSI manuale
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Ottieni dati Binance
df = get_binance_klines("BTCUSDT", "1h", 100)
df["RSI"] = compute_rsi(df["close"])

# Visualizzazione tabella
latest_price = df["close"].iloc[-1]
latest_rsi = df["RSI"].iloc[-1]

st.metric("🪙 BTC/USDT Price", f"${latest_price:,.2f}")
st.metric("📈 RSI (1h)", f"{latest_rsi:.2f}")

st.line_chart(df.set_index("timestamp")[["close", "RSI"]])

st.markdown("### ℹ️ Indicatori")
st.markdown("""
- **RSI (Relative Strength Index)**: misura la forza di un trend su scala 0–100.  
  Valori sopra 70 = ipercomprato, sotto 30 = ipervenduto.
- Dati aggiornati da [Binance Public API](https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data)
""")
