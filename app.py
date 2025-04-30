
import streamlit as st
import pandas as pd
import requests

st.set_page_config(layout="wide")
st.title("📊 BTC Live Dashboard – Binance API + RSI con fallback CoinGecko")

# === Funzione 1: Binance API (preferita)
def get_binance_klines(symbol="BTCUSDT", interval="1h", limit=100):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["close"] = pd.to_numeric(df["close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df[["timestamp", "close"]]
    except:
        return pd.DataFrame()

# === Funzione 2: CoinGecko fallback
def get_price_from_coingecko():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "bitcoin", "vs_currencies": "usd"}
        r = requests.get(url, params=params, timeout=5)
        return r.json().get("bitcoin", {}).get("usd", "N/A")
    except:
        return "N/A"

# RSI manuale
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# === DATI ===
df = get_binance_klines()

if not df.empty:
    df["RSI"] = compute_rsi(df["close"])
    latest_price = df["close"].iloc[-1]
    latest_rsi = df["RSI"].iloc[-1]
    st.success("✅ Dati da Binance API")
else:
    latest_price = get_price_from_coingecko()
    latest_rsi = "N/A"
    st.warning("⚠️ Binance non ha risposto. Usato CoinGecko solo per il prezzo.")

st.metric("🪙 BTC/USDT Price", f"${latest_price}")
st.metric("📈 RSI (1h)", f"{latest_rsi if latest_rsi != 'N/A' else 'N/A'}")

if not df.empty:
    st.line_chart(df.set_index("timestamp")[["close", "RSI"]])

st.markdown("### ℹ️ Indicatori")
st.markdown("""
- **RSI (Relative Strength Index)**: misura la forza del trend su scala 0–100.  
  Sopra 70 = ipercomprato, sotto 30 = ipervenduto.
- I dati provengono da [Binance](https://binance.com) con fallback su [CoinGecko](https://coingecko.com).
""")
