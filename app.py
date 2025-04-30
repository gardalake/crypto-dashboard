
import streamlit as st
import pandas as pd
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("📊 Crypto Dashboard LIVE – Indicatori Tecnici + GPT Signal")

@st.cache_data(ttl=60)
def get_current_prices():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin,ethereum,binancecoin,solana,ripple",
        "vs_currencies": "usd"
    }
    try:
        r = requests.get(url, params=params)
        data = r.json()
        return {
            "BTC-USD": data.get("bitcoin", {}).get("usd", "N/A"),
            "ETH-USD": data.get("ethereum", {}).get("usd", "N/A"),
            "BNB-USD": data.get("binancecoin", {}).get("usd", "N/A"),
            "SOL-USD": data.get("solana", {}).get("usd", "N/A"),
            "XRP-USD": data.get("ripple", {}).get("usd", "N/A")
        }
    except:
        return {}

@st.cache_data(ttl=60)
def get_price_history(symbol, period="2d", interval="1d"):
    try:
        data = yf.download(tickers=symbol, period=period, interval=interval)
        return data
    except:
        return pd.DataFrame()

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]
prices = get_current_prices()

results = []

for symbol in symbols:
    hist = get_price_history(symbol, period="2d", interval="1d")
    try:
        last_price = hist["Close"].iloc[-1]
        prev_price = hist["Close"].iloc[-2]
        change = ((last_price - prev_price) / prev_price) * 100
        price_display = f"${prices.get(symbol, "N/A")} ({change:+.2f}%)"
    except:
        price_display = "N/A"
        change = 0

    # GPT Signal
    if change >= 2:
        gpt = "🔶 Strong Buy"
    elif change >= 1:
        gpt = "🟢 Buy"
    elif change <= -2:
        gpt = "🔻 Strong Sell"
    elif change <= -1:
        gpt = "🔴 Sell"
    else:
        gpt = "🟡 Hold"

    # RSI values
    rsi1d = "N/A"
    rsi1h = rsi1w = rsi1mo = "N/A"

    if not hist.empty:
        rsi1d_calc = calculate_rsi(hist["Close"])
        if not rsi1d_calc.empty:
            rsi1d = round(rsi1d_calc.dropna().iloc[-1], 2)

    results.append({
        "Crypto": symbol,
        "Prezzo (CoinGecko)": price_display,
        "GPT Signal": gpt,
        "RSI 1h": rsi1h,
        "RSI 1d": rsi1d,
        "RSI 1w": rsi1w,
        "RSI 1mo": rsi1mo,
        "SRSI": "N/A",
        "MACD": "N/A",
        "MA": "N/A",
        "Doda Stoch": "N/A",
        "GChannel": "N/A",
        "Volume Flow": "N/A",
        "VWAP": "N/A"
    })

df = pd.DataFrame(results)

def highlight_pct(val):
    if isinstance(val, str) and '%' in val:
        try:
            num = float(val.split('(')[-1].replace('%','').replace(')',''))
            if num > 0:
                return 'color: green'
            elif num < 0:
                return 'color: red'
        except:
            return ''
    return ''

st.dataframe(df.style.applymap(highlight_pct, subset=["Prezzo (CoinGecko)"]), use_container_width=True)

st.markdown("### ℹ️ Indicatori Tecnici - Legenda")
st.markdown("""
- **RSI (Relative Strength Index)**: misura la forza di un trend (0–100), sopra 70 è ipercomprato, sotto 30 è ipervenduto.
- **SRSI**: stocastico dell'RSI.
- **MACD**: confronto di medie mobili, segnala cambi di tendenza.
- **MA**: media mobile semplice.
- **Doda Stoch**: versione personalizzata dello stocastico.
- **GChannel**: simile a Donchian/Gann Channel.
- **Volume Flow**: flusso di capitale basato sul volume.
- **VWAP**: prezzo medio ponderato per il volume.
- **GPT Signal**: decisione finale AI semplificata (Strong Buy → Strong Sell).
""")


st.markdown(f"⏱️ Ultimo aggiornamento prezzi da CoinGecko: **{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}**")
