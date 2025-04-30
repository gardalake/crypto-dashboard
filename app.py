
import streamlit as st
import pandas as pd
import requests

st.set_page_config(layout="wide")
st.title("📈 Crypto Live Dashboard - Prezzi e GPT Signal")

# Funzione per ottenere i dati da CoinGecko
@st.cache_data(ttl=300)
def get_live_data():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": "bitcoin,ethereum,binancecoin,solana,ripple",
        "order": "market_cap_desc"
    }
    r = requests.get(url, params=params)
    return r.json()

# Calcolo GPT Signal in base alla variazione %
def calculate_signal(change):
    if change >= 2:
        return "🔶 Strong Buy"
    elif change >= 1:
        return "🟢 Buy"
    elif change <= -2:
        return "🔻 Strong Sell"
    elif change <= -1:
        return "🔴 Sell"
    else:
        return "🟡 Hold"

data = get_live_data()
rows = []

for coin in data:
    price = f"${coin['current_price']:,}"
    change = coin['price_change_percentage_24h']
    price_pct = f"{price} ({change:+.2f}%)"
    signal = calculate_signal(change)
    rows.append({
        "Crypto": coin['symbol'].upper(),
        "Name": coin['name'],
        "Price (1d %)": price_pct,
        "GPT Signal": signal
    })

df = pd.DataFrame(rows)

def highlight_change(val):
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

st.dataframe(df.style.applymap(highlight_change, subset=['Price (1d %)']), use_container_width=True)
