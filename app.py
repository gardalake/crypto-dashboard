
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 Crypto Technical Dashboard (Full Version)")
st.markdown("Mostra indicatori completi per le principali 5 criptovalute")

# Simulazione dati per esempio statico
data = {
    "Crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"],
    "Price (1d %)": ["$63,500 (+2.35%)", "$3,280 (+1.20%)", "$570 (+0.25%)", "$140 (-1.10%)", "$0.58 (-2.40%)"],
    "RSI 1h": [65, 61, 59, 52, 47],
    "RSI 1d": [72, 69, 65, 60, 43],
    "RSI 1w": [68, 64, 61, 55, 40],
    "RSI 1mo": [63, 60, 58, 52, 39],
    "SRSI": [0.88, 0.73, 0.55, 0.49, 0.30],
    "MACD": [1.5, 1.2, 0.8, 0.3, -1.2],
    "MA": [61000, 3100, 560, 145, 0.62],
    "Doda Stoch": [0.9, 0.7, 0.6, 0.4, 0.2],
    "GChannel": [62500, 3200, 575, 155, 0.66],
    "Volume Flow": [1.8, 1.3, 0.9, 0.5, -0.6],
    "VWAP": [63100, 3250, 568, 142, 0.59],
    "GPT Signal": ["🔶 Strong Buy", "🟢 Buy", "🟡 Hold", "🔴 Sell", "🔻 Strong Sell"]
}

df = pd.DataFrame(data)

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

st.dataframe(df.style.applymap(highlight_pct, subset=['Price (1d %)']), use_container_width=True)
