
import streamlit as st
import pandas as pd
import requests
import random

st.set_page_config(layout="wide")
st.title("📊 Crypto Dashboard LIVE — Indicatori Tecnici & GPT Signal")

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

# Calcolo GPT Signal
def calculate_signal(change):
    if change is None:
        return "❔ N/A"
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

# Simulazione indicatori (in attesa di calcolo reale)
def safe_random():
    return round(random.uniform(0, 100), 2)

def simulate_indicator():
    return random.choice([round(random.uniform(0.1, 1.5), 2), 'N/A'])

data = get_live_data()
rows = []

for coin in data:
    try:
        price = coin['current_price']
        change = coin.get('price_change_percentage_24h', None)
        price_str = f"${price:,.2f}" if price else "N/A"
        pct_str = f"{price_str} ({change:+.2f}%)" if change is not None else "N/A"
    except:
        pct_str = "N/A"
        change = None

    rows.append({
        "Crypto": coin['symbol'].upper(),
        "Name": coin['name'],
        "Price (1d %)": pct_str,
        "GPT Signal": calculate_signal(change),
        "RSI 1h": safe_random(),
        "RSI 1d": safe_random(),
        "RSI 1w": safe_random(),
        "RSI 1mo": safe_random(),
        "SRSI": simulate_indicator(),
        "MACD": simulate_indicator(),
        "MA": simulate_indicator(),
        "Doda Stoch": simulate_indicator(),
        "GChannel": simulate_indicator(),
        "Volume Flow": simulate_indicator(),
        "VWAP": simulate_indicator()
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

# Legenda
st.markdown("### ℹ️ Legenda e Indicatori Tecnici")

st.markdown("""
**GPT Signal**: Sintesi automatica della forza del movimento giornaliero.
- 🔶 Strong Buy: trend molto positivo
- 🟢 Buy: crescita moderata
- 🟡 Hold: situazione stabile
- 🔴 Sell: calo moderato
- 🔻 Strong Sell: calo importante

**Indicatori Tecnici**:
- **RSI (1h, 1d, 1w, 1mo)**: indice di forza relativa (0–100), sopra 70 = ipercomprato, sotto 30 = ipervenduto.
- **SRSI**: RSI stocastico, misura la velocità del RSI.
- **MACD**: confronto tra 2 medie mobili esponenziali per segnalare cambi trend.
- **MA**: media mobile semplice, utile per capire la tendenza generale.
- **Doda Stoch**: derivato dello stocastico, utile per segnali di entrata/uscita (versione semplificata).
- **GChannel**: simile a Donchian o Gann, mostra bande di prezzo.
- **Volume Flow**: combina volume e prezzo per rilevare flussi di capitale.
- **VWAP**: prezzo medio ponderato per il volume, importante per trader istituzionali.
""")

st.markdown("⚠️ Valori simulati per indicatori: sarà possibile collegare librerie reali come `ta` sul server Hetzner.")
