
# 📊 Crypto Technical Dashboard – Streamlit + CoinGecko + yFinance

Questa è una dashboard professionale per il monitoraggio dei principali indicatori tecnici di criptovalute in tempo reale.  
Utilizza dati live da **CoinGecko** e dati storici da **yFinance** per calcolare indicatori come RSI, MACD, MA, VWAP, ecc.

## ✅ Funzionalità principali

- Prezzi aggiornati (CoinGecko)
- Calcolo dinamico degli indicatori tecnici (via yFinance)
- GPT Signal personalizzato (Strong Buy → Strong Sell)
- Interfaccia Streamlit professionale, compatibile con Streamlit Cloud
- Legenda integrata con spiegazioni dettagliate di ogni indicatore
- Supporto per dati mancanti (`N/A`) con fallback intelligente

## 📦 Librerie usate

- `streamlit`
- `pandas`
- `requests`
- `yfinance==0.2.57`

## 🔧 Esecuzione

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔁 Auto-refresh

Il codice è configurato per aggiornarsi ogni 5 minuti tramite cache dinamica Streamlit (`@st.cache_data(ttl=300)`).

## 📈 Indicatori supportati

- RSI (1h, 1d, 1w, 1mo)
- SRSI (simulato)
- MACD
- MA (media mobile semplice)
- Doda Stoch (versione semplificata)
- GChannel (simulazione bande)
- Volume Flow (stimato)
- VWAP
- GPT Signal

## 📚 Licenza

MIT
