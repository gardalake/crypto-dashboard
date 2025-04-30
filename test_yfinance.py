import yfinance as yf
import pandas as pd

# Lista di ticker da testare
tickers = ["BTC-USD", "ETH-USD", "MSFT", "GC=F"] # Aggiungo anche un'azione (Microsoft) e una commodity (Oro) per confronto

print(f"Versione yfinance installata: {yf.__version__}")
print("-" * 30)

for ticker_symbol in tickers:
    print(f"Tentativo di scaricare dati per: {ticker_symbol}")
    try:
        # Usa yf.Ticker per un controllo più granulare
        ticker_obj = yf.Ticker(ticker_symbol)
        # Scarica un piccolo periodo storico
        hist = ticker_obj.history(period="5d", interval="1d")

        if not hist.empty:
            print(f"✅ Successo! Scaricate {len(hist)} righe di dati.")
            # print(hist.tail(2)) # Opzionale: stampa le ultime 2 righe
        else:
            print(f"⚠️ Attenzione: Nessun dato restituito per {ticker_symbol}. (DataFrame vuoto)")
            # Controlla se ci sono news, potrebbero indicare problemi col ticker
            try:
               news = ticker_obj.news
               if news:
                   print(f"   (Trovate news per {ticker_symbol}, il ticker potrebbe essere valido ma senza dati storici recenti?)")
               else:
                    print(f"   (Nessuna news trovata per {ticker_symbol})")
            except Exception as news_e:
               print(f"   (Errore nel recupero news: {news_e})")


    except Exception as e:
        print(f"❌ Errore durante il download per {ticker_symbol}: {e}")
    print("-" * 30)

print("Test completato.")