
import pandas as pd
import streamlit as st
import yfinance as yf
import requests

@st.cache_data(ttl=600, show_spinner=False)
def get_top_100_crypto_symbols():
    try:
        url = 'https://api.coingecko.com/api/v3/coins/markets'
        params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 10, 'page': 1}
        response = requests.get(url, params=params)
        data = response.json()
        tickers = [f"{coin['symbol'].upper()}-USD" for coin in data]
        return tickers
    except:
        return ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    st.title("📈 Live Crypto Technical Dashboard (Simplified Version)")
    st.write("Auto-refresh every 5 minutes. Use your mouse scroll wheel or trackpad to move up/down.")

    crypto_symbols = get_top_100_crypto_symbols()
    results = []
    for symbol in crypto_symbols:
        st.markdown(f"➡️ Processing `{symbol}`...")
        try:
            price_df = yf.download(tickers=symbol, interval='1d', period='2d')
            if price_df is None or price_df.empty or 'Close' not in price_df.columns or price_df['Close'].isna().all():
                continue
            latest_price = price_df['Close'].iloc[-1]
            prev_price = price_df['Close'].iloc[-2]
            if pd.isna(latest_price) or pd.isna(prev_price):
                continue
            pct_change = ((latest_price - prev_price) / prev_price) * 100
            price_info = f"${latest_price:.2f} ({pct_change:+.2f}%)"

            if pct_change >= 2:
                gpt_decision = '**🔶 Strong Buy**'
            elif pct_change >= 1:
                gpt_decision = '**🟢 Buy**'
            elif pct_change <= -2:
                gpt_decision = '**🔻 Strong Sell**'
            elif pct_change <= -1:
                gpt_decision = '**🔴 Sell**'
            else:
                gpt_decision = '**🟡 Hold**'

            combined = {
                'Crypto': symbol,
                'Price (1d %)': price_info,
                'GPT Signal': gpt_decision
            }
            results.append(combined)

        except Exception as err:
            st.error(f"❌ Failed for {symbol}: {err}")
            continue

    if results:
        df = pd.DataFrame(results)

        def highlight_price(val):
            if isinstance(val, str) and '(' in val and '%' in val:
                try:
                    percent = float(val.split('(')[-1].replace('%', '').replace(')', ''))
                    if percent > 0:
                        return 'color: green'
                    elif percent < 0:
                        return 'color: red'
                except:
                    return ''
            return ''

        styled_df = df.style.applymap(highlight_price, subset=['Price (1d %)'])
        st.dataframe(styled_df, use_container_width=True, height=800)
    else:
        st.warning("⚠️ No crypto data available at the moment. Please try again later.")

if __name__ == '__main__':
    main()
