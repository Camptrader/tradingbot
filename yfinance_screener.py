import yfinance as yf
screen_list_col, screen_filter_col = st.columns([1, 1])
# Option A: Get S&P500 from Wikipedia (or use your own list/CSV)
with screen_list_col:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_df = pd.read_html(url)[0]
    symbol_list = sp500_df['Symbol'].tolist()

    # User can also upload a custom list
    uploaded_file = st.file_uploader("Upload symbol list (CSV, 'Symbol' column)", type=["csv"])
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        symbol_list = user_df['Symbol'].tolist()

    min_market_cap = st.number_input("Min Market Cap ($B)", 10, 3000, 100)
    max_pe = st.number_input("Max P/E Ratio", 1, 100, 40)
    min_cp_ratio = st.slider("Min Call/Put OI Ratio", 0.1, 5.0, 1.2)
    run_screen = st.button("Run Screener")

with screen_filter_col:
    if run_screen:
        results = []
        for sym in symbol_list:
            try:
                t = yf.Ticker(sym)
                info = t.info
                mktcap = info.get("marketCap", 0) / 1e9
                pe = info.get("trailingPE", None)
                expiries = t.options
                if not expiries or pe is None or mktcap < min_market_cap or pe > max_pe:
                    continue
                expiry = expiries[0]
                opt = t.option_chain(expiry)
                calls = opt.calls['openInterest'].sum()
                puts = opt.puts['openInterest'].sum()
                cp_ratio = calls / puts if puts else float('inf')
                if cp_ratio >= min_cp_ratio:
                    results.append({
                        "Symbol": sym,
                        "Market Cap ($B)": round(mktcap,2),
                        "P/E": round(pe,2),
                        "Call/Put OI Ratio": round(cp_ratio,2)
                    })
            except Exception as e:
                pass  # Optionally log errors
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)
            pick = st.selectbox("Pick a symbol:", df["Symbol"].tolist())
            if st.button("Set as selected_symbol"):
                st.session_state['selected_symbol'] = pick
                st.success(f"Selected symbol: {pick}")
        else:
            st.warning("No symbols passed the filters.")