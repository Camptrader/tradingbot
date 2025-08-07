import re
from datetime import datetime

import optuna
import pandas as pd
import streamlit as st

from datafeed import (
    load_csv, load_yfinance, load_alpaca,
    load_tvdatafeed, get_tv_ta
)
# ////////////////////////////////RMA Registration ///////////////////////////////////////////
from strategies.rma import rma_strategy
from strategies.sma_cross import sma_cross_strategy

STRATEGY_REGISTRY = {
    "RMA Strategy": {
        "function": rma_strategy,
        "params": [
            "rma_len", "barsForEntry", "barsForExit", "ATRLen",
            "normalizedUpper", "normalizedLower", "ema_fast_len", "ema_slow_len",
            "RiskLen", "TrailPct"
        ]
    },
    "SMA Cross": {
        "function": sma_cross_strategy,
        "params": [
            "fast_len", "slow_len"
        ]
    }
}

st.set_page_config(page_title="Universal Backtester", layout="wide")
st.title("📊 Universal Optuna Backtester")

# ---- Multi-CSV Symbol & Timeframe Search ----
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSVs (SYMBOL_TIMEFRAME.csv)", type=["csv"], accept_multiple_files=True
)
csv_symbols = {}
for file in uploaded_files or []:
    m = re.match(r"([A-Za-z0-9]+)_([0-9a-zA-Z]+)\.csv", file.name)
    if m:
        symbol, tf = m.group(1).upper(), m.group(2)
        key = f"{symbol}_{tf}"
        csv_symbols[key] = file

symbol_choices = sorted({key.split("_")[0] for key in csv_symbols.keys()})
tf_choices = sorted({key.split("_")[1] for key in csv_symbols.keys()})

if symbol_choices:
    symbol_query = st.sidebar.text_input("Search symbol", "")
    filtered_symbols = [s for s in symbol_choices if symbol_query.upper() in s]
    selected_symbol = st.sidebar.selectbox("Symbol", filtered_symbols)
    filtered_tfs = [key.split("_")[1] for key in csv_symbols if key.startswith(selected_symbol + "_")]
    selected_tf = st.sidebar.selectbox("Timeframe", sorted(set(filtered_tfs)))
    csv_key = f"{selected_symbol}_{selected_tf}"
else:
    selected_symbol = st.sidebar.text_input("Symbol (for APIs, if no CSVs yet)", value="AAPL")
    selected_tf = st.sidebar.selectbox("Timeframe", ["1m", "3m", "5m", "15m", "30m", "1h", "1d"], index=1)
    csv_key = f"{selected_symbol}_{selected_tf}"

feed_choices = ["csv", "tv", "yfinance", "alpaca", "tvdatafeed", "tradingview_ta"]
selected_feed = st.sidebar.selectbox("Data Source", feed_choices)

df = None
if selected_feed == "csv" and csv_key in csv_symbols:
    df = load_csv(csv_symbols[csv_key])
elif selected_feed == "yfinance":
    start = st.sidebar.date_input("Start Date", value=datetime(2023, 1, 1))
    end = st.sidebar.date_input("End Date")
    if st.sidebar.button("Load Yahoo Data"):
        df = load_yfinance(selected_symbol, selected_tf, start, end)
        if df is not None and not df.empty:
            st.session_state['df_loaded'] = df
elif selected_feed == "alpaca":
    start = st.sidebar.date_input("Start Date", value=datetime(2023, 1, 1))
    end = st.sidebar.date_input("End Date")
    key = st.secrets["alpaca"]["key"]
    secret = st.secrets["alpaca"]["secret"]
    url = st.secrets["alpaca"]["base_url"]
    if st.sidebar.button("Load Alpaca Data"):
        df = load_alpaca(selected_symbol, selected_tf, start, end, key, secret, url)
        if df is not None and not df.empty:
            st.session_state['df_loaded'] = df
elif selected_feed == "tvdatafeed":
    exchange = st.sidebar.text_input("Exchange", value="NASDAQ")
    n_bars = st.sidebar.number_input("Bars", value=1000, min_value=10, max_value=5000)
    if st.sidebar.button("Load TV Datafeed"):
        df = load_tvdatafeed(selected_symbol, exchange, selected_tf, n_bars)
        if df is not None and not df.empty:
            st.session_state['df_loaded'] = df
elif selected_feed == "tradingview_ta":
    exchange = st.sidebar.text_input("Exchange", value="NASDAQ")
    if st.sidebar.button("Get TV TA"):
        df = get_tv_ta(selected_symbol, exchange=exchange, interval=selected_tf)
        if df is not None and not df.empty:
            st.session_state['df_loaded'] = df

df = st.session_state.get('df_loaded', None)
# ---- Date Picker and Data Range ----
if df is not None and not df.empty and 'date' in df.columns:
    #    st.write("Columns:", df.columns.tolist())
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    data_start = df['date'].min().date()
    data_end = df['date'].max().date()
    test_range = st.sidebar.date_input(
        "Select Backtest Period",
        value=(data_start, data_end),
        min_value=data_start,
        max_value=data_end
    )
    mask = (df["date"].dt.date >= test_range[0]) & (df["date"].dt.date <= test_range[1])
    df = df.loc[mask]
    st.write("Rows after slicing:", len(df))
    st.write("Date min/max after slicing:", df['date'].min(), df['date'].max())
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # ... now your rest of pipeline will work!
    else:
        st.error("Loaded data has no 'date' column. Please check the loader.")
else:
    st.info("No data loaded yet or data has no 'date' column.")

# ---- Strategy Selector ----
strategy_choice = st.sidebar.selectbox(
    "Select Strategy",
    list(STRATEGY_REGISTRY.keys())
)
strategy_func = STRATEGY_REGISTRY[strategy_choice]["function"]
strategy_params = STRATEGY_REGISTRY[strategy_choice]["params"]

# --- Strategy/Optuna controls ---
maxtradesperday = st.sidebar.number_input("Max Trades Per Day", min_value=1, max_value=10, value=1, step=1)
session_start_h = st.sidebar.number_input("Session Start Hour", min_value=0, max_value=23, value=9)
session_start_m = st.sidebar.number_input("Session Start Minute", min_value=0, max_value=59, value=31)
session_end_h = st.sidebar.number_input("Session End Hour", min_value=0, max_value=23, value=15)
session_end_m = st.sidebar.number_input("Session End Minute", min_value=0, max_value=59, value=52)
optuna_trials = st.sidebar.number_input("Optuna Trials", min_value=10, max_value=1000, value=100, step=10)
optimize_for = st.sidebar.selectbox("Optimize For", ["Return", "Win", "Return with Win% tie-breaker"])
run_optuna = st.sidebar.button("Run Optuna Optimization")

# ----------- Run Optimization -----------
if run_optuna and df is not None and not df.empty:
    df_strategy = df.copy()
    df_strategy.rename(columns={
        "date": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume"
    }, inplace=True)

    session_start = f"{int(session_start_h):02d}:{int(session_start_m):02d}"
    session_end = f"{int(session_end_h):02d}:{int(session_end_m):02d}"


    def objective(trial):
        strategy_kwargs = {}
        for param in strategy_params:
            if param in ['rma_len', 'barsForEntry', 'barsForExit', 'ATRLen', 'ema_fast_len', 'ema_slow_len']:
                low, high = (1, 120) if param == 'ema_slow_len' else (1, 100)
                strategy_kwargs[param] = trial.suggest_int(param, low, high)
            elif param in ['normalizedUpper']:
                strategy_kwargs[param] = trial.suggest_float(param, 0, 2, step=0.1)
            elif param in ['normalizedLower']:
                strategy_kwargs[param] = trial.suggest_float(param, -2, 0, step=0.1)
            elif param in ['RiskLen', 'TrailPct']:
                strategy_kwargs[param] = trial.suggest_float(param, 0, 50, step=0.5)
                # --- SMA params ---
            elif param in ['fast_len']:
                strategy_kwargs[param] = trial.suggest_int(param, 2, 25)
            elif param in ['slow_len']:
                strategy_kwargs[param] = trial.suggest_int(param, 5, 60)
            # Add logic here for more strategies/params
            else:
                continue
        trades, _ = strategy_func(
            df_strategy,
            session_start=session_start,
            session_end=session_end,
            keepLime=True,
            initial_capital=15000,
            qty=1000,
            maxtradesperday=maxtradesperday,
            **strategy_kwargs
        )
        total_return = trades['Return'].sum() if not trades.empty else -999
        win_pct = (trades['PnL'] > 0).mean() if not trades.empty else 0
        if optimize_for == 'Return':
            return total_return
        elif optimize_for == 'Win':
            return win_pct
        else:
            return total_return + win_pct / 100


    st.info(f"Running Optuna optimizer for {optuna_trials} trials...")
    study = optuna.create_study(direction="maximize")
    progress_bar = st.progress(0)


    def optuna_callback(study, trial):
        progress_bar.progress(min(1.0, (trial.number + 1) / optuna_trials))


    study.optimize(objective, n_trials=optuna_trials, callbacks=[optuna_callback])
    progress_bar.empty()
    st.balloons()

    best_params = study.best_trial.params
    st.success("Best Parameters:")
    st.json(best_params)

    trades, df_all = strategy_func(
        df_strategy,
        session_start=session_start,
        session_end=session_end,
        keepLime=True,
        initial_capital=15000,
        qty=1000,
        maxtradesperday=maxtradesperday,
        **best_params
    )
    if not df_all.empty:
        test_start = df_all["Date"].min()
        test_end = df_all["Date"].max()
        st.markdown(
            f"**Test Period:** {test_start.strftime('%Y-%m-%d %H:%M')} &mdash; {test_end.strftime('%Y-%m-%d %H:%M')}")
    st.markdown("### Best Parameter Trade Log")
    st.dataframe(trades)
    st.markdown(f"**Total Trades:** {len(trades)}")
    st.markdown(f"**Total Return:** {trades['Return'].sum():.2f}%")
    st.markdown(f"**Win %:** {(trades['PnL'] > 0).mean() * 100:.2f}%")
    st.markdown(f"**Avg Bars In Trade:** {trades['BarsInTrade'].mean():.1f}")
    if not trades.empty:
        st.line_chart(df_all['Close'])

else:
    st.info("Upload data and select symbol/feed, set parameters, and click 'Run Optuna Optimization' to start.")
