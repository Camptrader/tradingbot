import re
from datetime import datetime

import optuna
import pandas as pd
import streamlit as st

from datafeed import (
    load_csv, load_yfinance, load_alpaca,
    load_tvdatafeed, get_tv_ta, load_ccxt
)
from strategies import crypto_intraday_multi
from strategies.rma import rma_strategy
from strategies.sma_cross import sma_cross_strategy

STRATEGY_REGISTRY = {
    "RMA Strategy": {
        "function": rma_strategy,
        "params": [
            "rma_len", "barsforentry", "barsforexit", "atrlen",
            "normalizedupper", "normalizedlower", "emasrc", "ema_fast_len", "ema_slow_len",
            "risklen", "trailpct"],
        "universal": ["session_start", "session_end", "keepLime", "initial_capital", "qty", "maxtradesperday"]
    },
    "Crypto Intraday Multi-Signal": {
        "function": crypto_intraday_multi,
        "params": [
            "breakout_len", "momentum_len", "momentum_thresh", "trend_len",
            "atr_len", "min_atr", "trailing_stop_pct", "max_hold_bars"],
        "universal": ["initial_capital", "qty"]
    },
    "SMA Cross": {
        "function": sma_cross_strategy,
        "params": [
            "fast_len", "slow_len"],
        "universal": ["initial_capital", "qty"]
    }
}

st.set_page_config(page_title="Universal Backtester", layout="wide")
st.title("📊 Universal Optuna Backtester")

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

feed_choices = ["csv", "ccxt", "yfinance", "alpaca", "tvdatafeed", "tradingview_ta"]
selected_feed = st.sidebar.selectbox("Data Source", feed_choices)

df = None
if selected_feed == "csv" and csv_key in csv_symbols:
    df = load_csv(csv_symbols[csv_key])
    if df is not None and not df.empty:
        st.session_state['df_loaded'] = df
elif selected_feed == "ccxt":
    ccxt_exchange = st.sidebar.selectbox(
        "Exchange", ["alpaca", "binanceus", "binance", "bybit", "okx", "kucoin", "coinbase", "kraken"], index=0)
    ccxt_symbol = st.sidebar.text_input("CCXT Symbol (e.g. BTC/USDT)", value="BTC/USDT")
    ccxt_tf = st.sidebar.selectbox(
        "CCXT Interval", ["1m", "3m", "5m", "15m", "1h", "4h", "1d"], index=2)
    ccxt_limit = st.sidebar.number_input("Bars", min_value=50, max_value=50000, value=5000)
    if st.sidebar.button("Load CCXT Data"):
        df = load_ccxt(
            symbol=ccxt_symbol,
            timeframe=ccxt_tf,
            limit=int(ccxt_limit),
            exchange=ccxt_exchange
        )
        st.success(f"Loaded CCXT {ccxt_symbol} {ccxt_tf} from {ccxt_exchange}")
        if df is not None and not df.empty:
            st.session_state['df_loaded'] = df
elif selected_feed == "yfinance":
    start = st.sidebar.date_input("Start Date", value=datetime(2025, 1, 1))
    end = st.sidebar.date_input("End Date")
    if st.sidebar.button("Load Yahoo Data"):
        df = load_yfinance(selected_symbol, selected_tf, start, end)
        if df is not None and not df.empty:
            st.session_state['df_loaded'] = df
elif selected_feed == "alpaca":
    start = st.sidebar.date_input("Start Date", value=datetime(2025, 1, 1))
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
if df is not None and not df.empty:
    # ENFORCE all columns lowercase ONCE
    df.columns = [c.lower() for c in df.columns]
    #    st.write(f"Lowercased columns from {selected_feed}:", list(df.columns))
    if 'date' not in df.columns:
        st.error("Loaded data has no 'date' column after normalization.")
    else:
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

        st.write("Selected Symbol:", selected_symbol)
        st.write("Rows after slicing:", len(df))
        st.write("Date min/max after slicing:", df['date'].min(), df['date'].max())
else:
    st.info("No data loaded yet or data has no 'date' column.")

# ---- Strategy Selector ----
strategy_choice = st.sidebar.selectbox(
    "Select Strategy",
    list(STRATEGY_REGISTRY.keys())
)
strategy_entry = STRATEGY_REGISTRY[strategy_choice]
strategy_func = strategy_entry["function"]
strategy_params = strategy_entry["params"]
universal_keys = strategy_entry.get("universal", [])

# --- Strategy/Optuna controls ---
maxtradesperday = st.sidebar.number_input("Max Trades Per Day", min_value=1, max_value=100, value=1, step=1)
session_start_h = st.sidebar.number_input("Session Start Hour", min_value=0, max_value=23, value=13)
session_start_m = st.sidebar.number_input("Session Start Minute", min_value=0, max_value=59, value=31)
session_end_h = st.sidebar.number_input("Session End Hour", min_value=0, max_value=23, value=19)
session_end_m = st.sidebar.number_input("Session End Minute", min_value=0, max_value=59, value=52)
optuna_trials = st.sidebar.number_input("Optuna Trials", min_value=10, max_value=1000, value=100, step=10)
optimize_for = st.sidebar.selectbox("Optimize For", ["return", "Win", "return with Win% tie-breaker"])
run_optuna = st.sidebar.button("Run Optuna Optimization")

# ----------- Run Optimization -----------
if run_optuna and df is not None and not df.empty:
    df_strategy = df.copy()
    df_strategy.columns = [c.lower() for c in df_strategy.columns]

    session_start = f"{int(session_start_h):02d}:{int(session_start_m):02d}"
    session_end = f"{int(session_end_h):02d}:{int(session_end_m):02d}"

    # Collect sidebar values (all lowercase, universal params)
    sidebar_values = {
        "session_start": session_start,
        "session_end": session_end,
        "keeplime": True,
        "initial_capital": 200000,
        "qty": 1,
        "maxtradesperday": maxtradesperday
    }


    def objective(trial):
        strategy_kwargs = {}
        for param in strategy_params:
            if param == 'rma_len':
                strategy_kwargs[param] = trial.suggest_int(param, 40, 100)
            elif param == 'barsforentry':
                strategy_kwargs[param] = trial.suggest_int(param, 1, 10)
            elif param == 'barsforexit':
                strategy_kwargs[param] = trial.suggest_int(param, 1, 10)
            elif param == 'atrlen':
                strategy_kwargs[param] = trial.suggest_int(param, 2, 20)
            elif param == 'emasrc':
                strategy_kwargs[param] = trial.suggest_categorical('emasrc', [30, 40, 45, 60, 120])
            elif param == 'ema_fast_len':
                strategy_kwargs[param] = trial.suggest_int(param, 2, 30)
            elif param == 'ema_slow_len':
                fast_val = strategy_kwargs.get('ema_fast_len', 2)  # fallback to 2 if not set
                strategy_kwargs[param] = trial.suggest_int(param, fast_val + 1, 100)
            elif param in ['normalizedupper']:
                strategy_kwargs[param] = trial.suggest_float(param, 0, 2, step=0.1)
            elif param in ['normalizedlower']:
                strategy_kwargs[param] = trial.suggest_float(param, -2, 0, step=0.1)
            elif param in ['risklen', 'trailpct']:
                strategy_kwargs[param] = trial.suggest_float(param, 0, 50, step=0.5)
            elif param in ['fast_len']:
                strategy_kwargs[param] = trial.suggest_int(param, 2, 25)
            elif param in ['slow_len']:
                strategy_kwargs[param] = trial.suggest_int(param, 5, 60)
            elif param == 'breakout_len':
                strategy_kwargs[param] = trial.suggest_int(param, 5, 50)
            elif param == 'momentum_len':
                strategy_kwargs[param] = trial.suggest_int(param, 3, 30)
            elif param == 'momentum_thresh':
                strategy_kwargs[param] = trial.suggest_float(param, 0.1, 3.0, step=0.1)
            elif param == 'trend_len':
                strategy_kwargs[param] = trial.suggest_int(param, 10, 100)
            elif param == 'atr_len':
                strategy_kwargs[param] = trial.suggest_int(param, 5, 50)
            elif param == 'min_atr':
                strategy_kwargs[param] = trial.suggest_float(param, 0.1, 5.0, step=0.1)
            elif param == 'trailing_stop_pct':
                strategy_kwargs[param] = trial.suggest_float(param, 0.2, 10, step=0.1)
            elif param == 'max_hold_bars':
                strategy_kwargs[param] = trial.suggest_int(param, 5, 50)
            else:
                continue

        call_kwargs = {k: sidebar_values[k] for k in universal_keys if k in sidebar_values}
        for k in strategy_kwargs:
            call_kwargs[k] = strategy_kwargs[k]

        trades, _ = strategy_func(df_strategy, **call_kwargs)

        total_return = trades['return'].sum() if not trades.empty and 'return' in trades else -999
        win_pct = (trades['pnl'] > 0).mean() if not trades.empty and 'pnl' in trades else 0
        if optimize_for.lower() == 'return':
            return total_return
        elif optimize_for.lower() == 'win':
            return win_pct
        else:
            return total_return + win_pct / 100


    st.info(
        f"((({strategy_choice}))) --------------- Running Optuna optimizer for {optuna_trials} trials...")  # /////////////////////////
    #    st.info(f"Running Optuna optimizer for {optuna_trials} trials...")
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

    call_kwargs = {k: sidebar_values[k] for k in universal_keys if k in sidebar_values}
    for k in best_params:
        call_kwargs[k] = best_params[k]
    trades, df_all = strategy_func(df_strategy, **call_kwargs)

    if not df_all.empty:
        test_start = df_all["date"].min()
        test_end = df_all["date"].max()
        st.markdown(
            f"**Test Period:** {test_start.strftime('%Y-%m-%d %H:%M')} &mdash; {test_end.strftime('%Y-%m-%d %H:%M')}")

    if 'return' in trades.columns:
        st.markdown(f"**Total return:** {trades['return'].sum():.2f}%")
        st.markdown(f"**Total P&L:** {trades['pnl'].sum():.2f}%")
        st.markdown(f"**Max Ddn:** {trades['pnl'].min():.2f}")
    else:
        st.warning("No 'return' column in trades.")
    if 'pnl' in trades.columns:
        st.markdown(f"**Win %:** {(trades['pnl'] > 0).mean() * 100:.2f}%")
    if 'barsintrade' in trades.columns:
        st.markdown(f"**Avg Bars In Trade:** {trades['barsintrade'].mean():.1f}")
    plot_col = None
    for col in ['close', 'Close']:
        if col in df_all.columns:
            plot_col = col
            break
    st.markdown("### Best Parameter Trade Log")
    st.dataframe(trades)
    st.markdown(f"**Total Trades:** {len(trades)}")

    if not trades.empty and plot_col is not None:
        st.line_chart(df_all[plot_col])
    else:
        st.info("No price chart to display. 'close' column missing or no trades.")

else:
    st.info("Upload data and select symbol/feed, set parameters, and click 'Run Optuna Optimization' to start.")
