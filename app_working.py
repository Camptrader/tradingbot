import os
import re
import json
import copy
import optuna
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from numpy.matlib import empty
from datafeed import (load_csv, load_yfinance, load_alpaca, load_tvdatafeed, get_tv_ta, load_ccxt)
from strategies.rma import rma_strategy
from strategies.sma_cross import sma_cross_strategy
from strategies.crypto_intraday_multi import crypto_intraday_multi
from strategies.hybrid_atr_mom_break import hybrid_atr_momentum_breakout
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass



#import yfinance as yf
# screen_list_col, screen_filter_col = st.columns([1, 1])
# # Option A: Get S&P500 from Wikipedia (or use your own list/CSV)
# with screen_list_col:
#     url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#     sp500_df = pd.read_html(url)[0]
#     symbol_list = sp500_df['Symbol'].tolist()
#
#     # User can also upload a custom list
#     uploaded_file = st.file_uploader("Upload symbol list (CSV, 'Symbol' column)", type=["csv"])
#     if uploaded_file is not None:
#         user_df = pd.read_csv(uploaded_file)
#         symbol_list = user_df['Symbol'].tolist()
#
#     min_market_cap = st.number_input("Min Market Cap ($B)", 10, 3000, 100)
#     max_pe = st.number_input("Max P/E Ratio", 1, 100, 40)
#     min_cp_ratio = st.slider("Min Call/Put OI Ratio", 0.1, 5.0, 1.2)
#     run_screen = st.button("Run Screener")
#
# with screen_filter_col:
#     if run_screen:
#         results = []
#         for sym in symbol_list:
#             try:
#                 t = yf.Ticker(sym)
#                 info = t.info
#                 mktcap = info.get("marketCap", 0) / 1e9
#                 pe = info.get("trailingPE", None)
#                 expiries = t.options
#                 if not expiries or pe is None or mktcap < min_market_cap or pe > max_pe:
#                     continue
#                 expiry = expiries[0]
#                 opt = t.option_chain(expiry)
#                 calls = opt.calls['openInterest'].sum()
#                 puts = opt.puts['openInterest'].sum()
#                 cp_ratio = calls / puts if puts else float('inf')
#                 if cp_ratio >= min_cp_ratio:
#                     results.append({
#                         "Symbol": sym,
#                         "Market Cap ($B)": round(mktcap,2),
#                         "P/E": round(pe,2),
#                         "Call/Put OI Ratio": round(cp_ratio,2)
#                     })
#             except Exception as e:
#                 pass  # Optionally log errors
#         if results:
#             df = pd.DataFrame(results)
#             st.dataframe(df)
#             pick = st.selectbox("Pick a symbol:", df["Symbol"].tolist())
#             if st.button("Set as selected_symbol"):
#                 st.session_state['selected_symbol'] = pick
#                 st.success(f"Selected symbol: {pick}")
#         else:
#             st.warning("No symbols passed the filters.")


api_key = st.secrets["alpaca"]["key"]
api_secret = st.secrets["alpaca"]["secret"]

@st.cache_data(show_spinner=False)
def get_all_alpaca_crypto_pairs(api_key, api_secret):
    client = TradingClient(api_key, api_secret, paper=True)
    all_assets = client.get_all_assets()
    pairs = [a.symbol for a in all_assets if getattr(a, "asset_class", None) == AssetClass.CRYPTO and a.status == "active"]
    # Optional: restrict to only /USD pairs:
    # pairs = [p for p in pairs if p.endswith("/USD")]
    return sorted(pairs)



PARAM_REGISTRY_FILE = "best_params.json"

def load_param_registry():
    if os.path.exists(PARAM_REGISTRY_FILE):
        with open(PARAM_REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_param_registry(registry):
    with open(PARAM_REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)

def get_saved_params(strategy, symbol):
    registry = load_param_registry()
    return registry.get(strategy, {}).get(symbol, None)

def register_params(strategy, symbol, params, results=None):
    registry = load_param_registry()
    if strategy not in registry:
        registry[strategy] = {}
    registry[strategy][symbol] = {"params": params, "results": results}
    save_param_registry(registry)

# /////////////////////////////Strategy registry and parameter definitions /////////////////////////////////////////////
STRATEGY_REGISTRY = {
    "RMA Strategy": {
        "function": rma_strategy,
        "params": [
            "rma_len", "barsforentry", "barsforexit", "atrlen",
            "normalizedupper", "normalizedlower", "ema_fast_len", "ema_slow_len", "trailpct"
        ],
        "universal": ["keeplime"]
    },
    "Crypto Intraday Multi-Signal": {
        "function": crypto_intraday_multi,
        "params": [
            "breakout_len", "momentum_len", "momentum_thresh", "trend_len",
            "atr_len", "min_atr", "trailing_stop_pct", "max_hold_bars"
        ],
        "universal": []
    },
    "SMA Cross": {
        "function": sma_cross_strategy,
        "params": [
            "fast_len", "slow_len"
        ],
        "universal": []
    },
    "Crypto Hybrid ATR-Mom Break": {
        "function": hybrid_atr_momentum_breakout,
        "params": [
            "breakout_len", "ema_len", "roc_thresh", "atr_len", "atr_mult"
        ],
        "universal": []
    }
}

# List of universal params to ignore during optimization for most crypto strategies
CRYPTO_DISABLE_IN_OPT = ["maxtradesperday", "session_start", "session_end", "initial_capital"]

# For convenience, you can add this to each crypto strategy registry entry if you want per-strategy overrides
CRYPTO_STRATEGY_KEYS = [
    "Crypto Intraday Multi-Signal",
    "Crypto Hybrid ATR-Mom Break"
]

# --- Per-strategy parameter search spaces for coordinate descent
CD_PARAM_SPACES = {
    "RMA Strategy": {
        "rma_len": (40, 100, 5),
        "barsforentry": (1, 10, 1),
        "barsforexit": (1, 10, 1),
        "atrlen": (2, 20, 1),
        "normalizedupper": (0, 2, 0.1),
        "normalizedlower": (-2, 0, 0.1),
        "ema_fast_len": (2, 30, 1),
        "ema_slow_len": (3, 100, 1),
        "trailpct": (40, 0, -0.5),
    },
    "Crypto Intraday Multi-Signal": {
        "breakout_len": (5, 51, 5),
        "momentum_len": (5, 51, 5),
        "momentum_thresh": (0.5, 3.0, 0.3),
        "trend_len": (10, 100, 1),
        "atr_len": (2, 20, 1),
        "min_atr": (0.1, 2.1, 0.2),
        "trailing_stop_pct": (0.5, 5.5, 0.5),
        "max_hold_bars": (5, 50, 1)
    },
    "SMA Cross": {
        "fast_len": (2, 16, 2),
        "slow_len": (5, 61, 5)
    },
    "Crypto Hybrid ATR-Mom Break": {
        "breakout_len": (5, 51, 5),
        "ema_len": (10, 100, 5),
        "roc_thresh": (0.5, 3.0, 0.1),
        "atr_len": (5, 30, 1),
        "atr_mult": (1.0, 3.0, 0.1)
    }
}

st.set_page_config(page_title="Universal  Backtester", layout="wide")
st.title("📊 Universal Trading Strategy Backtester")
progress_bar = st.empty()
progress_bar_widget = progress_bar.progress(0)
Coordinate_Descent_trials = 0

with st.sidebar:
    with st.expander("Data Feed", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload one or more CSVs (SYMBOL_TIMEFRAME.csv)",
            type=["csv"],
            accept_multiple_files=True
        )

        # FEED CHOICES
        feed_choices = ["alpaca stock", "alpaca crypto", "tvdatafeed", "ccxt", "csv", "yfinance", "tradingview_ta"]
        selected_feed = st.selectbox("Data Source", feed_choices)

        # PARSE CSVs
        csv_symbols = {}
        for file in uploaded_files or []:
            m = re.match(r"([A-Za-z0-9]+)_([0-9a-zA-Z]+)\.csv", file.name)
            if m:
                symbol, tf = m.group(1).upper(), m.group(2)
                key = f"{symbol}_{tf}"
                csv_symbols[key] = file

        # -- GET ALL ALPACA CRYPTO PAIRS FOR CRYPTO FEEDS --
        selected_pair = None
        if selected_feed in ["alpaca crypto", "ccxt"]:
            all_pairs = get_all_alpaca_crypto_pairs(api_key, api_secret)
            if all_pairs:
                default_idx = all_pairs.index("BTC/USD") if "BTC/USD" in all_pairs else 0
                selected_pair = st.selectbox("Crypto Pair", all_pairs, index=default_idx)
            else:
                selected_pair = st.text_input("Crypto Pair", value="BTC/USD")

        # SYMBOL/TIMEFRAME LOGIC FOR NON-CRYPTO FEEDS
        symbol_choices = sorted({key.split("_")[0] for key in csv_symbols.keys()})
        tf_choices = sorted({key.split("_")[1] for key in csv_symbols.keys()})

        if symbol_choices:
            symbol_query = st.text_input("Search symbol", "")
            filtered_symbols = [s for s in symbol_choices if symbol_query.upper() in s]
            selected_symbol = st.selectbox("Symbol", filtered_symbols)
            filtered_tfs = [key.split("_")[1] for key in csv_symbols if key.startswith(selected_symbol + "_")]
            selected_tf = st.selectbox("Timeframe", sorted(set(filtered_tfs)))
            csv_key = f"{selected_symbol}_{selected_tf}"
        else:
            selected_symbol = st.text_input("Symbol (for APIs, if no CSVs)", value="")
            selected_tf = st.selectbox("Timeframe", ["1m", "3m", "5m", "15m", "30m", "1h", "1d"], index=1)
            csv_key = f"{selected_symbol}_{selected_tf}"

        df = None

        # -- LOAD DATA BRANCHES --
        if selected_feed == "csv" and csv_key in csv_symbols:
            df = load_csv(csv_symbols[csv_key])
            if df is not None and not df.empty:
                st.session_state['df_loaded'] = df

        elif selected_feed == "ccxt":
            ccxt_exchange = st.selectbox(
                "Exchange", ["binanceus", "binance", "bybit", "okx", "kucoin", "coinbase", "kraken"], index=0
            )
            ccxt_tf = st.selectbox(
                "CCXT Interval", ["1m", "3m", "5m", "15m", "1h", "4h", "1d"], index=2
            )
            ccxt_limit = st.number_input("Bars", min_value=50, max_value=50000, value=5000)
            if st.button("Load CCXT Data"):
                df = load_ccxt(
                    symbol=selected_pair,
                    timeframe=ccxt_tf,
                    limit=int(ccxt_limit),
                    exchange=ccxt_exchange
                )
                st.success(f"Loaded CCXT {selected_pair} {ccxt_tf} from {ccxt_exchange}")
                if df is not None and not df.empty:
                    st.session_state['df_loaded'] = df

        elif selected_feed == "alpaca stock":
            start = st.date_input("Start Date", value=datetime(2025, 6, 1))
            end = st.date_input("End Date")
            key = st.secrets["alpaca"]["key"]
            secret = st.secrets["alpaca"]["secret"]
            url = st.secrets["alpaca"]["base_url"]
            if st.button("Load Alpaca Stock Data"):
                df = load_alpaca(selected_symbol, selected_tf, start, end, key, secret, url)
                if df is not None and not df.empty:
                    st.session_state['df_loaded'] = df

        elif selected_feed == "alpaca crypto":
            start = st.date_input("Start Date", value=datetime(2025, 6, 1))
            end = st.date_input("End Date")
            key = st.secrets["alpaca"]["key"]
            secret = st.secrets["alpaca"]["secret"]
            url = st.secrets["alpaca"]["base_url"]
            if st.button("Load Alpaca Crypto Data"):
                df = load_alpaca(selected_pair, selected_tf, start, end, key, secret, url)
                if df is not None and not df.empty:
                    st.session_state['df_loaded'] = df

        elif selected_feed == "yfinance":
            start = st.date_input("Start Date", value=datetime(2025, 6, 1))
            end = st.date_input("End Date")
            if st.button("Load Yahoo Data"):
                df = load_yfinance(selected_symbol, selected_tf, start, end)
                if df is not None and not df.empty:
                    st.session_state['df_loaded'] = df

        elif selected_feed == "tvdatafeed":
            exchange = st.text_input("Exchange", value="NASDAQ")
            n_bars = st.number_input("Bars", value=2600, min_value=10, max_value=5000)
            if st.button("Load TV Datafeed"):
                df = load_tvdatafeed(selected_symbol, exchange, selected_tf, n_bars)
                if df is not None and not df.empty:
                    st.session_state['df_loaded'] = df

        elif selected_feed == "tradingview_ta":
            exchange = st.text_input("Exchange", value="NASDAQ")
            if st.button("Get TV TA"):
                df = get_tv_ta(selected_symbol, exchange=exchange, interval=selected_tf)
                if df is not None and not df.empty:
                    st.session_state['df_loaded'] = df

        df = st.session_state.get('df_loaded', None)

        # --------------------- Date Filter UI ---------------------
        if df is not None and not df.empty:
            infodisplay = 0
            df.columns = [c.lower() for c in df.columns]
            if 'date' not in df.columns:
                st.error("Loaded data has no 'date' column after normalization.")
            else:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                data_start = df['date'].min().date()
                data_end = df['date'].max().date()
                test_range = st.sidebar.date_input("Select Backtest Period", value=(data_start, data_end),
                                                   min_value=data_start, max_value=data_end)
                mask = None
                try:
                    start_date, end_date = test_range
                    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
                except Exception:
                    # In case test_range is a single date, skip filtering
                    pass
                if mask is not None:
                    df = df.loc[mask]
        else:
            infodisplay = 1

    with st.expander("Session", expanded=False):
        session_start_h = st.number_input("Session Start Hour", min_value=0, max_value=23, value=13)
        session_start_m = st.number_input("Session Start Minute", min_value=0, max_value=59, value=30)
        session_end_h = st.number_input("Session End Hour", min_value=0, max_value=23, value=19)
        session_end_m = st.number_input("Session End Minute", min_value=0, max_value=59, value=52)
    with st.expander("Strategy Settings", expanded=False):
        maxtradesperday = st.number_input("Max Trades Per Day", min_value=1, max_value=100, value=1, step=1)
        initial_capital=st.number_input("Initial Capital", value=10000, min_value=100)
        qty=st.number_input("Order Size (qty)", value=1, min_value=1)
        session_rule = st.checkbox("Session Rule", value=True)

        strategy_choice = st.selectbox("Select Strategy", list(STRATEGY_REGISTRY.keys()))
        run_saved_params = st.button("▶️ Run With Saved Parameters")
    with st.expander("Optimizer Settings", expanded=False):
        optimize_for = st.selectbox("Optimize For", ["return", "win", "return with win% tie-breaker"])
        optimizer_choice = st.selectbox("Select Optimizer", ["Optuna (Bayesian)", "Coordinate Descent"])
        if optimizer_choice == "Optuna (Bayesian)":
            optuna_trials = st.number_input("Optuna Trials", min_value=10, max_value=1000, value=300, step=10)
            trialscount: int = optuna_trials
        else:
            Coordinate_Descent_trials = st.number_input("Coordinate Descent Trials", min_value=1, max_value=50, value=4,
                                                        step=1)
            trialscount: int =Coordinate_Descent_trials
        run_optuna = st.sidebar.button("Run Optimization")

# ///////////////////////////////////////////////Stock OR Crypto?/////////////////////////////////////////////

if selected_feed in ["alpaca crypto", "ccxt"]:
    symbol_for_data = selected_pair
else:
    symbol_for_data = selected_symbol

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////


def tf_to_minutes(tf):
    if 'm' in tf:
        return int(tf.replace('m', ''))
    if 'h' in tf:
        return int(tf.replace('h', '')) * 60
    if 'd' in tf:
        return int(tf.replace('d', '')) * 60 * 24
    return 1

# --- AUTO-LOAD saved best params for the selected strategy/symbol, if exists ---
saved_params_record = get_saved_params(strategy_choice, symbol_for_data)
if saved_params_record:
    best_params = saved_params_record.get('params', {})
    best_results = saved_params_record.get('results', {})
else:
    best_params = {}
    best_results = {}





# ///////////////////////////////////////////////display organizers/////////////////////////////////////////////
data_col, saved_col = st.columns([1, 3])

main_col, current_col= st.columns([1, 3])

table_col, comment_col = st.columns([10, 1])
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
with data_col: # /////////////////Clomn of Stored data
    with st.container(border=1, height=332):
        if infodisplay == 1:
            st.info("No data loaded yet or data has no 'date' column.")
        else:
            st.write("#### Selected Symbol: ( ", symbol_for_data, " )")
            st.write("##### TimeFrame ( ", selected_tf, " )")
            st.write("###### Rows after slicing:", len(df))
            st.write("###### Date min/max after slicing:")
            st.write(df['date'].min())
            st.write(df['date'].max())

                # --- Plot loaded data for visual inspection ---
            if df is not None and not df.empty:
                # --- Download loaded data ---
                st.download_button(
                    label="⬇️ Download Loaded Data (CSV)",
                    data=df.to_csv(index=False),
                    file_name=f"{symbol_for_data}_{selected_tf}_loaded_data.csv",
                    mime="text/csv"
                ) # /
with saved_col:
    with st.container(border=1, height=332):
        saved_results_col, saved_param_col = st.columns([1, 1])
        if saved_params_record:
            with saved_param_col:
                if saved_params_record and "params" in saved_params_record:
                    st.markdown(f"###### Saved Best Parameters for {strategy_choice}")
                    st.dataframe(pd.DataFrame(list(saved_params_record["params"].items()), columns=["Parameter", "Value"]),
                                 height=248)
                    def json_to_tv_preset(json_path):
                        registry = load_param_registry()
                        out_lines = []
                        param_map = {
                            "maxtradesperday": "_maxOrders",
                            "rma_len": "_len",
                            "barsforentry": "_barsForEntry",
                            "barsforexit": "_barsForExit",
                            "atrlen": "_ATRLen",
                            "normalizedupper": "_normalizedUpper",
                            "normalizedlower": "_normalizedLower",
                            "ema_fast_len": "_fastlen",
                            "ema_slow_len": "_slowlen",
                            "trailpct": "_TrailPct"
                        }
                        # Helper to handle "09:31" -> 9, 31
                        def time_to_hm(timestr):
                            try:
                                hour, minute = map(int, str(timestr).split(":"))
                                return hour, minute
                            except:
                                return 9, 30

                        for strategy, ticker_dict in registry.items():
                            for symbol, d in ticker_dict.items():
                                params = d.get("params", {})
                                out_lines.append(f'else if symbol == "{symbol}"')
                                for k, v in params.items():
                                    # Skip these as they're statically set above
                                    if k in ("maxtradesperday", "session_start", "session_end"):
                                        continue
                                    if k in param_map and isinstance(param_map[k], str):
                                        if k == "emasrc":
                                            v = f'"{v}"'
                                        out_lines.append(f"    {param_map[k]}       := {v}")
                                    elif k == "session_start" or k == "session_end":
                                        # handled above
                                        continue
                        return "\n".join(out_lines)

            with saved_results_col:
                if saved_params_record and "results" in saved_params_record:
                    st.markdown("###### Saved Best results")
                    st.dataframe(pd.DataFrame(list(saved_params_record["results"].items()),
                                              columns=["results", "Value"]), height=212)
                    tv_preset_str = json_to_tv_preset(PARAM_REGISTRY_FILE)
                    st.download_button(
                        label="⬇️ Export All TV Presets",
                        data=tv_preset_str,
                        file_name="tv_presets.txt",
                        mime="text/plain"
                    )
        else:
            st.info("No saved parameter set found for this strategy and symbol yet.")

# ---- Strategy Selector ----
strategy_entry = STRATEGY_REGISTRY[strategy_choice]
strategy_func = strategy_entry["function"]
strategy_params = strategy_entry["params"]
universal_keys = strategy_entry.get("universal", [])


# Build session start/end as "HH:MM" strings for all downstream logic
session_start = f"{int(session_start_h):02d}:{int(session_start_m):02d}"
session_end = f"{int(session_end_h):02d}:{int(session_end_m):02d}"

# ---- Always build sidebar_values after UI controls ----
sidebar_values = {
    "keeplime": True,
    "qty": qty,
    "session_rule": session_rule,
    "session_start": session_start,
    "session_end": session_end,
    "initial_capital": initial_capital,
    "maxtradesperday": maxtradesperday
}
# if strategy_choice not in CRYPTO_STRATEGY_KEYS:
#     session_start = f"{int(session_start_h):02d}:{int(session_start_m):02d}"
#     session_end = f"{int(session_end_h):02d}:{int(session_end_m):02d}"
#     sidebar_values["session_start"] = session_start
#     sidebar_values["session_end"] = session_end
#     sidebar_values["initial_capital"] = 200000
#     sidebar_values["maxtradesperday"] = maxtradesperday

#////////////////Implement/Update a universal simulator function////////////////////////////////////////////////////

def simulate_trades(
    trades,
    initial_capital,
    qty,
    maxtradesperday=None,
    session_rule=False,
    session_start=None,
    session_end=None
):
    if trades is None or trades.empty:
        trades = trades.copy() if trades is not None else pd.DataFrame()
        trades['filled_qty'] = []
        trades['cum_pnl'] = []
        trades['account_balance'] = []
        return trades
    trades = trades.copy()
    # --- Session rule filtering ---
    trades['EntryTime'] = pd.to_datetime(trades['EntryTime'])
    if session_rule and session_start and session_end:
        session_start_dt = datetime.strptime(session_start, "%H:%M").time()
        session_end_dt = datetime.strptime(session_end, "%H:%M").time()
        trades = trades[
            (trades['EntryTime'].dt.time >= session_start_dt) &
            (trades['EntryTime'].dt.time <= session_end_dt)
            ]
    # --- Trade count per day ---
    if maxtradesperday:
        if 'EntryTime' in trades:
            trades['date'] = trades['EntryTime'].dt.date
            trades = trades.groupby('date').head(maxtradesperday).reset_index(drop=True)
            trades.drop('date', axis=1, inplace=True)
    # --- Position sizing and account tracking ---
    trades['filled_qty'] = qty
    trades['trade_pnl'] = trades['pnl'] * qty
    trades['cum_pnl'] = trades['trade_pnl'].cumsum()
    trades['account_balance'] = initial_capital + trades['cum_pnl']
    return trades

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
call_kwargs = best_params.copy()  # Use only saved/best params
call_kwargs['bar_minutes'] = tf_to_minutes(selected_tf)
# If your strategy expects any always-on universal keys (like "keeplime"), add them here
# call_kwargs["keeplime"] = True

if run_saved_params:
    st.info("Running strategy with last saved parameters...")
    trades, df_all = strategy_func(df, **call_kwargs)
    trades = simulate_trades(
        trades,
        initial_capital=sidebar_values["initial_capital"],
        qty=sidebar_values["qty"],
        maxtradesperday=sidebar_values["maxtradesperday"],
        session_rule=sidebar_values.get("session_rule", False),
        session_start=sidebar_values.get("session_start", None),
        session_end=sidebar_values.get("session_end", None),
    )

    st.write("## Results with saved parameters")
    st.dataframe(trades)
    # Optional: show stats, plots, etc.

    main_results = {
        "Total Return": float(trades['return'].sum()) if not trades.empty and 'return' in trades else 0,
        "Win Rate": float((trades['pnl'] > 0).mean() *100) if not trades.empty and 'pnl' in trades else 0,
        "Total Trades": int(len(trades)),
        "Avg Bars In Trade": float(trades['BarsInTrade'].mean()),
        "Max Drawdown": float(trades['pnl'].min())
    }


    st.session_state['last_best_params'] = best_params
    st.session_state['last_main_results'] = main_results
    st.session_state['last_trades'] = trades
    st.session_state['last_df_all'] = df_all
    st.session_state['Symbol'] = symbol_for_data

# //////////////////////////////////////--- Coordinate Descent Optimizer --/////////////////////////////////////////////

def cd_score_fn(trades):
    if trades is None or trades.empty:
        return -99999
    if optimize_for.lower() == "return":
        return trades["return"].sum()
    elif optimize_for.lower() == "win":
        return (trades['pnl'] > 0).mean()
    elif optimize_for.lower() == "return with win% tie-breaker":
        win_pct = (trades['pnl'] > 0).mean()
        total_return = trades['return'].sum()
        return total_return + win_pct / 100
    else:
        return trades["return"].sum()

def coordinate_descent_optimizer(
    strategy_func,
    df_strategy,
    param_space,
    universal_kwargs,
    n_rounds=4,
    initial_params=None,
    show_progress=True,
    score_fn=None
):
    current_params = initial_params.copy() if initial_params else {}
    for p, v in param_space.items():
        if p not in current_params:
            if isinstance(v, list):
                current_params[p] = v[len(v) // 2]
            else:
                lo, hi, step_size = v
                current_params[p] = lo + ((hi - lo) // 2) if isinstance(step_size, int) else lo + ((hi - lo) / 2)

    # Compute total steps for progress bar
    total_steps = n_rounds * sum(
        len(v) if isinstance(v, list) else (int(round((v[1] - v[0]) / v[2])) + 1)
        for v in param_space.values()
    )
    step = 0

    for _ in range(n_rounds):
        for p in param_space:
            best_p = current_params[p]
            best_p_score = None
            if isinstance(param_space[p], list):
                vals = param_space[p]
            else:
                lo, hi, step_size = param_space[p]
                if isinstance(step_size, float):
                    vals = list(np.arange(lo, hi + step_size, step_size))
                else:
                    vals = list(range(lo, hi + 1, step_size))
            for val in vals:
                trial_params = copy.deepcopy(current_params)
                trial_params[p] = val
                # Enforce EMA constraint
                if "ema_fast_len" in trial_params and "ema_slow_len" in trial_params:
                    if trial_params["ema_slow_len"] <= trial_params["ema_fast_len"]:
                        continue
                call_kwargs = {**universal_kwargs, **trial_params}
                trades, _ = strategy_func(df_strategy, **call_kwargs)
                score = score_fn(trades) if score_fn is not None else (
                    trades["return"].sum() if not trades.empty and "return" in trades else -99999)
                if best_p_score is None or score > best_p_score:
                    best_p = val
                    best_p_score = score
                # ------ Update progress ------
                step += 1
                if progress_bar:
                    progress_bar_widget.progress(min(1.0, step / total_steps))
            current_params[p] = best_p
            best_score = best_p_score
    if progress_bar:
        progress_bar.empty()
    return current_params, best_score
optimized_indecator=0

# ////////////////////////////////////////////----------- Run Optimization -----------//////////////////////////////////
if run_optuna and df is not None and not df.empty:
    optimized_indecator = 1
    df_strategy = df.copy()
    df_strategy.columns = [c.lower() for c in df_strategy.columns]
    session_start = f"{int(session_start_h):02d}:{int(session_start_m):02d}"
    session_end = f"{int(session_end_h):02d}:{int(session_end_m):02d}"

    # Only use relevant params for the current strategy
    param_grid = CD_PARAM_SPACES[strategy_choice]
    strategy_param_list = strategy_entry["params"]
    cd_space = {k: param_grid[k] for k in strategy_param_list if k in param_grid}

    if optimizer_choice == "Optuna (Bayesian)":
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
                #    strategy_kwargs[param] = trial.suggest_categorical(param, [15, 30, 40, 45, 60, 120, 240])
                elif param == 'ema_fast_len':
                    strategy_kwargs[param] = trial.suggest_int(param, 2, 30)
                elif param == 'ema_slow_len':
                    fast_val = strategy_kwargs.get('ema_fast_len', 2)
                    strategy_kwargs[param] = trial.suggest_int(param, fast_val + 1, 100)
                elif param in ['normalizedupper']:
                    strategy_kwargs[param] = trial.suggest_float(param, 0, 2, step=0.1)
                elif param in ['normalizedlower']:
                    strategy_kwargs[param] = trial.suggest_float(param, -2, 0, step=0.1)
                elif param == 'fast_len':
                    strategy_kwargs[param] = trial.suggest_int(param, 2, 25)
                elif param == 'slow_len':
                    fast_val = strategy_kwargs.get('fast_len', 2)
                    strategy_kwargs[param] = trial.suggest_int(param, fast_val + 1, 60)
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
                elif param == 'barsforentry':
                    strategy_kwargs[param] = trial.suggest_int(param, 1, 10)
                elif param == 'barsforexit':
                    strategy_kwargs[param] = trial.suggest_int(param, 1, 10)
                elif param == 'atrlen':
                    strategy_kwargs[param] = trial.suggest_int(param, 2, 20)
                elif param == 'normalizedupper':
                    strategy_kwargs[param] = trial.suggest_float(param, 0, 2, step=0.1)
                elif param == 'normalizedlower':
                    strategy_kwargs[param] = trial.suggest_float(param, -2, 0, step=0.1)
                elif param == 'ema_fast_len':
                    strategy_kwargs[param] = trial.suggest_int(param, 2, 30)
                elif param == 'ema_slow_len':
                    strategy_kwargs[param] = trial.suggest_int(param, 3, 100)
                elif param == 'trailpct':
                    strategy_kwargs[param] = trial.suggest_float(param, 0, 40, step=0.5)
                elif param == 'lookback':
                    strategy_kwargs[param] = trial.suggest_int(param, 10, 50)
                elif param == 'ma_len':
                    strategy_kwargs[param] = trial.suggest_int(param, 10, 60)
                elif param == 'threshold':
                    strategy_kwargs[param] = trial.suggest_float(param, 1, 4, step=0.2)
                elif param == 'fast_len':
                    strategy_kwargs[param] = trial.suggest_int(param, 2, 20)
                elif param == 'slow_len':
                    strategy_kwargs[param] = trial.suggest_int(param, 5, 60)
                elif param == 'atr_len':
                    strategy_kwargs[param] = trial.suggest_int(param, 5, 30)
                elif param == 'mult':
                    strategy_kwargs[param] = trial.suggest_float(param, 1.0, 3.0, step=0.2)
                elif param == 'spread_lookback':
                    strategy_kwargs[param] = trial.suggest_int(param, 10, 60)
                elif param == 'vol_mult':
                    strategy_kwargs[param] = trial.suggest_float(param, 1.2, 4.0, step=0.2)
                elif param == 'breakout_len':
                    strategy_kwargs[param] = trial.suggest_int(param, 5, 51, step=5)
                elif param == 'momentum_len':
                    strategy_kwargs[param] = trial.suggest_int(param, 5, 51, step=5)
                elif param == 'momentum_thresh':
                    strategy_kwargs[param] = trial.suggest_float(param, 0.5, 3.0, step=0.3)
                elif param == 'trend_len':
                    strategy_kwargs[param] = trial.suggest_int(param, 10, 100)
                elif param == 'min_atr':
                    strategy_kwargs[param] = trial.suggest_float(param, 0.1, 2.1, step=0.2)
                elif param == 'trailing_stop_pct':
                    strategy_kwargs[param] = trial.suggest_float(param, 0.5, 5.5, step=0.5)
                elif param == 'max_hold_bars':
                    strategy_kwargs[param] = trial.suggest_int(param, 5, 50)
                elif param == 'maxtradesperday':
                    strategy_kwargs[param] = trial.suggest_int(param, 1, 20)
                elif param == 'threshold':  # for pairs trading
                    strategy_kwargs[param] = trial.suggest_float(param, 1.0, 3.0, step=0.2)
                else:
                    continue

            disable_in_optimizer = CRYPTO_DISABLE_IN_OPT if strategy_choice in CRYPTO_STRATEGY_KEYS else []
            enabled_universal_keys = [k for k in universal_keys if k not in disable_in_optimizer]
            call_kwargs = {k: sidebar_values[k] for k in enabled_universal_keys if k in sidebar_values}
            call_kwargs['bar_minutes'] = tf_to_minutes(selected_tf)

            for k in strategy_kwargs:
                call_kwargs[k] = strategy_kwargs[k]

            trades, _ = strategy_func(df_strategy, **call_kwargs)

            total_return = trades['return'].sum() if not trades.empty and 'return' in trades else -999
            win_pct = (trades['pnl'] > 0).mean() if not trades.empty and 'pnl' in trades else 0
            if optimize_for.lower() == 'return':
                return total_return
            elif optimize_for.lower() == 'win':
                return win_pct
            elif optimize_for.lower() == 'return with win% tie-breaker':
                return total_return + win_pct / 100
            else:
                return total_return
        study = optuna.create_study(direction="maximize")


        #        progress_bar = st.progress(0)
        def optuna_callback(study, trial):
            progress_bar_widget.progress(min(1.0, (trial.number + 1) / optuna_trials))


        study.optimize(objective, n_trials=optuna_trials, callbacks=[optuna_callback])
        progress_bar.empty()
        st.balloons()
        best_params = study.best_trial.params
    else:
        disable_in_optimizer = CRYPTO_DISABLE_IN_OPT if strategy_choice in CRYPTO_STRATEGY_KEYS else []
        enabled_universal_keys = [k for k in universal_keys if k not in disable_in_optimizer]
        call_kwargs = {k: sidebar_values[k] for k in enabled_universal_keys if k in sidebar_values}
        call_kwargs['bar_minutes'] = tf_to_minutes(selected_tf)


        # --- Inject initial parameters from JSON if available, with type-safe sanitization ---
        def sanitize_initial_params(param_space, initial_params):
            """Match initial values to their expected type for coordinate descent."""
            if not initial_params:
                return None
            cleaned = {}
            for k, v in initial_params.items():
                if k not in param_space:
                    continue
                space = param_space[k]
                if isinstance(space, list):
                    cleaned[k] = v if v in space else space[0]
                else:
                    lo, hi, step = space
                    if isinstance(step, int):
                        try:
                            v_int = int(round(v))
                        except Exception:
                            v_int = lo
                        cleaned[k] = max(lo, min(hi, v_int))
                    else:
                        try:
                            v_float = float(v)
                        except Exception:
                            v_float = lo
                        cleaned[k] = max(lo, min(hi, v_float))
            return cleaned if cleaned else None

        saved = get_saved_params(strategy_choice, symbol_for_data)
        initial_params = saved['params'] if saved and 'params' in saved else None
        initial_params = sanitize_initial_params(cd_space, initial_params)
#        st.write("Initial params used for coordinate descent:", initial_params)  # (Optional: for debug)
        best_params, best_score = coordinate_descent_optimizer(
            strategy_func, df_strategy, cd_space, call_kwargs,
            n_rounds=Coordinate_Descent_trials,
            initial_params=initial_params,
            score_fn=cd_score_fn
        )

# ////////////////////////////////FINAL run with Best Parameters Result Calculations////////////////////////////////////

    call_kwargs = {k: sidebar_values[k] for k in universal_keys if k in sidebar_values}
    call_kwargs['bar_minutes'] = tf_to_minutes(selected_tf)
    call_kwargs.update(best_params)
    trades, df_all = strategy_func(df_strategy, **call_kwargs)

    trades = simulate_trades(
        trades,
        initial_capital=sidebar_values["initial_capital"],
        qty=sidebar_values["qty"],
        maxtradesperday=sidebar_values["maxtradesperday"],
        session_rule=sidebar_values.get("session_rule", False),
        session_start=sidebar_values.get("session_start", None),
        session_end=sidebar_values.get("session_end", None),
    )

    # ---- Add this: Universal Simulation ----
    trades = simulate_trades(
        trades,
        initial_capital=sidebar_values["initial_capital"],
        qty=sidebar_values["qty"],
        maxtradesperday=sidebar_values["maxtradesperday"],
        session_rule=sidebar_values.get("session_rule", False),
        session_start=sidebar_values.get("session_start", None),
        session_end=sidebar_values.get("session_end", None),
    )

#/////////////////////////////////////////// --- Display results ---////////////////////////////////////////////////////
    main_results = {
        "Total Return": float(trades['return'].sum()) if not trades.empty and 'return' in trades else 0,
        "Win Rate": float((trades['pnl'] > 0).mean() *100) if not trades.empty and 'pnl' in trades else 0,
        "Total Trades": int(len(trades)),
        "Avg Bars In Trade": float(trades['BarsInTrade'].mean()),
        "Max Drawdown": float(trades['pnl'].min())
    }
    st.session_state['last_best_params'] = best_params
    st.session_state['last_main_results'] = main_results
    st.session_state['last_trades'] = trades
    st.session_state['last_df_all'] = df_all
    st.session_state['Symbol'] = symbol_for_data


Optimized_Symbol=st.session_state.get('Symbol')
trades = st.session_state.get('last_trades', pd.DataFrame())
df_all = st.session_state.get('last_df_all', pd.DataFrame())

if Optimized_Symbol==symbol_for_data:
    with main_col:
        with st.container(border=1, height=332):
            if 'df_all' in locals() and df_all is not None and not df_all.empty and optimized_indecator == 1:
                st.markdown("#### Optimization Results...")
                st.markdown(f"Optimizer: {optimizer_choice}")
                st.markdown(f"Trials: {trialscount}")
                st.markdown(f"Strategy: {strategy_choice}")
                if not df_all.empty:
                    test_start = df_all["date"].min()
                    test_end = df_all["date"].max()
                    st.markdown("Test Period:")
                    st.markdown(
                        f" {test_start.strftime('%Y-%m-%d %H:%M')} &mdash; {test_end.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.info("Optimization Results Not Available, Please run the optimizer first...")

    with current_col:
        with st.container(border=1, height=332):
            cur_result_col, curr_param_col = st.columns([1, 1])
            bp = st.session_state.get('last_best_params', None)
            mr = st.session_state.get('last_main_results', None)
            with curr_param_col:
                # Show last optimizer results (from session_state, if present)
                if bp:
                    st.markdown(f"###### Current Parameters for {strategy_choice}")
                    df_params = pd.DataFrame(list(bp.items()), columns=["Parameter", "Value"])
                    st.dataframe(df_params, height=248)
                else:
                    st.info("No optimizer run in this session yet.")
            with cur_result_col:
                if mr:
                    st.markdown("Current Best results")
                    mr_results = pd.DataFrame(list(mr.items()), columns=["Result", "Value"])
                    st.dataframe(mr_results, height=212)
                # Save button (always available)
                if  st.button("💾 Save Best Parameters", key="save_best_params"):
                    register_params(strategy_choice, symbol_for_data, bp, mr)
                    st.success("Parameters saved for this strategy/symbol (replaced old entry)!")
                # else:
                #     st.error("No optimizer results found in session. Please run the optimizer first.")

    with table_col:
        with st.container(border=1):
            st.markdown("### Trade Log")
            st.dataframe(trades)
            with st.expander("📈 RMA Strategy Indicators Plot", expanded=True):
                if 'df_all' in locals() and df_all is not None and not df_all.empty:
                    indicator_cols = []
                    # Always plot close/ema_fast/ema_slow if available
                    for col in ['close', 'ema_fast', 'ema_slow']:
                        if col in df_all.columns:
                            indicator_cols.append(col)
                    st.line_chart(df_all.set_index('date')[indicator_cols])

                    # Optional: export for TV/Python numeric comparison
                    st.download_button(
                        label="⬇️ Download Indicators CSV",
                        data=df_all[['date'] + indicator_cols].to_csv(index=False),
                        file_name="rma_indicators.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Run a strategy and optimizer to see indicator plots.")
else:
    st.info("Run a strategy and optimizer ...")

#///////////////////////////////////Show Trade Table and chart//////////////////////////////////////////////////////
