import re
import json
import optuna
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from streamlit import empty
from helper_visuals import render_visuals
# ---- data loading (unchanged) ----
from datafeed import (
    load_csv, load_yfinance, load_alpaca, load_tvdatafeed, get_tv_ta, load_ccxt)
from registry import CRYPTO_STRATEGY_KEYS
import os
from batch.batch_optimize import run_batch_optimization, PENDING_REGISTRY_FILE, STOP_FILE
from batch.merge_pending import merge_pending
# ---- new imports from helper modules ----
from helpers import (
    load_param_registry,
    save_param_registry,
    tf_to_minutes,
    sanitize_initial_params,
    json_to_tv_preset,
    get_all_alpaca_crypto_pairs,
    ensure_date_column,
    total_return_pct,
    score_from_trades,
    get_last_ohlcv,
    equity_curve_from_trades,
    max_drawdown,
    total_return_usd,
)
from param_registry import (
    load_best_params,          # returns a params dict (best by metric)
    load_all_best_params,      # returns history list with params+metrics
    register_best_params,      # saves (params, metrics) for strategy|symbol|tf
)
from optimization import coordinate_descent_optimizer, cd_score_fn
from registry import (
    STRATEGY_REGISTRY,
    CRYPTO_DISABLE_IN_OPT,
    CRYPTO_STRATEGY_KEYS,
    CD_PARAM_SPACES,
)

# =====================================================================================
# App config
# =====================================================================================
st.set_page_config(page_title="Universal  Backtester", layout="wide")
st.title("📊 Universal Trading Strategy Backtester")
progress_bar = st.empty()
progress_bar_widget = progress_bar.progress(0)


# Cached wrappers (keep Streamlit out of helpers)
# =====================================================================================
api_key = st.secrets["alpaca"]["key"]
api_secret = st.secrets["alpaca"]["secret"]

@st.cache_data(show_spinner=False)
def _cached_pairs(key, secret):
    return get_all_alpaca_crypto_pairs(key, secret)

# =====================================================================================
# Sidebar: Data feed and global controls
# =====================================================================================
Coordinate_Descent_trials = 0
with st.sidebar:
    if st.button("🚪 Exit App"):
        st.warning("Shutting down Streamlit app...")
        os._exit(0)
    with st.expander("Data Feed", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload one or more CSVs (SYMBOL_TIMEFRAME.csv)",
            type=["csv"],
            accept_multiple_files=True,
        )

        feed_choices = [
            "alpaca stock",
            "alpaca crypto",
            "tvdatafeed",
            "ccxt",
            "csv",
            "yfinance",
            "tradingview_ta",
        ]
        selected_feed = st.selectbox("Data Source", feed_choices)

        # Parse CSVs
        csv_symbols = {}
        for file in uploaded_files or []:
            m = re.match(r"([A-Za-z0-9]+)_([0-9a-zA-Z]+)\.csv", file.name)
            if m:
                symbol, tf = m.group(1).upper(), m.group(2)
                key = f"{symbol}_{tf}"
                csv_symbols[key] = file

        # Crypto pairs for crypto feeds
        selected_pair = None
        if selected_feed in ["alpaca crypto", "ccxt"]:
            all_pairs = _cached_pairs(api_key, api_secret)
            if all_pairs:
                default_idx = all_pairs.index("BTC/USD") if "BTC/USD" in all_pairs else 0
                selected_pair = st.selectbox("Crypto Pair", all_pairs, index=default_idx)
            else:
                selected_pair = st.text_input("Crypto Pair", value="BTC/USD")

        # Symbol / timeframe inputs for non-crypto feeds
        symbol_choices = sorted({key.split("_")[0] for key in csv_symbols.keys()})
        if symbol_choices:
            symbol_query = st.text_input("Search symbol", "")
            filtered_symbols = [s for s in symbol_choices if symbol_query.upper() in s]
            selected_symbol = st.selectbox("Symbol", filtered_symbols)
            filtered_tfs = [key.split("_")[1] for key in csv_symbols if key.startswith(selected_symbol + "_")]
            selected_tf = st.selectbox("Timeframe", sorted(set(filtered_tfs)))
            csv_key = f"{selected_symbol}_{selected_tf}"
        else:
            selected_symbol = st.text_input("Symbol (for APIs, if no CSVs)", value="RGTI")
            selected_tf = st.selectbox("Timeframe", ["1m", "3m", "5m", "15m", "30m", "1h", "1d"], index=1)
            csv_key = f"{selected_symbol}_{selected_tf}"

        df = None

        # Load branches
        if selected_feed == "csv" and csv_key in csv_symbols:
            df = load_csv(csv_symbols[csv_key])
            if df is not None and not df.empty:
                st.session_state["df_loaded"] = df

        elif selected_feed == "ccxt":
            ccxt_exchange = st.selectbox(
                "Exchange", ["binanceus", "binance", "bybit", "okx", "kucoin", "coinbase", "kraken"], index=0
            )
            ccxt_tf = st.selectbox("CCXT Interval", ["1m", "3m", "5m", "15m", "1h", "4h", "1d"], index=2)
            ccxt_limit = st.number_input("Bars", min_value=50, max_value=50000, value=5000)
            if st.button("Load CCXT Data"):
                df = load_ccxt(symbol=selected_pair, timeframe=ccxt_tf, limit=int(ccxt_limit), exchange=ccxt_exchange)
                st.success(f"Loaded CCXT {selected_pair} {ccxt_tf} from {ccxt_exchange}")
                if df is not None and not df.empty:
                    st.session_state["df_loaded"] = df

        elif selected_feed == "alpaca stock":
            start = st.date_input("Start Date", value=datetime(2025, 6, 1))
            end = st.date_input("End Date")
            key = st.secrets["alpaca"]["key"]
            secret = st.secrets["alpaca"]["secret"]
            url = st.secrets["alpaca"]["base_url"]
            if st.button("Load Alpaca Stock Data"):
                df = load_alpaca(selected_symbol, selected_tf, start, end, key, secret, url)
                if df is not None and not df.empty:
                    st.session_state["df_loaded"] = df

        elif selected_feed == "alpaca crypto":
            start = st.date_input("Start Date", value=datetime(2025, 6, 1))
            end = st.date_input("End Date")
            key = st.secrets["alpaca"]["key"]
            secret = st.secrets["alpaca"]["secret"]
            url = st.secrets["alpaca"]["base_url"]
            if st.button("Load Alpaca Crypto Data"):
                df = load_alpaca(selected_pair, selected_tf, start, end, key, secret, url)
                if df is not None and not df.empty:
                    st.session_state["df_loaded"] = df

        elif selected_feed == "yfinance":
            start = st.date_input("Start Date", value=datetime(2025, 6, 1))
            end = st.date_input("End Date")
            if st.button("Load Yahoo Data"):
                df = load_yfinance(selected_symbol, selected_tf, start, end)
                if df is not None and not df.empty:
                    st.session_state["df_loaded"] = df

        elif selected_feed == "tvdatafeed":
            exchange = st.text_input("Exchange", value="NASDAQ")
            n_bars = st.number_input("Bars", value=2600, min_value=10, max_value=5000)
            if st.button("Load TV Datafeed"):
                df = load_tvdatafeed(selected_symbol, exchange, selected_tf, n_bars)
                if df is not None and not df.empty:
                    st.session_state["df_loaded"] = df

        elif selected_feed == "tradingview_ta":
            exchange = st.text_input("Exchange", value="NASDAQ")
            if st.button("Get TV TA"):
                df = get_tv_ta(selected_symbol, exchange=exchange, interval=selected_tf)
                if df is not None and not df.empty:
                    st.session_state["df_loaded"] = df

        df = st.session_state.get("df_loaded", None)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]  # ensure lowercase
            last_values = get_last_ohlcv(df)
#        df2 = st.session_state.get("df2_loaded", None)

        # Date filter UI
        if df is not None and not df.empty:
            infodisplay = 0
            df.columns = [c.lower() for c in df.columns]
            if "date" not in df.columns:
                st.error("Loaded data has no 'date' column after normalization.")
            else:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                data_start = df["date"].min().date()
                data_end = df["date"].max().date()
                test_range = st.date_input(
                    "Select Backtest Period", value=(data_start, data_end), min_value=data_start, max_value=data_end
                )
                mask = None
                try:
                    start_date, end_date = test_range
                    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
                except Exception:
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
        use_session_end_rule = st.checkbox("Session End Exit Rule", value=True)
        initial_capital = st.number_input("Initial Capital", min_value=1, max_value=1_000_000, value=200_000, step=1)  # sidebar-only by design
        strategy_choice = st.selectbox("Select Strategy", list(STRATEGY_REGISTRY.keys()))
        if strategy_choice in CRYPTO_STRATEGY_KEYS:
            sizing_mode = st.radio("Position sizing (crypto)", ["Cash", "Quantity"], index=0, horizontal=True)
            if sizing_mode == "Cash":
                cash = st.number_input("Cash per trade ($)", min_value=1, max_value=1_000_000, value=1000,step=100)
                qty = st.number_input("Fallback Quantity (optional)", min_value=0.0, max_value=5_000.0, value=0.0,
                                           step=0.1,help = "Used only if entry price is missing or cash=0.")
            else:
                qty = st.number_input("Order Quantity", min_value=0.0, max_value=5_000.0, value=1.0, step=0.1)
                cash = st.number_input("Fallback Cash ($, optional)", min_value=0.0, max_value=1_000_000.0, value=0.0,
                                            step=100.0,help = "Ignored unless sizing mode flips to cash.")
        else:
            qty = st.number_input("Order Quantity", min_value=1, max_value=5_000, value=1, step=1)
            sizing_mode = "Quantity"
            cash = 0.0
        run_saved_params = st.button("▶️ Run With Saved Parameters")

    with st.expander("Optimizer Settings", expanded=False):
        optimize_for = st.selectbox("Optimize For", ["Penalized: Return - a*DD + b*Win%", "return", "win",
                                                     "return with win% tie-breaker", "Sharpe", "Sortino",
                                                     "MaxDD (minimize)"])
        optimizer_choice = st.selectbox("Select Optimizer", ["Optuna (Bayesian)", "Coordinate Descent"])
        if optimize_for == "Penalized: Return - a*DD + b*Win%":
            pen_alpha = st.number_input("α (penalty on MaxDD)", min_value=0.0, max_value=10.0, value=0.10, step=0.01)
            pen_beta = st.number_input("β (bonus per 1% Win Rate)", min_value=0.0, max_value=2.0, value=0.10, step=0.01)
        if optimizer_choice == "Optuna (Bayesian)":
            optuna_trials = st.number_input("Optuna Trials", min_value=1, max_value=1000, value=300, step=10)
            trialscount: int = optuna_trials
        else:
            Coordinate_Descent_trials = st.number_input(
                "Coordinate Descent Trials", min_value=1, max_value=50, value=4, step=1
            )
            trialscount: int = Coordinate_Descent_trials
        run_optuna = st.sidebar.button("Run Optimization")

# Determine symbol-for-data based on feed
symbol_for_data = selected_pair if selected_feed in ["alpaca crypto", "ccxt"] else selected_symbol

# Autoload saved params/metrics (if any) using (strategy, symbol, timeframe)
history_records = load_all_best_params(strategy_choice, symbol_for_data, selected_tf)
if history_records:
    top = history_records[0]  # already sorted best-first in param_registry
    best_params = top.get("params", {}) or {}
    best_results = top.get("metrics", {}) or {}
else:
    best_params, best_results = {}, {}

# ================================= Layout: top summary panels =================================
data_col, saved_col = st.columns([1, 3])
main_col, current_col = st.columns([1, 3])
table_col, comment_col = st.columns([10, 1])

with data_col:
    with st.container(border=1, height=332):
        if infodisplay == 1 :
            st.info("No data loaded yet or data has no 'date' column.")
        else:
            st.write("###### Selected Symbol: ( ", symbol_for_data, " )")
            st.write("###### TimeFrame ( ", selected_tf, " )")
            st.write("###### Rows after slicing:", len(df))
            st.write("###### Date after slicing:")
            st.write(df["date"].min())
            st.write(df["date"].max())
            if last_values:
                if last_values['open']>last_values['close']:
                    st.write(
                        f"Last Candle: " 
                        f"<span style='color:red; font-weight:bold;'> L: {last_values['low']},"f" H: {last_values['high']},  \n"f"O : {last_values['open']}" f", C: {last_values['close']}"
                        f", Vol:{last_values['volume']}</span>", unsafe_allow_html=True)
                else:
                    st.write(
                        f"Last Candle: "
                        f"<span style='color:green; font-weight:bold;'> L: {last_values['low']},"f" H: {last_values['high']},  \n"f"O : {last_values['open']}" f", C: {last_values['close']}"
                        f", Vol:{last_values['volume']}</span>", unsafe_allow_html=True)

            if df is not None and not df.empty:
                st.download_button(
                    label="⬇️ Download Loaded Data (CSV)",
                    data=df.to_csv(index=False),
                    file_name=f"{symbol_for_data}_{selected_tf}_loaded_data.csv",
                    mime="text/csv",
                )


with saved_col:
    with st.container(border=1, height=332):
        saved_results_col, saved_param_col = st.columns([1, 1])
        if history_records:
            top = history_records[0]
            best_params = top.get("params", {}) or {}
            best_results = top.get("metrics", {}) or {}

            with saved_param_col:
                st.markdown(f"###### Saved Best Parameters for {strategy_choice} ({symbol_for_data})({selected_tf})")
                st.dataframe(
                    pd.DataFrame(list(best_params.items()), columns=["Parameter", "Value"]),
                    height=248,
                )
            with saved_results_col:
                st.markdown(f"###### Saved Best results for {strategy_choice} ({symbol_for_data})({selected_tf})")
                st.dataframe(
                    pd.DataFrame(list(best_results.items()), columns=["Metric", "Value"]),
                    height=212,
                )
                tv_preset_str = json_to_tv_preset()
                st.download_button(
                    label="⬇️ Export All TV Presets",
                    data=tv_preset_str,
                    file_name="tv_presets.txt",
                    mime="text/plain",
                )
        else:
            st.info("No saved parameter set found yet for this strategy / symbol / timeframe.")

# ---- Strategy selector wiring ----
strategy_entry = STRATEGY_REGISTRY[strategy_choice]
strategy_func = strategy_entry["function"]
strategy_params = strategy_entry["params"]
universal_keys = strategy_entry.get("universal", [])

# ---- Build universal kwargs from UI ----
# ---- Build universal kwargs from UI ----
sidebar_values = {
    "use_session_end_rule": bool(use_session_end_rule),
    # keep initial_capital available if/when you want to use it in the future
    "initial_capital": int(initial_capital),
}
runtime = {
    "maxtradesperday": int(maxtradesperday),
    "use_session_end_rule": bool(use_session_end_rule),
}
if strategy_choice in CRYPTO_STRATEGY_KEYS:
    runtime["sizing_mode"] = "cash" if sizing_mode == "Cash" else "qty"
    runtime["cash"] = float(cash)
    runtime["qty"] = float(qty)
else:
    runtime["qty"] = int(qty)

if strategy_choice not in CRYPTO_STRATEGY_KEYS:
    session_start = f"{int(session_start_h):02d}:{int(session_start_m):02d}"
    session_end = f"{int(session_end_h):02d}:{int(session_end_m):02d}"
    runtime["session_start"] = session_start
    runtime["session_end"] = session_end

# =====================================================================================
# "Run with saved params" path
# =====================================================================================
if df is not None and not df.empty and run_saved_params:
    st.info("Running strategy with last saved parameters...")
    call_kwargs = {k: sidebar_values[k] for k in universal_keys if k in sidebar_values}
    call_kwargs.update(best_params)
    call_kwargs['runtime'] = runtime
    trades, df_all = strategy_func(df, **call_kwargs)

    with st.expander("Trades with saved parameters"):
        st.dataframe(trades)
    main_results = {
        "Total Return %": total_return_pct(trades),
        "Total Return $": total_return_usd(trades),
        "Win %": float(((trades["pnl"] > 0).mean()) * 100) if not trades.empty and "pnl" in trades else 0,
        "Max Drawdown": float(max_drawdown(trades)),
        "Total Trades": int(len(trades)),
        "Avg Bars In Trade": float(trades["BarsInTrade"].mean()) if "BarsInTrade" in trades else 0,
          # ← use helper
    }
    st.session_state["last_best_params"] = best_params
    st.session_state["last_main_results"] = main_results
    st.session_state["last_trades"] = trades
    st.session_state["last_df_all"] = df_all
    st.session_state["Symbol"] = symbol_for_data

# =====================================================================================
# Optimization
# =====================================================================================
optimized_indecator = 0

# Progress callback for CD

def _progress_cb(step, total):
    if progress_bar_widget:
        progress_bar_widget.progress(min(1.0, step / total))

if run_optuna and df is not None and not df.empty:
    optimized_indecator = 1
    df_strategy = df.copy()
    df_strategy.columns = [c.lower() for c in df_strategy.columns]

    # Per-strategy CD search space (used for CD and to seed Optuna bounds where relevant)
    param_grid = CD_PARAM_SPACES[strategy_choice]
    strategy_param_list = strategy_entry["params"]
    cd_space = {k: param_grid[k] for k in strategy_param_list if k in param_grid}

    if optimizer_choice == "Optuna (Bayesian)":
        def objective(trial):
            strategy_kwargs = {}
            for param in strategy_params:
                # (Keep your existing per-param suggest_* rules; trimmed for brevity)
                if param == "rma_len":
                    strategy_kwargs[param] = trial.suggest_int(param, 40, 100)
                elif param == "barsforentry":
                    strategy_kwargs[param] = trial.suggest_int(param, 1, 10)
                elif param == "barsforexit":
                    strategy_kwargs[param] = trial.suggest_int(param, 1, 10)
                elif param == "atrlen":
                    strategy_kwargs[param] = trial.suggest_int(param, 2, 20)
                elif param == "ema_fast_len":
                    strategy_kwargs[param] = trial.suggest_int(param, 2, 30)
                elif param == "ema_slow_len":
                    fast_val = strategy_kwargs.get("ema_fast_len", 2)
                    strategy_kwargs[param] = trial.suggest_int(param, fast_val + 1, 100)
                elif param == "normalizedupper":
                    strategy_kwargs[param] = trial.suggest_float(param, 0, 2, step=0.1)
                elif param == "normalizedlower":
                    strategy_kwargs[param] = trial.suggest_float(param, -2, 0, step=0.1)
                elif param == "fast_len":
                    strategy_kwargs[param] = trial.suggest_int(param, 2, 25)
                elif param == "slow_len":
                    fast_val = strategy_kwargs.get("fast_len", 2)
                    strategy_kwargs[param] = trial.suggest_int(param, fast_val + 1, 60)
                elif param == "breakout_len":
                    strategy_kwargs[param] = trial.suggest_int(param, 5, 50)
                elif param == "momentum_len":
                    strategy_kwargs[param] = trial.suggest_int(param, 3, 30)
                elif param == "momentum_thresh":
                    strategy_kwargs[param] = trial.suggest_float(param, 0.1, 3.0, step=0.1)
                elif param == "trend_len":
                    strategy_kwargs[param] = trial.suggest_int(param, 10, 100)
                elif param == "atr_len":
                    strategy_kwargs[param] = trial.suggest_int(param, 5, 50)
                elif param == "min_atr":
                    strategy_kwargs[param] = trial.suggest_float(param, 0.1, 5.0, step=0.1)
                elif param == "trailpct":
                    strategy_kwargs[param] = trial.suggest_float(param, 0.2, 10, step=0.1)
                elif param == "max_hold_bars":
                    strategy_kwargs[param] = trial.suggest_int(param, 5, 50)
                elif param == "lookback":
                    strategy_kwargs[param] = trial.suggest_int(param, 10, 50)
                elif param == "ma_len":
                    strategy_kwargs[param] = trial.suggest_int(param, 10, 60)
                elif param == "threshold":
                    strategy_kwargs[param] = trial.suggest_float(param, 1, 4, step=0.2)
                elif param == "mult":
                    strategy_kwargs[param] = trial.suggest_float(param, 1.0, 3.0, step=0.2)
                elif param == "spread_lookback":
                    strategy_kwargs[param] = trial.suggest_int(param, 10, 60)
                elif param == "vol_mult":
                    strategy_kwargs[param] = trial.suggest_float(param, 1.2, 4.0, step=0.2)
                else:
                    # Fallback: skip unknowns
                    continue

            disable_in_optimizer = CRYPTO_DISABLE_IN_OPT if strategy_choice in CRYPTO_STRATEGY_KEYS else []
            enabled_universal_keys = [k for k in universal_keys if k not in disable_in_optimizer]
            call_kwargs = {k: sidebar_values[k] for k in enabled_universal_keys if k in sidebar_values}
            call_kwargs['runtime'] = runtime
            call_kwargs.update(strategy_kwargs)
            # universal_kwargs already contains runtime
            trades, _ = strategy_func(df_strategy, **call_kwargs)

            score = score_from_trades(
                trades,
                optimize_for,
                pen_alpha = float(pen_alpha) if 'pen_alpha' in locals() else 0.1,
                pen_beta = float(pen_beta) if 'pen_beta' in locals() else 0.1,
            )
            return score

        study = optuna.create_study(direction="maximize")

        def optuna_callback(study, trial):
            progress_bar_widget.progress(min(1.0, (trial.number + 1) / optuna_trials))

        study.optimize(objective, n_trials=optuna_trials, callbacks=[optuna_callback])
        progress_bar.empty()
        st.balloons()
        best_params = study.best_trial.params

    else:
        # Coordinate Descent path
        disable_in_optimizer = CRYPTO_DISABLE_IN_OPT if strategy_choice in CRYPTO_STRATEGY_KEYS else []
        enabled_universal_keys = [k for k in universal_keys if k not in disable_in_optimizer]
        call_kwargs = {k: sidebar_values[k] for k in enabled_universal_keys if k in sidebar_values}
        call_kwargs['runtime'] = runtime
        initial_params = load_best_params(strategy_choice, symbol_for_data, selected_tf)
        initial_params = sanitize_initial_params(cd_space, initial_params)

        best_params, best_score = coordinate_descent_optimizer(
            strategy_func,
            df_strategy,
            cd_space,
            call_kwargs,
            n_rounds=Coordinate_Descent_trials,
            initial_params=initial_params,
            score_fn=lambda tr: score_from_trades(
                tr,
                optimize_for,
                pen_alpha=float(pen_alpha) if 'pen_alpha' in locals() else 0.1,
                pen_beta=float(pen_beta) if 'pen_beta' in locals() else 0.1,
            ),
            progress_cb=_progress_cb,
        )

    # Final run with best params
    call_kwargs = {k: sidebar_values[k] for k in universal_keys if k in sidebar_values}
    call_kwargs.update(best_params)
    call_kwargs['runtime'] = runtime
    trades, df_all = strategy_func(df_strategy, **call_kwargs)

    with st.expander("Trades with Optimized parameters"):
        #st.write("## Trades with Optimized parameters")
        st.dataframe(trades)

    main_results = {
        "Total Return %": total_return_pct(trades),
        "Total Return $": total_return_usd(trades),
        "Win %": float(((trades["pnl"] > 0).mean()) * 100) if not trades.empty and "pnl" in trades else 0,
        "Max Drawdown": float(max_drawdown(trades)),  # ← use helper
        "Total Trades": int(len(trades)),
        "Avg Bars In Trade": float(trades["BarsInTrade"].mean()) if "BarsInTrade" in trades else 0,

    }

    st.session_state["last_best_params"] = best_params
    st.session_state["last_main_results"] = main_results
    st.session_state["last_trades"] = trades
    st.session_state["last_df_all"] = df_all
    st.session_state["Symbol"] = symbol_for_data

# =====================================================================================
# Results / tables
# =====================================================================================
Optimized_Symbol = st.session_state.get("Symbol")
trades = st.session_state.get("last_trades", pd.DataFrame())
df_all = st.session_state.get("last_df_all", pd.DataFrame())

if Optimized_Symbol == symbol_for_data:
    with main_col:
        with st.container(border=1, height=332):
            if "df_all" in locals() and df_all is not None and not df_all.empty and optimized_indecator == 1:
                st.markdown("#### Optimization Results...")
                st.markdown(f"Optimizer: {optimizer_choice}")
                st.markdown(f"Trials: {trialscount}")
                st.markdown(f"Strategy: {strategy_choice}")
                df_all = ensure_date_column(df_all)
                test_start = None
                test_end = None
                if df_all is not None and not df_all.empty and "date" in df_all.columns:
                    s = pd.to_datetime(df_all["date"], errors="coerce")
                    s = s.dropna()
                    if not s.empty:
                        test_start = s.min()
                        test_end = s.max()
                    else:
                        test_start = test_end = None
                    st.markdown("Test Period:")
                    if test_start is not None and test_end is not None:
                        st.markdown(
                            f" {test_start.strftime('%Y-%m-%d %H:%M')} &mdash; {test_end.strftime('%Y-%m-%d %H:%M')}")
                    else:
                        st.markdown(" N/A")
            else:
                st.info("Optimization Results Not Available, Please run the optimizer first...")

    with current_col:
        with st.container(border=1, height=332):
            cur_result_col, curr_param_col = st.columns([1, 1])
            bp = st.session_state.get("last_best_params", None)
            mr = st.session_state.get("last_main_results", None)
            with curr_param_col:
                if bp:
                    st.markdown(f"###### Current Parameters for {strategy_choice} ({symbol_for_data})({selected_tf})")
                    df_params = pd.DataFrame(list(bp.items()), columns=["Parameter", "Value"])
                    st.dataframe(df_params, height=248)
                else:
                    st.info("No optimizer run in this session yet.")
            with cur_result_col:
                if mr:
                    st.markdown(f"###### Current Best results for {strategy_choice} ({symbol_for_data})({selected_tf})")
                    mr_results = pd.DataFrame(list(mr.items()), columns=["Result", "Value"])
                    st.dataframe(mr_results, height=212)
                if st.button("💾 Save Best Parameters", key="save_best_params"):
                    # Persist with (strategy, symbol, timeframe) key
                    register_best_params(
                        strategy=strategy_choice,
                        symbol=symbol_for_data,
                        tf=selected_tf,
                        params=bp,
                        metrics=mr,
                    )
                    #(strategy, symbol, tf, params, metrics, registry_path=REGISTRY_FILE)

                    st.success("Parameters saved for this strategy / symbol / timeframe!")

    render_visuals(
        st,
        df_all=df_all,
        trades=trades,
        runtime=runtime,
    )
else:
    st.info("Run a strategy and optimizer ...")



with st.expander("📅 Batch Optimization", expanded=False):
    if st.button("Run Weekly Optimizer"):
        try:
            run_batch_optimization()
            st.success("✅ Batch optimization completed. Review results below.")
        except Exception as e:
            st.error(f"❌ Batch optimization failed: {e}")
    if st.button("🛑 Stop Batch Optimization"):
        with open(STOP_FILE, "w") as f:
            f.write("stop")
        st.warning("Stop signal sent — will interrupt after current symbol.")

    # Show pending JSON if it exists
    try:
        with open(PENDING_REGISTRY_FILE, "r") as f:
            pending_data = json.load(f)

        if pending_data:
            st.markdown("### Pending Registry Preview")
            # Flatten preview for table view
            preview_rows = []
            for key, records in pending_data.items():
                for rec in records:
                    preview_rows.append({
                        "Key": key,
                        **rec["params"],
                        **rec["metrics"]
                    })
            st.dataframe(pd.DataFrame(preview_rows))

            if st.button("Merge Pending to Main Registry"):
                merge_pending()
                st.success("✅ Merge complete. Main registry updated.")

        else:
            st.info("No pending registry file found or it's empty.")
    except FileNotFoundError:
        st.info("No pending registry file found. Run the optimizer first.")