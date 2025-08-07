import copy
import re
from datetime import datetime
import numpy as np
import optuna
import pandas as pd
import streamlit as st
from datafeed import (load_csv, load_yfinance, load_alpaca, load_tvdatafeed, get_tv_ta, load_ccxt)
from strategies import crypto_intraday_multi
from strategies.rma import rma_strategy
from strategies.sma_cross import sma_cross_strategy
import json
import os

PARAM_REGISTRY_FILE = "../best_params.json"

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


# Strategy registry and parameter definitions
STRATEGY_REGISTRY = {
    "RMA Strategy": {
        "function": rma_strategy,
        "params": [
            "rma_len", "barsforentry", "barsforexit", "atrlen",
            "normalizedupper", "normalizedlower", "ema_fast_len", "ema_slow_len",
            "emasrc",  # Uncomment if you want to enable
            "risklen", "trailpct"
        ],
        "universal": ["session_start", "session_end", "keeplime", "initial_capital", "qty", "maxtradesperday",
                      "use_session_end_rule"]
    },
    "Crypto Intraday Multi-Signal": {
        "function": crypto_intraday_multi,
        "params": [
            "breakout_len", "momentum_len", "momentum_thresh", "trend_len",
            "atr_len", "min_atr", "trailing_stop_pct", "max_hold_bars"
        ],
        "universal": ["initial_capital", "qty"]
    },
    "SMA Cross": {
        "function": sma_cross_strategy,
        "params": [
            "fast_len", "slow_len"
        ],
        "universal": ["initial_capital", "qty"]
    }
}
# --- Per-strategy parameter search spaces for coordinate descent
CD_PARAM_SPACES = {
    "RMA Strategy": {
        "rma_len": (40, 100, 5),
        "barsforentry": (1, 10, 1),
        "barsforexit": (1, 10, 1),
        "atrlen": (2, 20, 1),
        "normalizedupper": (0, 2, 0.1),
        "normalizedlower": (0, -2, -0.1),
        "emasrc": [15, 30, 40, 45, 60, 120, 240],  # Uncomment if needed
        "ema_fast_len": (2, 30, 1),
        "ema_slow_len": (3, 100, 1),
        "risklen": (40, 0, -0.5),
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
    }
}

st.set_page_config(page_title="Universal Backtester", layout="wide")
st.title("📊 Universal  Backtester")
progress_bar = st.empty()
progress_bar_widget = progress_bar.progress(0)
Coordinate_Descent_trials = 0

with st.sidebar:
    with st.expander("Data Feed", expanded=False):
        uploaded_files = st.file_uploader("Upload one or more CSVs (SYMBOL_TIMEFRAME.csv)", type=["csv"],
                                          accept_multiple_files=True)
        feed_choices = ["alpaca", "tvdatafeed", "ccxt", "csv", "yfinance", "tradingview_ta"]
        selected_feed = st.selectbox("Data Source", feed_choices)
        # --------------------- Data Loading UI ---------------------
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

        #////////////////multi_symbol selection/////////////////////////////////////////////////////////////////////////
        multi_selected_symbols = st.multiselect(
            "Batch Optimize Symbols",
            filtered_symbols,
            default=[selected_symbol] if selected_symbol in filtered_symbols else []
        )
        run_multi_opt = st.button("Run Batch Optimization")
        df = None
        if selected_feed == "csv" and csv_key in csv_symbols:
            df = load_csv(csv_symbols[csv_key])
            if df is not None and not df.empty:
                st.session_state['df_loaded'] = df
        elif selected_feed == "ccxt":
            ccxt_exchange = st.sidebar.selectbox(
                "Exchange", ["alpaca", "binanceus", "binance", "bybit", "okx", "kucoin", "coinbase", "kraken"], index=0)
            ccxt_symbol = st.text_input("CCXT Symbol (e.g. BTC/USDT)", value="BTC/USDT")
            ccxt_tf = st.selectbox(
                "CCXT Interval", ["1m", "3m", "5m", "15m", "1h", "4h", "1d"], index=2)
            ccxt_limit = st.number_input("Bars", min_value=50, max_value=50000, value=5000)
            if st.button("Load CCXT Data"):
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
            start = st.date_input("Start Date", value=datetime(2025, 7, 1))
            end = st.date_input("End Date")
            if st.button("Load Yahoo Data"):
                df = load_yfinance(selected_symbol, selected_tf, start, end)
                if df is not None and not df.empty:
                    df['symbol'] = selected_symbol
                    st.session_state['df_loaded'] = df
        elif selected_feed == "alpaca":
            start = st.date_input("Start Date", value=datetime(2025, 7, 1))
            end = st.date_input("End Date")
            key = st.secrets["alpaca"]["key"]
            secret = st.secrets["alpaca"]["secret"]
            url = st.secrets["alpaca"]["base_url"]
            if st.button("Load Alpaca Data"):
                df = load_alpaca(selected_symbol, selected_tf, start, end, key, secret, url)
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
        session_start_m = st.number_input("Session Start Minute", min_value=0, max_value=59, value=31)
        session_end_h = st.number_input("Session End Hour", min_value=0, max_value=23, value=19)
        session_end_m = st.number_input("Session End Minute", min_value=0, max_value=59, value=52)
    with st.expander("Strategy Settings", expanded=False):
        maxtradesperday = st.number_input("Max Trades Per Day", min_value=1, max_value=100, value=1, step=1)
        use_session_end_rule = st.checkbox("Session End Exit Rule", value=True)
        optimize_for = st.selectbox("Optimize For", ["return with win% tie-breaker", "return", "win"])
        strategy_choice = st.selectbox("Select Strategy", list(STRATEGY_REGISTRY.keys()))
    with st.expander("Optimizer Settings", expanded=False):
        optimizer_choice = st.selectbox("Select Optimizer", ["Coordinate Descent", "Optuna (Bayesian)"])
        if optimizer_choice == "Optuna (Bayesian)":
            optuna_trials = st.number_input("Optuna Trials", min_value=10, max_value=1000, value=100, step=10)
        else:
            Coordinate_Descent_trials = st.number_input("Coordinate Descent Trials", min_value=1, max_value=50, value=4,
                                                        step=1)
        run_optuna = st.sidebar.button("Run Optimization")
data_col, main_col = st.columns([1, 3])
table_col, side_col = st.columns([2, 1])

if infodisplay == 1:
    st.info("No data loaded yet or data has no 'date' column.")
else:
    with data_col:
        with st.container(border=1):
            st.header("Loaded Data ")
            st.write("Selected Symbol:", selected_symbol)
            st.write("Rows after slicing:", len(df))
            st.write("Date min/max after slicing:")
            st.write(df['date'].min())
            st.write(df['date'].max())


# ---- Strategy Selector ----
strategy_entry = STRATEGY_REGISTRY[strategy_choice]
strategy_func = strategy_entry["function"]
strategy_params = strategy_entry["params"]
universal_keys = strategy_entry.get("universal", [])

# --- AUTO-LOAD saved best params for the selected strategy/symbol, if exists ---
saved_params_record = get_saved_params(strategy_choice, selected_symbol)
if saved_params_record:
    best_params = saved_params_record.get('params', {})
    best_results = saved_params_record.get('results', {})
else:
    best_params = {}
    best_results = {}

# --- Coordinate Descent Optimizer ---
def coordinate_descent_optimizer(
        strategy_func,
        df_strategy,
        param_space,
        universal_kwargs,
        maximize_metric=optimize_for,
        n_rounds=Coordinate_Descent_trials,
        show_progress=True
):
    current_params = {}
    for p, v in param_space.items():
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
    #    progress_bar = st.progress(0) if show_progress else None

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
                score = trades[maximize_metric].sum() if not trades.empty and maximize_metric in trades else -99999
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


# ----------- Run Optimization -----------
if run_optuna and df is not None and not df.empty:
    df_strategy = df.copy()
    df_strategy.columns = [c.lower() for c in df_strategy.columns]
    session_start = f"{int(session_start_h):02d}:{int(session_start_m):02d}"
    session_end = f"{int(session_end_h):02d}:{int(session_end_m):02d}"

    sidebar_values = {
        "session_start": session_start,
        "session_end": session_end,
        "keeplime": True,
        "initial_capital": 200000,
        "qty": 1,
        "maxtradesperday": maxtradesperday,
        "use_session_end_rule": use_session_end_rule  # <-- add here

    }

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
                elif param == 'emasrc':
                    strategy_kwargs[param] = trial.suggest_categorical(param, [15, 30, 40, 45, 60, 120, 240])
                elif param == 'ema_fast_len':
                    strategy_kwargs[param] = trial.suggest_int(param, 2, 30)
                elif param == 'ema_slow_len':
                    fast_val = strategy_kwargs.get('ema_fast_len', 2)
                    strategy_kwargs[param] = trial.suggest_int(param, fast_val + 1, 100)
                elif param in ['normalizedupper']:
                    strategy_kwargs[param] = trial.suggest_float(param, 0, 2, step=0.2)
                elif param in ['normalizedlower']:
                    strategy_kwargs[param] = trial.suggest_float(param, -2, 0, step=0.2)
                elif param in ['risklen', 'trailpct']:
                    strategy_kwargs[param] = trial.suggest_float(param, 0, 40, step=0.5)
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
            elif optimize_for.lower() == 'return with win% tie-breaker':
                return total_return + win_pct / 100
            else:
                return total_return


        st.info(f"Running Optuna optimizer for {optuna_trials} trials on {strategy_choice}...")
        study = optuna.create_study(direction="maximize")


        #        progress_bar = st.progress(0)
        def optuna_callback(study, trial):
            progress_bar_widget.progress(min(1.0, (trial.number + 1) / optuna_trials))


        study.optimize(objective, n_trials=optuna_trials, callbacks=[optuna_callback])
        progress_bar.empty()
        st.balloons()
        best_params = study.best_trial.params
    else:
        call_kwargs = {k: sidebar_values[k] for k in universal_keys if k in sidebar_values}
        best_params, best_score = coordinate_descent_optimizer(
            strategy_func, df_strategy, cd_space, call_kwargs,
            maximize_metric=optimize_for, n_rounds=Coordinate_Descent_trials
            # ///////////////////////////////////////////
        )

    # /////////////////////////////////////////Best Parameters Result Calculations///////////////////////////////////////////
    # ---- FINAL run with best params ----
    call_kwargs = {k: sidebar_values[k] for k in universal_keys if k in sidebar_values}
    for k in (best_params if optimizer_choice == "Optuna (Bayesian)" else best_params):
        call_kwargs[k] = best_params[k]
    trades, df_all = strategy_func(df_strategy, **call_kwargs)

    # --- Display results ---/////////////////////////////////////////////////////////////////////////////////////////

    with main_col:
        with st.container(border=1):
            st.markdown("### Optimization Results")
            # Your table and charts here
            if not df_all.empty:
                test_start = df_all["date"].min()
                test_end = df_all["date"].max()
                st.markdown(
                    f"**Test Period:** {test_start.strftime('%Y-%m-%d %H:%M')} &mdash; {test_end.strftime('%Y-%m-%d %H:%M')}")
            with st.container(border=1):
                tt, tr, wp, bnt, dd = st.columns([1, 1, 1, 1, 1])
                with tt:
                    st.markdown("Total Trades")
                    st.markdown(f"{len(trades)}")
                with tr:
                    if 'return' in trades.columns:
                        st.markdown("Total return", width="content")
                        st.markdown(f" {trades['return'].sum():.2f}%")
                with wp:
                    if 'pnl' in trades.columns:
                        st.markdown("win %", width="content")
                        st.markdown(f"{(trades['pnl'] > 0).mean() * 100:.2f}%")
                with bnt:
                    if 'BarsInTrade' in trades.columns:
                        st.markdown("Avg Bars In Trade", width="content")
                        st.markdown(f"{trades['BarsInTrade'].mean():.1f}")
                with dd:
                    if 'pnl' in trades.columns:
                        st.markdown("Max Drawdown", width="content")
                        st.markdown(f"{trades['pnl'].min():.2f}")
        st.info(f"Running {optimizer_choice} for {strategy_choice}")
    #//////////////////////////////Show Trade Table and chart//////////////////////////////////////////////////////
    with table_col:
        with st.container(border=1):
            st.markdown("### Best Parameter Trade Log")
            st.dataframe(trades)
            plot_col = None
            for col in ['close', 'Close']:
                if col in df_all.columns:
                    plot_col = col
                    break
            if not trades.empty and plot_col is not None:
                st.line_chart(df_all[plot_col])
            else:
                st.info("No price chart to display. 'close' column missing or no trades.")
    main_results = {
        "Total Return": float(trades['return'].sum()) if not trades.empty and 'return' in trades else 0,
        "Win Rate": float((trades['pnl'] > 0).mean()) if not trades.empty and 'pnl' in trades else 0,
        "Total Trades": int(len(trades))
    }
    st.session_state['last_best_params'] = best_params
    st.session_state['last_main_results'] = main_results
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////Multi symbol optimizer

# --- Batch Optimization Logic ---
if run_multi_opt and df is not None and not df.empty and multi_selected_symbols:
    st.info(f"Running optimizer for {len(multi_selected_symbols)} symbols...")
    summary_results = []
    for symbol in multi_selected_symbols:
        # Slice data for this symbol if possible
        df_symbol = df[df['symbol'] == symbol] if 'symbol' in df.columns else df.copy()
        if df_symbol.empty:
            st.warning(f"No data for {symbol}, skipping.")
            continue

        # Prepare strategy and universal params
        session_start = f"{int(session_start_h):02d}:{int(session_start_m):02d}"
        session_end = f"{int(session_end_h):02d}:{int(session_end_m):02d}"

        sidebar_values = {
            "session_start": session_start,
            "session_end": session_end,
            "keeplime": True,
            "initial_capital": 200000,
            "qty": 1,
            "maxtradesperday": maxtradesperday,
            "use_session_end_rule": use_session_end_rule
        }
        strategy_entry = STRATEGY_REGISTRY[strategy_choice]
        strategy_func = strategy_entry["function"]
        strategy_params = strategy_entry["params"]
        universal_keys = strategy_entry.get("universal", [])

        param_grid = CD_PARAM_SPACES[strategy_choice]
        strategy_param_list = strategy_entry["params"]
        cd_space = {k: param_grid[k] for k in strategy_param_list if k in param_grid}

        # Run optimizer (Coordinate Descent only; add Optuna support if you want)
        call_kwargs = {k: sidebar_values[k] for k in universal_keys if k in sidebar_values}
        best_params, best_score = coordinate_descent_optimizer(
            strategy_func, df_symbol, cd_space, call_kwargs,
            maximize_metric=optimize_for, n_rounds=Coordinate_Descent_trials
        )

        # Final run with best params
        call_kwargs = {k: sidebar_values[k] for k in universal_keys if k in sidebar_values}
        for k in best_params:
            call_kwargs[k] = best_params[k]
        trades, _ = strategy_func(df_symbol, **call_kwargs)
        main_results = {
            "Total Return": float(trades['return'].sum()) if not trades.empty and 'return' in trades else 0,
            "Win Rate": float((trades['pnl'] > 0).mean()) if not trades.empty and 'pnl' in trades else 0,
            "Total Trades": int(len(trades))
        }
        # Save to JSON registry
        register_params(strategy_choice, symbol, best_params, main_results)
        # Save summary for display
        summary_results.append({
            "Symbol": symbol,
            **best_params,
            **main_results
        })
    # Show summary table and export button
    summary_df = pd.DataFrame(summary_results)
    st.markdown("### Batch Optimization Summary")
    st.dataframe(summary_df)
    st.download_button(
        label="⬇️ Download Batch Optimization Results (CSV)",
        data=summary_df.to_csv(index=False),
        file_name="batch_optimized_params.csv",
        mime="text/csv"
    )
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////End of Multi symbol optimizer
with side_col:
    with st.container(border=1):
        # Show last optimizer results (from session_state, if present)
        bp = st.session_state.get('last_best_params', None)
        mr = st.session_state.get('last_main_results', None)
        if bp:
            st.markdown(f"##### Optimizer Results for {strategy_choice}")
            df_params = pd.DataFrame(list(bp.items()), columns=["Parameter", "Value"])
            st.dataframe(df_params, height=325)
            # bp: dict of current optimizer parameters
            export = pd.DataFrame(list(bp.items()), columns=["Parameter", "Value"]).to_csv(index=False)
            st.download_button(
                label="⬇️ Download Optimizer Params (CSV)",
                data=export,
                file_name="optimizer_params.csv",
                mime="text/csv"
            )

        else:
            st.info("No optimizer run in this session yet.")

        # Save button (always available)
        if st.button("💾 Save Best Parameters", key="save_best_params"):
            if bp is not None and mr is not None:
                register_params(strategy_choice, selected_symbol, bp, mr)
                st.success("Parameters saved for this strategy/symbol (replaced old entry)!")
            else:
                st.error("No optimizer results found in session. Please run the optimizer first.")

        # Show saved best parameters/results from JSON BELOW (always)
        true_saved = get_saved_params(strategy_choice, selected_symbol)
        if true_saved and "params" in true_saved:
            st.markdown("##### Saved Best Parameters")
            st.dataframe(pd.DataFrame(list(true_saved["params"].items()), columns=["Parameter", "Value"]), height=250)
            if "results" in true_saved:
                st.markdown("##### Saved Best Results")
                st.json(true_saved["results"])
            export_saved = pd.DataFrame(list(true_saved["params"].items()), columns=["Parameter", "Value"]).to_csv(
                index=False)
            st.download_button(
                label="⬇️ Download Saved Best Params (CSV)",
                data=export_saved,
                file_name="saved_params.csv",
                mime="text/csv"
            )
        else:
            st.info("No saved parameter set found for this strategy and symbol yet.")


