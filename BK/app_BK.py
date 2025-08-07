import io
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import streamlit as st


# ----------- RMA Strategy Functions -----------
def rma(x, length):
    a = np.full_like(x, np.nan)
    n = len(x)
    alpha = 1 / length
    a[0] = x[0]
    for i in range(1, n):
        a[i] = alpha * x[i] + (1 - alpha) * a[i - 1]
    return a


def pine_ema(src, length):
    return pd.Series(src).ewm(span=length, adjust=False).mean().values


def impulse(src, high, low, length):
    hi = rma(high, length)
    lo = rma(low, length)
    mi = 2 * pine_ema(src, length) - pine_ema(pine_ema(src, length), length)
    imp = np.where(mi > hi, mi - hi, np.where(mi < lo, mi - lo, 0))
    return imp, mi, hi, lo


def streak_bool(arr):
    streaks = np.zeros_like(arr, dtype=int)
    for i in range(1, len(arr)):
        if arr[i]:
            streaks[i] = streaks[i - 1] + 1
        else:
            streaks[i] = 0
    return streaks


def rma_strategy(
        df,
        rma_len=78,
        barsForEntry=6,
        barsForExit=8,
        ATRLen=9,
        normalizedUpper=1.7,
        normalizedLower=-2,
        ema_fast_len=4,
        ema_slow_len=60,
        RiskLen=50,
        TrailPct=50,
        session_start="09:31",
        session_end="15:52",
        keepLime=True,
        initial_capital=15000,
        qty=1000,
        maxtradesperday=1
):
    df = df.copy()
    df['HLC3'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['imp'], df['mi'], df['hi'], df['lo'] = impulse(
        df['HLC3'].values, df['High'].values, df['Low'].values, rma_len)
    df['lime'] = df['HLC3'] > df['hi']
    df['coreGreen'] = (df['HLC3'] > df['mi']) & (df['HLC3'] <= df['hi'])
    df['green'] = df['coreGreen'] | df['lime'] if keepLime else df['coreGreen']
    df['upCnt'] = streak_bool(df['green'].values)
    df['dnCnt'] = streak_bool(~df['green'].values)
    df['ATR'] = pd.Series(df['High'] - df['Low']).rolling(ATRLen).mean()
    df['normalizedImpulse'] = df['imp'] / df['ATR']
    df['impOutside'] = (df['normalizedImpulse'] >= normalizedUpper) | (df['normalizedImpulse'] <= normalizedLower)
    df['EMA_fast'] = pd.Series(df['Close']).ewm(span=ema_fast_len, adjust=False).mean()
    df['EMA_slow'] = pd.Series(df['Close']).ewm(span=ema_slow_len, adjust=False).mean()
    df['mtfTrend'] = df['EMA_fast'] > df['EMA_slow']
    df['time'] = df["Date"].apply(lambda x: x.time() if pd.notnull(x) else None)
    start_time = datetime.strptime(session_start, "%H:%M").time()
    end_time = datetime.strptime(session_end, "%H:%M").time()
    df['inSession'] = df['time'].apply(lambda t: (t is not None) and (start_time <= t <= end_time))

    pos = 0  # 0 = flat, 1 = long
    entry_price = 0
    peak_price = 0
    bars_in_trade = 0
    trade_log = []
    current_day = None
    trades_today = 0
    entry_taken = False

    for i in range(len(df)):
        day = df["Date"].iloc[i].date()
        bar_time = df["Date"].iloc[i].time()
        if day != current_day:
            current_day = day
            trades_today = 0
            entry_taken = False
        if not df['inSession'].iloc[i]:
            continue
        if (
                pos == 0 and
                not entry_taken and
                trades_today < maxtradesperday and
                start_time <= bar_time < end_time and
                (df['upCnt'].iloc[i] >= barsForEntry) and
                df['impOutside'].iloc[i] and
                df['mtfTrend'].iloc[i]
        ):
            pos = 1
            entry_idx = i
            entry_price = df['Close'].iloc[i]
            peak_price = entry_price
            bars_in_trade = 0
            entry_taken = True
            trades_today += 1
            trade_log.append({
                'EntryTime': df["Date"].iloc[i],
                'EntryPrice': entry_price,
                'BarsInTrade': 0,
                'ExitTime': None,
                'ExitPrice': None,
                'PnL': None,
                'ExitReason': None
            })
        is_session_end = (bar_time >= end_time)
        if pos == 1:
            bars_in_trade += 1
            peak_price = max(peak_price, df['High'].iloc[i])
            exit_reason = None
            exit_price = None
            if (df['dnCnt'].iloc[i] == barsForExit) or (not df['mtfTrend'].iloc[i]):
                exit_price = df['Close'].iloc[i]
                exit_reason = "Exit rule"
            elif df['Low'].iloc[i] <= entry_price * (1 - RiskLen / 100):
                exit_price = entry_price * (1 - RiskLen / 100)
                exit_reason = "Risk"
            elif df['Low'].iloc[i] <= peak_price * (1 - TrailPct / 100):
                exit_price = peak_price * (1 - TrailPct / 100)
                exit_reason = "Trail"
            elif is_session_end:
                exit_price = df['Close'].iloc[i]
                exit_reason = "Session Close"
            if exit_reason is not None:
                pos = 0
                trade_log[-1].update({
                    'ExitTime': df["Date"].iloc[i],
                    'ExitPrice': exit_price,
                    'PnL': exit_price - trade_log[-1]['EntryPrice'],
                    'BarsInTrade': bars_in_trade,
                    'ExitReason': exit_reason
                })
    trades = pd.DataFrame(trade_log)
    if not trades.empty and 'PnL' in trades.columns and 'EntryPrice' in trades.columns:
        trades['Return'] = trades['PnL'] / trades['EntryPrice'] * 100
    else:
        trades['Return'] = []
    return trades, df


# ----------- Streamlit UI -----------

st.set_page_config(page_title="RMA Universal Backtester", layout="wide")
st.title("📊 RMA Optuna Universal Backtester")

csv_symbol = st.sidebar.text_input("Label for Symbol", value="RGTI")
upload_csv = st.sidebar.file_uploader("Upload CSV (TV export)", type=["csv"])
uploaded_df = None

if upload_csv is not None:
    if 'csv_bytes' not in st.session_state or st.session_state['csv_name'] != upload_csv.name:
        st.session_state['csv_bytes'] = upload_csv.read()
        st.session_state['csv_name'] = upload_csv.name

    buffer = io.BytesIO(st.session_state['csv_bytes'])
    uploaded_df = pd.read_csv(buffer)
    st.write("Rows in uploaded_df (from BytesIO):", len(uploaded_df))

    # COLUMN RENAME FIX
    uploaded_df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "Volume": "Volume",
        "volume": "Volume"
    }, inplace=True)

    uploaded_df["Date"] = pd.to_datetime(uploaded_df["time"], utc=True, errors='coerce').dt.tz_convert(None)
    uploaded_df = uploaded_df[~uploaded_df["Date"].isna()]
    st.write("Date is_unique:", uploaded_df["Date"].is_unique)
    st.write("Date min/max:", uploaded_df["Date"].min(), uploaded_df["Date"].max())

    data_start = uploaded_df["Date"].min().date()
    data_end = uploaded_df["Date"].max().date()
    test_range = st.sidebar.date_input(
        "Select Backtest Period",
        value=(data_start, data_end),
        min_value=data_start,
        max_value=data_end
    )
    mask = (uploaded_df["Date"].dt.date >= test_range[0]) & (uploaded_df["Date"].dt.date <= test_range[1])
    df = uploaded_df.loc[mask]
    st.write("Rows after slicing:", len(df))
    st.write("Date min/max after slicing:", df['Date'].min(), df['Date'].max())
else:
    df = None

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
    session_start = f"{int(session_start_h):02d}:{int(session_start_m):02d}"
    session_end = f"{int(session_end_h):02d}:{int(session_end_m):02d}"


    def objective(trial):
        rma_len = trial.suggest_int('rma_len', 40, 100)
        entry_bars = trial.suggest_int('entry_bars', 1, 10)
        exit_bars = trial.suggest_int('exit_bars', 1, 10)
        ATRLen = trial.suggest_int('ATRLen', 2, 20)
        normalizedUpper = trial.suggest_float('normalizedUpper', 0, 2, step=0.1)
        normalizedLower = trial.suggest_float('normalizedLower', -2, 0, step=0.1)
        emasrc = trial.suggest_categorical('emasrc', [30, 40, 45, 60, 120])
        fastlen = trial.suggest_int('fastlen', 2, 30)
        slowlen = trial.suggest_int('slowlen', 20, 100)
        RiskLen = trial.suggest_float('RiskLen', 0, 50, step=0.5)
        TrailPct = trial.suggest_float('TrailPct', 0, 50, step=0.5)
        trades, _ = rma_strategy(
            df,
            rma_len=rma_len,
            barsForEntry=entry_bars,
            barsForExit=exit_bars,
            ATRLen=ATRLen,
            normalizedUpper=normalizedUpper,
            normalizedLower=normalizedLower,
            ema_fast_len=fastlen,
            ema_slow_len=slowlen,
            RiskLen=RiskLen,
            TrailPct=TrailPct,
            session_start=session_start,
            session_end=session_end,
            keepLime=True,
            initial_capital=15000,
            qty=1000,
            maxtradesperday=maxtradesperday
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

    trades, df_all = rma_strategy(
        df,
        rma_len=best_params['rma_len'],
        barsForEntry=best_params['entry_bars'],
        barsForExit=best_params['exit_bars'],
        ATRLen=best_params['ATRLen'],
        normalizedUpper=best_params['normalizedUpper'],
        normalizedLower=best_params['normalizedLower'],
        ema_fast_len=best_params['fastlen'],
        ema_slow_len=best_params['slowlen'],
        RiskLen=best_params['RiskLen'],
        TrailPct=best_params['TrailPct'],
        session_start=session_start,
        session_end=session_end,
        keepLime=True,
        initial_capital=15000,
        qty=1000,
        maxtradesperday=maxtradesperday
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
    st.info("Upload data and select range, set parameters, and click 'Run Optuna Optimization' to start.")
