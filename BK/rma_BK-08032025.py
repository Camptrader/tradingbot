# strategies/rma.py

from datetime import datetime

import numpy as np
import pandas as pd


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
        bar_minutes=3,  # default to 3 for 3m bars, or set from app
        rma_len=50,
        barsforentry=2,
        barsforexit=2,
        atrlen=9,
        normalizedupper=0.001,
        normalizedlower=-0.001,
        emasrc=45,
        ema_fast_len=5,
        ema_slow_len=60,
        risklen=50,
        trailpct=50,
        session_start="13:31",
        session_end="19:52",
        keeplime=True,
        initial_capital=15000,
        qty=1000,
        maxtradesperday=1,
        use_session_end_rule=False
):
    df = df.copy()
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['imp'], df['mi'], df['hi'], df['lo'] = impulse(
        df['hlc3'].values, df['high'].values, df['low'].values, rma_len)
    df['lime'] = df['hlc3'] > df['hi']
    df['coreGreen'] = (df['hlc3'] > df['mi']) & (df['hlc3'] <= df['hi'])
    df['green'] = df['coreGreen'] | df['lime'] if keeplime else df['coreGreen']
    df['upCnt'] = streak_bool(df['green'].values)
    df['dnCnt'] = streak_bool(~df['green'].values)
    df['ATR'] = pd.Series(df['high'] - df['low']).rolling(atrlen).mean()
    df['normalizedImpulse'] = df['imp'] / df['ATR']
    df['impOutside'] = (df['normalizedImpulse'] >= normalizedupper) | (df['normalizedImpulse'] <= normalizedlower)
    # Set your higher timeframe in minutes (emasrc, e.g. 120 for 2h)
    htf_minutes = int(emasrc)

    # Create a pandas offset string for resampling (e.g., '120min')
    htf_offset = f'{htf_minutes}min'

    # Ensure 'date' is datetime and set as index
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.set_index('date')

    # 1. Resample close to higher timeframe (using last value of each interval, like TV bar close)
    htf_close = df['close'].resample(htf_offset, label='right', closed='right').last()

    # 2. Calculate higher timeframe EMAs
    ema_fast_htf = htf_close.ewm(span=ema_fast_len, adjust=False).mean()
    ema_slow_htf = htf_close.ewm(span=ema_slow_len, adjust=False).mean()

    # 3. Re-align to original dataframe (forward-fill so every bar has the most recent HTF EMA)
    df['ema_fast'] = ema_fast_htf.reindex(df.index, method='ffill').values
    df['ema_slow'] = ema_slow_htf.reindex(df.index, method='ffill').values

    # Reset index for rest of your logic
    df = df.reset_index()

    df['mtfTrend'] = df['ema_fast'] > df['ema_slow']
    df['time'] = df["date"].apply(lambda x: x.time() if pd.notnull(x) else None)
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
        day = df["date"].iloc[i].date()
        bar_time = df["date"].iloc[i].time()
        # detect last bar of the day
        is_last_bar_of_day = (i == len(df) - 1) or (df["date"].iloc[i + 1].date() != day)
        if day != current_day:
            current_day = day
            trades_today = 0
            entry_taken = False
        if not df['inSession'].iloc[i] and pos == 0:
            continue
        if (
                pos == 0 and
                not entry_taken and
                trades_today < maxtradesperday and
                start_time <= bar_time < end_time and
                (df['upCnt'].iloc[i] >= barsforentry) and
                df['impOutside'].iloc[i] and
                df['mtfTrend'].iloc[i]
        ):
            if not df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i]:
                print(
                    f"Skipped entry at {df['date'].iloc[i]}: fast={df['ema_fast'].iloc[i]}, slow={df['ema_slow'].iloc[i]}")
                continue
            pos = 1
            entry_price = df['close'].iloc[i]
            peak_price = entry_price
            bars_in_trade = 0
            entry_taken = True
            trades_today += 1
            trade_log.append({
                'EntryTime': df["date"].iloc[i],
                'EntryPrice': entry_price,
                'BarsInTrade': 0,
                'ExitTime': None,
                'ExitPrice': None,
                'pnl': None,
                'ExitReason': None
            })
        is_session_end = (bar_time >= end_time)
        if pos == 1:
            bars_in_trade += 1
            peak_price = max(peak_price, df['high'].iloc[i])
            exit_reason = None
            exit_price = None
            if use_session_end_rule and (is_session_end or is_last_bar_of_day):
                exit_price = df['close'].iloc[i]
                exit_reason = "Session close"
            elif (df['dnCnt'].iloc[i] == barsforexit) or (not df['mtfTrend'].iloc[i]):
                exit_price = df['close'].iloc[i]
                exit_reason = "Exit rule"
            elif df['low'].iloc[i] <= entry_price * (1 - risklen / 100):
                exit_price = entry_price * (1 - risklen / 100)
                exit_reason = "Risk"
            elif df['low'].iloc[i] <= peak_price * (1 - trailpct / 100):
                exit_price = peak_price * (1 - trailpct / 100)
                exit_reason = "Trail"

            if exit_reason is not None:
                pos = 0
                trade_log[-1].update({
                    'ExitTime': df["date"].iloc[i],
                    'ExitPrice': exit_price,
                    'pnl': exit_price - trade_log[-1]['EntryPrice'],
                    'BarsInTrade': bars_in_trade,
                    'ExitReason': exit_reason
                })
    trades = pd.DataFrame(trade_log)
    if not trades.empty and 'pnl' in trades.columns and 'EntryPrice' in trades.columns:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    else:
        trades['return'] = []
    return trades, df
