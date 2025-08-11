from datetime import datetime
import numpy as np
import pandas as pd
from helpers import compute_order_qty, empty_trades_df




def tradingview_htf_ema(df, price_col, ema_len, htf_minutes):
    """
    TV-matching higher timeframe EMA 'broadcasted' to every LTF bar.
    - df: DataFrame with datetime index (UTC, no missing bars)
    - price_col: column to use for price (e.g. 'close')
    - ema_len: int, length for EMA
    - htf_minutes: int, higher timeframe in minutes (e.g., 120 for 2h)
    Returns: Series aligned to df.index
    """
    htf_str = f'{int(htf_minutes)}min'
    htf_closes = df[price_col].resample(htf_str, label='right', closed='right').last()
    htf_ema = htf_closes.ewm(span=ema_len, adjust=False).mean()
    return htf_ema.reindex(df.index, method='ffill')

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
        rma_len=50,
        barsforentry=2,
        barsforexit=2,
        atrlen=9,
        normalizedupper=0,
        normalizedlower=0,
        ema_fast_len=5,
        ema_slow_len=60,
        trailpct=50,
        keeplime=True,
        runtime: dict | None = None,
):
    runtime = dict(runtime or {})
      # runtime-sourced controls
    session_start = runtime.get("session_start", "13:31")
    session_end = runtime.get("session_end", "19:52")
    maxtradesperday = int(runtime.get("maxtradesperday", 1))
    use_session_end_rule = bool(runtime.get("use_session_end_rule", False))
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date')
    df = df.set_index('date')

    # --- TV-matching multi-timeframe EMAs
    df['ema_fast'] = df['close'].ewm(span=ema_fast_len, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow_len, adjust=False).mean()

    # --- Reset index for normal integer access
    df = df.reset_index()

    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['imp'], df['mi'], df['hi'], df['lo'] = impulse(
        df['hlc3'].values, df['high'].values, df['low'].values, rma_len)
    df['lime'] = df['hlc3'] > df['hi']
    df['coreGreen'] = (df['hlc3'] > df['mi']) & (df['hlc3'] <= df['hi'])
    df['green'] = df['coreGreen'] | df['lime'] if keeplime else df['coreGreen']
    df['upCnt'] = streak_bool(df['green'].values)
    df['dnCnt'] = streak_bool(~df['green'].values)
    tr = np.maximum(df['high'] - df['low'],
                    np.abs(df['high'] - df['close'].shift(1)),
                    np.abs(df['low'] - df['close'].shift(1)))
    atrlen = max(1, int(atrlen))
    df['ATR'] = pd.Series(tr).rolling(atrlen).mean()
    df['normalizedImpulse'] = df['imp'] / df['ATR']
    df['impOutside'] = (df['normalizedImpulse'] >= normalizedupper) | (df['normalizedImpulse'] <= normalizedlower)
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
            pos = 1
            entry_price = df['close'].iloc[i]
            peak_price = entry_price
            bars_in_trade = 0
            entry_taken = True
            trades_today += 1
            qty_filled = compute_order_qty(df['close'].iloc[i], runtime)  # NEW
            trade_log.append({
                'EntryTime': df["date"].iloc[i],
                'EntryPrice': entry_price,
                'EntryFast': df['ema_fast'].iloc[i],
                'EntrySlow': df['ema_slow'].iloc[i],
                'Qty': qty_filled,  # NEW
                'BarsInTrade': 0,
                'ExitTime': None,
                'ExitPrice': None,
                'ExitFast': None,
                'ExitSlow': None,
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
            elif df['low'].iloc[i] <= peak_price * (1 - trailpct / 100):
                exit_price = peak_price * (1 - trailpct / 100)
                exit_reason = "Trail"

            if exit_reason is not None:
                pos = 0
                trade_log[-1].update({
                    'ExitTime': df["date"].iloc[i],
                    'ExitPrice': exit_price,
                    'ExitFast': df['ema_fast'].iloc[i],
                    'ExitSlow': df['ema_slow'].iloc[i],
                    'pnl': (exit_price - trade_log[-1]['EntryPrice']) * trade_log[-1]['Qty'],  # FIX
                    'BarsInTrade': bars_in_trade,
                    'ExitReason': exit_reason
                })
    trades = pd.DataFrame(trade_log)
    if not trades.empty and 'pnl' in trades.columns and 'EntryPrice' in trades.columns:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    else:
        trades = empty_trades_df()  # ✅ always return correct structure
    return trades, df
