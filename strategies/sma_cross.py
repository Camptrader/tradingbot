import pandas as pd


def sma_cross_strategy(
        df,
        fast_len=5,
        slow_len=20,
        initial_capital=15000,
        qty=1000,
        session_start="09:31",
        session_end="15:52",
        maxtradesperday=1,
        **kwargs
):
    df = df.copy()
    df['FastSMA'] = df['close'].rolling(fast_len).mean()
    df['SlowSMA'] = df['close'].rolling(slow_len).mean()
    df['Signal'] = 0

    # Cross up: Fast SMA crosses above Slow SMA
    cross_up = (
            (df['FastSMA'] > df['SlowSMA']) &
            (df['FastSMA'].shift(1) <= df['SlowSMA'].shift(1))
    )
    cross_down = (
            (df['FastSMA'] < df['SlowSMA']) &
            (df['FastSMA'].shift(1) >= df['SlowSMA'].shift(1))
    )
    df.loc[cross_up, 'Signal'] = 1
    df.loc[cross_down, 'Signal'] = -1

    # --- Session, trade logic exactly as before ---
    df['time'] = df["date"].apply(lambda x: x.time() if pd.notnull(x) else None)
    from datetime import datetime
    start_time = datetime.strptime(session_start, "%H:%M").time()
    end_time = datetime.strptime(session_end, "%H:%M").time()
    df['inSession'] = df['time'].apply(lambda t: (t is not None) and (start_time <= t <= end_time))
    trade_log = []
    pos = 0
    entry_price = 0
    entry_idx = None
    current_day = None
    trades_today = 0
    for i in range(len(df)):
        day = df["date"].iloc[i].date()
        bar_time = df["date"].iloc[i].time()
        if day != current_day:
            current_day = day
            trades_today = 0
        if not df['inSession'].iloc[i]:
            continue
        signal = df['Signal'].iloc[i]
        if pos == 0 and signal == 1 and trades_today < maxtradesperday:
            pos = 1
            entry_price = df['close'].iloc[i]
            entry_idx = i
            trade_log.append({
                'EntryTime': df["date"].iloc[i],
                'EntryPrice': entry_price,
                'ExitTime': None,
                'ExitPrice': None,
                'pnl': None,
                'BarsInTrade': 0,
                'ExitReason': None
            })
            trades_today += 1
        elif pos == 1 and signal == -1:
            exit_price = df['close'].iloc[i]
            bars_in_trade = i - entry_idx
            pos = 0
            trade_log[-1].update({
                'ExitTime': df["date"].iloc[i],
                'ExitPrice': exit_price,
                'pnl': exit_price - trade_log[-1]['EntryPrice'],
                'BarsInTrade': bars_in_trade,
                'ExitReason': "Cross under"
            })
        elif pos == 1 and bar_time >= end_time:
            exit_price = df['close'].iloc[i]
            bars_in_trade = i - entry_idx
            pos = 0
            trade_log[-1].update({
                'ExitTime': df["date"].iloc[i],
                'ExitPrice': exit_price,
                'pnl': exit_price - trade_log[-1]['EntryPrice'],
                'BarsInTrade': bars_in_trade,
                'ExitReason': "Session close"
            })
    trades = pd.DataFrame(trade_log)
    if not trades.empty and 'pnl' in trades.columns and 'EntryPrice' in trades.columns:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    else:
        trades['return'] = []
    return trades, df
