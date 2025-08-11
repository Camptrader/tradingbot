import pandas as pd
from helpers import compute_order_qty, empty_trades_df

def sma_cross_strategy(
        df,
        fast_len=5,
        slow_len=20,
        session_start="09:31",
        session_end="15:52",
        maxtradesperday=1,
        runtime: dict | None = None,
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
            trades_today += 1
        elif pos == 1 and signal == -1:
            exit_price = df['close'].iloc[i]
            bars_in_trade = i - entry_idx
            pos = 0
            trade_log[-1].update({
                'ExitTime': df["date"].iloc[i],
                'ExitPrice': exit_price,
                'ExitFast': df['ema_fast'].iloc[i],
                'ExitSlow': df['ema_slow'].iloc[i],
                'pnl': (exit_price - trade_log[-1]['EntryPrice']) * trade_log[-1]['Qty'],  # FIX
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
                'ExitFast': df['ema_fast'].iloc[i],
                'ExitSlow': df['ema_slow'].iloc[i],
                'pnl': (exit_price - trade_log[-1]['EntryPrice']) * trade_log[-1]['Qty'],  # FIX
                'BarsInTrade': bars_in_trade,
                'ExitReason': "Session close"
            })
    trades = pd.DataFrame(trade_log)
    if not trades.empty and 'pnl' in trades.columns and 'EntryPrice' in trades.columns:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    else:
        trades = empty_trades_df()  # ✅ always return correct structure
    return trades, df

