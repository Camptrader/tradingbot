# strategies/momentum_breakout.py
import pandas as pd

def momentum_breakout_strategy(df, lookback=20, initial_capital=10000, qty=1):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date')
    df = df.set_index('date')

    df['roll_max_high'] = df['high'].rolling(lookback, min_periods=1).max().shift(1)
    pos = 0
    entry_price = 0
    trade_log = []

    for i in range(lookback, len(df)):
        if pos == 0 and df['close'].iloc[i] > df['roll_max_high'].iloc[i]:
            # Breakout long
            pos = 1
            entry_price = df['close'].iloc[i]
            trade_log.append({
                'EntryTime': df.index[i],
                'EntryPrice': entry_price,
                'ExitTime': None,
                'ExitPrice': None,
                'pnl': None,
                'ExitReason': None
            })
        elif pos == 1 and df['close'].iloc[i] < entry_price:  # Simple exit: close below entry
            exit_price = df['close'].iloc[i]
            pos = 0
            trade_log[-1].update({
                'ExitTime': df.index[i],
                'ExitPrice': exit_price,
                'pnl': (exit_price - entry_price) * qty,
                'ExitReason': 'Close below entry'
            })
    trades = pd.DataFrame(trade_log)
    if not trades.empty:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    return trades, df
