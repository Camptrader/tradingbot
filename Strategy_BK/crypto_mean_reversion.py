# strategies/mean_reversion.py
import pandas as pd

def mean_reversion_strategy(df, ma_len=20, threshold=2, initial_capital=10000, qty=1):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date')
    df = df.set_index('date')

    df['ma'] = df['close'].rolling(ma_len, min_periods=1).mean()
    df['ma_std'] = df['close'].rolling(ma_len, min_periods=1).std()
    pos = 0
    entry_price = 0
    trade_log = []

    for i in range(ma_len, len(df)):
        zscore = (df['close'].iloc[i] - df['ma'].iloc[i]) / df['ma_std'].iloc[i]
        if pos == 0 and zscore < -threshold:
            # Enter long
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
        elif pos == 1 and zscore > 0:  # Exit when price reverts above mean
            exit_price = df['close'].iloc[i]
            pos = 0
            trade_log[-1].update({
                'ExitTime': df.index[i],
                'ExitPrice': exit_price,
                'pnl': (exit_price - entry_price) * qty,
                'ExitReason': 'Reverted above mean'
            })
    trades = pd.DataFrame(trade_log)
    if not trades.empty:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    return trades, df
