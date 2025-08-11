# strategies/volume_price_action.py
import pandas as pd

def volume_price_action_strategy(df, ma_len=20, vol_mult=2, initial_capital=10000, qty=1):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date')
    df = df.set_index('date')

    df['ma'] = df['close'].rolling(ma_len, min_periods=1).mean()
    df['vol_ma'] = df['volume'].rolling(ma_len, min_periods=1).mean()
    pos = 0
    entry_price = 0
    trade_log = []

    for i in range(ma_len, len(df)):
        if pos == 0 and df['volume'].iloc[i] > df['vol_ma'].iloc[i] * vol_mult and df['close'].iloc[i] > df['ma'].iloc[i]:
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
        elif pos == 1 and df['close'].iloc[i] < df['ma'].iloc[i]:
            exit_price = df['close'].iloc[i]
            pos = 0
            trade_log[-1].update({
                'ExitTime': df.index[i],
                'ExitPrice': exit_price,
                'pnl': (exit_price - entry_price) * qty,
                'ExitReason': 'Close below MA'
            })
    trades = pd.DataFrame(trade_log)
    if not trades.empty:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    return trades, df
