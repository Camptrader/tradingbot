# strategies/ma_cross.py
import pandas as pd

def ma_cross_strategy(df, fast_len=10, slow_len=30, initial_capital=10000, qty=1):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date')
    df = df.set_index('date')

    df['fast_ma'] = df['close'].rolling(fast_len, min_periods=1).mean()
    df['slow_ma'] = df['close'].rolling(slow_len, min_periods=1).mean()
    pos = 0
    entry_price = 0
    trade_log = []

    for i in range(slow_len, len(df)):
        if pos == 0 and df['fast_ma'].iloc[i-1] < df['slow_ma'].iloc[i-1] and df['fast_ma'].iloc[i] > df['slow_ma'].iloc[i]:
            # Bullish cross
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
        elif pos == 1 and df['fast_ma'].iloc[i-1] > df['slow_ma'].iloc[i-1] and df['fast_ma'].iloc[i] < df['slow_ma'].iloc[i]:
            exit_price = df['close'].iloc[i]
            pos = 0
            trade_log[-1].update({
                'ExitTime': df.index[i],
                'ExitPrice': exit_price,
                'pnl': (exit_price - entry_price) * qty,
                'ExitReason': 'Bearish cross'
            })
    trades = pd.DataFrame(trade_log)
    if not trades.empty:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    return trades, df
