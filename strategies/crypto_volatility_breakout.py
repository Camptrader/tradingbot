# strategies/volatility_breakout.py
import pandas as pd

def volatility_breakout_strategy(df, atr_len=14, mult=1.5, initial_capital=10000, qty=1):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date')
    df = df.set_index('date')

    df['atr'] = df['high'].rolling(atr_len).max() - df['low'].rolling(atr_len).min()
    df['breakout_level'] = df['close'].shift(1) + df['atr'] * mult
    pos = 0
    entry_price = 0
    trade_log = []

    for i in range(atr_len, len(df)):
        if pos == 0 and df['close'].iloc[i] > df['breakout_level'].iloc[i]:
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
        elif pos == 1 and df['close'].iloc[i] < entry_price:
            exit_price = df['close'].iloc[i]
            pos = 0
            trade_log[-1].update({
                'ExitTime': df.index[i],
                'ExitPrice': exit_price,
                'pnl': (exit_price - entry_price) * qty,
                'ExitReason': 'Reversal below entry'
            })
    trades = pd.DataFrame(trade_log)
    if not trades.empty:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    return trades, df
