# strategies/pairs_trading.py
import pandas as pd

def pairs_trading_strategy(df1, df2, spread_lookback=30, threshold=1.5, qty=1):
    # df1, df2: DataFrames with 'date' and 'close' for each asset
    df1 = df1.copy()
    df2 = df2.copy()
    df1['date'] = pd.to_datetime(df1['date'], utc=True)
    df2['date'] = pd.to_datetime(df2['date'], utc=True)
    merged = pd.merge(df1[['date', 'close']], df2[['date', 'close']], on='date', suffixes=('_1', '_2'))
    merged = merged.set_index('date')

    merged['spread'] = merged['close_1'] - merged['close_2']
    merged['spread_ma'] = merged['spread'].rolling(spread_lookback, min_periods=1).mean()
    merged['spread_std'] = merged['spread'].rolling(spread_lookback, min_periods=1).std()

    pos = 0  # 0=flat, 1=long spread (buy 1, sell 2), -1=short spread
    entry_price = 0
    trade_log = []

    for i in range(spread_lookback, len(merged)):
        zscore = (merged['spread'].iloc[i] - merged['spread_ma'].iloc[i]) / merged['spread_std'].iloc[i]
        if pos == 0 and zscore > threshold:
            pos = -1
            entry_price = merged['spread'].iloc[i]
            trade_log.append({'EntryTime': merged.index[i], 'Side': 'Short Spread', 'EntrySpread': entry_price,
                              'ExitTime': None, 'ExitSpread': None, 'pnl': None})
        elif pos == 0 and zscore < -threshold:
            pos = 1
            entry_price = merged['spread'].iloc[i]
            trade_log.append({'EntryTime': merged.index[i], 'Side': 'Long Spread', 'EntrySpread': entry_price,
                              'ExitTime': None, 'ExitSpread': None, 'pnl': None})
        elif pos != 0 and abs(zscore) < 0.1:
            exit_spread = merged['spread'].iloc[i]
            trade_log[-1].update({
                'ExitTime': merged.index[i],
                'ExitSpread': exit_spread,
                'pnl': (entry_price - exit_spread) * qty * (1 if pos == -1 else -1)
            })
            pos = 0
    trades = pd.DataFrame(trade_log)
    if not trades.empty:
        trades['return'] = trades['pnl'] / trades['EntrySpread'] * 100
    return trades, merged
