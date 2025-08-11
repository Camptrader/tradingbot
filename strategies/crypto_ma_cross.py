# strategies/crypto_ma_cross.py
import pandas as pd
import numpy as np
from helpers import compute_order_qty, empty_trades_df

def ma_cross_strategy(df, fast_len=10, slow_len=30, runtime: dict | None = None):
    """
    Simple MA cross (long-only):
    - Enter long on fast MA crossing above slow MA.
    - Exit on fast MA crossing back below slow MA.
    Position sizing supports 'cash' or 'qty' via helpers.compute_order_qty().
    """
    runtime = runtime or {}

    df = df.copy()

    # Normalize datetime index (handles 'datetime' or 'date' column names)
    time_col = "datetime" if "datetime" in df.columns else "date"
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col).set_index(time_col)

    # MAs
    df["fast_ma"] = df["close"].rolling(int(fast_len), min_periods=1).mean()
    df["slow_ma"] = df["close"].rolling(int(slow_len), min_periods=1).mean()

    trade_log = []
    pos = 0
    entry_price = None
    entry_qty = None

    start_i = max(int(fast_len), int(slow_len))
    for i in range(start_i, len(df)):
        fast_prev = df["fast_ma"].iloc[i - 1]
        slow_prev = df["slow_ma"].iloc[i - 1]
        fast_now = df["fast_ma"].iloc[i]
        slow_now = df["slow_ma"].iloc[i]
        price_now = float(df["close"].iloc[i])
        tstamp = df.index[i]

        # Bullish crossover -> enter long
        if pos == 0 and fast_prev < slow_prev and fast_now > slow_now:
            entry_price = price_now
            entry_qty = compute_order_qty(entry_price, runtime)  # supports cash or qty
            if entry_qty <= 0:
                # skip invalid size
                continue
            pos = 1
            trade_log.append({
                "EntryTime": tstamp,
                "EntryPrice": entry_price,
                "Qty": entry_qty,
                "ExitTime": None,
                "ExitPrice": None,
                "pnl": None,
                "ExitReason": None,
            })

        # Bearish crossover while long -> exit
        elif pos == 1 and fast_prev > slow_prev and fast_now < slow_now:
            exit_price = price_now
            pos = 0
            last = trade_log[-1]
            qty = last.get("Qty", entry_qty or 0.0)
            pnl = (exit_price - (entry_price if entry_price is not None else last["EntryPrice"])) * qty
            last.update({
                "ExitTime": tstamp,
                "ExitPrice": exit_price,
                "pnl": pnl,
                "ExitReason": "Bearish cross",
            })
            entry_price, entry_qty = None, None

    # (Optional) if you want to force-close open positions at the final bar, do it here.
    # This implementation leaves open trades unclosed.

    trades = pd.DataFrame(trade_log)
    if not trades.empty and 'pnl' in trades.columns and 'EntryPrice' in trades.columns:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    else:
        trades = empty_trades_df()  # ✅ always return correct structure
    return trades, df

