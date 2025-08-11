# strategies/crypto_momentum_breakout.py
import pandas as pd
from helpers import compute_order_qty, empty_trades_df

def momentum_breakout_strategy(df, lookback=20, runtime: dict | None = None):
    """
    Long-only momentum breakout:
      - Enter when close > prior N-bar rolling max high.
      - Exit when close falls back below entry price (simple stop-back).
    Sizing supports 'cash' or 'qty' via helpers.compute_order_qty().
    """
    runtime = runtime or {}
    lb = int(lookback)

    # Normalize time column/index
    df = df.copy()
    time_col = "datetime" if "datetime" in df.columns else "date"
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

    # Signal: prior N-bar high (shifted to avoid lookahead)
    df["roll_max_high"] = df["high"].rolling(lb, min_periods=1).max().shift(1)

    trade_log = []
    pos = 0
    entry_price = None
    entry_qty = None
    trades = []

    for i in range(lb, len(df)):
        price = float(df["close"].iloc[i])
        max_high_prev = df["roll_max_high"].iloc[i]
        tstamp = df.index[i]
        # Enter: breakout above prior N-bar high
        if pos == 0 and pd.notna(max_high_prev) and price > float(max_high_prev):
            entry_price = price
            entry_qty = compute_order_qty(entry_price, runtime)  # cash or qty
            if entry_qty <= 0:
                continue
            pos = 1

            trade_log.append({
                "EntryTime": tstamp,
                "EntryPrice": entry_price,
                "Qty": float(entry_qty),
                "ExitTime": None,
                "ExitPrice": None,
                "pnl": None,
                "ExitReason": None,
            })

        # Exit: close back below entry price
        elif pos == 1 and entry_price is not None and price < entry_price:
            pos = 0
            last = trade_log[-1]
            qty = float(last.get("Qty", entry_qty or 0.0))
            pnl = (price - entry_price) * qty
            last.update({
                "ExitTime": tstamp,
                "ExitPrice": price,
                "pnl": pnl,
                "ExitReason": "Close below entry",
            })
            entry_price, entry_qty = None, None

    trades = pd.DataFrame(trade_log)
    if not trades.empty and 'pnl' in trades.columns and 'EntryPrice' in trades.columns:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    else:
        trades = empty_trades_df()  # ✅ always return correct structure
    return trades, df

