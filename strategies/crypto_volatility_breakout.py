# strategies/crypto_volatility_breakout.py
import pandas as pd
from helpers import compute_order_qty, empty_trades_df

def volatility_breakout_strategy(df, atr_len=14, mult=1.5, runtime: dict | None = None):
    """
    Long-only volatility breakout:
      - Breakout level: prior close + ATR * mult
      - Enter when close > breakout_level
      - Exit on close < entry (simple fail-safe)
    Sizing supports 'cash' or 'qty' via helpers.compute_order_qty().
    """
    runtime = runtime or {}
    n = int(atr_len)
    k = float(mult)

    # --- Normalize time/index ---
    df = df.copy()
    time_col = "datetime" if "datetime" in df.columns else "date"
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

    # --- ATR (True Range mean) ---
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(n, min_periods=1).mean()

    # --- Breakout level (use prior close to avoid lookahead) ---
    df["breakout_level"] = df["close"].shift(1) + df["atr"] * k

    pos = 0
    entry_price = None
    entry_qty = None
    trade_log = []

    start_i = n  # wait until ATR window is reasonably populated
    for i in range(start_i, len(df)):
        price = float(df["close"].iloc[i])
        bo_level = float(df["breakout_level"].iloc[i]) if pd.notna(df["breakout_level"].iloc[i]) else None
        ts = df.index[i]

        # ---- Entry: breakout over level ----
        if pos == 0 and bo_level is not None and price > bo_level:
            entry_price = price
            entry_qty = compute_order_qty(entry_price, runtime)  # ✅ cash or qty
            if entry_qty <= 0:
                continue
            pos = 1
            trade_log.append({
                "EntryTime": ts,
                "EntryPrice": entry_price,
                "Qty": float(entry_qty),
                "ExitTime": None,
                "ExitPrice": None,
                "pnl": None,
                "ExitReason": None,
            })

        # ---- Exit: simple fail-safe (close < entry) ----
        elif pos == 1 and entry_price is not None and price < entry_price:
            pos = 0
            last = trade_log[-1]
            qty = float(last.get("Qty", entry_qty or 0.0))
            pnl = (price - entry_price) * qty
            last.update({
                "ExitTime": ts,
                "ExitPrice": price,
                "pnl": pnl,
                "ExitReason": "Reversal below entry",
            })
            entry_price, entry_qty = None, None

    trades = pd.DataFrame(trade_log)
    if not trades.empty and 'pnl' in trades.columns and 'EntryPrice' in trades.columns:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    else:
        trades = empty_trades_df()  # ✅ always return correct structure
    return trades, df
