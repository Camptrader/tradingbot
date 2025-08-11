# strategies/crypto_volume_price_action.py
import pandas as pd
from helpers import compute_order_qty, empty_trades_df

def volume_price_action_strategy(df, ma_len=20, vol_mult=2, runtime: dict | None = None):
    """
    Volume + price action long-only:
      - Enter when volume > vol_mult * vol_MA AND close > MA
      - Exit when close < MA
    Sizing supports 'cash' or 'qty' via helpers.compute_order_qty().
    """
    runtime = runtime or {}
    m = int(ma_len)
    vmult = float(vol_mult)

    # --- Normalize time/index ---
    df = df.copy()
    time_col = "datetime" if "datetime" in df.columns else "date"
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

    # --- Indicators ---
    df["ma"] = df["close"].rolling(m, min_periods=1).mean()
    df["vol_ma"] = df["volume"].rolling(m, min_periods=1).mean()

    trade_log = []
    pos = 0
    entry_price = None
    entry_qty = None

    for i in range(m, len(df)):
        price = float(df["close"].iloc[i])
        vol = float(df["volume"].iloc[i])
        ma = float(df["ma"].iloc[i])
        vol_ma = float(df["vol_ma"].iloc[i])
        ts = df.index[i]

        # ---- Entry: volume expansion + price above MA ----
        if pos == 0 and vol > vol_ma * vmult and price > ma:
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

        # ---- Exit: price back below MA ----
        elif pos == 1 and price < ma:
            pos = 0
            last = trade_log[-1]
            qty = float(last.get("Qty", entry_qty or 0.0))
            pnl = (price - (entry_price if entry_price is not None else last["EntryPrice"])) * qty
            last.update({
                "ExitTime": ts,
                "ExitPrice": price,
                "pnl": pnl,
                "ExitReason": "Close below MA",
            })
            entry_price, entry_qty = None, None

    trades = pd.DataFrame(trade_log)
    if not trades.empty and 'pnl' in trades.columns and 'EntryPrice' in trades.columns:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    else:
        trades = empty_trades_df()  # ✅ always return correct structure
    return trades, df
