# strategies/crypto_mean_reversion.py
import pandas as pd
from helpers import compute_order_qty, empty_trades_df

def mean_reversion_strategy(df, ma_len=20, threshold=2, runtime: dict | None = None):
    """
    Long-only mean reversion:
      - Enter when z-score < -threshold
      - Exit when z-score > 0 (revert above mean)
    Sizing supports 'cash' or 'qty' via helpers.compute_order_qty().
    """
    runtime = runtime or {}
    df = df.copy()

    # Normalize time column
    time_col = "datetime" if "datetime" in df.columns else "date"
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

    # Rolling stats
    mlen = int(ma_len)
    df["ma"] = df["close"].rolling(mlen, min_periods=1).mean()
    df["ma_std"] = df["close"].rolling(mlen, min_periods=1).std()

    trade_log = []
    pos = 0
    entry_price = None
    entry_qty = None
    trades = []

    for i in range(mlen, len(df)):
        price = float(df["close"].iloc[i])
        mu = df["ma"].iloc[i]
        sig = df["ma_std"].iloc[i]
        tstamp = df.index[i]
        # Guard: no signal if std is 0 or NaN
        if not pd.notna(sig) or sig == 0:
            continue

        z = (price - mu) / sig
        ts = df.index[i]

        # Enter long when z < -threshold
        if pos == 0 and z < -float(threshold):
            entry_price = price
            entry_qty = compute_order_qty(entry_price, runtime)  # cash or qty
            if entry_qty <= 0:
                continue  # skip zero/invalid sizing
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

        # Exit when z > 0 (reversion)
        elif pos == 1 and z > 0:
            exit_price = price
            pos = 0
            last = trade_log[-1]
            qty = float(last.get("Qty", entry_qty or 0.0))
            pnl = (exit_price - (entry_price if entry_price is not None else last["EntryPrice"])) * qty
            last.update({
                "ExitTime": tstamp,
                "ExitPrice": exit_price,
                "pnl": pnl,
                "ExitReason": "Reverted above mean",
            })
            entry_price, entry_qty = None, None

    # Build trades DF
    trades = pd.DataFrame(trade_log)
    if not trades.empty and 'pnl' in trades.columns and 'EntryPrice' in trades.columns:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    else:
        trades = empty_trades_df()  # ✅ always return correct structure
    return trades, df
