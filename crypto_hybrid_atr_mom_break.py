import numpy as np
import pandas as pd
from helpers import compute_order_qty, empty_trades_df


def _extract_qty_from_kwargs(kwargs) -> float:
    """
    Accepts either:
      1) qty=<number> passed directly (preferred), or
      2) runtime={"qty": <number>} for backward-compat.
    Falls back to 1.0 if nothing provided.
    """
    if "qty" in kwargs and kwargs["qty"] is not None:
        try:
            return float(kwargs["qty"])
        except Exception:
            pass
    rt = kwargs.get("runtime")
    if isinstance(rt, dict) and "qty" in rt and rt["qty"] is not None:
        try:
            return float(rt["qty"])
        except Exception:
            pass
    return 1.0


def hybrid_atr_momentum_breakout(
        df,
        breakout_len=20,
        ema_len=50,
        roc_thresh=0.5,
        atr_len=14,
        atr_mult=2.0,
        qty=None,  # NEW: direct qty param
        runtime: dict | None = None,
        **kwargs
):
    # unified qty handling
    if qty is None:
        qty = _extract_qty_from_kwargs({**kwargs, "runtime": runtime, "qty": qty})

    """
    Lean Hybrid ATR-Momentum Breakout (long-only, partial + trailing)
    """
    rt = runtime or {}

    df = df.copy()
    time_col = "datetime" if "datetime" in df.columns else ("date" if "date" in df.columns else None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)
    else:
        df = df.sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    df["ema"] = df["close"].ewm(span=int(ema_len), adjust=False).mean()
    n = int(breakout_len)
    df["prior_high"] = df["high"].rolling(n, min_periods=1).max().shift(1)
    roc_len = 10
    df["roc_pct"] = (df["close"] / df["close"].shift(roc_len) - 1.0) * 100.0
    alen = int(atr_len)
    prev_close = df["close"].shift(1)
    tr = np.maximum.reduce([
        (df["high"] - df["low"]).to_numpy(),
        (df["high"] - prev_close).abs().to_numpy(),
        (df["low"] - prev_close).abs().to_numpy(),
    ])
    df["tr"] = tr
    df["atr"] = pd.Series(tr, index=df.index).rolling(alen, min_periods=1).mean()

    def _in_entry_session(ts):
        hhmm = ts.hour * 100 + ts.minute
        return 1200 <= hhmm <= 2000

    partial_rr = 1.5
    trail_atr_mult = 1.0

    start_i = max(n, alen, ema_len, roc_len)
    trade_log = []

    in_pos = False
    entry_price = None
    qty_total = None
    qty_rem = None
    took_partial = False
    peak = None
    entry_idx = None
    R_at_entry = None
    partial_realized = 0.0

    for i in range(start_i, len(df)):
        ts = df.index[i]
        price = float(df["close"].iloc[i])
        ema_now = float(df["ema"].iloc[i])
        phigh = df["prior_high"].iloc[i]
        atr_now = float(df["atr"].iloc[i]) if pd.notna(df["atr"].iloc[i]) else None
        roc_now = df["roc_pct"].iloc[i]

        entry_ok = (
                pd.notna(phigh) and atr_now is not None and
                (price > ema_now) and (price > float(phigh)) and (roc_now > float(roc_thresh))
        )
        can_enter_now = _in_entry_session(ts)

        if (not in_pos) and entry_ok and can_enter_now:
            entry_price = price
            order_qty = qty if qty is not None else compute_order_qty(entry_price, rt)
            if order_qty <= 0:
                continue
            in_pos = True
            qty_total = float(order_qty)
            qty_rem = float(order_qty)
            took_partial = False
            peak = float(df["high"].iloc[i])
            entry_idx = i
            R_at_entry = (atr_now or 0.0) * float(atr_mult)
            partial_realized = 0.0

            trade = {
                "EntryTime": ts,
                "EntryPrice": entry_price,
                "Qty": qty_total,
                "ExitTime": None,
                "ExitPrice": None,
                "BarsInTrade": None,
                "pnl": None,
                "ExitReason": None,
                "PartialQty": 0.0,
                "PartialPrice": np.nan,
            }

        if in_pos:
            high_i = float(df["high"].iloc[i])
            peak = max(peak, high_i) if peak is not None else high_i

            R = R_at_entry if R_at_entry is not None else (atr_now or 0.0) * float(atr_mult)
            init_stop = (entry_price or price) - R
            trail_stop_raw = (peak or price) - (atr_now or 0.0) * trail_atr_mult
            active_stop = max(init_stop, trail_stop_raw) if took_partial else init_stop

            target_price = (entry_price or price) + partial_rr * R
            if (not took_partial) and price >= target_price and qty_rem > 0:
                part_qty = qty_rem * 0.5
                partial_realized += (price - entry_price) * part_qty
                qty_rem -= part_qty
                took_partial = True
                trade_log.append({
                    "EntryTime": trade["EntryTime"],
                    "EntryPrice": trade["EntryPrice"],
                    "Qty": trade["Qty"],
                    "ExitTime": ts,
                    "ExitPrice": price,
                    "BarsInTrade": (i - entry_idx),
                    "pnl": (price - entry_price) * part_qty,
                    "ExitReason": "Partial",
                    "PartialQty": part_qty,
                    "PartialPrice": price,
                })

            exit_now = price <= active_stop or qty_rem <= 1e-12
            if exit_now:
                exit_price = active_stop if price <= active_stop else price
                bars_held = i - entry_idx
                total_pnl = partial_realized + (exit_price - entry_price) * max(0.0, qty_rem)

                trade_final = {
                    "EntryTime": trade["EntryTime"],
                    "EntryPrice": trade["EntryPrice"],
                    "Qty": trade["Qty"],
                    "ExitTime": ts,
                    "ExitPrice": exit_price,
                    "BarsInTrade": bars_held,
                    "pnl": total_pnl,
                    "ExitReason": "Stop" if price <= active_stop else "Flat",
                    "PartialQty": (trade_log[-1]["PartialQty"] if trade_log and trade_log[-1].get(
                        "ExitReason") == "Partial" else 0.0),
                    "PartialPrice": (trade_log[-1]["PartialPrice"] if trade_log and trade_log[-1].get(
                        "ExitReason") == "Partial" else np.nan),
                }
                if trade_log and trade_log[-1].get("ExitReason") == "Partial":
                    trade_log.pop()
                trade_log.append(trade_final)

                in_pos = False
                entry_price = None
                qty_total = None
                qty_rem = None
                took_partial = False
                peak = None
                entry_idx = None
                R_at_entry = None
                partial_realized = 0.0

    trades = pd.DataFrame(trade_log)
    if not trades.empty and 'pnl' in trades.columns and 'EntryPrice' in trades.columns:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100.0
    else:
        trades = empty_trades_df()

    return trades, df
