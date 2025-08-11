# strategies/hybrid_atr_mom_break.py
import numpy as np
import pandas as pd
from helpers import compute_order_qty, empty_trades_df

def hybrid_atr_momentum_breakout(
    df,
    breakout_len=20,   # N-bar breakout lookback
    ema_len=50,        # trend EMA length
    roc_thresh=0.5,    # ROC % threshold
    atr_len=14,        # ATR length
    atr_mult=2.0,      # initial stop = ATR * atr_mult
    runtime: dict | None = None,
):
    """
    Lean Hybrid ATR-Momentum Breakout (long-only, partial + trailing):
      Entry: close > EMA(ema_len) AND close > prior N-bar high AND ROC(10) > roc_thresh
      Stop : initial stop = entry - ATR*atr_mult; after partial, trail at max(init_stop, peak - ATR*1.0)
      Partial: take 50% at +1.5R (R = ATR*atr_mult at entry)
      Session: entries only 12:00–20:00 UTC (exits always allowed)

    Returns:
      trades: DataFrame with one row per completed trade
      df    : working DataFrame with indicators
    """
    rt = runtime or {}

    # ---- Normalize time/index (UTC) ----
    df = df.copy()
    time_col = "datetime" if "datetime" in df.columns else ("date" if "date" in df.columns else None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)
    else:
        df = df.sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    # ---- Indicators (no lookahead) ----
    # Trend
    df["ema"] = df["close"].ewm(span=int(ema_len), adjust=False).mean()

    # Prior N-bar high (shifted)
    n = int(breakout_len)
    df["prior_high"] = df["high"].rolling(n, min_periods=1).max().shift(1)

    # ROC% over fixed 10 bars
    roc_len = 10
    df["roc_pct"] = (df["close"] / df["close"].shift(roc_len) - 1.0) * 100.0

    # ATR (simple TR mean)
    alen = int(atr_len)
    prev_close = df["close"].shift(1)
    tr = np.maximum.reduce([
        (df["high"] - df["low"]).to_numpy(),
        (df["high"] - prev_close).abs().to_numpy(),
        (df["low"] - prev_close).abs().to_numpy(),
    ])
    df["tr"] = tr
    df["atr"] = pd.Series(tr, index=df.index).rolling(alen, min_periods=1).mean()

    # Session filter (entries only between 12:00 and 20:00 UTC)
    # Exits always allowed.
    def _in_entry_session(ts):
        # ts is tz-aware UTC
        hhmm = ts.hour * 100 + ts.minute
        return 1200 <= hhmm <= 2000

    # ---- Params/constants (non-tunable kept constant to avoid bloat) ----
    partial_rr = 1.5
    trail_atr_mult = 1.0

    # ---- Trade loop ----
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
    partial_realized = 0.0  # realized PnL from partials (if any)

    for i in range(start_i, len(df)):
        ts = df.index[i]
        price = float(df["close"].iloc[i])
        ema_now = float(df["ema"].iloc[i])
        phigh = df["prior_high"].iloc[i]
        atr_now = float(df["atr"].iloc[i]) if pd.notna(df["atr"].iloc[i]) else None
        roc_now = df["roc_pct"].iloc[i]

        # Build entry condition
        entry_ok = (
            pd.notna(phigh) and atr_now is not None and
            (price > ema_now) and (price > float(phigh)) and (roc_now > float(roc_thresh))
        )

        # Enforce entry session window
        can_enter_now = _in_entry_session(ts)

        # ---- Entry ----
        if (not in_pos) and entry_ok and can_enter_now:
            entry_price = price
            qty = compute_order_qty(entry_price, rt)
            if qty <= 0:
                continue
            in_pos = True
            qty_total = float(qty)
            qty_rem = float(qty)
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
                # Optional diagnostics:
                "PartialQty": 0.0,
                "PartialPrice": np.nan,
            }

        # ---- Manage/Exit ----
        if in_pos:
            # Update peak
            high_i = float(df["high"].iloc[i])
            peak = max(peak, high_i) if peak is not None else high_i

            # Current ATR / risk levels
            R = R_at_entry if R_at_entry is not None else (atr_now or 0.0) * float(atr_mult)
            init_stop = (entry_price or price) - R
            trail_stop_raw = (peak or price) - (atr_now or 0.0) * trail_atr_mult
            active_stop = max(init_stop, trail_stop_raw) if took_partial else init_stop

            # Partial at +1.5R (use close-based trigger)
            target_price = (entry_price or price) + partial_rr * R
            if (not took_partial) and price >= target_price and qty_rem > 0:
                part_qty = qty_rem * 0.5
                partial_realized += (price - entry_price) * part_qty
                qty_rem -= part_qty
                took_partial = True
                # decorate current trade dict
                trade_log.append({
                    # record a synthetic "event" row for auditing (optional to keep)
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
                # continue managing remaining position

            # Stop exit (close-based approximation)
            exit_now = price <= active_stop or qty_rem <= 1e-12
            if exit_now:
                exit_price = active_stop if price <= active_stop else price
                bars_held = i - entry_idx
                total_pnl = partial_realized + (exit_price - entry_price) * max(0.0, qty_rem)

                # finalize single trade row (aggregate)
                trade_final = {
                    "EntryTime": trade["EntryTime"],
                    "EntryPrice": trade["EntryPrice"],
                    "Qty": trade["Qty"],
                    "ExitTime": ts,
                    "ExitPrice": exit_price,
                    "BarsInTrade": bars_held,
                    "pnl": total_pnl,
                    "ExitReason": "Stop" if price <= active_stop else "Flat",
                    "PartialQty": (trade_log[-1]["PartialQty"] if trade_log and trade_log[-1].get("ExitReason") == "Partial" else 0.0),
                    "PartialPrice": (trade_log[-1]["PartialPrice"] if trade_log and trade_log[-1].get("ExitReason") == "Partial" else np.nan),
                }
                # Remove the last "Partial" event row if present (keep only the final consolidated row)
                if trade_log and trade_log[-1].get("ExitReason") == "Partial":
                    trade_log.pop()
                trade_log.append(trade_final)

                # reset state
                in_pos = False
                entry_price = None
                qty_total = None
                qty_rem = None
                took_partial = False
                peak = None
                entry_idx = None
                R_at_entry = None
                partial_realized = 0.0

    # ---- Build trades df ----
    trades = pd.DataFrame(trade_log)

    # Keep consistent schema with your other strategies
    if not trades.empty and 'pnl' in trades.columns and 'EntryPrice' in trades.columns:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100.0
    else:
        trades = empty_trades_df()

    return trades, df
