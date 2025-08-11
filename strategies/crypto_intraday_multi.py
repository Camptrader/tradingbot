import numpy as np
import pandas as pd
from helpers import compute_order_qty, empty_trades_df

def crypto_intraday_multi(
    df,
    breakout_len=20,         # Rolling high lookback
    momentum_len=10,         # Rolling ROC lookback
    momentum_thresh=0.5,     # ROC threshold for entry
    trend_len=50,            # EMA length for trend filter
    atr_len=14,              # ATR length for volatility filter
    min_atr=0.5,             # Minimum ATR as % of close
    trailing_stop_pct=2.0,   # Trailing stop (%)
    max_hold_bars=3000,      # Maximum bars to hold
    runtime: dict | None = None,
):
    rt = runtime or {}
    maxtradesperday = int(rt.get("maxtradesperday", 20))

    # --- Normalize time/index ---
    df = df.copy()
    time_col = "datetime" if "datetime" in df.columns else ("date" if "date" in df.columns else None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)
    else:
        # assume index is already datetime-like
        df = df.sort_index()

    # --- Signals/filters ---
    # 1) Breakout: close > prior N-bar high
    df["breakout"] = df["close"] > df["close"].rolling(int(breakout_len)).max().shift(1)

    # 2) Momentum: ROC over momentum_len
    df["roc"] = 100.0 * (df["close"] / df["close"].shift(int(momentum_len)) - 1.0)
    df["momentum"] = df["roc"] > float(momentum_thresh)

    # 3) Trend: EMA filter
    df["ema"] = df["close"].ewm(span=int(trend_len), adjust=False).mean()
    df["trend"] = df["close"] > df["ema"]

    # 4) Volatility: ATR% >= min_atr
    prev_close = df["close"].shift(1)
    tr = np.maximum.reduce([
        (df["high"] - df["low"]).to_numpy(),
        (df["close"] - prev_close).abs().to_numpy(),
        (df["low"] - prev_close).abs().to_numpy(),
    ])
    df["tr"] = tr
    df["atr"] = pd.Series(tr, index=df.index).rolling(int(atr_len)).mean()
    df["atr_pct"] = 100.0 * df["atr"] / df["close"]
    df["vol_ok"] = df["atr_pct"] >= float(min_atr)

    # Composite entry
    df["entry_signal"] = df["breakout"] & df["momentum"] & df["trend"] & df["vol_ok"]

    trade_log = []
    in_trade = False
    entry_price = None
    entry_qty = None
    entry_idx = None
    peak = None
    trades_today = 0
    last_trade_day = None

    for i in range(len(df)):
        ts = df.index[i]
        day = ts.date()

        if day != last_trade_day:
            trades_today = 0
            last_trade_day = day

        # ENTRY
        if (not in_trade) and bool(df["entry_signal"].iloc[i]) and trades_today < maxtradesperday:
            entry_price = float(df["close"].iloc[i])
            entry_qty = compute_order_qty(entry_price, rt)  # ✅ cash or qty
            if entry_qty <= 0:
                continue  # skip invalid sizing
            in_trade = True
            entry_idx = i
            peak = entry_price
            trades_today += 1
            trade = {
                "EntryTime": ts,
                "EntryPrice": entry_price,
                "Qty": float(entry_qty),
                "ExitTime": None,
                "ExitPrice": None,
                "BarsInTrade": None,
                "pnl": None,
                "ExitReason": None,
            }

        # MANAGE/EXIT
        if in_trade:
            bars_held = i - entry_idx
            peak = max(peak, float(df["high"].iloc[i])) if peak is not None else float(df["high"].iloc[i])

            exit_trade = False
            # 1) Trailing stop
            trail_level = peak * (1.0 - float(trailing_stop_pct) / 100.0)
            if float(df["low"].iloc[i]) <= trail_level:
                exit_price = trail_level
                reason = "Trailing Stop"
                exit_trade = True

            # 2) Max hold
            elif bars_held >= int(max_hold_bars):
                exit_price = float(df["close"].iloc[i])
                reason = "Max Hold"
                exit_trade = True

            if exit_trade:
                in_trade = False
                trade.update({
                    "ExitTime": ts,
                    "ExitPrice": exit_price,
                    "BarsInTrade": bars_held,
                    "pnl": (exit_price - trade["EntryPrice"]) * trade["Qty"],
                    "ExitReason": reason,
                })
                trade_log.append(trade)
                # reset state
                entry_price = entry_qty = entry_idx = peak = None

    trades = pd.DataFrame(trade_log)
    if not trades.empty and 'pnl' in trades.columns and 'EntryPrice' in trades.columns:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    else:
        trades = empty_trades_df()  # ✅ always return correct structure
    return trades, df

