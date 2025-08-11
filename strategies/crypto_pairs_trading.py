# strategies/crypto_pairs_trading.py
import pandas as pd
from helpers import compute_order_qty, empty_trades_df

def pairs_trading_strategy(
    df1,
    df2,
    spread_lookback=30,
    threshold=1.5,
    runtime: dict | None = None,
):
    """
    Simple spread pairs trading:
      - Build spread = close_1 - close_2
      - z-score over rolling lookback
      - Enter short-spread when z > +threshold  (sell leg1, buy leg2)
      - Enter long-spread  when z < -threshold (buy leg1, sell leg2)
      - Exit when |z| < 0.1

    Sizing:
      - 'cash' mode: treat runtime['cash'] as total cash per trade, split across both legs.
      - 'qty'  mode: same qty on both legs (units per leg).
    """
    rt = runtime or {}

    # ---- Normalize time columns, align both series ----
    def _prep(df):
        df = df.copy()
        tcol = "datetime" if "datetime" in df.columns else "date"
        df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        df = df.dropna(subset=[tcol]).sort_values(tcol)
        return df[[tcol, "close"]].rename(columns={"close": "close"})

    a = _prep(df1).rename(columns={"close": "close_1"})
    b = _prep(df2).rename(columns={"close": "close_2"})
    merged = pd.merge(a, b, on=("datetime" if "datetime" in a.columns else "date"), how="inner")
    tcol = "datetime" if "datetime" in merged.columns else "date"
    merged = merged.set_index(tcol)

    # ---- Spread & z-score ----
    lb = int(spread_lookback)
    merged["spread"] = merged["close_1"] - merged["close_2"]
    merged["spread_ma"] = merged["spread"].rolling(lb, min_periods=1).mean()
    merged["spread_std"] = merged["spread"].rolling(lb, min_periods=1).std()

    trade_log = []
    pos = 0            # 0=flat, +1=long spread (buy leg1, sell leg2), -1=short spread
    entry1 = entry2 = None
    qty_leg = None

    for i in range(lb, len(merged)):
        s = merged["spread"].iloc[i]
        mu = merged["spread_ma"].iloc[i]
        sd = merged["spread_std"].iloc[i]
        if not pd.notna(sd) or sd == 0:
            continue

        z = (s - mu) / sd
        ts = merged.index[i]
        p1 = float(merged["close_1"].iloc[i])
        p2 = float(merged["close_2"].iloc[i])

        # ---- Entries ----
        if pos == 0 and z > float(threshold):
            # Short spread: sell leg1, buy leg2
            # Sizing: cash mode splits across both legs; qty mode uses fixed qty per leg
            if (rt.get("sizing_mode", "").lower() == "cash") or (rt.get("sizing_mode", "") == "" and float(rt.get("cash", 0)) > 0):
                cash = float(rt.get("cash", 0.0))
                # Split cash evenly per leg; choose qty s.t. both legs are fundable
                qty1 = (cash / 2.0) / p1 if p1 > 0 else 0.0
                qty2 = (cash / 2.0) / p2 if p2 > 0 else 0.0
                qty_leg = max(0.0, min(qty1, qty2))
            else:
                qty_leg = float(rt.get("qty", 1.0))

            if qty_leg <= 0:
                continue
            pos = -1
            entry1, entry2 = p1, p2
            trade_log.append({
                "EntryTime": ts,
                "Side": "Short Spread",
                "EntryPrice1": entry1,
                "EntryPrice2": entry2,
                "QtyPerLeg": qty_leg,
                "ExitTime": None,
                "ExitPrice1": None,
                "ExitPrice2": None,
                "pnl": None,
                "ExitReason": None,
            })

        elif pos == 0 and z < -float(threshold):
            # Long spread: buy leg1, sell leg2
            if (rt.get("sizing_mode", "").lower() == "cash") or (rt.get("sizing_mode", "") == "" and float(rt.get("cash", 0)) > 0):
                cash = float(rt.get("cash", 0.0))
                qty1 = (cash / 2.0) / p1 if p1 > 0 else 0.0
                qty2 = (cash / 2.0) / p2 if p2 > 0 else 0.0
                qty_leg = max(0.0, min(qty1, qty2))
            else:
                qty_leg = float(rt.get("qty", 1.0))

            if qty_leg <= 0:
                continue
            pos = +1
            entry1, entry2 = p1, p2
            trade_log.append({
                "EntryTime": ts,
                "Side": "Long Spread",
                "EntryPrice1": entry1,
                "EntryPrice2": entry2,
                "QtyPerLeg": qty_leg,
                "ExitTime": None,
                "ExitPrice1": None,
                "ExitPrice2": None,
                "pnl": None,
                "ExitReason": None,
            })

        # ---- Exit: z reverts close to 0 ----
        elif pos != 0 and abs(z) < 0.1:
            exit1 = p1
            exit2 = p2
            last = trade_log[-1]
            qty = float(last["QtyPerLeg"])

            # PnL by legs, not approximate spread*qty
            if pos == +1:   # long spread: +leg1, -leg2
                pnl = (exit1 - entry1) * qty - (exit2 - entry2) * qty
                reason = "Reversion"
            else:           # short spread: -leg1, +leg2
                pnl = -(exit1 - entry1) * qty + (exit2 - entry2) * qty
                reason = "Reversion"

            last.update({
                "ExitTime": ts,
                "ExitPrice1": exit1,
                "ExitPrice2": exit2,
                "pnl": pnl,
                "ExitReason": reason,
            })

            # reset
            pos = 0
            entry1 = entry2 = None
            qty_leg = None

    trades_df = pd.DataFrame(trade_log)

    # Return %: use notional deployed at entry
    if not trades_df.empty:
        # If cash mode, use runtime cash; else approximate with (entry1+entry2)*qty
        if (rt.get("sizing_mode", "").lower() == "cash") or (rt.get("sizing_mode", "") == "" and float(rt.get("cash", 0)) > 0):
            notional = float(rt.get("cash", 0.0))
            trades_df["return_pct"] = trades_df["pnl"] / (notional if notional > 0 else pd.NA) * 100.0
        else:
            trades = empty_trades_df()  # ✅ always return correct structure

    return trades_df, merged
