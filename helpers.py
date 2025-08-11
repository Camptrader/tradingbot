# helpers.py
import json, os, math
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass

PARAM_REGISTRY_FILE = "best_params_registry.json"

TRADE_COLUMNS = [
    "EntryTime", "EntryPrice", "EntryFast", "EntrySlow",
    "Qty", "BarsInTrade",
    "ExitTime", "ExitPrice", "ExitFast", "ExitSlow",
    "pnl", "ExitReason", "return"
]

def empty_trades_df():
    """Return an empty trades DataFrame with the correct columns and dtypes."""
    df = pd.DataFrame({col: pd.Series(dtype=object) for col in TRADE_COLUMNS})
    # Ensure numeric columns have numeric dtype
    for col in ["EntryPrice", "EntryFast", "EntrySlow", "Qty", "BarsInTrade", "ExitPrice", "ExitFast", "ExitSlow", "pnl", "return"]:
        df[col] = pd.Series(dtype=float)
    return df

def load_param_registry() -> Dict[str, Any]:
    if os.path.exists(PARAM_REGISTRY_FILE):
        with open(PARAM_REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_param_registry(registry: Dict[str, Any]) -> None:
    with open(PARAM_REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)

def get_saved_params(strategy: str, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the most recent saved params for a given strategy, symbol, and timeframe
    from best_params_registry.json (flat key format).
    """
    registry = load_param_registry()
    key = f"{strategy}|{symbol}|{timeframe}"
    if key in registry and isinstance(registry[key], list) and registry[key]:
        return registry[key][0]  # Most recent/best is first in the list
    return None

def register_params(strategy: str, symbol: str, timeframe: str,
                    params: Dict[str, Any], results: Optional[Dict[str, Any]] = None) -> None:
    """
    Append a new params/results entry to best_params_registry.json for a given
    strategy, symbol, and timeframe.
    """
    registry = load_param_registry()
    key = f"{strategy}|{symbol}|{timeframe}"
    entry = {"params": params, "results": results}
    if key not in registry:
        registry[key] = []
    registry[key].insert(0, entry)  # Keep most recent first
    save_param_registry(registry)

def tf_to_minutes(tf: str) -> int:
    tf = str(tf).lower()
    if tf.endswith('m'): return int(tf[:-1])
    if tf.endswith('h'): return int(tf[:-1]) * 60
    if tf.endswith('d'): return int(tf[:-1]) * 60 * 24
    return 1

def sanitize_initial_params(param_space: Dict[str, Any], initial_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not initial_params:
        return None
    cleaned = {}
    for k, v in initial_params.items():
        if k not in param_space:
            continue
        space = param_space[k]
        if isinstance(space, list):
            cleaned[k] = v if v in space else space[0]
        else:
            lo, hi, step = space
            if isinstance(step, int):
                try: vi = int(round(v))
                except Exception: vi = lo
                cleaned[k] = max(lo, min(hi, vi))
            else:
                try: vf = float(v)
                except Exception: vf = lo
                cleaned[k] = max(lo, min(hi, vf))
    return cleaned or None

def json_to_tv_preset(param_registry_path: str = PARAM_REGISTRY_FILE) -> str:
    """
    Generate Pine Script 'else if' preset code from best_params_registry.json
    (new flat-key format with lists of {params, metrics}).
    """
    registry = load_param_registry()  # This now loads best_params_registry.json
    out_lines = []
    param_map = {
        "maxtradesperday": "_maxOrders",
        "rma_len": "_len",
        "barsforentry": "_barsForEntry",
        "barsforexit": "_barsForExit",
        "atrlen": "_ATRLen",
        "normalizedupper": "_normalizedUpper",
        "normalizedlower": "_normalizedLower",
        "ema_fast_len": "_fastlen",
        "ema_slow_len": "_slowlen",
        "trailpct": "_TrailPct",
        # session_* handled in your Pine side typically
    }

    for flat_key, records in registry.items():
        # flat_key is "Strategy|Symbol|Timeframe"
        try:
            strategy, symbol, tf = flat_key.split("|", 2)
        except ValueError:
            # In case older format keys are still in file
            continue

        if not isinstance(records, list) or not records:
            continue

        # Take the first/best record
        params = records[0].get("params", {})
        out_lines.append(f'else if symbol == "{symbol}" and timeframe == "{tf}"')
        for k, v in params.items():
            if k in ("maxtradesperday", "session_start", "session_end"):
                continue
            if k in param_map:
                out_lines.append(f"    {param_map[k]} := {v}")

    return "\n".join(out_lines)


def get_all_alpaca_crypto_pairs(api_key: str, api_secret: str):
    # Decorate with st.cache_data in app.py to avoid hard dependency here
    client = TradingClient(api_key, api_secret, paper=True)
    all_assets = client.get_all_assets()
    pairs = [a.symbol for a in all_assets if getattr(a, "asset_class", None) == AssetClass.CRYPTO and a.status == "active"]
    return sorted(pairs)

# helpers.py  (add near the top-level utils)


def compute_order_qty(entry_price: float, runtime: dict, lot_step: float = 0.000001) -> float:
    """
    Compute order size from runtime using one of two modes:
      - sizing_mode == 'cash'  -> qty = cash / entry_price
      - sizing_mode == 'qty'   -> qty = qty
    If sizing_mode is missing: prefer 'cash' if cash>0, else 'qty'.
    """
    entry_price = float(entry_price) if entry_price is not None else 0.0
    mode = (runtime.get("sizing_mode") or "").lower()

    cash = float(runtime.get("cash", 0) or 0)
    qty_raw = float(runtime.get("qty", 0) or 0)

    if not mode:
        mode = "cash" if cash > 0 else "qty"

    if mode == "cash":
        q = (cash / entry_price) if (entry_price > 0 and cash > 0) else 0.0
    else:
        q = qty_raw

    # lot step rounding (simple; exchange-specific steps can be added later)
    if lot_step and lot_step > 0:
        q = math.floor(q / lot_step) * lot_step
    return max(q, 0.0)

def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has a 'date' column in UTC-naive format.
    Tries common alternatives ('datetime', 'time') if missing.
    Returns a copy with the added/converted column.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    cols_lower = [c.lower() for c in df.columns]
    df.columns = cols_lower

    if "date" not in df.columns:
        if "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        elif "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        else:
            # No recognizable date column
            return df
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    # Drop NaT and strip tzinfo
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.tz_localize(None)
    return df

def equity_curve_from_trades(trades: pd.DataFrame) -> pd.Series:
    """
    Build a simple equity curve (PnL cumulative) with a time index from ExitTime or EntryTime.
    """
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return pd.Series(dtype=float, index=pd.to_datetime([]))
    ec = trades["pnl"].cumsum()
    ts = trades.get("ExitTime", trades.get("EntryTime"))
    try:
        idx = pd.to_datetime(ts)
    except Exception:
        idx = pd.RangeIndex(len(ec))
    ec.index = idx
    return ec.sort_index()

def max_drawdown(trades: pd.DataFrame) -> float:
    ec = equity_curve_from_trades(trades)
    if ec.empty:
        return 0.0
    peak = ec.cummax()
    return float((ec - peak).min())

def sharpe(trades: pd.DataFrame, rf: float = 0.0, period_scale: int = 252) -> float:
    ec = equity_curve_from_trades(trades)
    if ec.empty:
        return -1e9
    rets = ec.diff().fillna(0.0)
    mu, sd = rets.mean(), rets.std()
    if sd == 0:
        return -1e9
    return float(np.sqrt(period_scale) * (mu - rf / period_scale) / sd)

def sortino(trades: pd.DataFrame, target: float = 0.0, period_scale: int = 252) -> float:
    ec = equity_curve_from_trades(trades)
    if ec.empty:
        return -1e9
    rets = ec.diff().fillna(0.0)
    downside = np.clip(rets - target / period_scale, None, 0.0)
    dd = downside.std()
    if dd == 0:
        return -1e9
    return float(np.sqrt(period_scale) * (rets.mean() - target / period_scale) / dd)

def total_return_pct(trades: pd.DataFrame) -> float:
    """
    Sum of per-trade returns. Supports both 'return_pct' (new) and 'return' (legacy).
    """
    if trades is None or trades.empty:
        return 0.0
    if "return_pct" in trades.columns:
        return float(trades["return_pct"].sum())
    if "return" in trades.columns:
        return float(trades["return"].sum())
    return 0.0

def total_return_usd(trades: pd.DataFrame) -> float:
    """
    Sum of per-trade absolute PnL in dollars.
    Falls back to Qty × (ExitPrice - EntryPrice) if 'pnl' not present.
    """
    if trades is None or trades.empty:
        return 0.0
    if "pnl" in trades.columns:
        return float(trades["pnl"].sum())
    if all(col in trades.columns for col in ["EntryPrice", "ExitPrice", "Qty"]):
        return float(((trades["ExitPrice"] - trades["EntryPrice"]) * trades["Qty"]).sum())
    return 0.0


def score_from_trades(
    trades: pd.DataFrame,
    objective: str,
    pen_alpha: float = 0.1,
    pen_beta: float = 0.1,
) -> float:
    """
    Turn trades -> scalar score to MAXIMIZE.
    - 'Sharpe', 'Sortino'
    - 'MaxDD (minimize)' -> returns -abs(MaxDD)
    - 'Penalized: Return - a*DD + b*Win%'
    - 'return', 'win', 'return with win% tie-breaker'
    """
    if trades is None or trades.empty:
        return -1e9 if objective.lower() not in ("win",) else 0.0

    obj = objective.lower()
    if obj == "sharpe":
        return sharpe(trades)
    if obj == "sortino":
        return sortino(trades)
    if "maxdd" in obj:
        return -abs(max_drawdown(trades))
    if "penalized" in obj:
        tr = total_return_pct(trades)
        win = float(((trades["pnl"] > 0).mean()) * 100) if "pnl" in trades.columns else 0.0
        mdd = abs(max_drawdown(trades))
        return tr - pen_alpha * mdd + pen_beta * win
    if obj == "win":
        return float((trades["pnl"] > 0).mean())
    if obj == "return with win% tie-breaker":
        tr = total_return_pct(trades)
        win = float((trades["pnl"] > 0).mean())
        return tr + win / 100.0
    # default
    return total_return_pct(trades)

def get_last_ohlcv(
    df: pd.DataFrame,
    tz: str = "UTC",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    vol_col: str = "volume",
    date_col: str = "date",
):
    """
    Return aggregated OHLCV for the most recent *calendar day* in df.
    - Works for intraday data (1m/5m/etc) and daily data.
    - Interprets timestamps in UTC and *displays/aggregates* in `tz`.
    - Expects lowercase columns.
    """
    if df is None or df.empty or date_col not in df.columns:
        return None

    d = df.copy()
    # ensure datetime and tz-aware in UTC
    d[date_col] = pd.to_datetime(d[date_col], utc=True, errors="coerce")
    d = d.dropna(subset=[date_col])
    if d.empty:
        return None

    # Convert to target tz for “day” boundaries
    dt = d[date_col].dt.tz_convert(tz)
    # Most recent calendar day in that tz
    last_day = dt.max().date()

    # Mask rows that fall on that day
    day_mask = dt.dt.date == last_day
    day_df = d.loc[day_mask]
    if day_df.empty:
        return None

    # If data is already daily, just take last row
    if (day_df[date_col].dt.normalize().nunique() == 1) and (len(day_df) == 1):
        row = day_df.iloc[-1]
        return {
            "day": last_day,
            "open": float(row.get(open_col, float("nan"))),
            "high": float(row.get(high_col, float("nan"))),
            "low": float(row.get(low_col, float("nan"))),
            "close": float(row.get(close_col, float("nan"))),
            "volume": float(row.get(vol_col, 0.0)),
            "tz": tz,
        }

    # Aggregate intraday bars to daily OHLCV for that last day
    o = float(day_df[open_col].iloc[0]) if open_col in day_df else float("nan")
    h = float(day_df[high_col].max())    if high_col in day_df else float("nan")
    l = float(day_df[low_col].min())     if low_col in day_df else float("nan")
    c = float(day_df[close_col].iloc[-1])if close_col in day_df else float("nan")
    v = float(day_df[vol_col].sum())     if vol_col in day_df else 0.0

    return {"day": last_day, "open": o, "high": h, "low": l, "close": c, "volume": v, "tz": tz}



import pandas as pd
from math import isnan

def compute_avg_durations(trades: pd.DataFrame, df_all: pd.DataFrame, tf_str: str | None = None):
    if trades is None or trades.empty:
        return {"avg_closed_bars": 0.0, "avg_open_bars": 0.0,
                "avg_closed_minutes": 0.0, "avg_open_minutes": 0.0}

    # Try to infer bar size from data if tf_str not supplied
    if tf_str:
        # simple parse: "1m","3m","5m","15m","30m","1h","1d"
        unit = tf_str[-1].lower()
        n = float(tf_str[:-1])
        if unit == "m": bar_seconds = n * 60
        elif unit == "h": bar_seconds = n * 3600
        elif unit == "d": bar_seconds = n * 86400
        else: bar_seconds = pd.to_timedelta(df_all["date"].diff().median()).total_seconds()
    else:
        bar_seconds = pd.to_timedelta(df_all["date"].diff().median()).total_seconds()

    if not bar_seconds or isnan(bar_seconds) or bar_seconds <= 0:
        # fallback: assume 1 bar per row difference
        bar_seconds = pd.to_timedelta("1min").total_seconds()

    last_ts = pd.to_datetime(df_all["date"]).max()

    # Build per-trade duration in bars
    t = trades.copy()
    has_bars_col = "BarsInTrade" in t.columns

    if not has_bars_col:
        # compute from timestamps
        t["EntryTime"] = pd.to_datetime(t["EntryTime"], errors="coerce")
        t["ExitTime"] = pd.to_datetime(t["ExitTime"], errors="coerce")
        # closed duration
        t["dur_bars"] = (t["ExitTime"] - t["EntryTime"]).dt.total_seconds() / bar_seconds
        # open trades: use now/last bar
        open_mask = t["ExitTime"].isna()
        t.loc[open_mask, "dur_bars"] = (last_ts - t.loc[open_mask, "EntryTime"]).dt.total_seconds() / bar_seconds
    else:
        # already provided by strategy
        t["dur_bars"] = t["BarsInTrade"].astype(float)

    closed = t.dropna(subset=["ExitTime"]) if "ExitTime" in t.columns else t[t["dur_bars"].notna()]
    open_  = t[t["ExitTime"].isna()] if "ExitTime" in t.columns else t.iloc[0:0]

    avg_closed_bars = float(closed["dur_bars"].mean()) if not closed.empty else 0.0
    avg_open_bars   = float(open_["dur_bars"].mean())   if not open_.empty else 0.0

    avg_closed_minutes = avg_closed_bars * (bar_seconds / 60.0)
    avg_open_minutes   = avg_open_bars   * (bar_seconds / 60.0)

    return {
        "avg_closed_bars": round(avg_closed_bars, 2),
        "avg_open_bars": round(avg_open_bars, 2),
        "avg_closed_minutes": round(avg_closed_minutes, 2),
        "avg_open_minutes": round(avg_open_minutes, 2),
    }
