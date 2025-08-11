from __future__ import annotations
import math
from typing import Dict, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

# we reuse your helpers to be consistent
from helpers import ensure_date_column, equity_curve_from_trades

# ==== New helpers for filtering and detection ====
def filter_regular_session(df, start_time="08:00", end_time="23:59"):
    """Filter to UTC session hours."""
    if df is None or df.empty:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    start = pd.to_datetime(start_time).time()
    end = pd.to_datetime(end_time).time()
    return df[(df['date'].dt.time >= start) & (df['date'].dt.time <= end)]

def is_crypto_symbol(symbol: str) -> bool:
    if not symbol:
        return False
    s = symbol.upper()
    return s.endswith(("USDT", "USD", "BTC", "ETH")) or ":" in s

def apply_rangebreaks(fig: go.Figure, symbol: str):
    """Compress x-axis for stocks."""
    if is_crypto_symbol(symbol):
        return
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[23.9833, 8], pattern="hour")  # 23:59 → 08:00 UTC
        ]
    )
def _coerce_dt(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return s

def make_price_panel(df_all: pd.DataFrame, trades: pd.DataFrame | None, show_ema_fast=True, show_ema_slow=True, symbol="") -> go.Figure:
    fig = go.Figure()

    if df_all is None or df_all.empty:
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=0))
        apply_rangebreaks(fig, symbol)
        return fig

    df_all = ensure_date_column(df_all).sort_values("date")

    if not is_crypto_symbol(symbol):
        # Filter to session hours for stocks
        df_all = filter_regular_session(df_all)

        # Split only candlesticks per day to remove overnight lines
        for _, group in df_all.groupby(df_all['date'].dt.date):
            fig.add_trace(go.Candlestick(
                x=group['date'], open=group['open'], high=group['high'],
                low=group['low'], close=group['close'], name="Price", showlegend=False
            ))
    else:
        # Continuous candlestick for crypto
        fig.add_trace(go.Candlestick(
            x=df_all['date'], open=df_all['open'], high=df_all['high'],
            low=df_all['low'], close=df_all['close'], name="Price", showlegend=False
        ))

    # Keep EMAs continuous for both stocks & crypto
    if show_ema_fast and "ema_fast" in df_all.columns:
        fig.add_trace(go.Scatter(x=df_all['date'], y=df_all['ema_fast'], mode="lines", name="EMA Fast"))
    if show_ema_slow and "ema_slow" in df_all.columns:
        fig.add_trace(go.Scatter(x=df_all['date'], y=df_all['ema_slow'], mode="lines", name="EMA Slow", line=dict(dash="dash")))

    # Keep trades continuous for both stocks & crypto
    if trades is not None and not trades.empty:
        if "EntryTime" in trades.columns and "EntryPrice" in trades.columns:
            fig.add_trace(go.Scatter(x=trades["EntryTime"], y=trades["EntryPrice"], mode="markers", name="Entry",
                                     marker=dict(symbol="triangle-up", size=10)))
        if "ExitTime" in trades.columns and "ExitPrice" in trades.columns:
            fig.add_trace(go.Scatter(x=trades["ExitTime"], y=trades["ExitPrice"], mode="markers", name="Exit",
                                     marker=dict(symbol="triangle-down", size=10)))

    fig.update_layout(height=420, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=0))
    apply_rangebreaks(fig, symbol)
    return fig


def make_equity_panel(trades: pd.DataFrame | None, symbol="") -> go.Figure:
    fig = go.Figure()
    if trades is None or trades.empty:
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=30))
        apply_rangebreaks(fig, symbol)
        return fig

    try:
        ec = equity_curve_from_trades(trades)
    except Exception:
        ec = pd.Series(dtype=float)

    if is_crypto_symbol(symbol):
        fig.add_trace(go.Scatter(x=pd.Series(ec.index), y=ec.values, mode="lines", name="Equity"))
    else:
        for day, group in pd.DataFrame({"date": pd.Series(ec.index), "value": ec.values}).groupby(lambda x: pd.to_datetime(ec.index[x]).date()):
            fig.add_trace(go.Scatter(x=group["date"], y=group["value"], mode="lines", name=f"Equity {day}"))

    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=30))
    apply_rangebreaks(fig, symbol)
    return fig

def make_drawdown_panel(trades: pd.DataFrame | None, symbol="") -> go.Figure:
    fig = go.Figure()
    if trades is None or trades.empty:
        fig.update_layout(height=160, margin=dict(l=10, r=10, t=10, b=30))
        apply_rangebreaks(fig, symbol)
        return fig

    try:
        ec = equity_curve_from_trades(trades)
    except Exception:
        ec = pd.Series(dtype=float)

    peak = ec.cummax()
    dd = ec - peak

    if is_crypto_symbol(symbol):
        fig.add_trace(go.Scatter(x=pd.Series(dd.index), y=dd.values, mode="lines", name="Drawdown", fill="tozeroy"))
    else:
        for day, group in pd.DataFrame({"date": pd.Series(dd.index), "value": dd.values}).groupby(lambda x: pd.to_datetime(dd.index[x]).date()):
            fig.add_trace(go.Scatter(x=group["date"], y=group["value"], mode="lines", name=f"Drawdown {day}", fill="tozeroy"))

    fig.update_layout(height=160, margin=dict(l=10, r=10, t=10, b=30))
    apply_rangebreaks(fig, symbol)
    return fig

def make_3panel_figure(df_all, trades=None, show_ema_fast=True, show_ema_slow=True, symbol="") -> go.Figure:
    price = make_price_panel(df_all, trades, show_ema_fast, show_ema_slow, symbol)
    equity = make_equity_panel(trades, symbol)
    dd = make_drawdown_panel(trades, symbol)

    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                           row_heights=[0.62, 0.23, 0.15])
    for tr in price.data:
        fig.add_trace(tr, row=1, col=1)
    for tr in equity.data:
        fig.add_trace(tr, row=2, col=1)
    for tr in dd.data:
        fig.add_trace(tr, row=3, col=1)

    fig.update_layout(height=820, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def render_visuals(st, df_all, trades=None, runtime: Optional[Dict] = None, title="📈 Strategy Visuals", symbol="") -> None:
    df_all = ensure_date_column(df_all)
    trades = ensure_date_column(trades)

    if df_all is None or df_all.empty:
        st.info("No data to chart yet. Run a strategy first.")
        return

    with st.expander(title, expanded=False):
        cols = st.columns(3)
        with cols[0]:
            show_ema_fast = st.checkbox("Show EMA Fast", value=("ema_fast" in df_all.columns))
        with cols[1]:
            show_ema_slow = st.checkbox("Show EMA Slow", value=("ema_slow" in df_all.columns))
        with cols[2]:
            st.caption("Hover to inspect values • Scroll to zoom")

        fig = make_3panel_figure(df_all=df_all, trades=trades, show_ema_fast=show_ema_fast,
                                 show_ema_slow=show_ema_slow, symbol=symbol)
        st.plotly_chart(fig, use_container_width=True)
