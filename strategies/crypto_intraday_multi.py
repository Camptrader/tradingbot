import numpy as np
import pandas as pd


def crypto_intraday_multi(
        df,
        breakout_len=20,  # Rolling high lookback
        momentum_len=10,  # Rolling ROC lookback
        momentum_thresh=0.5,  # ROC threshold for entry
        trend_len=50,  # EMA length for trend filter
        atr_len=14,  # ATR length for volatility filter
        min_atr=0.5,  # Minimum ATR as % of close
        trailing_stop_pct=2.0,  # Trailing stop (%)
        max_hold_bars=3000,  # Maximum bars to hold
        maxtradesperday=20,  # Trades per day
        initial_capital=10000,
        qty=1,
):
    df = df.copy()
    # --- Signal 1: Breakout (close > N-bar high) ---
    df['breakout'] = df['close'] > df['close'].rolling(breakout_len).max().shift(1)
    # --- Signal 2: Momentum (Rate of Change) ---
    df['roc'] = 100 * (df['close'] / df['close'].shift(momentum_len) - 1)
    df['momentum'] = df['roc'] > momentum_thresh
    # --- Filter 1: Trend (EMA filter) ---
    df['ema'] = df['close'].ewm(span=trend_len, adjust=False).mean()
    df['trend'] = df['close'] > df['ema']
    # --- Filter 2: Volatility (ATR as % of close) ---
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.abs(df['close'] - df['close'].shift(1)),
                          np.abs(df['low'] - df['close'].shift(1)))
    df['atr'] = df['tr'].rolling(atr_len).mean()
    df['atr_pct'] = 100 * df['atr'] / df['close']
    df['vol_ok'] = df['atr_pct'] >= min_atr
    # --- Composite Entry ---
    df['entry_signal'] = df['breakout'] & df['momentum'] & df['trend'] & df['vol_ok']

    trade_log = []
    in_trade = False
    entry_price = 0
    entry_idx = 0
    peak = 0
    trades_today = 0
    last_trade_day = None

    for i in range(len(df)):
        day = df['date'].iloc[i].date() if 'date' in df.columns else df.index[i].date()
        if day != last_trade_day:
            trades_today = 0
            last_trade_day = day

        # ENTRY: all signals + max trades per day not hit
        if (not in_trade and df['entry_signal'].iloc[i]
                and trades_today < maxtradesperday):
            in_trade = True
            entry_price = df['close'].iloc[i]
            entry_idx = i
            peak = entry_price
            trades_today += 1
            trade = {
                "EntryTime": df['date'].iloc[i] if 'date' in df.columns else df.index[i],
                "EntryPrice": entry_price,
                "ExitTime": None,
                "ExitPrice": None,
                "BarsInTrade": None,
                "pnl": None,
                "ExitReason": None
            }
        # IN-TRADE: manage exits
        if in_trade:
            bars_held = i - entry_idx
            peak = max(peak, df['high'].iloc[i])
            exit_trade = False
            # Trailing stop
            if df['low'].iloc[i] <= peak * (1 - trailing_stop_pct / 100):
                exit_price = peak * (1 - trailing_stop_pct / 100)
                reason = "Trailing Stop"
                exit_trade = True
            # Max hold time
            elif bars_held >= max_hold_bars:
                exit_price = df['close'].iloc[i]
                reason = "Max Hold"
                exit_trade = True
            # Optional: Exit if momentum fades (uncomment if you want stricter exits)
            # elif not df['momentum'].iloc[i]:
            #     exit_price = df['close'].iloc[i]
            #     reason = "Momentum Fade"
            #     exit_trade = True
            if exit_trade:
                in_trade = False
                trade.update({
                    "ExitTime": df['date'].iloc[i] if 'date' in df.columns else df.index[i],
                    "ExitPrice": exit_price,
                    "BarsInTrade": bars_held,
                    "pnl": exit_price - trade['EntryPrice'],
                    "ExitReason": reason
                })
                trade_log.append(trade)
    trades = pd.DataFrame(trade_log)
    if not trades.empty:
        trades['return'] = trades['pnl'] / trades['EntryPrice'] * 100
    return trades, df
