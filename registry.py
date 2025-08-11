# registry.py

from strategies.rma import rma_strategy
from strategies.sma_cross import sma_cross_strategy
from strategies.crypto_intraday_multi import crypto_intraday_multi
from strategies.hybrid_atr_mom_break import hybrid_atr_momentum_breakout

STRATEGY_REGISTRY = {
    "RMA Strategy": {
        "function": rma_strategy,
        "params": [
            "rma_len", "barsforentry", "barsforexit", "atrlen",
            "normalizedupper", "normalizedlower", "ema_fast_len",
            "ema_slow_len", "trailpct"
        ],
        # Keep 'keeplime' universal for RMA only
        "universal": ["keeplime"]
    },
    "Crypto Intraday Multi-Signal": {
        "function": crypto_intraday_multi,
        "params": [
            "breakout_len", "momentum_len", "momentum_thresh", "trend_len",
            "atr_len", "min_atr", "trailing_stop_pct", "max_hold_bars"
        ],
        "universal": []
    },
    "SMA Cross": {
        "function": sma_cross_strategy,
        "params": ["fast_len", "slow_len"],
        "universal": ["qty"]
    },
    "Crypto Hybrid ATR-Mom Break": {
        "function": hybrid_atr_momentum_breakout,
        "params": ["breakout_len", "ema_len", "roc_thresh", "atr_len", "atr_mult"],
        "universal": ["qty"]
    },
}

# These runtime keys are disabled during optimization (UI-only / fixed)
CRYPTO_DISABLE_IN_OPT = ["session_start", "session_end", "initial_capital"]

# Order of crypto strategies shown in any crypto-specific lists/loops
CRYPTO_STRATEGY_KEYS = [
    "Crypto Intraday Multi-Signal",
    "Crypto Hybrid ATR-Mom Break",
]

# Coordinate Descent parameter spaces per strategy
CD_PARAM_SPACES = {
    "RMA Strategy": {
        "rma_len": (40, 100, 5),
        "barsforentry": (1, 10, 1),
        "barsforexit": (1, 10, 1),
        "atrlen": (2, 20, 1),
        "normalizedupper": (0, 2, 0.1),
        "normalizedlower": (-2, 0, 0.1),
        "ema_fast_len": (2, 30, 1),
        "ema_slow_len": (3, 100, 1),
        "trailpct": (40, 0, -0.5),
    },
    "Crypto Intraday Multi-Signal": {
        "breakout_len": (5, 51, 5),
        "momentum_len": (5, 51, 5),
        "momentum_thresh": (0.5, 3.0, 0.3),
        "trend_len": (10, 100, 1),
        "atr_len": (2, 20, 1),
        "min_atr": (0.1, 2.1, 0.2),
        "trailing_stop_pct": (0.5, 5.5, 0.5),
        "max_hold_bars": (5, 50, 1),
    },
    "SMA Cross": {"fast_len": (2, 16, 2), "slow_len": (5, 61, 5)},
    "Crypto Hybrid ATR-Mom Break": {
        "breakout_len": (5, 51, 5),
        "ema_len": (10, 100, 5),
        "roc_thresh": (0.5, 3.0, 0.1),
        "atr_len": (5, 30, 1),
        "atr_mult": (1.0, 3.0, 0.1),
    },
}
