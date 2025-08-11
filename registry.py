# registry.py

from strategies.rma import rma_strategy
from strategies.sma_cross import sma_cross_strategy
from strategies.crypto_momentum_breakout import momentum_breakout_strategy
from strategies.crypto_mean_reversion import mean_reversion_strategy
from strategies.crypto_ma_cross import ma_cross_strategy
from strategies.crypto_volatility_breakout import volatility_breakout_strategy
from strategies.crypto_volume_price_action import volume_price_action_strategy
from strategies.crypto_intraday_multi import crypto_intraday_multi

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
    "Crypto Momentum Breakout": {
        "function": momentum_breakout_strategy,
        "params": ["lookback"],
        "universal": ["qty"]
    },
    "Crypto Mean Reversion": {
        "function": mean_reversion_strategy,
        "params": ["ma_len", "threshold"],
        "universal": ["qty"]
    },
    "Crypto MA Cross": {
        "function": ma_cross_strategy,
        "params": ["fast_len", "slow_len"],
        "universal": ["qty"]
    },
    "Crypto Volatility Breakout": {
        "function": volatility_breakout_strategy,
        "params": ["atr_len", "mult"],
        "universal": ["qty"]
    },
    "Crypto Volume Price Action": {
        "function": volume_price_action_strategy,
        "params": ["ma_len", "vol_mult"],
        "universal": ["qty"]
    },
}

# These runtime keys are disabled during optimization (UI-only / fixed)
CRYPTO_DISABLE_IN_OPT = ["session_start", "session_end", "initial_capital"]

# Order of crypto strategies shown in any crypto-specific lists/loops
CRYPTO_STRATEGY_KEYS = [
    "Crypto Intraday Multi-Signal",
    "Crypto Momentum Breakout",
    "Crypto Mean Reversion",
    "Crypto MA Cross",
    "Crypto Volatility Breakout",
    "Crypto Volume Price Action",
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
    "Crypto Momentum Breakout": {"lookback": (10, 50, 2)},
    "Crypto Mean Reversion": {"ma_len": (10, 60, 2), "threshold": (1, 4, 0.2)},
    "Crypto MA Cross": {"fast_len": (2, 20, 2), "slow_len": (5, 60, 5)},
    "Crypto Volatility Breakout": {"atr_len": (5, 30, 2), "mult": (1.0, 3.0, 0.2)},
    "Crypto Volume Price Action": {"ma_len": (10, 60, 2), "vol_mult": (1.2, 4.0, 0.2)},
}
