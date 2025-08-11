import copy
import numpy as np
from typing import Callable, Optional, Dict, Any
from helpers import score_from_trades
def cd_score_fn(trades, optimize_for: str = "return"):
    return score_from_trades(trades, optimize_for)

def coordinate_descent_optimizer(
    strategy_func,
    df_strategy,
    param_space: Dict[str, Any],
    universal_kwargs: Dict[str, Any],
    n_rounds: int = 4,
    initial_params: Optional[Dict[str, Any]] = None,
    score_fn: Optional[Callable] = None,
    optimize_for: str = "return",
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    score_fn = score_fn or (lambda tr: cd_score_fn(tr, optimize_for))
    current_params = (initial_params or {}).copy()

    # seed defaults at midpoints
    for p, v in param_space.items():
        if p not in current_params:
            if isinstance(v, list):
                current_params[p] = v[len(v)//2]
            else:
                lo, hi, step = v
                current_params[p] = lo + ((hi - lo) / 2.0)

    total_steps = n_rounds * sum(
        len(v) if isinstance(v, list) else (int(round((v[1]-v[0]) / v[2])) + 1)
        for v in param_space.values()
    )
    step = 0

    for _ in range(n_rounds):
        for p in param_space:
            best_val = current_params[p]
            best_score = None
            space = param_space[p]
            if isinstance(space, list):
                vals = space
            else:
                lo, hi, s = space
                vals = list(np.arange(lo, hi + s, s)) if isinstance(s, float) else list(range(lo, hi + 1, s))
            for val in vals:
                trial = copy.deepcopy(current_params)
                trial[p] = val
                # keep fast/slow sane if present
                if "ema_fast_len" in trial and "ema_slow_len" in trial and trial["ema_slow_len"] <= trial["ema_fast_len"]:
                    step += 1
                    if progress_cb: progress_cb(step, total_steps)
                    continue
                trades, _ = strategy_func(df_strategy, **{**universal_kwargs, **trial})
                score = score_fn(trades)
                if best_score is None or score > best_score:
                    best_score, best_val = score, val
                step += 1
                if progress_cb: progress_cb(step, total_steps)
            current_params[p] = best_val
    return current_params, best_score
