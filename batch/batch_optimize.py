import os
from datetime import datetime, timedelta
from registry import STRATEGY_REGISTRY, CD_PARAM_SPACES
from param_registry import register_best_params, load_best_params
from optimization import coordinate_descent_optimizer
from helpers import score_from_trades, sanitize_initial_params
from datafeed import load_alpaca
# ==== CONFIG ====

WATCHLIST = [
    "RGTI",
    "SOUN",
    "QUBT",
    "SLV",
    "QBTS",
    "PONY",
    "TRON",
    "SMCI",
    "QS"
]
TIMEFRAME = "3m"
STRATEGY = "RMA Strategy"
DAYS_LOOKBACK = 30
TRIALS = 1
PENDING_REGISTRY_FILE = "../best_params_registry_pending.json"
STOP_FILE = "stop_batch.txt"  # stop flag file

# Alpaca credentials (use same as in your Streamlit secrets)
API_KEY = "PK99OKUJAAI9KIINMT9G"
API_SECRET = "isFedDOxxFjl5KWuzSnPoe8Zb7aHMUYig4VDsuyI"
BASE_URL = "https://data.alpaca.markets"  # Change if using paper/live account


def run_batch_optimization():
    strategy_entry = STRATEGY_REGISTRY[STRATEGY]
    strategy_func = strategy_entry["function"]
    cd_space = CD_PARAM_SPACES[STRATEGY]

    for symbol in WATCHLIST:
        # Check for stop signal
        if os.path.exists(STOP_FILE):
            print("🛑 Batch optimization stopped by user.")
            os.remove(STOP_FILE)
            break

        print(f"📈 Optimizing {symbol} ...")
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=DAYS_LOOKBACK)

        df = load_alpaca(
            symbol=symbol,
            timeframe=TIMEFRAME,
            start=start_date,
            end=end_date,
            key=API_KEY,
            secret=API_SECRET,
            # base_url=BASE_URL
        )

        if df is None or df.empty:
            print(f"⚠️ Skipping {symbol} — no data.")
            continue

        initial_params = load_best_params(STRATEGY, symbol, TIMEFRAME)
        initial_params = sanitize_initial_params(cd_space, initial_params)

        best_params, best_score = coordinate_descent_optimizer(
            strategy_func,
            df,
            cd_space,
            {"runtime": {"maxtradesperday": 1, "qty": 1}},
            n_rounds=TRIALS,
            initial_params=initial_params,
            score_fn=lambda tr: score_from_trades(tr, "return"),
        )

        metrics = {"Score": best_score}
        register_best_params(
            strategy=STRATEGY,
            symbol=symbol,
            tf=TIMEFRAME,
            params=best_params,
            metrics=metrics,
            registry_path=PENDING_REGISTRY_FILE
        )

    print(f"✅ Batch optimization finished or stopped. Review {PENDING_REGISTRY_FILE}.")