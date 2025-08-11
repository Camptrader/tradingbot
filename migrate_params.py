# migrate_params.py
import json
from pathlib import Path
from datetime import datetime
from param_registry import register_best_params

LEGACY_PATH = Path("best_params.json")                # old file
NEW_PATH     = Path("best_params_registry.json")      # new file (created/updated)

# If your old file had no timeframe info, set a sensible default here.
DEFAULT_TF = "3m"

# Optional: provide a per-(strategy,symbol) override to nail the correct TFs.
# e.g., {("RMA Strategy","RGTI"): "3m", ("SMA Cross","AAPL"): "15m"}
TF_MAP: dict[tuple[str, str], str] = {
    # ("Strategy Name", "SYMBOL"): "TF",
}

def pick_tf(strategy: str, symbol: str) -> str:
    return TF_MAP.get((strategy, symbol), DEFAULT_TF)

def main():
    if not LEGACY_PATH.exists():
        print(f"[skip] {LEGACY_PATH} not found.")
        return

    with LEGACY_PATH.open("r") as f:
        legacy = json.load(f)

    migrated = 0
    for strategy, by_symbol in (legacy or {}).items():
        if not isinstance(by_symbol, dict):
            continue
        for symbol, rec in by_symbol.items():
            params  = (rec or {}).get("params")  or {}
            metrics = (rec or {}).get("results") or {}
            if not params:
                continue
            tf = pick_tf(strategy, symbol)
            register_best_params(
                strategy=strategy,
                symbol=symbol,
                tf=tf,
                params=params,
                metrics=metrics,
            )
            migrated += 1
            print(f"[ok] {strategy} | {symbol} | {tf}  → saved")

    print(f"Done. Migrated {migrated} entries into {NEW_PATH.resolve()}")

if __name__ == "__main__":
    main()
