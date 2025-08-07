import json
import os
from datetime import datetime

REGISTRY_FILE = "best_params_registry.json"

def register_best_params(strategy, symbol, tf, params, metrics, registry_path=REGISTRY_FILE):
    """Save/update the best parameter set for (strategy, symbol, tf) in a deduplicated JSON registry."""
    # Load registry or create
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    key = f"{strategy}|{symbol}|{tf}"

    # Always store as a list of dicts (for historical analysis)
    record = {
        "date": datetime.utcnow().isoformat(),
        "params": params,
        "metrics": metrics
    }

    if key not in registry:
        registry[key] = [record]
    else:
        # Deduplicate: only add if different params or better metric (you decide, e.g. higher 'Sortino')
        exists = any(r["params"] == params for r in registry[key])
        if not exists:
            registry[key].append(record)
        else:
            # Optionally, update if new record is better (e.g., higher Sortino)
            best_metric = metrics.get("Sortino") or metrics.get("Sharpe") or metrics.get("PnL") or 0
            for i, r in enumerate(registry[key]):
                r_metric = r["metrics"].get("Sortino") or r["metrics"].get("Sharpe") or r["metrics"].get("PnL") or 0
                if r["params"] == params and best_metric > r_metric:
                    registry[key][i] = record  # Replace with better metrics

    # Sort entries so best is first (e.g., by Sortino)
    registry[key].sort(key=lambda r: r["metrics"].get("Sortino", 0), reverse=True)

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

def load_best_params(strategy, symbol, tf, registry_path=REGISTRY_FILE):
    """Load the best (highest metric) params for (strategy, symbol, tf), or None."""
    if not os.path.exists(registry_path):
        return None
    with open(registry_path, "r") as f:
        registry = json.load(f)
    key = f"{strategy}|{symbol}|{tf}"
    if key not in registry or not registry[key]:
        return None
    # Return best (first in list, after sorting)
    return registry[key][0]["params"]

def load_all_best_params(strategy, symbol, tf, registry_path=REGISTRY_FILE):
    """Load all saved parameter sets (history) for (strategy, symbol, tf)."""
    if not os.path.exists(registry_path):
        return []
    with open(registry_path, "r") as f:
        registry = json.load(f)
    key = f"{strategy}|{symbol}|{tf}"
    return registry.get(key, [])
