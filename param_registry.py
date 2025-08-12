import json
import os
from datetime import datetime

REGISTRY_FILE = "best_params_registry.json"

def register_best_params(strategy, symbol, tf, params, metrics, registry_path=REGISTRY_FILE):
    """Save/update ONLY the latest parameter set for (strategy, symbol, tf) in the JSON registry."""
    # Load registry or create
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    key = f"{strategy}|{symbol}|{tf}"

    # Always store as a list of ONE dict (latest only)
    record = {
        "date": datetime.utcnow().isoformat(),
        "params": params,
        "metrics": metrics
    }

    # Replace any existing record for this key with the new one
    registry[key] = [record]

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
