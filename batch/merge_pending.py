import json
import os

MAIN_FILE = "../best_params_registry.json"
PENDING_FILE = "../best_params_registry_pending.json"
BACKUP_FILE = "best_params_registry_backup.json"

def load_registry(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def save_registry(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def best_metric_value(metrics):
    return (
        metrics.get("Sortino")
        or metrics.get("Sharpe")
        or metrics.get("PnL")
        or metrics.get("Score")
        or 0
    )

def merge_pending():
    main_registry = load_registry(MAIN_FILE)
    pending_registry = load_registry(PENDING_FILE)

    if not pending_registry:
        print("⚠️ Pending registry is empty — nothing to merge.")
        return

    # Backup
    if os.path.exists(MAIN_FILE):
        save_registry(BACKUP_FILE, main_registry)
        print(f"💾 Backup saved: {BACKUP_FILE}")

    updated_count = 0
    for key, pending_records in pending_registry.items():
        if not pending_records:
            continue
        pending_best = pending_records[0]

        if key not in main_registry or not main_registry[key]:
            main_registry[key] = [pending_best]
            updated_count += 1
            continue

        current_best = main_registry[key][0]
        if best_metric_value(pending_best["metrics"]) > best_metric_value(current_best["metrics"]):
            main_registry[key].insert(0, pending_best)  # Keep history
            updated_count += 1

    save_registry(MAIN_FILE, main_registry)
    print(f"✅ Merge complete: {updated_count} entries updated.")

if __name__ == "__main__":
    merge_pending()
