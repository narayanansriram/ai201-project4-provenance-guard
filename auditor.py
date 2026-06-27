import json
import os
from datetime import datetime, timezone

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "audit.jsonl")


def log_entry(entry: dict) -> None:
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_log(n: int = 20) -> list:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        lines = f.readlines()
    entries = []
    for line in lines[-n:]:
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries
