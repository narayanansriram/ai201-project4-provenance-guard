import json
import os
from collections import defaultdict

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "audit.jsonl")


def compute_analytics() -> dict:
    if not os.path.exists(LOG_PATH):
        return _empty()

    submissions = []
    appeals = []

    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") == "submission":
                submissions.append(entry)
            elif entry.get("type") == "appeal":
                appeals.append(entry)

    total = len(submissions)
    if total == 0:
        return _empty()

    # 1. Detection distribution
    counts = {"likely_ai": 0, "uncertain": 0, "likely_human": 0}
    for s in submissions:
        attr = s.get("attribution", "uncertain")
        if attr in counts:
            counts[attr] += 1

    distribution = {
        k: {"count": v, "pct": round(v / total * 100, 1)}
        for k, v in counts.items()
    }

    # 2. Appeal rate
    total_appeals = len(appeals)
    appeal_pct = round(total_appeals / total * 100, 1) if total else 0.0

    # 3. Average confidence over time (group by date)
    by_date = defaultdict(list)
    for s in submissions:
        ts = s.get("timestamp", "")
        date = ts[:10] if ts else "unknown"
        confidence = s.get("confidence")
        if confidence is not None:
            by_date[date].append(confidence)

    confidence_over_time = [
        {
            "date": date,
            "avg_confidence": round(sum(vals) / len(vals), 4),
            "count": len(vals),
        }
        for date, vals in sorted(by_date.items())
    ]

    return {
        "total_submissions": total,
        "detection_distribution": distribution,
        "appeal_rate": {
            "total_appeals": total_appeals,
            "pct_of_submissions": appeal_pct,
        },
        "confidence_over_time": confidence_over_time,
    }


def _empty() -> dict:
    return {
        "total_submissions": 0,
        "detection_distribution": {
            "likely_ai": {"count": 0, "pct": 0.0},
            "uncertain": {"count": 0, "pct": 0.0},
            "likely_human": {"count": 0, "pct": 0.0},
        },
        "appeal_rate": {"total_appeals": 0, "pct_of_submissions": 0.0},
        "confidence_over_time": [],
    }
