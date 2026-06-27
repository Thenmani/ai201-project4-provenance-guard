"""
Structured audit log (JSON-file backed).

Every submission writes one entry. Extended in Milestone 4 (stylometry score,
fused confidence) and Milestone 5 (appeals). Entries are appended to a JSON file
so they survive restarts and can be surfaced via GET /log.
"""

import os
import json
from datetime import datetime, timezone
from threading import Lock

LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "audit_log.json")
_lock = Lock()


def utc_now_iso():
    """UTC timestamp, e.g. 2025-04-01T14:32:10.123456Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_all():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def write_entry(entry):
    """Append a structured entry to the audit log (thread-safe)."""
    with _lock:
        entries = _read_all()
        entries.append(entry)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    return entry


def get_log(limit=50):
    """Return the most recent `limit` entries (oldest first within the slice)."""
    return _read_all()[-limit:]
