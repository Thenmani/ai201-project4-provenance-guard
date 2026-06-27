"""
Structured audit log (JSON-file backed).

Every submission writes one entry. An appeal updates that content's entry in
place (status -> under_review, appeal_reasoning populated). Entries are stored in
a JSON file so they survive restarts and can be surfaced via GET /log.
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


def _write_all(entries):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def write_entry(entry):
    """Append a structured entry to the audit log (thread-safe)."""
    with _lock:
        entries = _read_all()
        entries.append(entry)
        _write_all(entries)
    return entry


def find_entry(content_id):
    """Return the most recent entry for `content_id`, or None."""
    for entry in reversed(_read_all()):
        if entry.get("content_id") == content_id:
            return entry
    return None


def update_entry(content_id, updates):
    """Apply `updates` (a dict) to the entry matching `content_id`.

    Returns the updated entry, or None if no matching content_id exists.
    Thread-safe; the find + write happen under one lock so an appeal can't race.
    """
    with _lock:
        entries = _read_all()
        target = None
        for entry in entries:
            if entry.get("content_id") == content_id:
                target = entry  # keep last match (there should only be one)
        if target is None:
            return None
        target.update(updates)
        _write_all(entries)
        return target


def get_log(limit=50):
    """Return the most recent `limit` entries (oldest first within the slice)."""
    return _read_all()[-limit:]
