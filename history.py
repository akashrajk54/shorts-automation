"""Track past video topics so we never repeat the same theme/tool."""
import json
from datetime import datetime

import config

HISTORY_FILE = config.BASE_DIR / "history.json"
MAX_ENTRIES = 60


def load_history() -> list[dict]:
    """Return the list of past video entries (newest last)."""
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def recent_topics(limit: int = 25) -> list[str]:
    """Return a list of recent 'topic - title' strings for the avoid-list."""
    entries = load_history()[-limit:]
    out = []
    for e in entries:
        topic = e.get("topic") or ""
        title = e.get("title") or ""
        label = f"{topic} — {title}".strip(" —")
        if label:
            out.append(label)
    return out


def add_entry(title: str, topic: str) -> None:
    """Append a new video entry and trim to the most recent MAX_ENTRIES."""
    entries = load_history()
    entries.append({
        "date": datetime.now().isoformat(timespec="seconds"),
        "title": title,
        "topic": topic,
    })
    entries = entries[-MAX_ENTRIES:]
    HISTORY_FILE.write_text(json.dumps(entries, indent=2))
