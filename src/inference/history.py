"""Helpers for persisting inference results to a local history."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HISTORY_DIR = Path("data/history")


def _utc_now_iso() -> str:
    """Return a compact ISO8601 timestamp (UTC, second precision)."""
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _slugify(value: Optional[str]) -> str:
    """Create a filesystem-friendly slug from track_id."""
    if not value:
        return "track"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = slug.strip("-").lower()
    return slug or "track"


def save_result(result: Dict[str, Any]) -> str:
    """
    Persist a prediction result to disk.

    Returns:
        history_id that can be used to retrieve the entry later.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = result.get("timestamp")
    if not timestamp:
        timestamp = _utc_now_iso()
        result["timestamp"] = timestamp

    track_slug = _slugify(result.get("track_id"))
    time_compact = timestamp.replace(":", "").replace("-", "")
    history_id = f"{time_compact}_{track_slug}"
    history_path = HISTORY_DIR / f"{history_id}.json"

    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    return history_id


def list_results(limit: int = 20) -> List[Dict[str, Any]]:
    """Return the most recent history entries (metadata only)."""
    if not HISTORY_DIR.exists():
        return []

    entries: List[Dict[str, Any]] = []
    for path in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            continue

        entries.append(
            {
                "history_id": path.stem,
                "track_id": data.get("track_id"),
                "timestamp": data.get("timestamp"),
                "duration": data.get("duration"),
                "segment_count": len(data.get("segments", [])),
            }
        )
        if len(entries) >= limit:
            break

    return entries


def load_result(history_id: str) -> Optional[Dict[str, Any]]:
    """Load a stored history entry."""
    history_path = HISTORY_DIR / f"{history_id}.json"
    if not history_path.exists():
        return None
    try:
        with history_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None
