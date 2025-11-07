"""Helpers for persisting inference results to a local history."""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HISTORY_DIR = Path("data/history")
AUDIO_DIR = HISTORY_DIR / "audio"


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


def _find_audio_file(history_id: str) -> Optional[Path]:
    """Return the path to a stored audio file for the given history id."""
    if not AUDIO_DIR.exists():
        return None
    for candidate in AUDIO_DIR.glob(f"{history_id}.*"):
        if candidate.is_file():
            return candidate
    return None


def save_result(result: Dict[str, Any], audio_source: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Persist a prediction result (and optional audio) to disk.

    Returns:
        Tuple of (history_id, relative_audio_path or None)
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

    relative_audio: Optional[str] = None
    if audio_source:
        try:
            src_path = Path(audio_source)
            if src_path.exists():
                AUDIO_DIR.mkdir(parents=True, exist_ok=True)
                suffix = src_path.suffix or ".wav"
                audio_dest = AUDIO_DIR / f"{history_id}{suffix}"
                shutil.copy2(src_path, audio_dest)
                relative_audio = str(audio_dest.relative_to(HISTORY_DIR))
        except Exception as exc:
            logger.warning("Failed to store history audio for %s: %s", history_id, exc)

    result.setdefault("history_id", history_id)
    if relative_audio:
        result.setdefault("history_audio_path", relative_audio)

    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    return history_id, relative_audio


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

        history_id = path.stem
        has_audio = _find_audio_file(history_id) is not None

        entries.append(
            {
                "history_id": history_id,
                "track_id": data.get("track_id"),
                "timestamp": data.get("timestamp"),
                "duration": data.get("duration"),
                "segment_count": len(data.get("segments", [])),
                "audio_url": f"/history/{history_id}/audio" if has_audio else None,
                "has_audio": has_audio,
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
            data = json.load(fp)
    except Exception as exc:
        logger.error("Failed to load history %s: %s", history_id, exc)
        return None

    data.setdefault("history_id", history_id)
    audio_path = _find_audio_file(history_id)
    if audio_path:
        data["audio_url"] = f"/history/{history_id}/audio"
    return data


def get_audio_path(history_id: str) -> Optional[Path]:
    """Return filesystem path to stored audio for a history entry."""
    return _find_audio_file(history_id)
