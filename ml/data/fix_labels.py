#!/usr/bin/env python
"""Fix CCMusic label quality issues in split JSONL files.

- Remap mid-song INTRO (position > 0) to BRIDGE
- Report changes
"""

import json
import sys
from pathlib import Path


def fix_track(track):
    labels = track.get("boundary_labels", [])
    changes = 0
    for i, label in enumerate(labels):
        if label == "INTRO" and i > 0:
            labels[i] = "BRIDGE"
            changes += 1
    track["boundary_labels"] = labels
    return changes


def fix_file(path):
    path = Path(path)
    with open(path) as f:
        tracks = [json.loads(line) for line in f]

    total_changes = 0
    for t in tracks:
        total_changes += fix_track(t)

    with open(path, "w") as f:
        for t in tracks:
            f.write(json.dumps(t) + "\n")

    return len(tracks), total_changes


if __name__ == "__main__":
    splits_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/splits")
    for name in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
        p = splits_dir / name
        if p.exists():
            tracks, changes = fix_file(p)
            print(f"{name}: {tracks} tracks, {changes} mid-song INTROs → BRIDGE")
