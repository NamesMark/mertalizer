#!/usr/bin/env python
"""Ingest Harmonix annotations into our JSONL format.

Reads the corrected HarmonixSet annotations and outputs normalized JSONL.
Audio paths point to BigVGAN reconstructions (must be downloaded separately).
"""

import json
import sys
from pathlib import Path

LABEL_MAP = {
    "intro": "INTRO",
    "verse": "VERSE",
    "chorus": "CHORUS",
    "bridge": "BRIDGE",
    "outro": "OUTRO",
    "solo": "SOLO",
    "pre-chorus": "PRE",
    "pre_chorus": "PRE",
    "prechorus": "PRE",
    "interlude": "BRIDGE",
    "instrumental": "BRIDGE",
    "break": "BRIDGE",
    "transition": "OTHER",
    "silence": "OTHER",
    "fade-out": "OUTRO",
    "fadeout": "OUTRO",
    "coda": "OUTRO",
}


def normalize_label(label):
    return LABEL_MAP.get(label.lower().strip(), "OTHER")


def ingest(annotations_path, audio_dir=None, output_path=None):
    annotations_path = Path(annotations_path)
    with open(annotations_path) as f:
        records = [json.loads(line) for line in f]

    tracks = []
    for record in records:
        data_id = record["data_id"]
        split = record.get("split", "train")
        msa_info = record["msa_info"]

        if len(msa_info) < 2:
            continue

        # Extract boundaries and labels
        boundary_times = [entry[0] for entry in msa_info]
        raw_labels = [entry[1] for entry in msa_info]

        # Labels are per-segment (between consecutive boundaries)
        # Last entry is the final boundary (end of last segment), its label starts the last segment
        boundary_labels = [normalize_label(l) for l in raw_labels[:-1]]
        # Add end time (estimate from last boundary + typical segment length)
        # The dataset doesn't include duration, so we'll need audio for that
        # For now, estimate last segment as 15s
        last_boundary = boundary_times[-1]
        end_time = last_boundary + 15.0  # rough estimate

        # Actually, the last entry IS a segment start, so:
        boundary_labels = [normalize_label(l) for l in raw_labels]
        boundary_times.append(end_time)

        audio_path = None
        if audio_dir:
            # BigVGAN reconstructions are named like: 0001_12step.wav
            for ext in [".wav", ".mp3", ".flac"]:
                candidate = Path(audio_dir) / f"{data_id}{ext}"
                if candidate.exists():
                    audio_path = str(candidate)
                    break

        track = {
            "track_id": f"harmonix_{data_id}",
            "dataset": "harmonix",
            "split": split,
            "sr": 22050,
            "duration": end_time,
            "boundary_times": boundary_times,
            "boundary_labels": boundary_labels,
            "audio_path": audio_path,
            "original_track_id": data_id,
        }
        tracks.append(track)

    # Write output
    if output_path is None:
        output_path = annotations_path.parent.parent.parent / "processed" / "harmonix.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for t in tracks:
            f.write(json.dumps(t) + "\n")

    # Stats
    all_labels = [l for t in tracks for l in t["boundary_labels"]]
    from collections import Counter
    print(f"Ingested {len(tracks)} Harmonix tracks → {output_path}")
    print(f"Splits: {Counter(t['split'] for t in tracks)}")
    print(f"Labels: {Counter(all_labels).most_common()}")
    print(f"With audio: {sum(1 for t in tracks if t['audio_path'])}")

    return tracks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", help="Path to harmonixset.corrected.*.jsonl")
    parser.add_argument("--audio-dir", help="Directory with BigVGAN audio files")
    parser.add_argument("--output", help="Output JSONL path")
    args = parser.parse_args()
    ingest(args.annotations, args.audio_dir, args.output)
