#!/usr/bin/env python
"""
Extract embeddings for Rust inference.
Called as a subprocess by the Rust predictor.
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure ml is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import AudioPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from audio")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--model-type", default="mert", choices=["mert", "w2v"])

    args = parser.parse_args()

    try:
        preprocessor = AudioPreprocessor()
        audio, sr = preprocessor.load_audio(args.audio_path)
        beats, _ = preprocessor.detect_beats(audio, sr)
        embeddings, frame_times = preprocessor.extract_ssl_embeddings(
            audio, sr, args.model_type
        )

        duration = len(audio) / sr

        result = {
            "embeddings": embeddings.tolist(),
            "sr": int(sr),
            "duration": float(duration),
            "beats": [float(b) for b in beats],
        }

        with open(args.output, "w") as f:
            json.dump(result, f)

        print(f"SUCCESS: Extracted {embeddings.shape} embeddings", file=sys.stderr)

    except Exception as e:
        import traceback

        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
