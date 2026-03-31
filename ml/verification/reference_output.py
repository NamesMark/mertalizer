#!/usr/bin/env python
"""
Generate reference outputs for verifying the Rust pipeline.

Runs the full Python pipeline on an audio file and saves all intermediate
and final results as JSON for comparison with Rust.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing import AudioPreprocessor, AudioConfig
from modeling.system import MusicStructureModel

logger = logging.getLogger(__name__)


def generate_reference(
    audio_path: str,
    checkpoint_path: str,
    output_path: str,
    model_type: str = "mert",
):
    """Generate reference output for an audio file."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    config = AudioConfig(target_sr=24000, model_name=model_type)
    preprocessor = AudioPreprocessor(config)

    # --- Step 1: Load and preprocess audio ---
    logger.info("Loading audio: %s", audio_path)
    audio, sr = preprocessor.load_audio(str(audio_path))
    duration = len(audio) / sr
    logger.info("Audio: %.2fs, sr=%d, samples=%d", duration, sr, len(audio))

    # --- Step 2: Beat detection ---
    logger.info("Detecting beats...")
    beats, downbeats = preprocessor.detect_beats(audio, sr)
    if beats:
        bpm = 60.0 / np.median(np.diff(beats)) if len(beats) > 1 else 0.0
    else:
        bpm = 0.0
    logger.info("Beats: %d, BPM: %.1f", len(beats), bpm)

    # --- Step 3: MERT embedding extraction ---
    logger.info("Extracting MERT embeddings...")
    embeddings_np, frame_times = preprocessor.extract_ssl_embeddings(audio, sr, model_type)
    logger.info("Embeddings shape: %s", embeddings_np.shape)

    # --- Step 4: Compute embedding checksum ---
    emb_bytes = embeddings_np.tobytes()
    emb_hash = hashlib.sha256(emb_bytes).hexdigest()[:16]

    # --- Step 5: Run boundary/label heads ---
    logger.info("Loading checkpoint: %s", checkpoint_path)
    model = MusicStructureModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()

    embeddings_t = torch.from_numpy(embeddings_np).unsqueeze(0)
    mask = torch.ones(1, embeddings_t.shape[1], dtype=torch.bool)

    with torch.no_grad():
        outputs = model.forward(embeddings_t, mask)

    boundary_logits = outputs["boundary_logits"]
    label_logits = outputs["label_logits"]

    # --- Step 6: Post-process ---
    boundary_times = model.predict_boundaries(
        embeddings_t,
        frame_times,
        threshold=0.5,
        mask=mask,
        boundary_logits=boundary_logits,
        smooth=False,
        smoothing_window=1,
        min_gap_seconds=3.0,
    )

    # Ensure start and end
    if not boundary_times or boundary_times[0] > 0.1:
        boundary_times = [0.0] + boundary_times
    if not boundary_times or duration - boundary_times[-1] > 0.1:
        boundary_times.append(float(duration))

    labels = model.predict_labels(
        embeddings_t,
        frame_times,
        boundary_times,
        mask=mask,
        label_logits=label_logits,
        position_bias=True,
        min_segment_seconds=1.5,
    )

    # --- Step 7: Also get raw logits for verification ---
    boundary_probs = torch.sigmoid(boundary_logits).squeeze(0).numpy()
    label_probs = torch.softmax(label_logits.squeeze(0), dim=-1).numpy()

    # --- Build reference output ---
    # Save normalized audio for Rust to read (ensures identical input)
    normalized_audio = (audio - audio.mean()) / (audio.std() + 1e-8)

    reference = {
        "audio_path": str(audio_path.resolve()),
        "audio_samples": len(audio),
        "sr": int(sr),
        "duration": float(duration),
        # Normalized audio (first 100 + last 100 samples for spot-check)
        "normalized_audio_head": normalized_audio[:100].tolist(),
        "normalized_audio_tail": normalized_audio[-100:].tolist(),
        "normalized_audio_mean": float(normalized_audio.mean()),
        "normalized_audio_std": float(normalized_audio.std()),
        # Embeddings
        "embeddings_shape": list(embeddings_np.shape),
        "embeddings_hash": emb_hash,
        "embeddings_head": embeddings_np[0, :10].tolist(),  # first frame, first 10 dims
        "embeddings_tail": embeddings_np[-1, :10].tolist(),  # last frame, first 10 dims
        "embeddings_mean": float(embeddings_np.mean()),
        "embeddings_std": float(embeddings_np.std()),
        # Boundary logits (raw)
        "boundary_probs_shape": list(boundary_probs.shape),
        "boundary_probs_head": boundary_probs[:10].tolist(),
        "boundary_probs_max": float(boundary_probs.max()),
        "boundary_probs_mean": float(boundary_probs.mean()),
        # Final results
        "boundaries": [float(b) for b in boundary_times],
        "labels": labels,
        "num_segments": len(labels),
        "beats": [float(b) for b in beats],
        "bpm": float(bpm),
        # Label distribution
        "label_distribution": {
            label: labels.count(label) for label in set(labels)
        } if labels else {},
    }

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(reference, f, indent=2)

    logger.info("Reference output saved to %s", output_path)
    print(f"\nReference output summary:")
    print(f"  Duration:      {duration:.2f}s")
    print(f"  BPM:           {bpm:.1f}")
    print(f"  Embeddings:    {embeddings_np.shape}")
    print(f"  Boundaries:    {len(boundary_times)}")
    print(f"  Segments:      {len(labels)}")
    print(f"  Labels:        {labels}")
    print(f"  Emb hash:      {emb_hash}")

    return reference


def main():
    parser = argparse.ArgumentParser(description="Generate reference output for Rust verification")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", default="ml/verification/reference.json", help="Output JSON path")
    parser.add_argument("--model-type", default="mert", choices=["mert", "w2v"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_reference(args.audio_path, args.checkpoint, args.output, args.model_type)


if __name__ == "__main__":
    main()
