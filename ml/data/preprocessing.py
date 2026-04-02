"""
Audio preprocessing utilities for music structure recognition.

This module handles:
  * audio loading and resampling
  * beat and downbeat estimation
  * SSL embedding extraction (MERT / w2v baseline)
  * beat-synchronous pooling
  * dataset-level embedding extraction pipelines
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, AutoModel

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio preprocessing."""

    target_sr: int = 24000
    target_channels: int = 1
    trim_silence: bool = True
    trim_top_db: float = 30.0
    beat_hop_length: int = 512
    beat_start_bpm: Optional[float] = None
    model_name: str = "mert"


class AudioPreprocessor:
    """Handles audio preprocessing and SSL embedding extraction."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-loaded SSL models
        self._mert_model: Optional[torch.nn.Module] = None
        self._mert_extractor: Optional[AutoFeatureExtractor] = None

        self._w2v_model: Optional[torch.nn.Module] = None
        self._w2v_extractor: Optional[AutoFeatureExtractor] = None

    # ------------------------------------------------------------------
    # Audio utilities

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and optionally trim an audio file.

        Args:
            audio_path: path to an audio file supported by librosa

        Returns:
            mono audio array (float32) and sample rate
        """
        audio_path = str(audio_path)
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, sr = librosa.load(
            audio_path,
            sr=self.config.target_sr,
            mono=True,
        )

        if self.config.trim_silence and audio.size:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=self.config.trim_top_db,
                frame_length=2048,
                hop_length=512,
            )

        logger.debug("Loaded audio %s (%.2fs @ %d Hz)", audio_path, len(audio) / sr, sr)
        return audio.astype(np.float32), sr

    def detect_beats(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[List[float], List[float]]:
        """Detect beats and pseudo-downbeats using librosa."""
        if audio.size == 0:
            return [], []

        try:
            beat_kwargs = {
                "y": audio,
                "sr": sr,
                "hop_length": self.config.beat_hop_length,
                "units": "time",
            }
            if self.config.beat_start_bpm is not None:
                beat_kwargs["start_bpm"] = self.config.beat_start_bpm

            tempo, beat_frames = librosa.beat.beat_track(**beat_kwargs)
            beat_times = beat_frames.tolist()

            # Downbeat heuristic: every 4th beat
            downbeats = beat_times[::4]
            return beat_times, downbeats
        except Exception as exc:
            logger.warning("Beat detection failed: %s", exc)
            return [], []

    # ------------------------------------------------------------------
    # Embedding extraction

    def extract_ssl_embeddings(
        self, audio: np.ndarray, sr: int, model_name: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract frame-level embeddings and corresponding frame times.

        Returns:
            embeddings: (num_frames, feature_dim)
            frame_times: (num_frames,) in seconds
        """
        model_name = model_name or self.config.model_name

        if model_name == "mert":
            return self._extract_mert_embeddings(audio, sr)
        if model_name == "w2v":
            return self._extract_w2v_embeddings(audio, sr)
        raise ValueError(f"Unsupported model: {model_name}")

    def _extract_mert_embeddings(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract embeddings using the MERT SSL model."""
        if self._mert_extractor is None or self._mert_model is None:
            logger.info("Loading MERT model (m-a-p/MERT-v1-95M)")
            self._mert_extractor = AutoFeatureExtractor.from_pretrained(
                "m-a-p/MERT-v1-95M",
                trust_remote_code=True,
            )
            self._mert_model = AutoModel.from_pretrained(
                "m-a-p/MERT-v1-95M",
                trust_remote_code=True,
            )
            self._mert_model.to(self.device).eval()
            self._mert_model.config.output_hidden_states = True

        inputs = self._mert_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._mert_model(**inputs, output_hidden_states=True)

        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError("MERT model did not return hidden states.")

        hidden_stack = torch.stack(outputs.hidden_states[-4:])  # (4, B, T, D)
        embeddings = hidden_stack.mean(dim=0).squeeze(0)  # (T, D)

        duration = len(audio) / sr if len(audio) else 0.0
        num_frames = embeddings.shape[0]
        if num_frames == 0 or duration == 0.0:
            frame_times = np.array([], dtype=np.float32)
        else:
            frame_times = np.linspace(
                0.0,
                duration,
                num=num_frames,
                endpoint=False,
                dtype=np.float32,
            )

        return embeddings.cpu().numpy(), frame_times

    def _extract_w2v_embeddings(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract embeddings using wav2vec2 baseline."""
        if self._w2v_extractor is None or self._w2v_model is None:
            logger.info("Loading w2v-BERT-2.0 baseline model")
            self._w2v_extractor = AutoFeatureExtractor.from_pretrained(
                "facebook/w2v-bert-2.0"
            )
            self._w2v_model = AutoModel.from_pretrained("facebook/w2v-bert-2.0")
            self._w2v_model.to(self.device).eval()
            self._w2v_model.config.output_hidden_states = True

        inputs = self._w2v_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._w2v_model(**inputs, output_hidden_states=True)

        hidden_stack = torch.stack(outputs.hidden_states[-4:])
        embeddings = hidden_stack.mean(dim=0).squeeze(0)

        duration = len(audio) / sr if len(audio) else 0.0
        num_frames = embeddings.shape[0]
        frame_times = (
            np.linspace(0.0, duration, num=num_frames, endpoint=False, dtype=np.float32)
            if num_frames and duration
            else np.array([], dtype=np.float32)
        )

        return embeddings.cpu().numpy(), frame_times

    # ------------------------------------------------------------------
    # Pooling utilities

    def beat_sync_pooling(
        self,
        embeddings: np.ndarray,
        frame_times: np.ndarray,
        beats: Iterable[float],
    ) -> np.ndarray:
        """Average pool embeddings between detected beats."""
        beat_list = list(beats or [])
        if embeddings.size == 0 or not beat_list:
            return embeddings

        beat_indices = np.searchsorted(frame_times, beat_list, side="left")
        beat_indices = np.clip(beat_indices, 0, len(frame_times) - 1)

        segment_boundaries = np.unique(
            np.concatenate([beat_indices, [len(frame_times)]]).astype(int)
        )

        pooled: List[np.ndarray] = []
        start = 0
        for end in segment_boundaries:
            if end <= start:
                continue
            pooled.append(embeddings[start:end].mean(axis=0))
            start = end

        if not pooled:
            return embeddings
        return np.stack(pooled, axis=0)

    # ------------------------------------------------------------------
    # Track & dataset processing

    def process_track(
        self,
        audio_path: str,
        output_dir: str,
        model_name: Optional[str] = None,
        track_id: Optional[str] = None,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a single track into embeddings.

        Args:
            audio_path: path to audio file
            output_dir: directory where `.npz` will be written
            model_name: SSL model identifier
            track_id: optional explicit track id for output filename
            skip_existing: skip processing if file already exists
        """
        model_name = model_name or self.config.model_name
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        resolved_track_id = track_id or Path(audio_path).stem
        artifact_path = output_path / f"{resolved_track_id}.npz"

        if skip_existing and artifact_path.exists():
            logger.info("Skipping %s (already processed)", resolved_track_id)
            return {
                "track_id": resolved_track_id,
                "output_file": str(artifact_path),
                "skipped": True,
            }

        audio, sr = self.load_audio(audio_path)
        beats, downbeats = self.detect_beats(audio, sr)
        embeddings, frame_times = self.extract_ssl_embeddings(audio, sr, model_name)
        beat_embeddings = self.beat_sync_pooling(embeddings, frame_times, beats)

        duration = len(audio) / sr if sr else 0.0

        np.savez(
            artifact_path,
            embeddings=embeddings.astype(np.float32),
            frame_times=frame_times.astype(np.float32),
            beat_embeddings=beat_embeddings.astype(np.float32),
            beats=np.asarray(beats, dtype=np.float32),
            downbeats=np.asarray(downbeats, dtype=np.float32),
            sr_audio=np.array([sr], dtype=np.int32),
            duration=np.array([duration], dtype=np.float32),
            model=np.array([model_name]),
        )

        logger.info(
            "Processed track %s (frames=%d, duration=%.2fs) -> %s",
            resolved_track_id,
            embeddings.shape[0],
            duration,
            artifact_path,
        )

        return {
            "track_id": resolved_track_id,
            "frames": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1] if embeddings.size else 0,
            "duration": duration,
            "beats": len(beats),
            "output_file": str(artifact_path),
            "skipped": False,
        }

    def process_annotations(
        self,
        annotations_path: str,
        output_dir: str,
        dataset_name: Optional[str] = None,
        limit: Optional[int] = None,
        model_name: Optional[str] = None,
        skip_existing: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process an annotations file (JSON lines) and extract embeddings.

        Args:
            annotations_path: path to normalized annotations JSONL
            output_dir: root directory for embeddings
            dataset_name: optional explicit dataset sub-directory
            limit: optional max number of tracks
            model_name: SSL model identifier
            skip_existing: skip tracks with existing artifacts
        """
        annotations_file = Path(annotations_path)
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

        df = pd.read_json(annotations_file, orient="records", lines=True)
        if limit is not None:
            df = df.head(limit)

        dataset_dir = Path(output_dir) / (dataset_name or annotations_file.stem)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        for record in tqdm(df.to_dict("records"), desc=f"Processing {dataset_dir.name}"):
            audio_path = record.get("audio_path")
            track_id = record.get("track_id")
            if not audio_path:
                logger.warning("Skipping %s (missing audio_path)", track_id)
                continue

            try:
                result = self.process_track(
                    audio_path=str(audio_path),
                    output_dir=str(dataset_dir),
                    model_name=model_name,
                    track_id=track_id,
                    skip_existing=skip_existing,
                )
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Failed to process track %s (%s): %s", track_id, audio_path, exc
                )
                results.append(
                    {
                        "track_id": track_id,
                        "audio_path": audio_path,
                        "error": str(exc),
                    }
                )

        summary_path = dataset_dir / "embedding_summary.json"
        with summary_path.open("w") as fp:
            json.dump(results, fp, indent=2)

        logger.info("Saved embedding summary to %s", summary_path)
        return results


# ----------------------------------------------------------------------
# CLI


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract SSL embeddings for music structure recognition."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        help="Single audio file to process (skip when using --annotations).",
    )
    parser.add_argument(
        "--annotations",
        help="Path to normalized annotations JSONL file for batch processing.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/embeddings",
        help="Directory where embeddings will be stored.",
    )
    parser.add_argument(
        "--model",
        choices=["mert", "w2v"],
        default="mert",
        help="SSL model to use.",
    )
    parser.add_argument(
        "--dataset-name",
        help="Optional dataset name (sub-directory) for batch processing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of tracks to process.",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable silence trimming before embedding extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings even if artifacts already exist.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = AudioConfig(
        target_sr=24000,
        trim_silence=not args.no_trim,
        model_name=args.model,
    )
    preprocessor = AudioPreprocessor(config)

    if args.annotations:
        preprocessor.process_annotations(
            annotations_path=args.annotations,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            limit=args.limit,
            model_name=args.model,
            skip_existing=not args.force,
        )
    elif args.audio_path:
        result = preprocessor.process_track(
            audio_path=args.audio_path,
            output_dir=args.output_dir,
            model_name=args.model,
            skip_existing=not args.force,
        )
        print(json.dumps(result, indent=2))
    else:
        parser.error("Provide either an audio_path or --annotations.")


if __name__ == "__main__":
    main()
