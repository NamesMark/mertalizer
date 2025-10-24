"""
CLI inference tool for music structure recognition.

Provides command-line interface for predicting structure from audio files.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import librosa
import soundfile as sf
from dataclasses import dataclass

from modeling.system import MusicStructureModel, ModelConfig
from data.preprocessing import AudioPreprocessor, AudioConfig
from evaluation.metrics import SegmentEvaluator

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    model_path: str
    model_type: str = "mert"  # "mert" or "w2v"
    device: str = "auto"
    batch_size: int = 1
    threshold: float = 0.5
    output_format: str = "json"  # "json" or "csv"


class MusicStructureInference:
    """Inference engine for music structure recognition."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto"
            else torch.device(config.device)
        )

        # Load model
        self.model = self._load_model()

        # Initialize audio preprocessor
        self.preprocessor = AudioPreprocessor()

        # Label names
        self.label_names = [
            "INTRO",
            "VERSE",
            "PRE",
            "CHORUS",
            "BRIDGE",
            "SOLO",
            "OUTRO",
            "OTHER",
        ]

    def _load_model(self) -> MusicStructureModel:
        """Load the trained model."""
        try:
            model = MusicStructureModel.load_from_checkpoint(
                self.config.model_path, map_location=self.device
            )
            model.to(self.device)
            model.eval()
            logger.info(f"Loaded model from {self.config.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, audio_path: str) -> Dict[str, Any]:
        """
        Predict music structure for an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with predictions
        """
        # Load and preprocess audio
        audio, sr = self.preprocessor.load_audio(audio_path)
        beats, _ = self.preprocessor.detect_beats(audio, sr)
        embeddings_np, frame_times = self.preprocessor.extract_ssl_embeddings(
            audio, sr, self.config.model_type
        )

        if embeddings_np.size == 0:
            raise RuntimeError("No embeddings were produced for the input audio")

        embeddings = torch.from_numpy(embeddings_np).unsqueeze(0).to(self.device)
        mask = torch.ones(embeddings.size(0), embeddings.size(1), dtype=torch.bool, device=self.device)

        boundary_times = self.model.predict_boundaries(
            embeddings, frame_times, threshold=self.config.threshold, mask=mask
        )

        duration = len(audio) / sr if sr else frame_times[-1]
        boundary_times = self._postprocess_boundaries(boundary_times, duration)

        labels = self.model.predict_labels(
            embeddings,
            frame_times,
            boundary_times,
            mask=mask,
        )

        # Create segments
        segments = []
        for idx, (start, end) in enumerate(
            zip(boundary_times[:-1], boundary_times[1:])
        ):
            segments.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "label": labels[idx] if idx < len(labels) else "OTHER",
                }
            )

        return {
            "track_id": Path(audio_path).stem,
            "sr": sr,
            "duration": duration,
            "boundaries": boundary_times,
            "labels": labels,
            "segments": segments,
            "version": f"{self.config.model_type}@2025-01-24",
            "beats": beats,
        }

    def _postprocess_boundaries(
        self, boundary_times: List[float], duration: float
    ) -> List[float]:
        """Ensure boundary list includes start/end and is sorted."""
        if not boundary_times:
            return [0.0, float(duration)]

        boundaries = sorted(set(float(b) for b in boundary_times))
        if boundaries[0] > 1e-3:
            boundaries.insert(0, 0.0)
        if duration - boundaries[-1] > 1e-3:
            boundaries.append(float(duration))
        return boundaries

    def predict_batch(self, audio_paths: List[str]) -> List[Dict[str, Any]]:
        """Predict structure for multiple audio files."""
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                results.append(result)
                logger.info(f"Processed {audio_path}")
            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                results.append({"track_id": Path(audio_path).stem, "error": str(e)})
        return results

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to file."""
        output_path = Path(output_path)

        if self.config.output_format == "json":
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        elif self.config.output_format == "csv":
            # Convert to CSV format
            import pandas as pd

            if "segments" in results:
                df = pd.DataFrame(results["segments"])
                df.to_csv(output_path, index=False)
            else:
                # Create empty CSV
                pd.DataFrame().to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unknown output format: {self.config.output_format}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Music Structure Recognition Inference"
    )
    parser.add_argument("audio_path", help="Path to audio file or directory")
    parser.add_argument(
        "--model", required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument("--output", help="Output file path")
    parser.add_argument(
        "--format", choices=["json", "csv"], default="json", help="Output format"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Boundary detection threshold"
    )
    parser.add_argument(
        "--device", default="auto", help="Device to use (cpu, cuda, auto)"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Process directory in batch"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level)

    # Create inference config
    config = InferenceConfig(
        model_path=args.model,
        device=args.device,
        threshold=args.threshold,
        output_format=args.format,
    )

    # Create inference engine
    inference = MusicStructureInference(config)

    # Process audio
    audio_path = Path(args.audio_path)

    if audio_path.is_file():
        # Single file
        result = inference.predict(str(audio_path))

        if args.output:
            inference.save_results(result, args.output)
        else:
            print(json.dumps(result, indent=2))

    elif audio_path.is_dir() and args.batch:
        # Batch processing
        audio_files = (
            list(audio_path.glob("*.wav"))
            + list(audio_path.glob("*.mp3"))
            + list(audio_path.glob("*.flac"))
        )

        if not audio_files:
            print(f"No audio files found in {audio_path}")
            return

        results = inference.predict_batch([str(f) for f in audio_files])

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
        else:
            for result in results:
                print(json.dumps(result, indent=2))

    else:
        print(
            "Error: Provide a single audio file or use --batch for directory processing"
        )
        return


if __name__ == "__main__":
    main()
