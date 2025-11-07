"""
CLI inference tool for music structure recognition.

Provides command-line interface for predicting structure from audio files.
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import librosa
import soundfile as sf
from dataclasses import dataclass
from datetime import datetime, timezone

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
    min_gap_seconds: float = 3.0
    prominence: Optional[float] = None
    smooth_boundaries: bool = False
    smoothing_window: int = 1
    min_segment_seconds: float = 1.5
    use_position_bias: bool = True


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

    def predict(
        self,
        audio_path: str,
        threshold: Optional[float] = None,
        include_debug: bool = False,
        smooth: Optional[bool] = None,
        smoothing_window: Optional[int] = None,
        min_gap: Optional[float] = None,
        prominence: Optional[float] = None,
        min_segment: Optional[float] = None,
        position_bias: Optional[bool] = None,
    ) -> Dict[str, Any]:
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

        with torch.no_grad():
            outputs = self.model.forward(embeddings)

        boundary_logits = outputs["boundary_logits"]
        label_logits = outputs["label_logits"]

        effective_threshold = (
            self.config.threshold if threshold is None else float(threshold)
        )
        effective_smooth = (
            self.config.smooth_boundaries if smooth is None else bool(smooth)
        )
        effective_window = smoothing_window
        if effective_window is None:
            effective_window = self.config.smoothing_window
        effective_window = max(1, int(effective_window))
        effective_min_gap = (
            self.config.min_gap_seconds if min_gap is None else max(0.0, float(min_gap))
        )
        effective_prominence = (
            self.config.prominence if prominence is None else prominence
        )
        effective_min_segment = (
            self.config.min_segment_seconds
            if min_segment is None
            else max(0.0, float(min_segment))
        )
        effective_position_bias = (
            self.config.use_position_bias
            if position_bias is None
            else bool(position_bias)
        )

        boundary_times = self.model.predict_boundaries(
            embeddings,
            frame_times,
            threshold=effective_threshold,
            mask=mask,
            boundary_logits=boundary_logits,
            smooth=effective_smooth,
            smoothing_window=effective_window,
            min_gap_seconds=effective_min_gap,
            prominence=effective_prominence,
        )

        duration = len(audio) / sr if sr else frame_times[-1]
        boundary_times = self._postprocess_boundaries(
            boundary_times, duration, min_segment_seconds=effective_min_segment
        )

        labels = self.model.predict_labels(
            embeddings,
            frame_times,
            boundary_times,
            mask=mask,
            label_logits=label_logits,
            position_bias=effective_position_bias,
            min_segment_seconds=effective_min_segment,
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

        debug_info = None
        if include_debug:
            boundary_probs = torch.sigmoid(boundary_logits).squeeze(0).detach().cpu().numpy()
            if mask is not None:
                mask_np = mask.squeeze(0).detach().cpu().numpy().astype(bool)
                boundary_probs = np.where(mask_np, boundary_probs, 0.0)

            max_prob = float(boundary_probs.max())
            min_prob = float(boundary_probs.min())
            mean_prob = float(boundary_probs.mean())
            top_indices = np.argsort(boundary_probs)[-10:][::-1]
            top_peaks = [
                {"time": float(frame_times[idx]), "prob": float(boundary_probs[idx])}
                for idx in top_indices
            ]
            debug_info = {
                "max_boundary_prob": max_prob,
                "mean_boundary_prob": mean_prob,
                "min_boundary_prob": min_prob,
                "top_boundary_peaks": top_peaks,
                "threshold": effective_threshold,
                "min_gap_seconds": effective_min_gap,
                "prominence": effective_prominence,
                "smoothed": effective_smooth,
                "smoothing_window": effective_window,
                "min_segment_seconds": effective_min_segment,
                "position_bias": effective_position_bias,
            }

        timestamp = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()

        result = {
            "track_id": Path(audio_path).stem,
            "sr": sr,
            "duration": duration,
            "boundaries": boundary_times,
            "labels": labels,
            "segments": segments,
            "version": f"{self.config.model_type}@2025-01-24",
            "beats": beats,
            "threshold": effective_threshold,
            "smooth": effective_smooth,
            "smoothing_window": effective_window,
            "min_gap_seconds": effective_min_gap,
            "min_segment_seconds": effective_min_segment,
            "position_bias": effective_position_bias,
            "timestamp": timestamp,
        }

        if debug_info is not None:
            result["debug"] = debug_info

        return result

    def _postprocess_boundaries(
        self,
        boundary_times: List[float],
        duration: float,
        min_segment_seconds: float = 0.0,
    ) -> List[float]:
        """Ensure boundary list includes start/end and is sorted."""
        if not boundary_times:
            return [0.0, float(duration)]

        # Sort and deduplicate with tolerance
        sorted_bounds = sorted(float(b) for b in boundary_times)
        boundaries: List[float] = []
        for b in sorted_bounds:
            if not boundaries or abs(b - boundaries[-1]) > 1e-3:
                boundaries.append(b)

        if not boundaries:
            boundaries = [0.0]

        if boundaries[0] > 1e-3:
            boundaries.insert(0, 0.0)
        if duration - boundaries[-1] > 1e-3:
            boundaries.append(float(duration))
        # Final pass to ensure strictly increasing and remove micro segments
        cleaned: List[float] = []
        last = None
        for b in boundaries:
            if last is None or b - last > 1e-3:
                cleaned.append(b)
                last = b
        if len(cleaned) == 1:
            cleaned.append(float(duration))

        if min_segment_seconds > 0 and len(cleaned) > 2:
            min_len = float(min_segment_seconds)
            filtered = [cleaned[0]]
            for boundary in cleaned[1:-1]:
                if boundary - filtered[-1] >= min_len:
                    filtered.append(boundary)
            filtered.append(cleaned[-1])
            cleaned = filtered

            # ensure final segment meets minimum, else merge into previous
            if len(cleaned) >= 3 and cleaned[-1] - cleaned[-2] < min_len:
                cleaned.pop(-2)

            if len(cleaned) < 2:
                cleaned = [0.0, float(duration)]

        return cleaned

    def predict_batch(
        self,
        audio_paths: List[str],
        include_debug: bool = False,
        threshold: Optional[float] = None,
        smooth: Optional[bool] = None,
        smoothing_window: Optional[int] = None,
        min_gap: Optional[float] = None,
        prominence: Optional[float] = None,
        min_segment: Optional[float] = None,
        position_bias: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Predict structure for multiple audio files."""
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(
                    audio_path,
                    threshold=threshold,
                    include_debug=include_debug,
                    smooth=smooth,
                    smoothing_window=smoothing_window,
                    min_gap=min_gap,
                    prominence=prominence,
                    min_segment=min_segment,
                    position_bias=position_bias,
                )
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
    parser.add_argument(
        "--min-gap",
        type=float,
        default=3.0,
        help="Minimum gap between consecutive boundaries in seconds",
    )
    parser.add_argument(
        "--min-segment",
        type=float,
        default=1.5,
        help="Minimum duration (seconds) for individual segments",
    )
    parser.add_argument(
        "--prominence",
        type=float,
        help="Minimum peak prominence for boundary detection (default: None)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable smoothing of boundary probabilities",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=1,
        help="Window size for boundary probability smoothing (odd integer recommended)",
    )
    parser.add_argument(
        "--no-position-bias",
        action="store_true",
        help="Disable positional bias when assigning labels",
    )

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
        min_gap_seconds=args.min_gap,
        prominence=args.prominence,
        smooth_boundaries=not args.no_smooth,
        smoothing_window=max(1, args.smoothing_window),
        min_segment_seconds=args.min_segment,
        use_position_bias=not args.no_position_bias,
    )

    # Create inference engine
    inference = MusicStructureInference(config)

    # Process audio
    audio_path = Path(args.audio_path)

    if audio_path.is_file():
        # Single file
        result = inference.predict(
            str(audio_path),
            threshold=args.threshold,
            include_debug=args.verbose,
            smooth=not args.no_smooth,
            smoothing_window=max(1, args.smoothing_window),
            min_gap=args.min_gap,
            prominence=args.prominence,
            min_segment=args.min_segment,
            position_bias=not args.no_position_bias,
        )

        if args.verbose and "debug" in result:
            debug = result["debug"]
            print(
                "# Boundary probability summary:\n"
                f"  max: {debug['max_boundary_prob']:.3f}  "
                f"mean: {debug['mean_boundary_prob']:.3f}  "
                f"min: {debug['min_boundary_prob']:.3f}  "
                f"(threshold={debug['threshold']:.3f})",
                file=sys.stderr,
            )
            for peak in debug["top_boundary_peaks"]:
                print(
                    f"  peak @ {peak['time']:.2f}s -> {peak['prob']:.3f}",
                    file=sys.stderr,
                )
            print(
                f"  min_gap={debug['min_gap_seconds']:.2f}s  "
                f"prominence={debug['prominence']}  "
                f"smoothed={debug['smoothed']}  "
                f"window={debug['smoothing_window']}  "
                f"min_segment={debug['min_segment_seconds']:.2f}s  "
                f"position_bias={debug['position_bias']}",
                file=sys.stderr,
            )

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

        results = inference.predict_batch(
            [str(f) for f in audio_files],
            include_debug=args.verbose,
            threshold=args.threshold,
            smooth=not args.no_smooth,
            smoothing_window=max(1, args.smoothing_window),
            min_gap=args.min_gap,
            prominence=args.prominence,
            min_segment=args.min_segment,
            position_bias=not args.no_position_bias,
        )

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
