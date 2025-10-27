"""
Data ingestion pipeline for music structure datasets.

Normalizes annotations from various datasets into a unified CSV format.
"""

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - support both package and script execution
    from .ontology import LabelMapper, map_labels
except ImportError:  # When running as a script (python src/data/ingestion.py)
    from data.ontology import LabelMapper, map_labels

logger = logging.getLogger(__name__)


@dataclass
class TrackAnnotation:
    """Unified track annotation format."""

    track_id: str
    dataset: str
    sr: int
    duration: float
    boundary_times: List[float]
    boundary_labels: List[str]
    split: Optional[str] = None
    source_index: Optional[int] = None
    audio_path: Optional[str] = None
    original_track_id: Optional[str] = None
    beats: Optional[List[float]] = None
    downbeats: Optional[List[float]] = None


class DatasetIngester:
    """Handles ingestion of various music structure datasets."""

    def __init__(self, data_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_mapper = LabelMapper()

    def ingest_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Ingest a specific dataset.

        Args:
            dataset_name: Name of the dataset to ingest

        Returns:
            DataFrame with unified annotations
        """
        dataset_dir = self.data_dir / dataset_name

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        logger.info(f"Ingesting dataset: {dataset_name}")

        df: pd.DataFrame
        if dataset_name == "harmonix":
            df = self._ingest_harmonix(dataset_dir)
        elif dataset_name == "salami":
            df = self._ingest_salami(dataset_dir)
        elif dataset_name == "beatles":
            df = self._ingest_beatles(dataset_dir)
        elif dataset_name == "spam":
            df = self._ingest_spam(dataset_dir)
        elif dataset_name == "ccmusic":
            df = self._ingest_ccmusic(dataset_dir)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Persist normalized annotations for reproducibility
        output_path = self.output_dir / f"{dataset_name}.jsonl"
        df.to_json(output_path, orient="records", lines=True)
        logger.info(
            f"Saved normalized annotations for {dataset_name} "
            f"to {output_path} ({len(df)} tracks)"
        )

        return df

    def _ingest_harmonix(self, dataset_dir: Path) -> pd.DataFrame:
        """Ingest Harmonix Set dataset."""
        annotations_file = dataset_dir / "annotations.json"

        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

        with open(annotations_file, "r") as f:
            data = json.load(f)

        tracks = []
        for track_id, track_data in data.items():
            try:
                # Extract structure annotations
                structure = track_data.get("structure", {})
                boundaries = structure.get("boundaries", [])
                labels = structure.get("labels", [])

                # Map labels to canonical form
                mapped_labels = map_labels(labels, "harmonix")

                # Get audio info
                audio_path = track_data.get("audio_path")
                duration = track_data.get("duration", 0.0)
                sr = track_data.get("sr", 22050)

                # Get beats if available
                beats = track_data.get("beats")
                downbeats = track_data.get("downbeats")

                track = TrackAnnotation(
                    track_id=track_id,
                    dataset="harmonix",
                    sr=sr,
                    duration=duration,
                    boundary_times=boundaries,
                    boundary_labels=mapped_labels,
                    audio_path=audio_path,
                    beats=beats,
                    downbeats=downbeats,
                )
                tracks.append(track)

            except Exception as e:
                logger.warning(f"Failed to process track {track_id}: {e}")
                continue

        return self._tracks_to_dataframe(tracks)

    def _ingest_salami(self, dataset_dir: Path) -> pd.DataFrame:
        """Ingest SALAMI dataset."""
        annotations_dir = dataset_dir / "annotations"

        if not annotations_dir.exists():
            raise FileNotFoundError(
                f"Annotations directory not found: {annotations_dir}"
            )

        tracks = []
        for txt_file in annotations_dir.glob("*.txt"):
            try:
                track_id = txt_file.stem

                # Parse SALAMI format
                boundaries, labels = self._parse_salami_file(txt_file)

                # Map labels
                mapped_labels = map_labels(labels, "salami")

                track = TrackAnnotation(
                    track_id=track_id,
                    dataset="salami",
                    sr=22050,  # Default
                    duration=max(boundaries) if boundaries else 0.0,
                    boundary_times=boundaries,
                    boundary_labels=mapped_labels,
                )
                tracks.append(track)

            except Exception as e:
                logger.warning(f"Failed to process SALAMI file {txt_file}: {e}")
                continue

        return self._tracks_to_dataframe(tracks)

    def _parse_salami_file(self, file_path: Path) -> Tuple[List[float], List[str]]:
        """Parse SALAMI annotation file."""
        boundaries = []
        labels = []

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                if len(parts) >= 2:
                    try:
                        time = float(parts[0])
                        label = parts[1]
                        boundaries.append(time)
                        labels.append(label)
                    except ValueError:
                        continue

        return boundaries, labels

    def _ingest_beatles(self, dataset_dir: Path) -> pd.DataFrame:
        """Ingest Beatles/Isophonics dataset."""
        annotations_file = dataset_dir / "annotations.json"

        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

        with open(annotations_file, "r") as f:
            data = json.load(f)

        tracks = []
        for track_id, track_data in data.items():
            try:
                boundaries = track_data.get("boundaries", [])
                labels = track_data.get("labels", [])

                mapped_labels = map_labels(labels, "beatles")

                track = TrackAnnotation(
                    track_id=track_id,
                    dataset="beatles",
                    sr=22050,
                    duration=max(boundaries) if boundaries else 0.0,
                    boundary_times=boundaries,
                    boundary_labels=mapped_labels,
                )
                tracks.append(track)

            except Exception as e:
                logger.warning(f"Failed to process Beatles track {track_id}: {e}")
                continue

        return self._tracks_to_dataframe(tracks)

    def _ingest_spam(self, dataset_dir: Path) -> pd.DataFrame:
        """Ingest SPAM dataset."""
        annotations_dir = dataset_dir / "annotations"

        if not annotations_dir.exists():
            raise FileNotFoundError(
                f"Annotations directory not found: {annotations_dir}"
            )

        tracks = []
        for txt_file in annotations_dir.glob("*.txt"):
            try:
                track_id = txt_file.stem
                boundaries, labels = self._parse_spam_file(txt_file)

                mapped_labels = map_labels(labels, "spam")

                track = TrackAnnotation(
                    track_id=track_id,
                    dataset="spam",
                    sr=22050,
                    duration=max(boundaries) if boundaries else 0.0,
                    boundary_times=boundaries,
                    boundary_labels=mapped_labels,
                )
                tracks.append(track)

            except Exception as e:
                logger.warning(f"Failed to process SPAM file {txt_file}: {e}")
                continue

        return self._tracks_to_dataframe(tracks)

    def _parse_spam_file(self, file_path: Path) -> Tuple[List[float], List[str]]:
        """Parse SPAM annotation file."""
        boundaries = []
        labels = []

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    try:
                        time = float(parts[0])
                        label = " ".join(parts[1:])
                        boundaries.append(time)
                        labels.append(label)
                    except ValueError:
                        continue

        return boundaries, labels

    def _ingest_ccmusic(self, dataset_dir: Path) -> pd.DataFrame:
        """Ingest CCMUSIC dataset."""
        from datasets import load_from_disk, Audio

        tracks: List[TrackAnnotation] = []
        dataset_root = dataset_dir / "default"

        if not dataset_root.exists():
            raise FileNotFoundError(f"CCMUSIC dataset not found at {dataset_root}")

        dataset = load_from_disk(str(dataset_root))
        dataset = dataset.cast_column("audio", Audio(decode=False))

        for split_name, split_dataset in dataset.items():
            logger.info(
                f"Processing CCMUSIC split '{split_name}' with {len(split_dataset)} tracks"
            )
            for idx, row in enumerate(split_dataset):
                try:
                    audio_info = row["audio"]
                    label_info = row["label"]

                    onset_times = label_info.get("onset_time", [])
                    offset_times = label_info.get("offset_time", [])
                    structures = label_info.get("structure", [])

                    if (
                        not onset_times
                        or not offset_times
                        or not structures
                        or len(structures) != len(onset_times)
                    ):
                        logger.warning(
                            f"Missing annotations for CCMUSIC track idx={idx} split={split_name}"
                        )
                        continue

                    # Convert centiseconds to seconds
                    segment_starts = [t / 100.0 for t in onset_times]
                    segment_ends = [t / 100.0 for t in offset_times]

                    if not segment_ends:
                        continue

                    duration = segment_ends[-1]
                    boundary_times = segment_starts + [duration]
                    boundary_times = self._ensure_strictly_increasing(boundary_times)

                    if len(boundary_times) != len(structures) + 1:
                        logger.warning(
                            "Boundary/label mismatch for CCMUSIC track idx=%s split=%s",
                            idx,
                            split_name,
                        )
                        continue

                    mapped_labels = map_labels(structures, "ccmusic")

                    original_name = audio_info.get("path") or structures[0]
                    track_id = self._build_track_id(
                        dataset_name="ccmusic",
                        split=split_name,
                        index=idx,
                        original_name=original_name,
                    )

                    try:
                        audio_path = self._save_ccmusic_audio(
                            audio_info=audio_info,
                            dataset_dir=dataset_dir,
                            split_name=split_name,
                            track_id=track_id,
                        )
                    except Exception as audio_error:
                        logger.warning(
                            "Failed to materialize audio for CCMUSIC track %s: %s",
                            track_id,
                            audio_error,
                        )
                        continue

                    track = TrackAnnotation(
                        track_id=track_id,
                        dataset="ccmusic",
                        split=split_name,
                        sr=audio_info.get("sampling_rate", 22050),
                        duration=duration,
                        boundary_times=boundary_times,
                        boundary_labels=mapped_labels,
                        source_index=idx,
                        audio_path=audio_path,
                        original_track_id=original_name,
                    )
                    tracks.append(track)
                except Exception as e:
                    logger.warning(f"Failed to process CCMUSIC track {idx}: {e}")

        return self._tracks_to_dataframe(tracks)

    def _tracks_to_dataframe(self, tracks: List[TrackAnnotation]) -> pd.DataFrame:
        """Convert track annotations to DataFrame."""
        data = []
        for track in tracks:
            data.append(
                {
                    "track_id": track.track_id,
                    "dataset": track.dataset,
                    "split": track.split,
                    "sr": track.sr,
                    "duration": track.duration,
                    "boundary_times": track.boundary_times,
                    "boundary_labels": track.boundary_labels,
                    "audio_path": track.audio_path,
                    "original_track_id": track.original_track_id,
                    "source_index": track.source_index,
                    "beats": track.beats,
                    "downbeats": track.downbeats,
                }
            )

        return pd.DataFrame(data)

    def ingest_all_datasets(self) -> pd.DataFrame:
        """Ingest all available datasets."""
        all_tracks = []

        for dataset_name in ["harmonix", "salami", "beatles", "spam", "ccmusic"]:
            dataset_dir = self.data_dir / dataset_name
            if dataset_dir.exists():
                try:
                    df = self.ingest_dataset(dataset_name)
                    all_tracks.append(df)
                    logger.info(f"Ingested {len(df)} tracks from {dataset_name}")
                except Exception as e:
                    logger.error(f"Failed to ingest {dataset_name}: {e}")

        if not all_tracks:
            raise RuntimeError("No datasets could be ingested")

        combined_df = pd.concat(all_tracks, ignore_index=True)

        # Save combined dataset
        output_file = self.output_dir / "combined_annotations.csv"
        combined_df.to_csv(output_file, index=False)
        logger.info(
            f"Saved combined dataset with {len(combined_df)} tracks to {output_file}"
        )

        return combined_df

    def create_splits(
        self, df: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits grouped by artist."""
        # For now, use random splits (in practice, group by artist)
        np.random.seed(42)
        indices = np.random.permutation(len(df))

        n_test = int(len(df) * test_size)
        n_val = int(len(df) * val_size)

        test_indices = indices[:n_test]
        val_indices = indices[n_test : n_test + n_val]
        train_indices = indices[n_test + n_val :]

        splits = {
            "train": df.iloc[train_indices],
            "validation": df.iloc[val_indices],
            "test": df.iloc[test_indices],
        }

        # Save splits
        splits_dir = self.output_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        for split_name, split_df in splits.items():
            split_file = splits_dir / f"{split_name}.jsonl"
            split_df.to_json(split_file, orient="records", lines=True)
            logger.info(f"Saved {split_name} split with {len(split_df)} tracks")

        return splits

    # ------------------------------------------------------------------
    # Helper utilities

    def _slugify(self, value: str) -> str:
        """Create a filesystem-safe slug from an arbitrary string."""
        if value is None:
            return ""
        value = value.lower()
        value = re.sub(r"[^a-z0-9]+", "-", value)
        value = re.sub(r"-+", "-", value)
        return value.strip("-")

    def _build_track_id(
        self, dataset_name: str, split: str, index: int, original_name: Optional[str]
    ) -> str:
        """Build a deterministic, filesystem-friendly track identifier."""
        base = self._slugify(original_name or "")
        suffix = f"{dataset_name}_{split}_{index:05d}"
        if base:
            base = base[:80]
            return f"{suffix}_{base}"
        return suffix

    def _save_ccmusic_audio(
        self,
        audio_info: Dict[str, Any],
        dataset_dir: Path,
        split_name: str,
        track_id: str,
    ) -> str:
        """Materialize CCMUSIC audio bytes to disk and return the local path."""
        audio_dir = dataset_dir / "audio" / split_name
        audio_dir.mkdir(parents=True, exist_ok=True)

        raw_path = audio_info.get("path") if isinstance(audio_info, dict) else None
        extension = Path(raw_path).suffix if raw_path else ".mp3"
        if not extension:
            extension = ".mp3"

        target_path = audio_dir / f"{track_id}{extension}"

        if target_path.exists():
            return str(target_path.resolve())

        audio_bytes = (
            audio_info.get("bytes") if isinstance(audio_info, dict) else None
        )
        if audio_bytes:
            target_path.write_bytes(audio_bytes)
            return str(target_path.resolve())

        if raw_path and Path(raw_path).exists():
            shutil.copyfile(raw_path, target_path)
            return str(target_path.resolve())

        raise FileNotFoundError("No audio data available for CCMUSIC track")

    def _ensure_strictly_increasing(self, times: List[float]) -> List[float]:
        """Ensure boundary times are strictly increasing."""
        result: List[float] = []
        last_time = -float("inf")
        for time_val in times:
            if time_val <= last_time:
                if time_val + 1e-3 <= last_time:
                    continue
                time_val = last_time + 1e-3
            result.append(time_val)
            last_time = time_val
        return result


def main():
    """CLI interface for data ingestion."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest music structure datasets")
    parser.add_argument("--dataset", help="Specific dataset to ingest")
    parser.add_argument("--all", action="store_true", help="Ingest all datasets")
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    parser.add_argument(
        "--output-dir", default="data/processed", help="Processed data directory"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    ingester = DatasetIngester(data_dir=args.data_dir, output_dir=args.output_dir)

    if args.all:
        df = ingester.ingest_all_datasets()
        splits = ingester.create_splits(df)
        print(f"Ingested {len(df)} tracks total")
        for split_name, split_df in splits.items():
            print(f"  {split_name}: {len(split_df)} tracks")
    elif args.dataset:
        df = ingester.ingest_dataset(args.dataset)
        print(f"Ingested {len(df)} tracks from {args.dataset}")
    else:
        print("Available datasets: harmonix, salami, beatles, spam, ccmusic")


if __name__ == "__main__":
    main()
