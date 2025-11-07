"""
Training script for music structure recognition models.

Implements LightningModule training with proper data loading and evaluation.
"""

import os
import yaml
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import json

from modeling.system import MusicStructureModel, ModelConfig, create_model
from data.preprocessing import AudioPreprocessor, AudioConfig
from data.ontology import LabelMapper

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Model settings
    model_config: ModelConfig

    # Data settings
    data_dir: str = "data/processed"
    batch_size: int = 8
    max_epochs: int = 100
    num_workers: int = 4

    # Training settings
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0

    # Validation settings
    val_check_interval: float = 0.25
    early_stopping_patience: int = 10
    monitor_metric: str = "val/total_loss"
    monitor_mode: str = "min"

    # Logging settings
    use_wandb: bool = True
    project_name: str = "mertalizer"
    experiment_name: Optional[str] = None

    # Checkpoint settings
    checkpoint_dir: str = "models/checkpoints"
    save_top_k: int = 3

    # Audio settings
    target_sr: int = 22050
    max_audio_length: float = 600.0  # seconds per batch


class MusicStructureDataset(Dataset):
    """Dataset wrapping precomputed SSL embeddings and annotations."""

    def __init__(self, data_file: str, embeddings_dir: str, max_length: float = 600.0):
        self.data_file = str(data_file)
        self.embeddings_dir = Path(embeddings_dir)
        self.max_length = max_length
        self.label_mapper = LabelMapper()

        if self.data_file.endswith(".jsonl"):
            with open(self.data_file, "r") as f:
                self.data = [json.loads(line) for line in f]
        else:
            self.data = pd.read_csv(self.data_file).to_dict("records")

        filtered_data = []
        missing_embeddings = 0
        for record in self.data:
            if self._embedding_path(record).exists():
                filtered_data.append(record)
            else:
                missing_embeddings += 1

        self.data = filtered_data
        logger.info(
            "Loaded %d tracks from %s (%d missing embeddings)",
            len(self.data),
            data_file,
            missing_embeddings,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        track = self.data[idx]
        track_id = track["track_id"]
        embedding_file = self._embedding_path(track)

        npz = np.load(embedding_file)
        embeddings = npz["embeddings"]
        frame_times = npz.get("frame_times")
        beats = npz.get("beats", np.array([]))
        duration = float(npz.get("duration", [track.get("duration", 0.0)])[0])

        if frame_times is None or len(frame_times) != len(embeddings):
            # Derive frame times assuming uniform spacing
            if duration > 0 and len(embeddings):
                frame_times = np.linspace(0.0, duration, num=len(embeddings), endpoint=False)
            else:
                frame_times = np.arange(len(embeddings))

        # Optional truncation based on max_length (seconds)
        if self.max_length and duration > 0:
            approx_frame_rate = len(frame_times) / max(duration, 1e-6)
            max_frames = int(self.max_length * approx_frame_rate)
            if max_frames and len(embeddings) > max_frames:
                embeddings = embeddings[:max_frames]
                frame_times = frame_times[:max_frames]

        boundary_times = track.get("boundary_times", [])
        boundary_labels = track.get("boundary_labels", [])

        boundary_targets = self._create_boundary_targets(boundary_times, frame_times)
        label_targets = self._create_label_targets(
            boundary_times, boundary_labels, frame_times
        )

        return {
            "track_id": track_id,
            "embeddings": torch.from_numpy(embeddings).float(),
            "boundary_targets": torch.from_numpy(boundary_targets).float(),
            "label_targets": torch.from_numpy(label_targets).long(),
            "frame_times": torch.from_numpy(frame_times).float(),
            "beats": torch.from_numpy(beats).float(),
            "duration": duration,
        }

    def _create_boundary_targets(
        self, boundary_times: List[float], frame_times: np.ndarray
    ) -> np.ndarray:
        targets = np.zeros(len(frame_times), dtype=np.float32)
        if not boundary_times or len(frame_times) == 0:
            return targets

        indices = np.searchsorted(frame_times, boundary_times, side="left")
        indices = np.clip(indices, 0, len(frame_times) - 1)
        for idx in np.unique(indices):
            targets[idx] = 1.0
        return targets

    def _embedding_path(self, track: Dict[str, Any]) -> Path:
        dataset_name = track.get("dataset", "default")
        track_id = track["track_id"]
        return self.embeddings_dir / dataset_name / f"{track_id}.npz"

    def _create_label_targets(
        self,
        boundary_times: List[float],
        boundary_labels: List[str],
        frame_times: np.ndarray,
    ) -> np.ndarray:
        num_frames = len(frame_times)
        if num_frames == 0:
            return np.zeros(0, dtype=np.int64)

        label_to_idx = {
            label: i for i, label in enumerate(self.label_mapper.get_canonical_labels())
        }
        default_idx = label_to_idx.get("OTHER", len(label_to_idx) - 1)

        targets = np.full(num_frames, default_idx, dtype=np.int64)

        if not boundary_labels or len(boundary_times) < len(boundary_labels) + 1:
            return targets

        for seg_idx, label in enumerate(boundary_labels):
            start_time = boundary_times[seg_idx]
            end_time = boundary_times[seg_idx + 1]

            start_frame = int(np.searchsorted(frame_times, start_time, side="left"))
            end_frame = int(np.searchsorted(frame_times, end_time, side="left"))
            end_frame = max(end_frame, start_frame + 1)
            end_frame = min(end_frame, num_frames)

            label_idx = label_to_idx.get(label, default_idx)
            targets[start_frame:end_frame] = label_idx

        return targets


class MusicStructureDataModule(pl.LightningDataModule):
    """Data module for music structure recognition."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        data_dir = Path(self.config.data_dir)
        splits_dir = data_dir / "splits"
        embeddings_dir = data_dir / "embeddings"

        if stage == "fit" or stage is None:
            self.train_dataset = MusicStructureDataset(
                splits_dir / "train.jsonl", embeddings_dir, self.config.max_audio_length
            )
            self.val_dataset = MusicStructureDataset(
                splits_dir / "validation.jsonl",
                embeddings_dir,
                self.config.max_audio_length,
            )
            if len(self.val_dataset) == 0:
                logger.warning("Validation dataset is empty; disabling validation phase.")
                self.val_dataset = None

        if stage == "test" or stage is None:
            self.test_dataset = MusicStructureDataset(
                splits_dir / "test.jsonl", embeddings_dir, self.config.max_audio_length
            )
            if len(self.test_dataset) == 0:
                logger.warning("Test dataset is empty; disabling test phase.")
                self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return []
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        if self.test_dataset is None or len(self.test_dataset) == 0:
            return []
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for variable-length sequences."""
        # Pad sequences to same length
        max_length = max(item["embeddings"].size(0) for item in batch)
        padded_embeddings = []
        padded_boundary_targets = []
        padded_label_targets = []
        masks = []
        track_ids = []
        frame_times = []
        beats = []
        durations = []

        for item in batch:
            embeddings = item["embeddings"]
            boundary_targets = item["boundary_targets"]
            label_targets = item["label_targets"]

            length = embeddings.size(0)
            pad_length = max_length - length

            if pad_length > 0:
                embeddings = F.pad(embeddings, (0, 0, 0, pad_length))
                boundary_targets = F.pad(boundary_targets, (0, pad_length))
                label_targets = F.pad(
                    label_targets, (0, pad_length), value=-100
                )

            mask = torch.zeros(max_length, dtype=torch.bool)
            mask[:length] = True

            padded_embeddings.append(embeddings)
            padded_boundary_targets.append(boundary_targets)
            padded_label_targets.append(label_targets)
            masks.append(mask)
            track_ids.append(item.get("track_id"))
            frame_times.append(item.get("frame_times"))
            beats.append(item.get("beats"))
            durations.append(item.get("duration", 0.0))

        return {
            "embeddings": torch.stack(padded_embeddings),
            "boundary_targets": torch.stack(padded_boundary_targets),
            "label_targets": torch.stack(padded_label_targets),
            "mask": torch.stack(masks),
            "track_ids": track_ids,
            "frame_times": frame_times,
            "beats": beats,
            "durations": torch.tensor(durations, dtype=torch.float32),
        }


def load_config(config_path: str) -> TrainingConfig:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Create model config
    model_config = ModelConfig(**config_dict.get("model", {}))

    # Create training config
    training_config = TrainingConfig(
        model_config=model_config,
        **{k: v for k, v in config_dict.items() if k != "model"},
    )

    return training_config


def create_callbacks(config: TrainingConfig) -> List[pl.Callback]:
    """Create training callbacks."""
    callbacks = []

    # Model checkpoint
    monitor_tag = config.monitor_metric.replace("/", "_")

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename=f"best-{{epoch:02d}}-{monitor_tag}",
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=config.save_top_k,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    if config.early_stopping_patience and config.early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            monitor=config.monitor_metric,
            mode=config.monitor_mode,
            patience=config.early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stopping)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    return callbacks


def create_logger(config: TrainingConfig) -> Optional[pl.loggers.Logger]:
    """Create logger for training."""
    if config.use_wandb:
        wandb_key = os.getenv("WANDB_KEY")
        if wandb_key:
            return WandbLogger(
                project=config.project_name,
                name=config.experiment_name,
                api_key=wandb_key,
            )
        else:
            logger.warning("WANDB_KEY not found, using TensorBoard logger")

    return pl.loggers.TensorBoardLogger("logs/")


def train_model(config: TrainingConfig):
    """Train the music structure recognition model."""
    # Set random seeds
    pl.seed_everything(42)

    # Create data module
    data_module = MusicStructureDataModule(config)

    # Create model
    model = create_model(config.model_config)

    # Create callbacks
    callbacks = create_callbacks(config)

    # Create logger
    logger = create_logger(config)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.gradient_clip_val,
        val_check_interval=config.val_check_interval,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32",
    )

    # Train model
    trainer.fit(model, data_module)

    # Test model
    trainer.test(model, data_module)

    return model, trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train music structure recognition model"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    parser.add_argument("--test-only", action="store_true", help="Only run testing")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load configuration
    config = load_config(args.config)

    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if args.test_only:
        # Load model from checkpoint
        model = MusicStructureModel.load_from_checkpoint(args.resume)
        data_module = MusicStructureDataModule(config)
        trainer = pl.Trainer()
        trainer.test(model, data_module)
    else:
        # Train model
        model, trainer = train_model(config)

        # Save final model
        final_model_path = Path(config.checkpoint_dir) / "final_model.ckpt"
        trainer.save_checkpoint(final_model_path)
        logger.info(f"Saved final model to {final_model_path}")


if __name__ == "__main__":
    main()
