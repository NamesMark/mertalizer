"""
Two-head architecture for music structure recognition.

Implements TCN/Transformer boundary detection and CRF labeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.signal import find_peaks
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the two-head model."""

    # Encoder settings
    encoder_model: str = "mert"  # "mert" or "w2v"
    encoder_frozen: bool = True
    encoder_lr: float = 1e-5
    embedding_dim: int = 768

    # Boundary head settings
    boundary_head_type: str = "tcn"  # "tcn" or "transformer"
    boundary_hidden_dim: int = 256
    boundary_num_layers: int = 4
    boundary_dropout: float = 0.1
    boundary_kernel_size: int = 3
    boundary_dilation_base: int = 2

    # Transformer-specific settings
    transformer_num_heads: int = 8
    transformer_num_layers: int = 2

    # Label head settings
    label_hidden_dim: int = 128
    num_labels: int = 8  # INTRO, VERSE, PRE, CHORUS, BRIDGE, SOLO, OUTRO, OTHER

    # Loss settings
    boundary_loss_weight: float = 1.0
    label_loss_weight: float = 1.0
    focal_gamma: float = 2.0

    # Training settings
    learning_rate: float = 2e-4
    weight_decay: float = 0.01


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Apply residual connection
        if self.residual is not None:
            residual = self.residual(residual)

        # Crop to match input length
        out = out[..., : x.size(-1)]
        residual = residual[..., : x.size(-1)]

        return out + residual


class TCNBoundaryHead(nn.Module):
    """TCN-based boundary detection head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dilation_base: int,
        dropout: float,
    ):
        super().__init__()

        layers = []
        in_channels = input_dim

        for i in range(num_layers):
            dilation = dilation_base**i
            layers.append(
                TCNBlock(in_channels, hidden_dim, kernel_size, dilation, dropout)
            )
            in_channels = hidden_dim

        self.tcn_layers = nn.ModuleList(layers)
        self.classifier = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)

        for layer in self.tcn_layers:
            x = layer(x)

        # Classify each time step
        logits = self.classifier(x)  # (batch, 1, seq_len)
        logits = logits.transpose(1, 2)  # (batch, seq_len, 1)

        return logits.squeeze(-1)  # (batch, seq_len)


class TransformerBoundaryHead(nn.Module):
    """Transformer-based boundary detection head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_projection(x)

        # Add positional encoding
        seq_len = x.size(1)
        pos_encoding = self._get_positional_encoding(seq_len, x.size(-1), x.device)
        x = x + pos_encoding

        # Apply transformer
        x = self.transformer(x)

        # Classify each time step
        logits = self.classifier(x)  # (batch, seq_len, 1)

        return logits.squeeze(-1)  # (batch, seq_len)

    def _get_positional_encoding(
        self, seq_len: int, d_model: int, device: torch.device
    ) -> torch.Tensor:
        """Generate positional encoding."""
        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(
            1
        )

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=device)
            * -(np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)


class LabelHead(nn.Module):
    """Linear label classification head."""

    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        return self.classifier(x)  # (batch, seq_len, num_labels)


class MusicStructureModel(pl.LightningModule):
    """Two-head model operating on precomputed embeddings."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.embedding_dim = config.embedding_dim

        # Boundary detection head
        if config.boundary_head_type == "tcn":
            self.boundary_head = TCNBoundaryHead(
                input_dim=self.embedding_dim,
                hidden_dim=config.boundary_hidden_dim,
                num_layers=config.boundary_num_layers,
                kernel_size=config.boundary_kernel_size,
                dilation_base=config.boundary_dilation_base,
                dropout=config.boundary_dropout,
            )
        elif config.boundary_head_type == "transformer":
            self.boundary_head = TransformerBoundaryHead(
                input_dim=self.embedding_dim,
                hidden_dim=config.boundary_hidden_dim,
                num_layers=config.transformer_num_layers,
                num_heads=config.transformer_num_heads,
                dropout=config.boundary_dropout,
            )
        else:
            raise ValueError(f"Unknown boundary head type: {config.boundary_head_type}")

        # Label classification head
        self.label_head = LabelHead(
            input_dim=self.embedding_dim,
            hidden_dim=config.label_hidden_dim,
            num_labels=config.num_labels,
        )

        # Loss functions
        self.boundary_criterion = nn.BCEWithLogitsLoss()
        self.label_criterion = nn.CrossEntropyLoss()

    def forward(
        self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass on precomputed embeddings."""
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)

        # Boundary detection
        boundary_logits = self.boundary_head(embeddings)

        # Label classification
        label_logits = self.label_head(embeddings)

        return {
            "boundary_logits": boundary_logits,
            "label_logits": label_logits,
            "embeddings": embeddings,
        }

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        embeddings = batch["embeddings"]
        boundary_targets = batch["boundary_targets"]
        label_targets = batch["label_targets"]
        mask = batch["mask"].bool()

        outputs = self.forward(embeddings, mask)
        boundary_logits = outputs["boundary_logits"]
        label_logits = outputs["label_logits"]

        boundary_loss = self._boundary_loss(boundary_logits, boundary_targets, mask)
        label_loss = self._label_loss(label_logits, label_targets, mask)

        # Combined loss
        total_loss = (
            self.config.boundary_loss_weight * boundary_loss
            + self.config.label_loss_weight * label_loss
        )

        # Logging
        self.log("train/boundary_loss", boundary_loss, on_step=True, on_epoch=True)
        self.log("train/label_loss", label_loss, on_step=True, on_epoch=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        embeddings = batch["embeddings"]
        boundary_targets = batch["boundary_targets"]
        label_targets = batch["label_targets"]
        mask = batch["mask"].bool()

        outputs = self.forward(embeddings, mask)
        boundary_logits = outputs["boundary_logits"]
        label_logits = outputs["label_logits"]

        boundary_loss = self._boundary_loss(boundary_logits, boundary_targets, mask)
        label_loss = self._label_loss(label_logits, label_targets, mask)

        total_loss = (
            self.config.boundary_loss_weight * boundary_loss
            + self.config.label_loss_weight * label_loss
        )

        # Compute metrics
        boundary_preds = torch.sigmoid(boundary_logits) > 0.5
        boundary_acc = (
            boundary_preds[mask] == (boundary_targets[mask] > 0.5)
        ).float()
        boundary_acc = boundary_acc.mean() if boundary_acc.numel() else torch.tensor(0.0)

        label_preds = torch.argmax(label_logits, dim=-1)
        valid_label_mask = mask & (label_targets != -100)
        if valid_label_mask.any():
            label_acc = (
                label_preds[valid_label_mask] == label_targets[valid_label_mask]
            ).float().mean()
        else:
            label_acc = torch.tensor(0.0, device=self.device)

        # Logging
        self.log("val/boundary_loss", boundary_loss, on_step=False, on_epoch=True)
        self.log("val/label_loss", label_loss, on_step=False, on_epoch=True)
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("val/boundary_acc", boundary_acc, on_step=False, on_epoch=True)
        self.log("val/label_acc", label_acc, on_step=False, on_epoch=True)

        return total_loss

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.validation_step(batch, batch_idx)

    def _boundary_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        active_logits = logits[mask]
        active_targets = targets[mask]

        if self.config.focal_gamma > 0:
            loss = self._focal_loss(active_logits, active_targets)
        else:
            loss = F.binary_cross_entropy_with_logits(active_logits, active_targets)
        return loss

    def _label_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = mask & (targets != -100)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device)

        active_logits = logits[valid_mask]
        active_targets = targets[valid_mask]
        return F.cross_entropy(active_logits, active_targets, ignore_index=-100)

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.config.focal_gamma * ce_loss
        return focal_loss.mean()

    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def predict_boundaries(
        self,
        embeddings: torch.Tensor,
        frame_times: np.ndarray,
        threshold: float = 0.5,
        mask: Optional[torch.Tensor] = None,
    ) -> List[float]:
        """Predict boundary times from embeddings."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(embeddings)
            boundary_probs = torch.sigmoid(outputs["boundary_logits"])

            if boundary_probs.dim() == 2:
                boundary_probs = boundary_probs.squeeze(0)

            if mask is not None:
                boundary_probs = boundary_probs.masked_fill(~mask, 0.0)

            prob_numpy = boundary_probs.detach().cpu().numpy().astype(np.float32)
            prob_numpy = np.reshape(prob_numpy, (-1,))

            if len(frame_times) > 1:
                frame_step = max(frame_times[1] - frame_times[0], 1e-3)
            else:
                frame_step = 1.0
            min_spacing_seconds = 1.0
            distance = max(1, int(min_spacing_seconds / frame_step))

            peak_indices, _ = find_peaks(
                prob_numpy, height=threshold, distance=distance
            )
            boundary_indices = peak_indices

            boundary_times = frame_times[boundary_indices]
            return boundary_times.tolist()

    def predict_labels(
        self,
        embeddings: torch.Tensor,
        frame_times: np.ndarray,
        boundaries: List[float],
        mask: Optional[torch.Tensor] = None,
    ) -> List[str]:
        """Predict labels for segments defined by boundaries."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(embeddings)
            label_logits = outputs["label_logits"]

            if label_logits.dim() == 3:
                label_logits = label_logits.squeeze(0)

            labels = []
            label_names = [
                "INTRO",
                "VERSE",
                "PRE",
                "CHORUS",
                "BRIDGE",
                "SOLO",
                "OUTRO",
                "OTHER",
            ]

            if not boundaries:
                return labels

            boundary_frames = np.searchsorted(frame_times, boundaries, side="left")
            boundary_frames = np.clip(boundary_frames, 0, len(frame_times) - 1)

            for start_frame, end_frame in zip(boundary_frames[:-1], boundary_frames[1:]):
                end_frame = max(end_frame, start_frame + 1)
                segment_logits = label_logits[start_frame:end_frame]
                predicted_label = torch.argmax(segment_logits.mean(dim=0)).item()
                labels.append(label_names[predicted_label])

            return labels


def create_model(config: ModelConfig, ssl_model=None) -> MusicStructureModel:
    """Create a music structure model with given configuration."""
    if ssl_model is not None:
        logger.warning("ssl_model argument is deprecated and ignored.")
    return MusicStructureModel(config)


if __name__ == "__main__":
    # Test the model
    config = ModelConfig()
    model = create_model(config)

    # Test forward pass
    batch_size, seq_len = 2, 1000
    embeddings = torch.randn(batch_size, seq_len, config.embedding_dim)

    outputs = model(embeddings)
    print(f"Boundary logits shape: {outputs['boundary_logits'].shape}")
    print(f"Label logits shape: {outputs['label_logits'].shape}")
