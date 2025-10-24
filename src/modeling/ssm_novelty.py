"""
Classic baseline model for music structure recognition.

Implements self-similarity matrix novelty detection and majority vote labeling.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import logging
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SelfSimilarityMatrix:
    """Self-similarity matrix computation for novelty detection."""

    def __init__(self, window_size: int = 30, hop_length: int = 1):
        self.window_size = window_size
        self.hop_length = hop_length

    def compute_ssm(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute self-similarity matrix.

        Args:
            embeddings: Feature embeddings of shape (time_frames, feature_dim)

        Returns:
            Self-similarity matrix of shape (time_frames, time_frames)
        """
        # Normalize embeddings
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute cosine similarity
        ssm = cosine_similarity(embeddings_norm)

        return ssm

    def compute_novelty(self, ssm: np.ndarray) -> np.ndarray:
        """
        Compute novelty function from self-similarity matrix.

        Args:
            ssm: Self-similarity matrix

        Returns:
            Novelty function
        """
        # Create checkerboard kernel
        kernel_size = self.window_size
        kernel = np.ones((kernel_size, kernel_size))

        # Checkerboard pattern
        for i in range(kernel_size):
            for j in range(kernel_size):
                if (i + j) % 2 == 0:
                    kernel[i, j] = 1
                else:
                    kernel[i, j] = -1

        # Normalize kernel
        kernel = kernel / np.sum(np.abs(kernel))

        # Convolve with kernel
        novelty = ndimage.convolve(ssm, kernel, mode="constant")

        # Take diagonal (lag = 0)
        novelty_diag = np.diag(novelty)

        return novelty_diag


class ClassicBaseline:
    """Classic baseline model for music structure recognition."""

    def __init__(
        self,
        window_size: int = 30,
        hop_length: int = 1,
        novelty_threshold: float = 0.1,
        min_segment_length: float = 2.0,
    ):
        self.window_size = window_size
        self.hop_length = hop_length
        self.novelty_threshold = novelty_threshold
        self.min_segment_length = min_segment_length

        self.ssm_computer = SelfSimilarityMatrix(window_size, hop_length)
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

    def detect_boundaries(self, embeddings: np.ndarray, sr: int = 22050) -> List[float]:
        """
        Detect boundaries using novelty detection.

        Args:
            embeddings: Feature embeddings
            sr: Sample rate

        Returns:
            List of boundary times in seconds
        """
        # Compute self-similarity matrix
        ssm = self.ssm_computer.compute_ssm(embeddings)

        # Compute novelty function
        novelty = self.ssm_computer.compute_novelty(ssm)

        # Find peaks in novelty function
        boundaries = self._find_novelty_peaks(novelty, sr)

        return boundaries

    def _find_novelty_peaks(self, novelty: np.ndarray, sr: int) -> List[float]:
        """Find peaks in novelty function."""
        # Convert to time
        frame_rate = sr / 100  # Assuming 100 frames per second
        times = np.arange(len(novelty)) / frame_rate

        # Find peaks above threshold
        peaks = []
        for i in range(1, len(novelty) - 1):
            if (
                novelty[i] > self.novelty_threshold
                and novelty[i] > novelty[i - 1]
                and novelty[i] > novelty[i + 1]
            ):
                peaks.append(times[i])

        # Filter by minimum segment length
        filtered_peaks = []
        for peak in peaks:
            if (
                not filtered_peaks
                or peak - filtered_peaks[-1] >= self.min_segment_length
            ):
                filtered_peaks.append(peak)

        return filtered_peaks

    def predict_labels(
        self, embeddings: np.ndarray, boundaries: List[float], sr: int = 22050
    ) -> List[str]:
        """
        Predict labels using majority vote from frame classifier.

        Args:
            embeddings: Feature embeddings
            boundaries: Boundary times
            sr: Sample rate

        Returns:
            List of predicted labels
        """
        if len(boundaries) < 2:
            return ["OTHER"]

        # Convert boundaries to frame indices
        frame_rate = sr / 100
        boundary_frames = [int(b * frame_rate) for b in boundaries]

        labels = []

        for i in range(len(boundary_frames) - 1):
            start_frame = boundary_frames[i]
            end_frame = boundary_frames[i + 1]

            # Extract segment embeddings
            segment_embeddings = embeddings[start_frame:end_frame]

            # Predict label for segment
            label = self._classify_segment(segment_embeddings)
            labels.append(label)

        return labels

    def _classify_segment(self, segment_embeddings: np.ndarray) -> str:
        """Classify a segment using simple heuristics."""
        if len(segment_embeddings) == 0:
            return "OTHER"

        # Simple heuristic based on segment characteristics
        # In practice, you'd train a proper classifier

        # Compute segment statistics
        mean_energy = np.mean(np.linalg.norm(segment_embeddings, axis=1))
        segment_length = len(segment_embeddings)

        # Simple rules (these are just examples)
        if segment_length < 50:  # Short segment
            return "INTRO" if mean_energy < 0.5 else "OUTRO"
        elif segment_length > 200:  # Long segment
            return "VERSE"
        elif mean_energy > 0.8:  # High energy
            return "CHORUS"
        else:
            return "OTHER"

    def predict(self, embeddings: np.ndarray, sr: int = 22050) -> Dict[str, List]:
        """
        Complete prediction pipeline.

        Args:
            embeddings: Feature embeddings
            sr: Sample rate

        Returns:
            Dictionary with boundaries and labels
        """
        boundaries = self.detect_boundaries(embeddings, sr)
        labels = self.predict_labels(embeddings, boundaries, sr)

        return {"boundaries": boundaries, "labels": labels}


class FrameClassifier(nn.Module):
    """Simple frame-level classifier for label prediction."""

    def __init__(self, input_dim: int, num_labels: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class ImprovedBaseline:
    """Improved baseline with trained frame classifier."""

    def __init__(self, input_dim: int, num_labels: int = 8):
        self.input_dim = input_dim
        self.num_labels = num_labels
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

        # Initialize components
        self.ssm_computer = SelfSimilarityMatrix()
        self.frame_classifier = FrameClassifier(input_dim, num_labels)

    def train_frame_classifier(self, embeddings: np.ndarray, labels: np.ndarray):
        """Train the frame classifier."""
        # Convert to torch tensors
        X = torch.from_numpy(embeddings).float()
        y = torch.from_numpy(labels).long()

        # Simple training loop (in practice, use proper training)
        optimizer = torch.optim.Adam(self.frame_classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        self.frame_classifier.train()
        for epoch in range(10):  # Simple training
            optimizer.zero_grad()
            outputs = self.frame_classifier(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict_labels(
        self, embeddings: np.ndarray, boundaries: List[float], sr: int = 22050
    ) -> List[str]:
        """Predict labels using trained frame classifier."""
        if len(boundaries) < 2:
            return ["OTHER"]

        # Convert boundaries to frame indices
        frame_rate = sr / 100
        boundary_frames = [int(b * frame_rate) for b in boundaries]

        labels = []

        self.frame_classifier.eval()
        with torch.no_grad():
            for i in range(len(boundary_frames) - 1):
                start_frame = boundary_frames[i]
                end_frame = boundary_frames[i + 1]

                # Extract segment embeddings
                segment_embeddings = embeddings[start_frame:end_frame]

                if len(segment_embeddings) == 0:
                    labels.append("OTHER")
                    continue

                # Convert to torch tensor
                segment_tensor = torch.from_numpy(segment_embeddings).float()

                # Predict labels for each frame
                frame_logits = self.frame_classifier(segment_tensor)
                frame_preds = torch.argmax(frame_logits, dim=1)

                # Majority vote
                predicted_label = torch.mode(frame_preds).values.item()
                labels.append(self.label_names[predicted_label])

        return labels

    def predict(self, embeddings: np.ndarray, sr: int = 22050) -> Dict[str, List]:
        """Complete prediction pipeline."""
        # Use classic boundary detection
        classic_baseline = ClassicBaseline()
        boundaries = classic_baseline.detect_boundaries(embeddings, sr)

        # Use trained classifier for labels
        labels = self.predict_labels(embeddings, boundaries, sr)

        return {"boundaries": boundaries, "labels": labels}


def main():
    """Test the baseline models."""
    # Create dummy embeddings
    np.random.seed(42)
    embeddings = np.random.randn(1000, 768)  # 1000 frames, 768-dim embeddings

    # Test classic baseline
    print("Testing Classic Baseline:")
    baseline = ClassicBaseline()
    result = baseline.predict(embeddings, sr=22050)
    print(f"Boundaries: {result['boundaries']}")
    print(f"Labels: {result['labels']}")

    # Test improved baseline
    print("\nTesting Improved Baseline:")
    improved = ImprovedBaseline(input_dim=768)

    # Create dummy training data
    dummy_labels = np.random.randint(0, 8, size=1000)
    improved.train_frame_classifier(embeddings, dummy_labels)

    result = improved.predict(embeddings, sr=22050)
    print(f"Boundaries: {result['boundaries']}")
    print(f"Labels: {result['labels']}")


if __name__ == "__main__":
    main()
