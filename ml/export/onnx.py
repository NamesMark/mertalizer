"""
ONNX export functionality for music structure recognition models.

Exports trained PyTorch models to ONNX format for deployment.
"""

import torch
import torch.onnx
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import argparse
import onnx
import onnxruntime as ort

from modeling.system import MusicStructureModel, ModelConfig

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Exports PyTorch models to ONNX format."""

    def __init__(self, model: MusicStructureModel):
        self.model = model
        self.model.eval()

    def export(
        self,
        output_path: str,
        input_shape: tuple = (1, 220500),
        opset_version: int = 11,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            input_shape: Input tensor shape (batch_size, sequence_length)
            opset_version: ONNX opset version

        Returns:
            Path to exported ONNX model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create dummy input
        dummy_input = torch.randn(input_shape)
        sr = torch.tensor(22050)

        # Define input names
        input_names = ["audio", "sr"]
        output_names = ["boundary_logits", "label_logits"]

        # Export to ONNX
        torch.onnx.export(
            self.model,
            (dummy_input, sr),
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "audio": {0: "batch_size", 1: "sequence_length"},
                "boundary_logits": {0: "batch_size", 1: "sequence_length"},
                "label_logits": {
                    0: "batch_size",
                    1: "sequence_length",
                    2: "num_labels",
                },
            },
        )

        logger.info(f"Exported ONNX model to {output_path}")

        # Verify the exported model
        self._verify_onnx_model(str(output_path))

        return str(output_path)

    def _verify_onnx_model(self, onnx_path: str):
        """Verify the exported ONNX model."""
        try:
            # Load and check the model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model verification passed")

            # Test inference with ONNX Runtime
            self._test_onnx_inference(onnx_path)

        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            raise

    def _test_onnx_inference(self, onnx_path: str):
        """Test inference with ONNX Runtime."""
        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(onnx_path)

            # Create test input
            test_input = np.random.randn(1, 1000).astype(np.float32)
            test_sr = np.array([22050], dtype=np.int64)

            # Run inference
            outputs = session.run(
                ["boundary_logits", "label_logits"],
                {"audio": test_input, "sr": test_sr},
            )

            logger.info(
                f"ONNX inference test passed. Output shapes: {[o.shape for o in outputs]}"
            )

        except Exception as e:
            logger.error(f"ONNX inference test failed: {e}")
            raise


class ONNXInference:
    """ONNX-based inference engine."""

    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(onnx_path)

        # Get input/output info
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(f"Loaded ONNX model from {onnx_path}")
        logger.info(f"Input names: {self.input_names}")
        logger.info(f"Output names: {self.output_names}")

    def predict(self, audio: np.ndarray, sr: int = 22050) -> Dict[str, np.ndarray]:
        """
        Predict using ONNX model.

        Args:
            audio: Audio array
            sr: Sample rate

        Returns:
            Dictionary with predictions
        """
        # Prepare inputs
        audio_input = audio.astype(np.float32)
        sr_input = np.array([sr], dtype=np.int64)

        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_names[0]: audio_input, self.input_names[1]: sr_input},
        )

        return {"boundary_logits": outputs[0], "label_logits": outputs[1]}

    def predict_boundaries(
        self, audio: np.ndarray, sr: int = 22050, threshold: float = 0.5
    ) -> List[float]:
        """Predict boundary times."""
        outputs = self.predict(audio, sr)
        boundary_logits = outputs["boundary_logits"]

        # Apply sigmoid and find peaks
        boundary_probs = 1 / (1 + np.exp(-boundary_logits))  # Sigmoid
        boundary_frames = np.where(boundary_probs > threshold)[1]

        # Convert to time
        frame_rate = sr / 100  # Assuming 100 frames per second
        boundary_times = boundary_frames / frame_rate

        return boundary_times.tolist()

    def predict_labels(
        self, audio: np.ndarray, boundaries: List[float], sr: int = 22050
    ) -> List[str]:
        """Predict labels for segments."""
        outputs = self.predict(audio, sr)
        label_logits = outputs["label_logits"]

        # Convert boundaries to frame indices
        frame_rate = sr / 100
        boundary_frames = [int(b * frame_rate) for b in boundaries]

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

        for i in range(len(boundary_frames) - 1):
            start_frame = boundary_frames[i]
            end_frame = boundary_frames[i + 1]

            # Average logits over segment
            segment_logits = label_logits[0, start_frame:end_frame].mean(axis=0)
            predicted_label = np.argmax(segment_logits)

            labels.append(label_names[predicted_label])

        return labels


def export_model(
    checkpoint_path: str, output_path: str, input_shape: tuple = (1, 220500)
) -> str:
    """
    Export a trained model to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save ONNX model
        input_shape: Input tensor shape

    Returns:
        Path to exported ONNX model
    """
    # Load PyTorch model
    model = MusicStructureModel.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Export to ONNX
    exporter = ONNXExporter(model)
    onnx_path = exporter.export(output_path, input_shape)

    return onnx_path


def main():
    """Main function for ONNX export."""
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to PyTorch checkpoint"
    )
    parser.add_argument("--output", required=True, help="Output ONNX file path")
    parser.add_argument(
        "--input-shape",
        nargs=2,
        type=int,
        default=[1, 220500],
        help="Input shape (batch_size, sequence_length)",
    )
    parser.add_argument(
        "--opset-version", type=int, default=11, help="ONNX opset version"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test ONNX model after export"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Export model
        onnx_path = export_model(args.checkpoint, args.output, tuple(args.input_shape))

        print(f"✓ Successfully exported model to {onnx_path}")

        # Test ONNX model if requested
        if args.test:
            print("Testing ONNX model...")

            # Create test audio
            test_audio = np.random.randn(args.input_shape[1]).astype(np.float32)

            # Test ONNX inference
            onnx_inference = ONNXInference(onnx_path)
            boundaries = onnx_inference.predict_boundaries(test_audio)
            labels = onnx_inference.predict_labels(test_audio, boundaries)

            print(f"✓ ONNX inference test passed")
            print(f"  Predicted boundaries: {len(boundaries)}")
            print(f"  Predicted labels: {len(labels)}")

    except Exception as e:
        print(f"✗ Export failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
