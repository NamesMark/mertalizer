#!/usr/bin/env python
"""
Export MERT-v1-95M encoder to TorchScript format for Rust inference via tch.

Creates:
  models/mert_encoder.pt           -- TorchScript traced encoder
  models/mert_encoder_config.json  -- feature extractor config for Rust
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Prevent ml/export/onnx.py from shadowing packages
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MERTEncoderForExport(nn.Module):
    """
    Wrapper around MERT that returns averaged last-N hidden states.

    TorchScript-compatible: uses scripting-friendly ops only.
    """

    def __init__(self, mert_model, n_last_layers: int = 4):
        super().__init__()
        self.mert = mert_model
        self.n_last_layers = n_last_layers
        # Force hidden states output
        self.mert.config.output_hidden_states = True

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_values: normalized audio tensor [batch, samples]
        Returns:
            embeddings: [batch, frames, hidden_dim]
        """
        outputs = self.mert(input_values=input_values, output_hidden_states=True)
        # outputs.hidden_states is a tuple of (num_layers+1) tensors
        # Take last n_last_layers and average
        hidden_states = outputs.hidden_states
        stack = torch.stack(hidden_states[-self.n_last_layers :])  # [N, B, T, D]
        return stack.mean(dim=0)  # [B, T, D]


def export_mert_torchscript(
    output_dir: str = "models",
    model_name: str = "m-a-p/MERT-v1-95M",
    n_last_layers: int = 4,
    test_length: int = 48000,  # 2 seconds at 24kHz
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_path = output_dir / "mert_encoder.pt"
    config_path = output_dir / "mert_encoder_config.json"

    # ---- Load model and feature extractor ----
    from transformers import AutoFeatureExtractor, AutoModel

    logger.info("Loading MERT model: %s", model_name)
    fe = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.config.output_hidden_states = True
    model.eval()

    wrapper = MERTEncoderForExport(model, n_last_layers=n_last_layers)
    wrapper.eval()

    # ---- Save feature extractor config for Rust ----
    fe_config = {
        "sampling_rate": fe.sampling_rate,
        "do_normalize": fe.do_normalize,
        "feature_size": fe.feature_size,
        "padding_value": fe.padding_value,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "n_last_layers": n_last_layers,
        "normalization": "zero_mean_unit_variance",
    }
    with open(config_path, "w") as f:
        json.dump(fe_config, f, indent=2)
    logger.info("Saved feature extractor config to %s", config_path)

    # ---- Prepare dummy input ----
    dummy_audio = np.random.randn(test_length).astype(np.float32)
    inputs = fe(dummy_audio, sampling_rate=fe.sampling_rate, return_tensors="pt")
    input_values = inputs["input_values"]  # [1, test_length]

    logger.info("Input shape: %s", input_values.shape)

    # ---- Reference output ----
    with torch.no_grad():
        ref_output = wrapper(input_values)
    logger.info("Reference output shape: %s (expect [1, frames, 768])", ref_output.shape)

    # ---- Trace the model ----
    logger.info("Tracing MERT encoder...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, input_values)

    # Freeze to inline all ops and remove custom module references
    # This is critical for loading in Rust without Python dependencies
    logger.info("Freezing traced model (inlining all ops)...")
    frozen = torch.jit.freeze(traced)

    # ---- Verify frozen output matches ----
    with torch.no_grad():
        frozen_output = frozen(input_values)

    max_diff = (ref_output - frozen_output).abs().max().item()
    logger.info("Frozen vs eager max diff: %.8f", max_diff)
    assert max_diff < 1e-4, f"Frozen output diverged: max_diff={max_diff}"

    # ---- Test with different length ----
    logger.info("Testing with variable length input...")
    dummy2 = np.random.randn(test_length * 3).astype(np.float32)
    inputs2 = fe(dummy2, sampling_rate=fe.sampling_rate, return_tensors="pt")
    with torch.no_grad():
        ref2 = wrapper(inputs2["input_values"])
        frozen2 = frozen(inputs2["input_values"])
    max_diff2 = (ref2 - frozen2).abs().max().item()
    logger.info("Variable length: ref shape=%s, frozen shape=%s, max_diff=%.8f",
                ref2.shape, frozen2.shape, max_diff2)
    assert max_diff2 < 1e-4, f"Variable length frozen diverged: max_diff={max_diff2}"

    # ---- Verify it loads in a clean context ----
    logger.info("Verifying clean load (simulating Rust)...")
    frozen.save(str(ts_path))
    reloaded = torch.jit.load(str(ts_path))
    with torch.no_grad():
        reloaded_output = reloaded(input_values)
    max_diff3 = (ref_output - reloaded_output).abs().max().item()
    logger.info("Reloaded vs eager max diff: %.8f", max_diff3)
    assert max_diff3 < 1e-4, f"Reloaded output diverged: max_diff={max_diff3}"
    logger.info("Saved TorchScript model to %s", ts_path)

    print(f"\nExport complete:")
    print(f"  TorchScript model: {ts_path} ({ts_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Config:            {config_path}")
    print(f"  Hidden dim:        {model.config.hidden_size}")
    print(f"  Last layers avg:   {n_last_layers}")
    print(f"  Sample rate:       {fe.sampling_rate}")
    return str(ts_path), str(config_path)


def main():
    parser = argparse.ArgumentParser(description="Export MERT encoder to TorchScript")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--model", default="m-a-p/MERT-v1-95M", help="HuggingFace model name")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of last hidden layers to average")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    export_mert_torchscript(
        output_dir=args.output_dir,
        model_name=args.model,
        n_last_layers=args.n_layers,
    )


if __name__ == "__main__":
    main()
