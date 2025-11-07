"""
TorchScript export functionality for music structure recognition models.

Exports trained PyTorch Lightning models to TorchScript format for Rust inference.
"""

import torch
import argparse
import logging
from pathlib import Path
from modeling.system import MusicStructureModel

logger = logging.getLogger(__name__)


class TracedMusicStructureModel(torch.nn.Module):
    """Wrapper for tracing - extracts only inference components."""
    
    def __init__(self, lightning_model: MusicStructureModel):
        super().__init__()
        # Extract only the nn.Module components needed for inference
        self.boundary_head = lightning_model.boundary_head
        self.label_head = lightning_model.label_head
        self.eval()
    
    def forward(self, embeddings: torch.Tensor) -> tuple:
        """
        Forward pass for inference.
        
        Args:
            embeddings: Pre-computed SSL embeddings [batch, seq_len, embed_dim]
            
        Returns:
            (boundary_logits, label_logits) tuple
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        
        boundary_logits = self.boundary_head(embeddings)
        label_logits = self.label_head(embeddings)
        
        return (boundary_logits, label_logits)


def export_torchscript(
    checkpoint_path: str,
    output_path: str,
    batch_size: int = 1,
    seq_len: int = 100,
    embed_dim: int = 768,
) -> str:
    """
    Export Lightning checkpoint to TorchScript.
    
    Args:
        checkpoint_path: Path to Lightning .ckpt file
        output_path: Path to save .pt TorchScript file
        batch_size: Batch size for tracing
        seq_len: Sequence length for tracing
        embed_dim: Embedding dimension (768 for MERT, 1024 for w2v-BERT)
        
    Returns:
        Path to exported TorchScript model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading Lightning checkpoint from {checkpoint_path}")
    
    # Load checkpoint state dict directly to avoid Lightning dependencies
    # Note: weights_only=False is needed for Lightning checkpoints with custom classes
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract hyperparameters if available
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        logger.info(f"Found hyperparameters: {list(hparams.keys())}")
    else:
        hparams = {}
    
    # Create model with loaded config
    model = MusicStructureModel(**hparams)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Wrap model for tracing
    traced_wrapper = TracedMusicStructureModel(model)
    
    # Create dummy input (pre-computed embeddings)
    dummy_embeddings = torch.randn(batch_size, seq_len, embed_dim)
    
    logger.info(f"Tracing model with input shape: {dummy_embeddings.shape}")
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(traced_wrapper, dummy_embeddings)
    
    # Save the traced model
    traced_model.save(str(output_path))
    logger.info(f"✓ Exported TorchScript model to {output_path}")
    
    # Verify the export
    verify_torchscript_model(str(output_path), embed_dim)
    
    return str(output_path)


def verify_torchscript_model(model_path: str, embed_dim: int):
    """Verify the exported TorchScript model."""
    try:
        logger.info("Verifying TorchScript model...")
        
        # Load the model
        loaded_model = torch.jit.load(model_path)
        loaded_model.eval()
        
        # Test with dummy input
        test_input = torch.randn(1, 150, embed_dim)
        
        with torch.no_grad():
            outputs = loaded_model(test_input)
            boundary_logits, label_logits = outputs
        
        logger.info(f"  Boundary logits shape: {boundary_logits.shape}")
        logger.info(f"  Label logits shape: {label_logits.shape}")
        logger.info("✓ TorchScript model verification passed")
        
    except Exception as e:
        logger.error(f"✗ TorchScript model verification failed: {e}")
        raise


def main():
    """Main function for TorchScript export."""
    parser = argparse.ArgumentParser(
        description="Export PyTorch Lightning model to TorchScript"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to PyTorch Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output TorchScript file path (.pt)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=768,
        help="Embedding dimension (768 for MERT, 1024 for w2v-BERT)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for tracing",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=100,
        help="Sequence length for tracing",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Export model
        torchscript_path = export_torchscript(
            args.checkpoint,
            args.output,
            args.batch_size,
            args.seq_len,
            args.embed_dim,
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Successfully exported model to {torchscript_path}")
        print(f"{'='*60}")
        print("\nTo use in Rust:")
        print(f"  export MODEL_PATH={torchscript_path}")
        print(f"  cargo run --release")
        
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

