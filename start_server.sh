#!/bin/bash
# Start Mertalizer server with native Rust inference

set -e

cd "$(dirname "$0")"

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Run ./scripts/setup.sh first"
    exit 1
fi

# Set up libtorch from PyTorch venv
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")/lib

# Check MERT encoder exists
if [ ! -f "models/mert_encoder.pt" ]; then
    echo "MERT encoder not found. Exporting..."
    PYTHONPATH=ml python ml/export/export_mert_onnx.py --output-dir models
fi

# Check head model exists
if [ ! -f "models/mertalizer_traced.pt" ]; then
    echo "Head model not found. Exporting..."
    PYTHONPATH=ml python ml/export/torchscript.py \
        --checkpoint models/checkpoints/mertalizer_final_model.ckpt \
        --output models/mertalizer_traced.pt \
        --embed-dim 768
fi

# Load .env if present
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

export RUST_LOG=${RUST_LOG:-info}

echo "Starting Mertalizer on http://0.0.0.0:${PORT:-3000}"
echo "  MERT encoder: native Rust (TorchScript)"
echo "  Audio features: Python/librosa sidecar"
echo "  Chat: $([ -n "$ANTHROPIC_API_KEY" ] && echo 'enabled' || echo 'disabled (set ANTHROPIC_API_KEY)')"
echo ""

cargo run --release
